"""Configuration-level experiment dispatch."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch

from .data import make_rare_witness_batch
from .rare_witness import (
    HypercubeAggregator,
    decode_hypercube,
    exhaustive_scan,
    fixed_point_round,
    sampled_inspection_prefixes,
)
from .theory import sampling_success
from .training import run_frontier_experiment, run_pointer_experiment


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _sampling_budgets(branches: int) -> tuple[int, ...]:
    return tuple(sorted({0, 1, branches // 4, branches // 2, branches, 2 * branches, 4 * branches}))


def run_rare_witness_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run paired sampling, exhaustive construction, and precision checks."""

    sampling_dimensions = tuple(config.get("sampling_dimensions", [4, 6, 8, 10]))
    construction_dimensions = tuple(
        config.get("construction_dimensions", [4, 6, 8, 10, 12])
    )
    trials = int(config.get("trials", 5000))
    seed = int(config.get("seed", 17))
    chunk_size = int(config.get("construction_chunk_size", 256))
    precision_offsets = tuple(config.get("precision_offsets", [-2, -1, 0, 1]))
    if trials < 1 or chunk_size < 1:
        raise ValueError("trials and construction_chunk_size must be positive")
    if any(dimension < 4 for dimension in sampling_dimensions + construction_dimensions):
        raise ValueError("rare-witness dimensions must be at least four")

    sampling_results = []
    for dimension in sampling_dimensions:
        branches = 1 << dimension
        budgets = _sampling_budgets(branches)
        batch = make_rare_witness_batch(branches, trials, seed=seed + branches)
        generator = torch.Generator().manual_seed(seed + 100_000 + branches)
        curve = sampled_inspection_prefixes(
            batch.markers, budgets, generator=generator
        )
        for index, inspections in enumerate(budgets):
            hits = int(curve.success[index].sum())
            empirical = hits / trials
            theoretical = sampling_success(1.0 / branches, inspections)
            ci_low, ci_high = _wilson_interval(hits, trials)
            variance = theoretical * (1.0 - theoretical) / trials
            standardized_error = (
                abs(empirical - theoretical) / math.sqrt(variance)
                if variance > 0.0
                else 0.0
            )
            sampling_results.append(
                {
                    "dimension": dimension,
                    "branches": branches,
                    "inspections": inspections,
                    "inspection_ratio": inspections / branches,
                    "hits": hits,
                    "trials": trials,
                    "empirical_success": empirical,
                    "theoretical_success": theoretical,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "absolute_error": abs(empirical - theoretical),
                    "standardized_error": standardized_error,
                    "branch_reads_per_instance": inspections,
                    "total_branch_reads": inspections * trials,
                }
            )

    construction_results = []
    precision_results = []
    for dimension in construction_dimensions:
        branches = 1 << dimension
        aggregator = HypercubeAggregator(dimension)
        aggregation_correct = scan_correct = noise_correct = 0
        precision_correct = {
            (normalized, offset): 0
            for normalized in (False, True)
            for offset in precision_offsets
        }
        for start in range(0, branches, chunk_size):
            stop = min(start + chunk_size, branches)
            targets = torch.arange(start, stop)
            markers = torch.zeros(stop - start, branches)
            markers[torch.arange(stop - start), targets] = 1.0
            aggregated = aggregator.aggregate(markers)
            scanned = exhaustive_scan(markers)
            aggregation_correct += int((aggregated.prediction == targets).sum())
            scan_correct += int((scanned.prediction == targets).sum())

            # Moving every one-hot marker across all positions provides the
            # relocation check. This perturbation moves each coordinate toward
            # zero by the largest conveniently representable amount below 1/2.
            perturbed = aggregated.state * (1.0 - 0.499)
            noise_prediction = decode_hypercube(perturbed, signal_scale=1.0)
            noise_correct += int((noise_prediction == targets).sum())

            for normalized in (False, True):
                scale = 1.0 / branches if normalized else 1.0
                state = aggregated.state * scale
                for offset in precision_offsets:
                    fractional_bits = dimension + offset
                    quantized = fixed_point_round(state, fractional_bits)
                    prediction = decode_hypercube(quantized, signal_scale=scale)
                    precision_correct[(normalized, offset)] += int(
                        (prediction == targets).sum()
                    )

        null_markers = torch.zeros(1, branches)
        null_aggregate = aggregator.aggregate(null_markers)
        null_scan = exhaustive_scan(null_markers)
        null_noise = torch.full((1, dimension), 0.499)
        null_noise_prediction = decode_hypercube(null_noise, signal_scale=1.0)
        construction_results.append(
            {
                "dimension": dimension,
                "branches": branches,
                "positions_tested": branches,
                "aggregation_accuracy": aggregation_correct / branches,
                "exhaustive_scan_accuracy": scan_correct / branches,
                "bounded_noise_accuracy": noise_correct / branches,
                "aggregation_null_correct": bool(
                    null_aggregate.prediction.item() == branches
                ),
                "scan_null_correct": bool(null_scan.prediction.item() == branches),
                "null_noise_correct": bool(null_noise_prediction.item() == branches),
                "aggregation_branch_reads": branches,
                "aggregation_scalar_work": branches * dimension,
                "exhaustive_scan_branch_reads": branches,
            }
        )
        for normalized in (False, True):
            for offset in precision_offsets:
                fractional_bits = dimension + offset
                precision_results.append(
                    {
                        "dimension": dimension,
                        "branches": branches,
                        "normalized": normalized,
                        "fractional_bits": fractional_bits,
                        "precision_offset": offset,
                        "accuracy": precision_correct[(normalized, offset)] / branches,
                        "smallest_quantum": 2.0 ** (-fractional_bits),
                        "signal_magnitude": 1.0 / branches if normalized else 1.0,
                    }
                )

    errors = [row["absolute_error"] for row in sampling_results]
    standardized_errors = [row["standardized_error"] for row in sampling_results]
    construction_passed = all(
        row["aggregation_accuracy"] == 1.0
        and row["exhaustive_scan_accuracy"] == 1.0
        and row["bounded_noise_accuracy"] == 1.0
        and row["aggregation_null_correct"]
        and row["scan_null_correct"]
        and row["null_noise_correct"]
        for row in construction_results
    )
    precision_passed = all(
        row["accuracy"]
        == (1.0 if not row["normalized"] or row["precision_offset"] >= 0 else 0.0)
        for row in precision_results
    )
    sampling_passed = max(errors) < 0.04 and max(standardized_errors) < 4.0
    return {
        "task": "rare_witness",
        "evidence_type": "fixed construction and Monte Carlo implementation checks",
        "seed": seed,
        "sampling_dimensions": list(sampling_dimensions),
        "construction_dimensions": list(construction_dimensions),
        "trials_per_sampling_dimension": trials,
        "paired_sampling_prefixes": True,
        "sampling": sampling_results,
        "construction": construction_results,
        "precision": precision_results,
        "work_accounting": [
            {
                "procedure": "uniform_sampling",
                "branch_reads": "r",
                "additional_work": "sample generation, marker checks, output decoding",
            },
            {
                "procedure": "exhaustive_scan",
                "branch_reads": "K",
                "additional_work": "marker checks and witness recording",
            },
            {
                "procedure": "direct_vector_aggregation",
                "branch_reads": "K",
                "additional_work": "K*d scalar aggregation operations",
            },
        ],
        "summary": {
            "sampling_configuration_count": len(sampling_results),
            "mean_absolute_error": sum(errors) / len(errors),
            "maximum_absolute_error": max(errors),
            "maximum_standardized_error": max(standardized_errors),
            "sampling_tolerance": 0.04,
            "sampling_passed": sampling_passed,
            "construction_passed": construction_passed,
            "precision_passed": precision_passed,
        },
        "passed": sampling_passed and construction_passed and precision_passed,
    }


def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    task = config.get("task")
    if task == "frontier_retrieval":
        return run_frontier_experiment(config)
    if task == "pointer_chase":
        return run_pointer_experiment(config)
    if task == "rare_witness":
        return run_rare_witness_experiment(config)
    raise ValueError(f"unknown task: {task!r}")
