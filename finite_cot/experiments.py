"""Configuration-level experiment dispatch."""

from __future__ import annotations

from typing import Any, Dict

import torch

from .data import make_rare_witness_batch
from .rare_witness import HypercubeAggregator, sampled_inspection
from .resources import ResourceLedger
from .theory import sampling_success
from .training import run_frontier_experiment, run_pointer_experiment


def run_rare_witness_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    dimension = int(config.get("dimension", 8))
    branches = 1 << dimension
    trials = int(config.get("trials", 10000))
    seed = int(config.get("seed", 17))
    batch = make_rare_witness_batch(branches, trials, seed)
    aggregator = HypercubeAggregator(dimension)
    full = aggregator.aggregate(batch.markers)
    full_accuracy = (full.prediction == batch.target).float().mean().item()

    inspection_results = []
    for inspections in config.get("inspections", [1, branches // 4, branches, 2 * branches]):
        generator = torch.Generator().manual_seed(seed + int(inspections))
        sampled = sampled_inspection(
            batch.markers, int(inspections), generator=generator
        )
        empirical = sampled.success.float().mean().item()
        inspection_results.append(
            {
                "inspections": int(inspections),
                "empirical_success": empirical,
                "theoretical_success": sampling_success(1.0 / branches, int(inspections)),
                "absolute_error": abs(
                    empirical - sampling_success(1.0 / branches, int(inspections))
                ),
            }
        )

    precision_results = []
    for fractional_bits in config.get(
        "fractional_bits", list(range(max(0, dimension - 3), dimension + 3))
    ):
        output = aggregator.aggregate(
            batch.markers,
            normalized=True,
            fractional_bits=int(fractional_bits),
        )
        precision_results.append(
            {
                "fractional_bits": int(fractional_bits),
                "accuracy": (output.prediction == batch.target).float().mean().item(),
                "smallest_quantum": 2.0 ** (-int(fractional_bits)),
                "signal_magnitude": 1.0 / branches,
            }
        )

    ledger = ResourceLedger(
        state_dim=dimension,
        bits_per_coordinate=1,
        access_model="full_hypercube_aggregation",
        input_reads=full.marker_accesses,
        scalar_additions=full.scalar_additions,
        notes="persistent identity is compact; aggregation work is exponential",
    )
    return {
        "task": "rare_witness",
        "seed": seed,
        "dimension": dimension,
        "branches": branches,
        "trials": trials,
        "full_aggregation_accuracy": full_accuracy,
        "full_aggregation_ledger": ledger.to_dict(),
        "sampled_inspection": inspection_results,
        "normalized_precision": precision_results,
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
