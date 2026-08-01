#!/usr/bin/env python3
"""Deterministic construction checks; these are not learned evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from finite_cot.data import make_rare_witness_batch
from finite_cot.models import BinaryPointerMachine, boolean_bfs
from finite_cot.rare_witness import HypercubeAggregator, sampled_inspection
from finite_cot.theory import sampling_success


def reference_bfs(adjacency: torch.Tensor, source: int, steps: int) -> torch.Tensor:
    reached = {source}
    for _ in range(steps):
        reached |= {
            v
            for u in list(reached)
            for v in adjacency[u].nonzero(as_tuple=False).view(-1).tolist()
        }
    result = torch.zeros(adjacency.shape[0], dtype=torch.bool)
    result[list(reached)] = True
    return result


def check_bfs() -> dict:
    cases = agreements = 0
    for nodes in (16, 32):
        for edge_probability in (0.02, 0.10):
            for steps in (2, 4, 8):
                for seed in (17, 42, 137):
                    generator = torch.Generator().manual_seed(seed + nodes + steps)
                    adjacency = (
                        torch.rand(nodes, nodes, generator=generator) < edge_probability
                    )
                    source = torch.tensor([seed % nodes])
                    predicted = boolean_bfs(adjacency.unsqueeze(0), source, steps)[0]
                    expected = reference_bfs(adjacency, int(source), steps)
                    agreements += int(torch.equal(predicted.cpu(), expected))
                    cases += 1
    return {"cases": cases, "exact_frontier_agreements": agreements, "passed": cases == agreements}


def check_pointer() -> dict:
    nodes, depth = 16, 4
    function = torch.arange(nodes).add(1).clamp_max(nodes - 1).unsqueeze(0)
    source = torch.tensor([8])
    expected = torch.tensor([12])
    enough = BinaryPointerMachine(4).run(function, source, depth)
    too_narrow = BinaryPointerMachine(3).run(function, source, depth)
    too_shallow = BinaryPointerMachine(4).run(function, source, depth - 1)
    return {
        "d4_t4_exact": bool(torch.equal(enough.vertex, expected)),
        "d3_collision_control_fails": bool(not torch.equal(too_narrow.vertex, expected)),
        "d4_t3_depth_control_fails": bool(not torch.equal(too_shallow.vertex, expected)),
        "queries_at_threshold": enough.total_queries,
    }


def check_rare_witness() -> dict:
    dimension = 6
    branches = 1 << dimension
    batch = make_rare_witness_batch(branches, 4096, seed=17)
    full = HypercubeAggregator(dimension).aggregate(batch.markers)
    generator = torch.Generator().manual_seed(42)
    sampled = sampled_inspection(batch.markers, branches, generator=generator)
    empirical = sampled.success.float().mean().item()
    theoretical = sampling_success(1.0 / branches, branches)
    return {
        "full_aggregation_accuracy": (full.prediction == batch.target).float().mean().item(),
        "sampled_empirical_at_r_eq_k": empirical,
        "sampled_theory_at_r_eq_k": theoretical,
        "absolute_error": abs(empirical - theoretical),
        "passed": abs(empirical - theoretical) < 0.04,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "evidence_type": "deterministic construction and formula checks",
        "bfs": check_bfs(),
        "pointer": check_pointer(),
        "rare_witness": check_rare_witness(),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
