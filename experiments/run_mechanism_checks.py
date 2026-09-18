#!/usr/bin/env python3
"""Deterministic construction checks; these are not learned evidence."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from finite_cot.experiments import run_rare_witness_experiment
from finite_cot.models import BinaryPointerMachine, boolean_bfs


def reference_bfs(adjacency: torch.Tensor, source: int, steps: int) -> torch.Tensor:
    """Classical queue-based BFS, truncated after ``steps`` edges."""

    distance = [-1] * adjacency.shape[0]
    distance[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        if distance[u] == steps:
            continue
        for v in adjacency[u].nonzero(as_tuple=False).view(-1).tolist():
            if distance[v] == -1:
                distance[v] = distance[u] + 1
                queue.append(v)

    result = torch.zeros(adjacency.shape[0], dtype=torch.bool)
    result[[v for v, depth in enumerate(distance) if depth != -1]] = True
    return result


def check_bfs() -> dict:
    """Reproduce the 81 configurations reported in the paper."""

    graph_sizes = (32, 64, 128)
    edge_probabilities = (0.02, 0.05, 0.10)
    propagation_depths = (4, 8, 16)
    seeds = (17, 42, 137)
    cases = exact_agreements = correct_decisions = total_decisions = 0
    failures = []
    for nodes in graph_sizes:
        for edge_probability in edge_probabilities:
            for seed in seeds:
                # Reuse one graph across depths so the depth sweep changes only
                # the number of recurrent updates.
                generator = torch.Generator().manual_seed(seed)
                adjacency = (
                    torch.rand(nodes, nodes, generator=generator) < edge_probability
                )
                source = torch.tensor([seed % nodes])
                for steps in propagation_depths:
                    predicted = boolean_bfs(adjacency.unsqueeze(0), source, steps)[0]
                    expected = reference_bfs(adjacency, int(source), steps)
                    matches = predicted.cpu() == expected
                    correct_decisions += int(matches.sum())
                    total_decisions += nodes
                    exact = bool(matches.all())
                    exact_agreements += int(exact)
                    cases += 1
                    if not exact:
                        failures.append(
                            {
                                "nodes": nodes,
                                "edge_probability": edge_probability,
                                "steps": steps,
                                "seed": seed,
                                "mismatched_vertices": int((~matches).sum()),
                            }
                        )
    return {
        "graph_sizes": list(graph_sizes),
        "edge_probabilities": list(edge_probabilities),
        "propagation_depths": list(propagation_depths),
        "seeds": list(seeds),
        "cases": cases,
        "correct_decisions": correct_decisions,
        "total_decisions": total_decisions,
        "decision_accuracy": correct_decisions / total_decisions,
        "exact_frontier_agreements": exact_agreements,
        "frontier_vector_agreement": exact_agreements / cases,
        "failures": failures,
        "passed": cases == exact_agreements,
    }


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
    """Run the paper's paired sampling, construction, and precision protocol."""

    return run_rare_witness_experiment({})


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
