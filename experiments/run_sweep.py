#!/usr/bin/env python3
"""Expand a Cartesian YAML sweep and write one JSONL record per run."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from finite_cot.experiments import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text())
    base = dict(spec["base"])
    grid = spec["sweep"]
    keys = list(grid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for values in itertools.product(*(grid[key] for key in keys)):
            config = {**base, **dict(zip(keys, values))}
            if (
                config.get("task") == "frontier_retrieval"
                and int(config.get("state_dim", 0)) == 0
                and int(config.get("transcript_length", 0)) == 0
            ):
                handle.write(
                    json.dumps(
                        {
                            "config": config,
                            "skipped": "no retained channel",
                        }
                    )
                    + "\n"
                )
                continue
            result = run_experiment(config)
            handle.write(json.dumps({"config": config, "result": result}) + "\n")
            handle.flush()
            metrics = result.get("metrics", {})
            ledger = metrics.get("ledger") or {}
            print(
                {key: config[key] for key in keys},
                {
                    "accuracy": metrics.get("accuracy"),
                    "retained_summary_bits": ledger.get("retained_summary_bits"),
                },
            )


if __name__ == "__main__":
    main()
