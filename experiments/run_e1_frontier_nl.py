#!/usr/bin/env python3
"""Run one learned, sealed-prefix Frontier Retrieval-NL E1 configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from finite_cot.frontier_nl import (
    is_complete_e1_result,
    result_filename,
    run_e1_experiment,
)


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--n", type=int)
    parser.add_argument("--state-dim", type=int)
    parser.add_argument("--bits", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-ratio", type=float)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--dev-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--force", action="store_true",
        help="replace an existing result (never used by the sweep driver)",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    for key, value in (
        ("n", args.n), ("state_dim", args.state_dim), ("bits", args.bits),
        ("seed", args.seed), ("target_ratio", args.target_ratio),
        ("train_samples", args.train_samples), ("dev_samples", args.dev_samples),
        ("test_samples", args.test_samples), ("batch_size", args.batch_size),
        ("gradient_accumulation_steps", args.gradient_accumulation_steps),
        ("max_steps", args.max_steps),
    ):
        if value is not None:
            config[key] = value
    finite = dict(config.get("finite_state") or {})
    if args.bits is not None:
        finite["bits_per_coordinate"] = args.bits
    config["finite_state"] = finite

    n, d = int(config["n"]), int(config["state_dim"])
    p, seed = int(config.get("bits", 2)), int(config.get("seed", 17))
    output_dir = Path(config.get("output_dir", "results/e1_frontier_nl"))
    output = args.output or output_dir / "runs" / result_filename(n, d, p, seed)
    if output.exists() and not args.force:
        try:
            previous = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError):
            previous = None
        if is_complete_e1_result(previous):
            print(f"Complete result already exists; leaving it unchanged: {output}")
            return
        raise FileExistsError(
            f"refusing to replace existing non-valid result without --force: {output}"
        )

    result = run_e1_experiment(config)
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
