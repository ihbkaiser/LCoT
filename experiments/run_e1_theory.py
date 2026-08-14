#!/usr/bin/env python3
"""Run one deterministic, training-free E1-Theory capacity point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from finite_cot.frontier_theory import (
    is_complete_theory_result,
    run_e1_theory,
    theory_result_filename,
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
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    for key, value in (
        ("n", args.n),
        ("state_dim", args.state_dim),
        ("bits", args.bits),
        ("seed", args.seed),
        ("target_ratio", args.target_ratio),
        ("test_samples", args.test_samples),
    ):
        if value is not None:
            config[key] = value
    n, d = int(config["n"]), int(config["state_dim"])
    p, seed = int(config.get("bits", 2)), int(config.get("seed", 17))
    output_dir = Path(config.get("output_dir", "results/e1_theory"))
    output = args.output or output_dir / "runs" / theory_result_filename(
        n, d, p, seed
    )
    if output.exists() and not args.force:
        try:
            previous = json.loads(output.read_text())
        except (OSError, json.JSONDecodeError):
            previous = None
        if is_complete_theory_result(previous):
            print(f"Complete result already exists; leaving it unchanged: {output}")
            return
        raise FileExistsError(
            f"refusing to replace existing invalid result without --force: {output}"
        )
    result = run_e1_theory(config)
    _atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
