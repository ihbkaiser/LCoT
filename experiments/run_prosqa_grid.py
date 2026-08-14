#!/usr/bin/env python3
"""Launch a sequential ProsQA grid over state dimension and model QAT bits."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterator

import yaml


def expand_grid(spec: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    base_path = Path(spec["base_config"])
    base = yaml.safe_load(base_path.read_text())
    grid = spec["grid"]
    dimensions = grid["state_dim"]
    model_bits = grid["model_bits"]
    training_dtype = str(spec.get("training_dtype", "float32"))
    if training_dtype not in {"float16", "float32", "bfloat16"}:
        raise ValueError("training_dtype must be float16, float32, or bfloat16")
    name_prefix = spec.get("name_prefix", base["name"])

    for dimension, bits in itertools.product(dimensions, model_bits):
        config = dict(base)
        finite_state = dict(config.get("finite_state", {}))
        finite_state.update(
            {
                "enabled": True,
                "state_dim": int(dimension),
                "model_bits": int(bits),
                # Keep the legacy key synchronized for ledgers and old tooling.
                "bits_per_coordinate": int(bits),
            }
        )
        config["finite_state"] = finite_state
        config["training_dtype"] = training_dtype
        config["name"] = (
            f"{name_prefix}-d{int(dimension)}"
            f"-mb{int(bits)}-{training_dtype}"
        )
        yield config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    spec = yaml.safe_load(args.spec.read_text())
    output_dir = Path(spec.get("output_dir", "results/prosqa_grid"))
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    nproc = int(spec.get("nproc_per_node", 1))

    mode = "w" if args.dry_run else "a"
    with manifest_path.open(mode) as manifest:
        for config in expand_grid(spec):
            config_path = config_dir / f"{config['name']}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={nproc}",
                "run.py",
                str(config_path),
            ]
            record = {
                "name": config["name"],
                "state_dim": config["finite_state"]["state_dim"],
                "model_bits": config["finite_state"]["model_bits"],
                "training_dtype": config["training_dtype"],
                "config": str(config_path),
                "command": command,
            }
            print(json.dumps(record))
            if args.dry_run:
                record["status"] = "dry_run"
                manifest.write(json.dumps(record) + "\n")
                continue

            result = subprocess.run(command, check=False)
            record["returncode"] = result.returncode
            record["status"] = "complete" if result.returncode == 0 else "failed"
            manifest.write(json.dumps(record) + "\n")
            manifest.flush()
            if result.returncode != 0 and not args.continue_on_error:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
