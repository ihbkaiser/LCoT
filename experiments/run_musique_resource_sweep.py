#!/usr/bin/env python3
"""Launch MuSiQue finite-CoT runs while swapping compute and state resources."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

import yaml


def expand_sweep(spec):
    base = yaml.safe_load(Path(spec["base_config"]).read_text())
    prefix = spec.get("name_prefix", base["name"])
    axes = itertools.product(
        spec["budgets"],
        spec["bits_per_coordinate"],
        spec["latent_steps"],
        spec.get("access_modes", ["sealed_prefix"]),
    )
    for budget, bits, steps, access_mode in axes:
        if int(budget) % int(bits):
            continue
        config = dict(base)
        finite_state = dict(config["finite_state"])
        finite_state.update(
            state_dim=int(budget) // int(bits),
            bits_per_coordinate=int(bits),
            access_mode=access_mode,
        )
        config["finite_state"] = finite_state
        config["latent_steps"] = int(steps)
        config["max_latent_stage"] = int(steps)
        config["name"] = (
            f"{prefix}-b{budget}-p{bits}-t{steps}-{access_mode.replace('_', '-')}"
        )
        yield config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    spec = yaml.safe_load(args.spec.read_text())
    output_dir = Path(spec.get("output_dir", "results/musique_resource_sweep"))
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    mode = "w" if args.dry_run else "a"
    with manifest_path.open(mode) as manifest:
        for config in expand_sweep(spec):
            config_path = config_dir / f"{config['name']}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={int(spec.get('nproc_per_node', 1))}",
                "run.py",
                str(config_path),
            ]
            record = {
                "name": config["name"],
                "budget_bits": (
                    config["finite_state"]["state_dim"]
                    * config["finite_state"]["bits_per_coordinate"]
                ),
                "state_dim": config["finite_state"]["state_dim"],
                "bits_per_coordinate": config["finite_state"][
                    "bits_per_coordinate"
                ],
                "latent_steps": config["latent_steps"],
                "access_mode": config["finite_state"]["access_mode"],
                "config": str(config_path),
                "command": command,
            }
            if args.dry_run:
                record["status"] = "dry_run"
            else:
                result = subprocess.run(command, check=False)
                record["returncode"] = result.returncode
                record["status"] = "complete" if result.returncode == 0 else "failed"
            print(json.dumps(record))
            manifest.write(json.dumps(record) + "\n")
            manifest.flush()
            if record["status"] == "failed" and not args.continue_on_error:
                raise SystemExit(record["returncode"])


if __name__ == "__main__":
    main()
