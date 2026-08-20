#!/usr/bin/env python3
"""Launch a sequential ProsQA grid with deterministic config-derived names."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Iterator, Mapping, Sequence

import yaml


GRID_PATH_ALIASES = {
    "state_dim": "finite_state.state_dim",
    "model_bits": "finite_state.model_bits",
    "bits_per_coordinate": "finite_state.bits_per_coordinate",
}

NAME_ALIASES = {
    "state_dim": "d",
    "finite_state.state_dim": "d",
    "model_bits": "mb",
    "finite_state.model_bits": "mb",
    "bits_per_coordinate": "bp",
    "finite_state.bits_per_coordinate": "bp",
    "training_dtype": "dtype",
    "batch_size_training": "bs",
    "batch_size_eval": "ebs",
    "gradient_accumulation_steps": "ga",
    "distributed_strategy": "dist",
    "finite_state.access_mode": "access",
    "lr": "lr",
    "weight_decay": "wd",
}


def _set_nested(config: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    current = config
    for key in keys[:-1]:
        child = current.get(key)
        if child is None:
            child = {}
            current[key] = child
        elif not isinstance(child, dict):
            raise ValueError(
                f"cannot set grid key {path!r}: {key!r} is not a mapping"
            )
        current = child
    current[keys[-1]] = value


def _get_nested(config: Mapping[str, Any], path: str) -> Any:
    current: Any = config
    for key in path.split("."):
        current = current[key]
    return current


def _slug(value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = "none"
    else:
        text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9.+-]+", "-", text).strip("-.")
    return text or "empty"


def _name_tag(key: str, value: Any) -> str:
    alias = NAME_ALIASES.get(key)
    if alias is None:
        alias = _slug(key.split(".")[-1])
    return f"{alias}{_slug(value)}"


def _validate_grid(grid: Mapping[str, Sequence[Any]]) -> None:
    if not isinstance(grid, Mapping) or not grid:
        raise ValueError("grid must be a non-empty mapping")
    for key, values in grid.items():
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(f"grid value for {key!r} must be a non-empty list")
        if not values:
            raise ValueError(f"grid value for {key!r} cannot be empty")


def expand_grid(spec: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    base_path = Path(spec["base_config"])
    base = yaml.safe_load(base_path.read_text())
    grid = spec["grid"]
    _validate_grid(grid)
    grid_keys = list(grid)
    overrides = spec.get("overrides", {}) or {}
    if not isinstance(overrides, Mapping):
        raise ValueError("overrides must be a mapping")
    name_prefix = spec.get("name_prefix", base["name"])

    for combination in itertools.product(*(grid[key] for key in grid_keys)):
        config = copy.deepcopy(base)
        for key, value in overrides.items():
            _set_nested(config, GRID_PATH_ALIASES.get(key, key), value)

        tags = []
        for key, value in zip(grid_keys, combination):
            target_path = GRID_PATH_ALIASES.get(key, key)
            normalized_value = (
                int(value)
                if target_path
                in {
                    "finite_state.state_dim",
                    "finite_state.model_bits",
                    "finite_state.bits_per_coordinate",
                }
                else value
            )
            _set_nested(config, target_path, normalized_value)
            tags.append(_name_tag(key, normalized_value))

            # Keep the legacy precision alias synchronized with model_bits.
            if target_path == "finite_state.model_bits":
                _set_nested(
                    config, "finite_state.bits_per_coordinate", int(value)
                )

        if "state_dim" in grid or "finite_state.state_dim" in grid:
            _set_nested(config, "finite_state.enabled", True)

        # Backward-compatible fixed dtype. A dtype inside grid takes priority.
        dtype_is_gridded = "training_dtype" in grid_keys
        if not dtype_is_gridded and "training_dtype" in spec:
            training_dtype = str(spec["training_dtype"])
            config["training_dtype"] = training_dtype
            tags.append(_slug(training_dtype))

        training_dtype = str(config.get("training_dtype", "float32"))
        if training_dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError("training_dtype must be float16, float32, or bfloat16")

        config["name"] = "-".join([_slug(name_prefix), *tags])
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
    if nproc < 1:
        raise ValueError("nproc_per_node must be at least 1")

    mode = "w" if args.dry_run else "a"
    seen_names = set()
    with manifest_path.open(mode) as manifest:
        for config in expand_grid(spec):
            if config["name"] in seen_names:
                raise ValueError(
                    f"grid generated duplicate run name: {config['name']}"
                )
            seen_names.add(config["name"])
            config_path = config_dir / f"{config['name']}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            checkpoint_dir = Path(config["save_path"]) / config["name"]
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
                "grid_values": {
                    key: _get_nested(config, GRID_PATH_ALIASES.get(key, key))
                    for key in spec["grid"]
                },
                "training_dtype": config["training_dtype"],
                "config": str(config_path),
                "checkpoint_dir": str(checkpoint_dir),
                "nproc_per_node": nproc,
                "command": command,
            }
            if "finite_state" in config:
                record["state_dim"] = config["finite_state"].get("state_dim")
                record["model_bits"] = config["finite_state"].get("model_bits")
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
