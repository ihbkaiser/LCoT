#!/usr/bin/env python3
"""Run one finite-CoT experiment from a YAML file."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    result = run_experiment(config)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    output = args.output or (
        Path(config.get("output", "results")) / f"{config['task']}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
