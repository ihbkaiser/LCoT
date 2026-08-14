#!/usr/bin/env python3
"""Run and aggregate the resumable 63-run E1 capacity sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import traceback
from typing import Any, Dict, Iterable, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from finite_cot.frontier_nl import (
    is_complete_e1_result,
    result_filename,
    run_e1_experiment,
)
from finite_cot.theory import fano_error_lower_bound


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def expand_sweep(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    base = dict(spec["base"])
    sweep = spec["sweep"]
    for n in sweep["n"]:
        for ratio in sweep["target_ratio"]:
            retained_bits = round(int(n) * float(ratio))
            if retained_bits % 2:
                raise ValueError(
                    f"n={n}, ratio={ratio} gives R={retained_bits}, not divisible by p=2"
                )
            d = retained_bits // 2
            if d * 2 != retained_bits:
                raise AssertionError("invalid E1 dp accounting")
            for seed in sweep["seed"]:
                yield {
                    **base,
                    "n": int(n),
                    "state_dim": d,
                    "bits": 2,
                    "seed": int(seed),
                    "target_ratio": float(ratio),
                }


def load_complete_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    results = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if is_complete_e1_result(value):
            results.append(value)
    return results


def aggregate(output_dir: Path) -> List[Dict[str, Any]]:
    runs = load_complete_runs(output_dir / "runs")
    _atomic_text(
        output_dir / "results.jsonl",
        "".join(json.dumps(run, sort_keys=True) + "\n" for run in runs),
    )
    grouped: Dict[tuple[int, float], List[Dict[str, Any]]] = {}
    for run in runs:
        key = (int(run["n"]), float(run["target_ratio"]))
        grouped.setdefault(key, []).append(run)

    rows: List[Dict[str, Any]] = []
    for (n, target_ratio), values in sorted(grouped.items()):
        train_errors = [float(value["train_error"]) for value in values]
        dev_errors = [float(value["dev_error"]) for value in values]
        test_errors = [float(value["test_error"]) for value in values]
        train_accuracies = [float(value["train_accuracy"]) for value in values]
        dev_accuracies = [float(value["dev"]["accuracy"]) for value in values]
        test_accuracies = [float(value["accuracy"]) for value in values]
        exemplar = values[0]
        rows.append(
            {
                "n": n,
                "target_ratio": target_ratio,
                "R": int(exemplar["R"]),
                "d": int(exemplar["d"]),
                "p": int(exemplar["p"]),
                "actual_ratio": float(exemplar["actual_ratio"]),
                "number_seeds": len(values),
                "seeds": ";".join(
                    str(value["seed"])
                    for value in sorted(values, key=lambda x: x["seed"])
                ),
                "mean_train_accuracy": statistics.fmean(train_accuracies),
                "std_train_accuracy": (
                    statistics.stdev(train_accuracies)
                    if len(train_accuracies) > 1 else 0.0
                ),
                "mean_train_error": statistics.fmean(train_errors),
                "std_train_error": (
                    statistics.stdev(train_errors)
                    if len(train_errors) > 1 else 0.0
                ),
                "mean_dev_accuracy": statistics.fmean(dev_accuracies),
                "std_dev_accuracy": (
                    statistics.stdev(dev_accuracies)
                    if len(dev_accuracies) > 1 else 0.0
                ),
                "mean_dev_error": statistics.fmean(dev_errors),
                "std_dev_error": (
                    statistics.stdev(dev_errors)
                    if len(dev_errors) > 1 else 0.0
                ),
                "mean_test_accuracy": statistics.fmean(test_accuracies),
                "std_test_accuracy": (
                    statistics.stdev(test_accuracies)
                    if len(test_accuracies) > 1 else 0.0
                ),
                "mean_test_error": statistics.fmean(test_errors),
                "std_test_error": (
                    statistics.stdev(test_errors)
                    if len(test_errors) > 1 else 0.0
                ),
                "chance_error": 0.5,
                "e1_theory_deterministic_codec_error": max(
                    0, n - int(exemplar["R"])
                ) / (2 * n),
                "fano_error_lower_bound": fano_error_lower_bound(
                    float(exemplar["actual_ratio"])
                ),
            }
        )
    fields = [
        "n", "target_ratio", "R", "d", "p", "actual_ratio",
        "number_seeds", "seeds", "mean_train_accuracy", "std_train_accuracy",
        "mean_train_error", "std_train_error", "mean_dev_accuracy",
        "std_dev_accuracy", "mean_dev_error", "std_dev_error",
        "mean_test_accuracy", "std_test_accuracy", "mean_test_error",
        "std_test_error", "chance_error", "e1_theory_deterministic_codec_error",
        "fano_error_lower_bound",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "aggregate.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_dir / "aggregate.csv")
    return rows


def plot(rows: List[Dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for n in sorted({int(row["n"]) for row in rows}):
        selected = sorted(
            (row for row in rows if int(row["n"]) == n),
            key=lambda row: float(row["actual_ratio"]),
        )
        axis.errorbar(
            [row["actual_ratio"] for row in selected],
            [row["mean_test_error"] for row in selected],
            yerr=[row["std_test_error"] for row in selected],
            marker="o", capsize=3, label=f"n = {n}",
        )
    xs = [index / 400 for index in range(0, 501)]
    axis.axhline(
        0.5, color="tab:red", linestyle="--", linewidth=1.2,
        label="chance error",
    )
    axis.plot(
        xs, [max(0.0, 1.0 - x) / 2.0 for x in xs],
        color="tab:blue", linestyle="-.", linewidth=1.4,
        label="E1-Theory deterministic codec",
    )
    axis.plot(
        xs, [fano_error_lower_bound(x) for x in xs], "k--",
        linewidth=1.5, label="Fano lower bound",
    )
    axis.axvline(1.0, color="gray", linestyle=":", linewidth=1.3, label="R/n = 1")
    axis.set(xlabel="Retained capacity R/n", ylabel="Held-out membership error",
             xlim=(0.2, 1.3), ylim=(-0.02, 0.52))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text())
    output_dir = Path(spec.get("output_dir", "results/e1_frontier_nl"))
    runs_dir = output_dir / "runs"
    failures_dir = output_dir / "failures"
    runs_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        for config in expand_sweep(spec):
            filename = result_filename(
                config["n"], config["state_dim"], 2, config["seed"]
            )
            run_path = runs_dir / filename
            expected = {
                "n": config["n"], "d": config["state_dim"], "p": 2,
                "R": config["state_dim"] * 2, "seed": config["seed"],
            }
            if run_path.exists():
                try:
                    previous = json.loads(run_path.read_text())
                except (OSError, json.JSONDecodeError):
                    previous = None
                if is_complete_e1_result(previous, expected):
                    print(f"skip complete: {run_path.name}")
                    continue
                error = {
                    "status": "failed", "config": config,
                    "error": f"invalid existing run file was not overwritten: {run_path}",
                }
                _atomic_json(failures_dir / filename, error)
                print(error["error"], file=sys.stderr)
                continue
            try:
                print(f"run: {run_path.name}")
                result = run_e1_experiment(config)
                _atomic_json(run_path, result)
            except Exception as exception:
                failure = {
                    "status": "failed", "config": config,
                    "error_type": type(exception).__name__,
                    "error": str(exception), "traceback": traceback.format_exc(),
                }
                _atomic_json(failures_dir / filename, failure)
                print(f"failed: {run_path.name}: {exception}", file=sys.stderr)

    rows = aggregate(output_dir)
    plot(rows, output_dir / "error_vs_capacity.png")
    print(
        f"Aggregated {len(load_complete_runs(runs_dir))} complete runs "
        f"into {output_dir}"
    )


if __name__ == "__main__":
    main()
