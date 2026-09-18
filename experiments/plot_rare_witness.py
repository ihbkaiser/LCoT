#!/usr/bin/env python3
"""Render the rare-witness sampling and fixed-point precision checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rare_witness.png"),
    )
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    if result.get("task") != "rare_witness":
        raise ValueError("results file is not a rare-witness experiment")

    figure, (sampling_axis, precision_axis) = plt.subplots(
        1, 2, figsize=(11.5, 4.4), constrained_layout=True
    )
    colors = plt.get_cmap("tab10")
    for color_index, branches in enumerate(result["sampling_dimensions"]):
        branch_count = 1 << branches
        rows = [
            row for row in result["sampling"] if row["branches"] == branch_count
        ]
        x = [row["inspection_ratio"] for row in rows]
        empirical = [row["empirical_success"] for row in rows]
        lower = [
            row["empirical_success"] - row["ci95_low"] for row in rows
        ]
        upper = [
            row["ci95_high"] - row["empirical_success"] for row in rows
        ]
        color = colors(color_index)
        sampling_axis.errorbar(
            x,
            empirical,
            yerr=[lower, upper],
            fmt="o",
            markersize=4,
            capsize=2,
            color=color,
            label=f"K={branch_count} empirical",
        )
        theory_inspections = list(
            range(0, 4 * branch_count + 1, max(1, branch_count // 100))
        )
        theory_x = [value / branch_count for value in theory_inspections]
        theory_y = [
            1.0 - (1.0 - 1.0 / branch_count) ** value
            for value in theory_inspections
        ]
        sampling_axis.plot(theory_x, theory_y, color=color, linewidth=1.2)

    sampling_axis.set(
        title="Uniform sampling coverage",
        xlabel=r"Inspection budget $r/K$",
        ylabel="Probability of finding the witness",
        xlim=(-0.05, 4.05),
        ylim=(-0.03, 1.03),
    )
    sampling_axis.grid(alpha=0.25)
    sampling_axis.legend(fontsize=8, loc="lower right")

    for normalized, label, marker in (
        (False, "Unnormalized aggregate", "o"),
        (True, "Normalized aggregate", "s"),
    ):
        rows = [row for row in result["precision"] if row["normalized"] == normalized]
        offsets = sorted({row["precision_offset"] for row in rows})
        accuracies = [
            sum(row["accuracy"] for row in rows if row["precision_offset"] == offset)
            / sum(1 for row in rows if row["precision_offset"] == offset)
            for offset in offsets
        ]
        precision_axis.plot(
            offsets,
            accuracies,
            marker=marker,
            linewidth=1.8,
            markersize=6,
            label=label,
        )
    precision_axis.set(
        title="Fixed-point witness decoding",
        xlabel=r"Precision offset $q-\log_2 K$",
        ylabel="Exact decoding accuracy",
        xticks=[-2, -1, 0, 1],
        ylim=(-0.05, 1.05),
    )
    precision_axis.grid(alpha=0.25)
    precision_axis.legend(fontsize=8, loc="lower right")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
