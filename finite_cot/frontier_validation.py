"""Validation utilities for Experiment 1 (FrontierRate), Section 4.5.

These functions intentionally fail closed when passed ordinary task-level
accuracy rows. FrontierRate validation requires prefix/query-level records and
exact hard-channel identifiers.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

QUERY_REQUIRED = {
    "model", "seed", "split", "n", "d", "p", "L", "T",
    "prefix_id", "query", "label", "prediction", "prob_1",
    "state_key", "transcript_key", "bit_hash", "saturation_rate",
}
PREFIX_REQUIRED = {
    "model", "seed", "split", "n", "d", "p", "L", "T",
    "prefix_id", "channel_codes", "bits",
}
INTERVENTION_REQUIRED = {
    "model", "seed", "n", "d", "p", "L", "T", "prefix_id",
    "patch_channel", "recipient_prediction", "patched_prediction",
    "recipient_label", "donor_label",
}
CONFIG_COLUMNS = ["model", "split", "n", "d", "p", "L", "T"]


def _require(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def binary_entropy(error: float | np.ndarray) -> float | np.ndarray:
    x = np.asarray(error, dtype=float)
    clipped = np.clip(x, 1e-15, 1 - 1e-15)
    value = -(clipped * np.log2(clipped) + (1 - clipped) * np.log2(1 - clipped))
    value = np.where((x <= 0) | (x >= 1), 0.0, value)
    return float(value) if value.ndim == 0 else value


def inverse_binary_entropy(value: float) -> float:
    """Inverse h2 on [0, 1/2], evaluated by stable bisection."""
    target = float(np.clip(value, 0.0, 1.0))
    if target == 0.0:
        return 0.0
    if target == 1.0:
        return 0.5
    lo, hi = 0.0, 0.5
    for _ in range(80):
        mid = (lo + hi) / 2
        if binary_entropy(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def theoretical_error_envelope(rate: float) -> float:
    return inverse_binary_entropy(max(0.0, 1.0 - float(rate)))


def _hierarchical_bootstrap_error(
    frame: pd.DataFrame, draws: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Resample seeds, then prefixes within each sampled seed."""
    seeds = frame["seed"].drop_duplicates().to_numpy()
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        errors = []
        for sampled_seed in rng.choice(seeds, size=len(seeds), replace=True):
            seed_rows = frame[frame["seed"] == sampled_seed]
            prefixes = seed_rows["prefix_id"].drop_duplicates().to_numpy()
            for sampled_prefix in rng.choice(prefixes, size=len(prefixes), replace=True):
                block = seed_rows[seed_rows["prefix_id"] == sampled_prefix]
                errors.extend((block["prediction"].to_numpy() != block["label"].to_numpy()).astype(float))
        samples[draw] = np.mean(errors)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def primary_summary(
    query_rows: pd.DataFrame, *, bootstrap_draws: int = 2000, bootstrap_seed: int = 2026
) -> pd.DataFrame:
    """Mean queried-bit error, hierarchical 95% CI, and theorem envelope."""
    _require(query_rows, QUERY_REQUIRED, "query_rows")
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    rows = query_rows.copy()
    rows["error"] = (rows["prediction"] != rows["label"]).astype(float)
    rows["B"] = rows["d"] * rows["p"] + rows["L"]
    rows["B_over_n"] = rows["B"] / rows["n"]
    rng = np.random.default_rng(bootstrap_seed)
    output = []
    for keys, group in rows.groupby(CONFIG_COLUMNS, dropna=False, sort=True):
        low, high = _hierarchical_bootstrap_error(group, bootstrap_draws, rng)
        B = int(group["B"].iloc[0])
        rate = float(group["B_over_n"].iloc[0])
        output.append({
            **dict(zip(CONFIG_COLUMNS, keys)), "B": B, "B_over_n": rate,
            "queries": len(group), "prefixes": group["prefix_id"].nunique(),
            "mean_bit_error": float(group["error"].mean()),
            "ci_low": float(low), "ci_high": float(high),
            "theoretical_envelope": theoretical_error_envelope(rate),
            "below_envelope_reproducibly": bool(high < theoretical_error_envelope(rate)),
        })
    return pd.DataFrame(output)


def secondary_summary(query_rows: pd.DataFrame) -> pd.DataFrame:
    """NLL, all-continuation exact recovery, collisions, and saturation."""
    _require(query_rows, QUERY_REQUIRED, "query_rows")
    rows = query_rows.copy()
    rows["B"] = rows["d"] * rows["p"] + rows["L"]
    probability = np.where(rows["label"].to_numpy() == 1, rows["prob_1"], 1 - rows["prob_1"])
    rows["nll"] = -np.log(np.clip(probability.astype(float), 1e-12, 1.0))
    output = []
    for keys, group in rows.groupby(CONFIG_COLUMNS, dropna=False, sort=True):
        prefix_stats = group.groupby(["seed", "prefix_id"], sort=False).agg(
            unique_queries=("query", "nunique"), correct=("prediction", lambda x: True),
        )
        # Compute exactness against labels without relying on aggregation alignment.
        exact = group.assign(ok=group.prediction.eq(group.label)).groupby(["seed", "prefix_id"]).agg(
            all_correct=("ok", "all"), unique_queries=("query", "nunique")
        )
        complete = exact["unique_queries"] == int(group["n"].iloc[0])
        exact_complete = exact.loc[complete, "all_correct"]

        prefix_channels = group.drop_duplicates(["seed", "prefix_id"])[
            ["seed", "prefix_id", "state_key", "transcript_key", "bit_hash"]
        ]
        channel_groups = prefix_channels.groupby(["seed", "state_key", "transcript_key"], dropna=False)
        channel_collision = channel_groups["bit_hash"].nunique() > 1

        channel_query = group.groupby(["seed", "state_key", "transcript_key", "query"], dropna=False)["label"].nunique()
        conflict = channel_query > 1
        output.append({
            **dict(zip(CONFIG_COLUMNS, keys)), "B": int(group["B"].iloc[0]),
            "negative_log_likelihood": float(group["nll"].mean()),
            "all_continuations_prefix_fraction": float(complete.mean()),
            "exact_recovery": float(exact_complete.mean()) if len(exact_complete) else np.nan,
            "unique_channel_codes": int(channel_groups.ngroups),
            "channel_collision_rate": float(channel_collision.mean()) if len(channel_collision) else 0.0,
            "collision_conflict_rate": float(conflict.mean()) if len(conflict) else 0.0,
            "saturation_rate": float(group["saturation_rate"].mean()),
        })
    return pd.DataFrame(output)


def frozen_frontier_reconstruction(
    development_prefixes: pd.DataFrame,
    test_prefixes: pd.DataFrame,
    *,
    random_state: int = 2026,
    max_iter: int = 1000,
) -> dict:
    """Fit on development codes, freeze, then reconstruct every test bit."""
    _require(development_prefixes, PREFIX_REQUIRED, "development_prefixes")
    _require(test_prefixes, PREFIX_REQUIRED, "test_prefixes")
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_n = development_prefixes["n"].unique()
    test_n = test_prefixes["n"].unique()
    if len(train_n) != 1 or len(test_n) != 1 or train_n[0] != test_n[0]:
        raise ValueError("frozen reconstruction requires one matching n in development and test")
    X_dev = np.stack(development_prefixes["channel_codes"].map(np.asarray))
    y_dev = np.stack(development_prefixes["bits"].map(np.asarray)).astype(int)
    X_test = np.stack(test_prefixes["channel_codes"].map(np.asarray))
    y_test = np.stack(test_prefixes["bits"].map(np.asarray)).astype(int)
    decoder = make_pipeline(
        StandardScaler(),
        MultiOutputClassifier(LogisticRegression(max_iter=max_iter, random_state=random_state)),
    )
    decoder.fit(X_dev, y_dev)
    prediction = decoder.predict(X_test)
    return {
        "n": int(train_n[0]), "development_prefixes": len(y_dev), "test_prefixes": len(y_test),
        "bit_accuracy": float((prediction == y_test).mean()),
        "bit_error": float((prediction != y_test).mean()),
        "exact_recovery": float(np.all(prediction == y_test, axis=1).mean()),
        "decoder": decoder,
    }


def donor_flip_summary(intervention_rows: pd.DataFrame) -> pd.DataFrame:
    """Donor-consistent output flips, separately for state and transcript patches."""
    _require(intervention_rows, INTERVENTION_REQUIRED, "intervention_rows")
    rows = intervention_rows.copy()
    rows["donor_consistent_flip"] = (
        rows["patched_prediction"].eq(rows["donor_label"])
        & rows["patched_prediction"].ne(rows["recipient_prediction"])
    )
    group_columns = ["model", "n", "d", "p", "L", "T", "patch_channel"]
    return rows.groupby(group_columns, dropna=False).agg(
        pairs=("prefix_id", "size"),
        donor_consistent_flip_rate=("donor_consistent_flip", "mean"),
    ).reset_index()


def decision_report(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    donor_flips: pd.DataFrame | None = None,
    *,
    shape_tolerance: float = 0.02,
    collision_tolerance: float = 0.0,
    donor_flip_threshold: float | None = None,
) -> dict:
    """Apply explicit, preregisterable decision thresholds without hiding failures."""
    if primary.empty or secondary.empty:
        raise ValueError("primary and secondary summaries must be non-empty")
    shape_spread = primary.groupby(["model", "split", "n", "B", "T"])["mean_bit_error"].agg(
        lambda x: float(x.max() - x.min())
    )
    above = secondary[(secondary["B"] / secondary["n"]) >= 1.0]
    report = {
        "no_reproducible_below_envelope": bool(~primary["below_envelope_reproducibly"].any()),
        "max_matched_budget_shape_spread": float(shape_spread.max()) if len(shape_spread) else np.nan,
        "shape_control_pass": bool(len(shape_spread) and shape_spread.max() <= shape_tolerance),
        "above_boundary_collisions_absent": bool(len(above) and (above["collision_conflict_rate"] <= collision_tolerance).all()),
        "thresholds": {
            "shape_tolerance": shape_tolerance,
            "collision_tolerance": collision_tolerance,
            "donor_flip_threshold": donor_flip_threshold,
        },
    }
    if donor_flip_threshold is None:
        report["causal_patch_pass"] = None
        report["causal_patch_note"] = "Set a preregistered donor_flip_threshold before evaluation."
    elif donor_flips is None or donor_flips.empty:
        report["causal_patch_pass"] = False
        report["causal_patch_note"] = "No intervention rows were supplied."
    else:
        report["causal_patch_pass"] = bool(
            (donor_flips["donor_consistent_flip_rate"] >= donor_flip_threshold).all()
        )
    report["strong_positive_result"] = bool(
        report["no_reproducible_below_envelope"]
        and report["shape_control_pass"]
        and report["above_boundary_collisions_absent"]
        and report["causal_patch_pass"] is True
    )
    if not report["no_reproducible_below_envelope"]:
        report["mandatory_action"] = "STOP: run leakage audit before scientific interpretation."
    elif not report["above_boundary_collisions_absent"]:
        report["mandatory_action"] = "Inspect above-boundary collisions/optimization; this is not a sufficiency contradiction."
    else:
        report["mandatory_action"] = "None. Report all component outcomes and thresholds."
    return report
