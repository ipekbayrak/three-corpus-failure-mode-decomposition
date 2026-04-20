#!/usr/bin/env python3
"""Covariate shift analysis: measures benign feature drift separate from attack novelty.

The reviewer demands that the paper analyze feature drift as a component of failure,
rather than lumping all shift into attack-label novelty. This script:
1. Computes per-feature KS test statistics between source and target benign traffic.
2. Reports aggregate covariate shift scores.
3. Correlates covariate shift with FPR drift (separate from TPR drop).
4. Produces data for the two-panel decomposition figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from iot_ids.config import load_config
from iot_ids.experiment import build_split_manifest, prepare_feature_frames
from iot_ids.features import FEATURE_COLUMNS


def compute_benign_covariate_shift(
    source_benign: pd.DataFrame,
    target_benign: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:
    """Compute per-feature KS statistics between source and target benign traffic.

    Returns a dict with per-feature KS statistics and aggregate measures.
    """
    ks_stats = {}
    for col in feature_columns:
        source_vals = source_benign[col].dropna().to_numpy()
        target_vals = target_benign[col].dropna().to_numpy()
        if len(source_vals) < 2 or len(target_vals) < 2:
            ks_stats[col] = float("nan")
            continue
        stat, _ = scipy_stats.ks_2samp(source_vals, target_vals)
        ks_stats[col] = float(stat)

    valid_stats = [v for v in ks_stats.values() if not np.isnan(v)]
    aggregate = {
        "mean_ks": float(np.mean(valid_stats)) if valid_stats else float("nan"),
        "max_ks": float(np.max(valid_stats)) if valid_stats else float("nan"),
        "median_ks": float(np.median(valid_stats)) if valid_stats else float("nan"),
        "n_features_shifted_gt_0_3": sum(1 for v in valid_stats if v > 0.3),
        "n_features_shifted_gt_0_5": sum(1 for v in valid_stats if v > 0.5),
    }
    return {**{f"ks_{col}": v for col, v in ks_stats.items()}, **aggregate}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze benign covariate shift across transfer directions."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to benchmark YAML config.",
    )
    parser.add_argument(
        "--settings",
        default="artifacts/benchmark/settings.csv",
        help="Path to benchmark settings CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/covariate_shift",
        help="Directory for covariate shift outputs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    settings = pd.read_csv(args.settings)

    # Load and prepare feature frames.
    frames = prepare_feature_frames(config)
    split_frames, _ = build_split_manifest(frames, config)

    # Get unique direction pairs from settings.
    natural = settings.loc[settings["setting_type"] == "natural_transfer"].copy()
    directions = (
        natural[["source_dataset", "target_dataset"]]
        .drop_duplicates()
        .values.tolist()
    )

    feature_columns = FEATURE_COLUMNS
    direction_rows = []
    feature_rows = []

    for source_name, target_name in directions:
        source_test = split_frames[source_name].test
        target_test = split_frames[target_name].test

        source_benign = source_test.loc[source_test["label"] == 0]
        target_benign = target_test.loc[target_test["label"] == 0]

        shift_results = compute_benign_covariate_shift(
            source_benign, target_benign, feature_columns
        )

        # Aggregate direction-level row.
        direction_rows.append(
            {
                "source_dataset": source_name,
                "target_dataset": target_name,
                "n_source_benign": len(source_benign),
                "n_target_benign": len(target_benign),
                "mean_ks": shift_results["mean_ks"],
                "max_ks": shift_results["max_ks"],
                "median_ks": shift_results["median_ks"],
                "n_features_shifted_gt_0_3": shift_results["n_features_shifted_gt_0_3"],
                "n_features_shifted_gt_0_5": shift_results["n_features_shifted_gt_0_5"],
            }
        )

        # Per-feature rows.
        for col in feature_columns:
            feature_rows.append(
                {
                    "source_dataset": source_name,
                    "target_dataset": target_name,
                    "feature": col,
                    "ks_statistic": shift_results.get(f"ks_{col}", float("nan")),
                }
            )

    direction_df = pd.DataFrame(direction_rows)
    feature_df = pd.DataFrame(feature_rows)

    # Merge covariate shift with per-direction transfer metrics.
    direction_metrics = (
        natural.groupby(["source_dataset", "target_dataset"])
        .agg(
            mean_target_realized_fpr=("target_realized_fpr", "mean"),
            mean_relative_tpr_drop=("relative_tpr_drop", "mean"),
            mean_oracle_alpha=("oracle_alpha", "mean"),
            mean_fpr_severity=("fpr_severity", "mean"),
        )
        .reset_index()
    )
    merged = direction_df.merge(
        direction_metrics, on=["source_dataset", "target_dataset"], how="left"
    )

    # Compute correlations.
    print("\n=== Covariate Shift Summary by Direction ===")
    print(merged.to_string(index=False))

    if len(merged) >= 3:
        ks_fpr_corr = merged["mean_ks"].corr(merged["mean_fpr_severity"], method="spearman")
        ks_tpr_corr = merged["mean_ks"].corr(merged["mean_relative_tpr_drop"], method="spearman")
        alpha_tpr_corr = merged["mean_oracle_alpha"].corr(
            merged["mean_relative_tpr_drop"], method="spearman"
        )
        print(f"\nSpearman correlations (direction-level, n={len(merged)}):")
        print(f"  benign_covariate_shift (mean KS) vs FPR severity: {ks_fpr_corr:.3f}")
        print(f"  benign_covariate_shift (mean KS) vs TPR drop:     {ks_tpr_corr:.3f}")
        print(f"  oracle_alpha vs TPR drop:                         {alpha_tpr_corr:.3f}")

    # Save outputs.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    direction_path = output_dir / "direction_covariate_shift.csv"
    merged.to_csv(direction_path, index=False)
    print(f"\nWrote direction-level shift to {direction_path}")

    feature_path = output_dir / "feature_covariate_shift.csv"
    feature_df.to_csv(feature_path, index=False)
    print(f"Wrote per-feature shift to {feature_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
