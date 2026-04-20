#!/usr/bin/env python3
"""Coverage threshold sensitivity analysis.

Sweeps the ontology coverage floor from 0.60 to 1.00 and reports how the
primary IoT subset size and oracle novelty Spearman correlation change.
This addresses the reviewer concern that the 0.80 threshold is post-hoc
cherry-picking.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def sweep_coverage_thresholds(
    settings: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    """Evaluate oracle novelty association at each coverage floor."""
    natural = settings.loc[settings["setting_type"] == "natural_transfer"].copy()
    rows = []
    for floor in thresholds:
        subset = natural.loc[
            (natural["source_mapped_attack_coverage"].astype(float) >= floor)
            & (natural["target_mapped_attack_coverage"].astype(float) >= floor)
        ].copy()

        n_settings = len(subset)
        n_directions = subset[["source_dataset", "target_dataset"]].drop_duplicates().shape[0]

        if n_settings < 3 or subset["oracle_alpha"].nunique() < 2:
            spearman = float("nan")
            p_value = float("nan")
        else:
            spearman, p_value = scipy_stats.spearmanr(
                subset["oracle_alpha"].astype(float),
                subset["relative_tpr_drop"].astype(float),
            )

        mean_alpha = subset["oracle_alpha"].astype(float).mean() if n_settings > 0 else float("nan")
        mean_tpr_drop = (
            subset["relative_tpr_drop"].astype(float).mean() if n_settings > 0 else float("nan")
        )

        rows.append(
            {
                "coverage_floor": floor,
                "n_settings": n_settings,
                "n_directions": n_directions,
                "spearman_alpha_vs_tpr_drop": float(spearman),
                "p_value": float(p_value),
                "mean_oracle_alpha": float(mean_alpha),
                "mean_relative_tpr_drop": float(mean_tpr_drop),
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep ontology coverage floor and report oracle novelty sensitivity."
    )
    parser.add_argument(
        "--settings",
        default="artifacts/benchmark/settings.csv",
        help="Path to benchmark settings CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/coverage_sensitivity",
        help="Directory for output.",
    )
    args = parser.parse_args()

    settings = pd.read_csv(args.settings)
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    result = sweep_coverage_thresholds(settings, thresholds)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "coverage_sensitivity.csv"
    result.to_csv(output_path, index=False)

    print("=== Coverage Threshold Sensitivity ===")
    print(result.to_string(index=False))
    print(f"\nWrote {len(result)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
