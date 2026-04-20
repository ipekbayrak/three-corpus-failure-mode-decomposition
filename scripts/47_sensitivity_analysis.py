#!/usr/bin/env python3
"""Sensitivity analysis: headline statistics with/without weak source models.

The reviewer notes that ToN-IoT logistic regression with holdout TPR 0.080
contaminates the global median. This script reports metrics both including
and excluding settings where source competence is below a floor.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from iot_ids.decision import add_operational_metrics


def compute_pAUC(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.05) -> float:
    """Compute partial AUC at FPR <= max_fpr, normalized to [0, 1]."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores, max_fpr=max_fpr))
    except ValueError:
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report benchmark with/without weak source models and compute pAUC."
    )
    parser.add_argument("--settings", default="artifacts/benchmark/settings.csv")
    parser.add_argument("--output-dir", default="artifacts/sensitivity")
    args = parser.parse_args()

    settings = pd.read_csv(args.settings)
    settings = add_operational_metrics(settings)
    natural = settings[settings["setting_type"] == "natural_transfer"].copy()

    # Full benchmark.
    full_stats = {
        "subset": "all_settings",
        "n_settings": len(natural),
        "median_target_tpr": natural["target_tpr_at_1pct_fpr"].median(),
        "median_realized_fpr": natural["target_realized_fpr"].median(),
        "median_target_auroc": natural["target_auroc"].median() if "target_auroc" in natural.columns else float("nan"),
        "n_catastrophic": int(natural["catastrophic_operational_failure"].sum()),
    }

    # Excluding weak source models (source holdout TPR < 0.20).
    strong = natural[natural["source_holdout_tpr_at_1pct_fpr"] >= 0.20].copy()
    strong_stats = {
        "subset": "source_tpr_ge_0.20",
        "n_settings": len(strong),
        "median_target_tpr": strong["target_tpr_at_1pct_fpr"].median(),
        "median_realized_fpr": strong["target_realized_fpr"].median(),
        "median_target_auroc": float("nan"),
        "n_catastrophic": int(strong["catastrophic_operational_failure"].sum()),
    }

    # Excluding weak source models (source holdout TPR < 0.50).
    competent = natural[natural["source_holdout_tpr_at_1pct_fpr"] >= 0.50].copy()
    competent_stats = {
        "subset": "source_tpr_ge_0.50",
        "n_settings": len(competent),
        "median_target_tpr": competent["target_tpr_at_1pct_fpr"].median(),
        "median_realized_fpr": competent["target_realized_fpr"].median(),
        "median_target_auroc": float("nan"),
        "n_catastrophic": int(competent["catastrophic_operational_failure"].sum()),
    }

    result = pd.DataFrame([full_stats, strong_stats, competent_stats])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "source_competence_sensitivity.csv", index=False)

    print("=== Source Competence Sensitivity ===")
    print(result.to_string(index=False))
    print(f"\nWrote to {output_dir / 'source_competence_sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
