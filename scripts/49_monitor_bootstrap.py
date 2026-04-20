#!/usr/bin/env python3
"""Bootstrap uncertainty for monitor AUROC/AUPRC on the 24-cell benchmark.

The reviewer correctly notes that AUROC 0.579 computed on 24 settings
with no uncertainty is not a robust empirical result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from iot_ids.decision import add_operational_metrics


MONITOR_SCORES = [
    "alpha_hat",
    "proxy_a_distance",
    "avg_wasserstein_1d",
]


def bootstrap_monitor_metrics(
    settings: pd.DataFrame,
    score_column: str,
    severity_column: str = "operational_severity",
    catastrophic_column: str = "catastrophic_operational_failure",
    n_boot: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap AUROC/AUPRC/Spearman for a monitor on the settings."""
    rng = np.random.default_rng(seed)
    values = settings[score_column].astype(float).to_numpy()
    severity = settings[severity_column].astype(float).to_numpy()
    catastrophic = settings[catastrophic_column].astype(int).to_numpy()
    n = len(settings)

    auroc_draws = []
    auprc_draws = []
    spearman_draws = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_vals = values[idx]
        boot_cat = catastrophic[idx]
        boot_sev = severity[idx]

        # AUROC/AUPRC.
        if len(np.unique(boot_cat)) >= 2 and len(np.unique(boot_vals)) >= 2:
            auroc_draws.append(float(roc_auc_score(boot_cat, boot_vals)))
            auprc_draws.append(float(average_precision_score(boot_cat, boot_vals)))
        # Spearman.
        s_vals = pd.Series(boot_vals)
        s_sev = pd.Series(boot_sev)
        if s_vals.nunique() >= 2 and s_sev.nunique() >= 2:
            spearman_draws.append(float(s_vals.corr(s_sev, method="spearman")))

    def ci(draws: list[float]) -> tuple[float, float, float]:
        if not draws:
            return (float("nan"), float("nan"), float("nan"))
        arr = np.array(draws)
        return (float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))

    auroc_mean, auroc_lo, auroc_hi = ci(auroc_draws)
    auprc_mean, auprc_lo, auprc_hi = ci(auprc_draws)
    spear_mean, spear_lo, spear_hi = ci(spearman_draws)

    return {
        "catastrophic_auroc": auroc_mean,
        "catastrophic_auroc_ci_lo": auroc_lo,
        "catastrophic_auroc_ci_hi": auroc_hi,
        "catastrophic_auprc": auprc_mean,
        "catastrophic_auprc_ci_lo": auprc_lo,
        "catastrophic_auprc_ci_hi": auprc_hi,
        "spearman_vs_severity": spear_mean,
        "spearman_ci_lo": spear_lo,
        "spearman_ci_hi": spear_hi,
        "n_boot": n_boot,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute bootstrap uncertainty for monitor AUROC/AUPRC."
    )
    parser.add_argument("--settings", default="artifacts/benchmark/settings.csv")
    parser.add_argument("--output-dir", default="artifacts/monitor_uncertainty")
    args = parser.parse_args()

    settings = pd.read_csv(args.settings)
    settings = add_operational_metrics(settings)
    natural = settings[settings["setting_type"] == "natural_transfer"].copy()

    # Add source-val risk column.
    natural["risk_from_source_validation_tpr"] = (
        -natural["source_validation_tpr_at_1pct_fpr"].astype(float)
    )

    all_scores = MONITOR_SCORES + ["risk_from_source_validation_tpr"]
    rows = []

    for subset_name, subset_df in [("all_natural", natural)]:
        for score_name in all_scores:
            if score_name not in subset_df.columns:
                continue
            metrics = bootstrap_monitor_metrics(subset_df, score_name)
            rows.append({
                "subset": subset_name,
                "score_name": score_name,
                **metrics,
            })

    result = pd.DataFrame(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "monitor_bootstrap.csv", index=False)

    print("=== Monitor Bootstrap Uncertainty (n=5000) ===")
    for _, r in result.iterrows():
        print(
            f"  {r['score_name']:40s}  "
            f"AUROC={r['catastrophic_auroc']:.3f} [{r['catastrophic_auroc_ci_lo']:.3f}, {r['catastrophic_auroc_ci_hi']:.3f}]  "
            f"Spearman={r['spearman_vs_severity']:.3f} [{r['spearman_ci_lo']:.3f}, {r['spearman_ci_hi']:.3f}]"
        )

    print(f"\nWrote to {output_dir / 'monitor_bootstrap.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
