#!/usr/bin/env python3
"""SUBA: Score-distribution Unsupervised Baseline Alignment.

A deployable diagnostic that predicts ranking collapse vs. threshold drift
using only the detector's score distribution on target benign traffic.

Core idea: if the detector's score CDF on target benign traffic differs
substantially from the source-holdout benign CDF, the model's internal
representation of "benign" has shifted — a necessary condition for
ranking collapse. This is measured via a 2-sample KS test on the
**score space** (not the feature space).

Decision rule:
  - if score_ks >= threshold_d → predict ranking collapse → recommend RETRAIN
  - if score_ks <  threshold_d → predict threshold drift  → recalibrate on target benign
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score


def compute_score_ks(
    source_benign_scores: np.ndarray,
    target_benign_scores: np.ndarray,
) -> tuple[float, float]:
    """2-sample KS test on detector scores for benign traffic only."""
    stat, pvalue = ks_2samp(source_benign_scores, target_benign_scores)
    return float(stat), float(pvalue)


def _load_predictions_chunked(
    path: str, chunksize: int = 500000
) -> dict[tuple[str, str], pd.DataFrame]:
    """Load predictions grouped by (setting_id, eval_role)."""
    grouped: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    for chunk in pd.read_csv(
        path,
        usecols=["setting_id", "eval_role", "label", "score"],
        chunksize=chunksize,
    ):
        for (sid, role), sub in chunk.groupby(["setting_id", "eval_role"], sort=False):
            grouped[(str(sid), str(role))].append(
                sub[["label", "score"]].reset_index(drop=True)
            )
    return {
        key: pd.concat(parts, ignore_index=True) for key, parts in grouped.items()
    }


def run_suba(
    settings: pd.DataFrame,
    pred_groups: dict[tuple[str, str], pd.DataFrame],
    target_fpr: float = 0.01,
) -> pd.DataFrame:
    """Run SUBA diagnostic on all natural-transfer settings."""
    natural = settings[settings["setting_type"] == "natural_transfer"].copy()
    rows = []

    for _, row in natural.iterrows():
        sid = str(row["setting_id"])
        threshold = float(row["threshold"])

        # Get source-holdout and target predictions.
        sh_key = (sid, "source_holdout")
        tgt_key = (sid, "target_transfer")

        if sh_key not in pred_groups or tgt_key not in pred_groups:
            continue

        sh = pred_groups[sh_key]
        tgt = pred_groups[tgt_key]

        sh_labels = sh["label"].astype(int).to_numpy()
        sh_scores = sh["score"].astype(float).to_numpy()
        tgt_labels = tgt["label"].astype(int).to_numpy()
        tgt_scores = tgt["score"].astype(float).to_numpy()

        # SUBA: KS test on benign-only scores.
        sh_benign = sh_scores[sh_labels == 0]
        tgt_benign = tgt_scores[tgt_labels == 0]

        if len(sh_benign) < 10 or len(tgt_benign) < 10:
            continue

        score_ks, score_ks_p = compute_score_ks(sh_benign, tgt_benign)

        # Ground truth: target AUROC.
        if len(np.unique(tgt_labels)) >= 2:
            target_auroc = float(roc_auc_score(tgt_labels, tgt_scores))
        else:
            target_auroc = float("nan")

        # Ground truth diagnosis.
        is_collapsed = 1 if target_auroc < 0.6 else 0

        # Recalibration TPR (benign-only).
        quantile = float(np.clip(1.0 - target_fpr, 0.0, 1.0))
        recal_threshold = float(np.quantile(tgt_benign, quantile, method="higher"))
        recal_preds = tgt_scores >= recal_threshold
        recal_tpr = float(recal_preds[tgt_labels == 1].mean()) if (tgt_labels == 1).sum() > 0 else 0.0
        recal_fpr = float(recal_preds[tgt_labels == 0].mean()) if (tgt_labels == 0).sum() > 0 else 0.0

        # Frozen-threshold metrics.
        frozen_preds = tgt_scores >= threshold
        frozen_tpr = float(frozen_preds[tgt_labels == 1].mean()) if (tgt_labels == 1).sum() > 0 else 0.0

        rows.append({
            "setting_id": sid,
            "source_dataset": str(row["source_dataset"]),
            "target_dataset": str(row["target_dataset"]),
            "model": str(row["model"]),
            "score_ks": score_ks,
            "score_ks_p": score_ks_p,
            "target_auroc": target_auroc,
            "is_collapsed": is_collapsed,
            "frozen_tpr": frozen_tpr,
            "recal_tpr": recal_tpr,
            "recal_fpr": recal_fpr,
            "source_holdout_tpr": float(row["source_holdout_tpr_at_1pct_fpr"]),
        })

    result = pd.DataFrame(rows)

    # Evaluate SUBA as a collapse predictor.
    print("=== SUBA Score-Distribution KS Test ===")
    print(result[["setting_id", "score_ks", "target_auroc", "is_collapsed"]].to_string(index=False))

    # How well does score_ks predict ranking collapse?
    if result["is_collapsed"].nunique() >= 2:
        pred_auroc = roc_auc_score(result["is_collapsed"], result["score_ks"])
        print(f"\n  SUBA AUROC for detecting ranking collapse: {pred_auroc:.3f}")

        # Find optimal threshold (Youden's J).
        from sklearn.metrics import roc_curve
        fpr_curve, tpr_curve, thresholds = roc_curve(result["is_collapsed"], result["score_ks"])
        j_scores = tpr_curve - fpr_curve
        best_idx = int(np.argmax(j_scores))
        best_thresh = float(thresholds[best_idx])
        print(f"  Optimal score_ks threshold (Youden's J): {best_thresh:.4f}")
        print(f"    Sensitivity (collapsed detected): {tpr_curve[best_idx]:.3f}")
        print(f"    Specificity (drift correctly kept): {1.0 - fpr_curve[best_idx]:.3f}")

        # End-to-end: what if we apply SUBA decision rule?
        result["suba_predicts_collapse"] = (result["score_ks"] >= best_thresh).astype(int)
        result["suba_action"] = result["suba_predicts_collapse"].map(
            {0: "recalibrate", 1: "retrain"}
        )

        # For "recalibrate" settings, use recal_tpr. For "retrain", refuse to deploy.
        recal_subset = result[result["suba_action"] == "recalibrate"]
        retrain_subset = result[result["suba_action"] == "retrain"]

        print(f"\n  SUBA decisions:")
        print(f"    Recalibrate: {len(recal_subset)} settings (actual collapsed: {recal_subset['is_collapsed'].sum()})")
        print(f"    Retrain:     {len(retrain_subset)} settings (actual collapsed: {retrain_subset['is_collapsed'].sum()})")

        if len(recal_subset) > 0:
            print(f"    Mean recal TPR (recalibrate group): {recal_subset['recal_tpr'].mean():.3f}")
            print(f"    Mean frozen TPR (recalibrate group): {recal_subset['frozen_tpr'].mean():.3f}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SUBA: Score-distribution Unsupervised Baseline Alignment."
    )
    parser.add_argument("--settings", default="artifacts/benchmark/settings.csv")
    parser.add_argument("--predictions", default="artifacts/benchmark/predictions.csv")
    parser.add_argument("--output-dir", default="artifacts/suba")
    args = parser.parse_args()

    settings = pd.read_csv(args.settings)
    print("Loading predictions...", flush=True)
    pred_groups = _load_predictions_chunked(args.predictions)
    print(f"Loaded {len(pred_groups)} groups.", flush=True)

    result = run_suba(settings, pred_groups)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "suba_results.csv", index=False)
    print(f"\nWrote {len(result)} rows to {output_dir / 'suba_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
