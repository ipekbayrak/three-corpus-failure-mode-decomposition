#!/usr/bin/env python3
"""Ranking-failure diagnosis.

Decomposes transfer failure into ranking failure vs. threshold drift.

For each benchmark setting, this script:
1. Computes target AUROC (measures ranking quality independent of threshold).
2. Computes the target-retuned TPR (oracle threshold on target).
3. Classifies the failure mode:
   - Ranking collapse: target AUROC <= 0.5 (at-chance or inverted ranking).
   - Ranking degraded: 0.5 < target AUROC < 0.6.
   - Threshold drift: target AUROC >= 0.6 but frozen threshold is wrong.
   - Non-collapsed, low TPR loss: target AUROC survives and fixed-threshold TPR loss stays bounded.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from iot_ids.decision import relative_transfer_loss
from iot_ids.metrics import compute_metrics, threshold_at_fpr


def compute_pauc(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01) -> float:
    """Compute standardized partial AUC at FPR <= max_fpr."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores, max_fpr=max_fpr))
    except ValueError:
        return float("nan")


def _median_or_nan(values: np.ndarray) -> float:
    """Return the median of a score vector or NaN when empty."""
    if len(values) == 0:
        return float("nan")
    return float(np.median(values))


def _load_prediction_groups(
    path: str, chunksize: int = 250000
) -> dict[tuple[str, str], pd.DataFrame]:
    """Load predictions grouped by (setting_id, eval_role) in chunks."""
    grouped_chunks: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    for chunk in pd.read_csv(
        path,
        usecols=["setting_id", "eval_role", "label", "score"],
        chunksize=chunksize,
    ):
        chunk["setting_id"] = chunk["setting_id"].astype(str)
        chunk["eval_role"] = chunk["eval_role"].astype(str)
        chunk["label"] = chunk["label"].astype(int)
        chunk["score"] = chunk["score"].astype(float)
        for (setting_id, eval_role), subset in chunk.groupby(
            ["setting_id", "eval_role"], sort=False
        ):
            grouped_chunks[(str(setting_id), str(eval_role))].append(
                subset[["label", "score"]].reset_index(drop=True)
            )
    return {
        key: pd.concat(parts, ignore_index=True)
        for key, parts in grouped_chunks.items()
    }


def diagnose_ranking(
    settings: pd.DataFrame,
    prediction_groups: dict[tuple[str, str], pd.DataFrame],
    target_fpr: float = 0.01,
    ranking_floor: float = 0.6,
) -> pd.DataFrame:
    """Diagnose ranking failure vs. threshold drift for each setting.

    Parameters
    ----------
    ranking_floor : float
        AUROC below this value is classified as ranking failure.
    """
    natural = settings.loc[
        settings["setting_type"].astype(str) == "natural_transfer"
    ].copy()
    for column in ["target_auroc_ci_lo", "target_auroc_ci_hi"]:
        if column not in natural.columns:
            natural[column] = float("nan")

    rows = []
    for row in natural.itertuples(index=False):
        sid = str(row.setting_id)
        target_key = (sid, "target_transfer")
        source_holdout_key = (sid, "source_holdout")

        if target_key not in prediction_groups:
            continue

        target = prediction_groups[target_key]
        target_labels = target["label"].astype(int).to_numpy()
        target_scores = target["score"].astype(float).to_numpy()

        # Target AUROC and low-FPR pAUC (ranking quality).
        n_unique_labels = len(np.unique(target_labels))
        if n_unique_labels < 2:
            target_auroc = float("nan")
        else:
            target_auroc = float(roc_auc_score(target_labels, target_scores))
        target_pauc = compute_pauc(target_labels, target_scores, max_fpr=target_fpr)

        # Oracle-retuned threshold on target.
        try:
            oracle_threshold = threshold_at_fpr(target_labels, target_scores, target_fpr)
            oracle_metrics = compute_metrics(target_labels, target_scores, oracle_threshold)
            oracle_target_tpr = float(oracle_metrics["tpr_at_1pct_fpr"])
        except ValueError:
            oracle_threshold = float("nan")
            oracle_target_tpr = float("nan")

        # Source holdout AUROC for comparison.
        source_holdout_auroc = float("nan")
        source_holdout_benign_scores = np.array([], dtype=float)
        source_holdout_attack_scores = np.array([], dtype=float)
        if source_holdout_key in prediction_groups:
            sh = prediction_groups[source_holdout_key]
            sh_labels = sh["label"].astype(int).to_numpy()
            sh_scores = sh["score"].astype(float).to_numpy()
            source_holdout_benign_scores = sh_scores[sh_labels == 0]
            source_holdout_attack_scores = sh_scores[sh_labels == 1]
            if len(np.unique(sh_labels)) >= 2:
                source_holdout_auroc = float(roc_auc_score(sh_labels, sh_scores))
            source_holdout_pauc = compute_pauc(sh_labels, sh_scores, max_fpr=target_fpr)
        else:
            source_holdout_pauc = float("nan")

        # Benign-only score statistics (for covariate shift signal).
        target_benign_scores = target_scores[target_labels == 0]
        target_attack_scores = target_scores[target_labels == 1]

        # Score overlap (how separable are attacks from benign on target?).
        if len(target_benign_scores) > 0 and len(target_attack_scores) > 0:
            # Mann-Whitney U statistic = AUROC.
            score_separation = target_auroc
        else:
            score_separation = float("nan")

        source_holdout_benign_score_median = _median_or_nan(source_holdout_benign_scores)
        source_holdout_attack_score_median = _median_or_nan(source_holdout_attack_scores)
        target_benign_score_median = _median_or_nan(target_benign_scores)
        target_attack_score_median = _median_or_nan(target_attack_scores)
        source_holdout_median_score_gap = (
            source_holdout_attack_score_median - source_holdout_benign_score_median
            if not np.isnan(source_holdout_attack_score_median)
            and not np.isnan(source_holdout_benign_score_median)
            else float("nan")
        )
        target_median_score_gap = (
            target_attack_score_median - target_benign_score_median
            if not np.isnan(target_attack_score_median)
            and not np.isnan(target_benign_score_median)
            else float("nan")
        )

        # Classification of failure mode and ranking orientation.
        fixed_tpr = float(row.target_tpr_at_1pct_fpr)
        fixed_target_realized_fpr = float(row.target_realized_fpr)
        source_holdout_tpr = float(row.source_holdout_tpr_at_1pct_fpr)
        fixed_loss = relative_transfer_loss(fixed_tpr, source_holdout_tpr)

        if np.isnan(target_auroc):
            diagnosis = "insufficient_data"
            ranking_orientation = "insufficient_data"
        elif np.isclose(target_auroc, 0.50, atol=1e-12, rtol=0.0):
            ranking_orientation = "at_chance"
            diagnosis = "ranking_collapse"
        elif target_auroc < 0.50:
            ranking_orientation = "inverted"
            diagnosis = "ranking_collapse"
        else:
            ranking_orientation = "positive"
            if target_auroc < ranking_floor:
                diagnosis = "ranking_degraded"
            elif fixed_loss > 0.5:
                # Good ranking but bad TPR -> threshold is the problem.
                diagnosis = "threshold_drift"
            else:
                diagnosis = "noncollapsed_low_tpr_loss"

        # AUROC drop from source holdout.
        auroc_drop = float("nan")
        if not np.isnan(source_holdout_auroc) and not np.isnan(target_auroc):
            auroc_drop = source_holdout_auroc - target_auroc

        rows.append(
            {
                "setting_id": sid,
                "source_dataset": str(row.source_dataset),
                "target_dataset": str(row.target_dataset),
                "model": str(row.model),
                "source_holdout_auroc": source_holdout_auroc,
                "source_holdout_pauc_at_1pct": source_holdout_pauc,
                "target_auroc": target_auroc,
                "target_auroc_ci_lo": float(row.target_auroc_ci_lo),
                "target_auroc_ci_hi": float(row.target_auroc_ci_hi),
                "target_pauc_at_1pct": target_pauc,
                "source_holdout_benign_score_median": source_holdout_benign_score_median,
                "source_holdout_attack_score_median": source_holdout_attack_score_median,
                "target_benign_score_median": target_benign_score_median,
                "target_attack_score_median": target_attack_score_median,
                "source_holdout_median_score_gap": source_holdout_median_score_gap,
                "target_median_score_gap": target_median_score_gap,
                "auroc_drop": auroc_drop,
                "fixed_threshold": float(row.threshold),
                "fixed_target_tpr": fixed_tpr,
                "fixed_target_realized_fpr": fixed_target_realized_fpr,
                "oracle_target_threshold": oracle_threshold,
                "oracle_target_tpr": oracle_target_tpr,
                "source_holdout_tpr": source_holdout_tpr,
                "fixed_relative_tpr_drop": max(fixed_loss, 0.0),
                "oracle_relative_tpr_drop": max(
                    relative_transfer_loss(oracle_target_tpr, source_holdout_tpr), 0.0
                )
                if not np.isnan(oracle_target_tpr)
                else float("nan"),
                "retuned_tpr_gain": oracle_target_tpr - fixed_tpr
                if not np.isnan(oracle_target_tpr)
                else float("nan"),
                "diagnosis": diagnosis,
                "ranking_orientation": ranking_orientation,
                "score_separation": score_separation,
            }
        )

    result = pd.DataFrame(rows)

    # Summary statistics by diagnosis category.
    print("\n=== Ranking Diagnosis Summary ===")
    print(result.groupby("diagnosis").agg(
        count=("setting_id", "count"),
        median_target_auroc=("target_auroc", "median"),
        median_target_pauc_at_1pct=("target_pauc_at_1pct", "median"),
        median_fixed_tpr=("fixed_target_tpr", "median"),
        median_fixed_target_realized_fpr=("fixed_target_realized_fpr", "median"),
        median_oracle_tpr=("oracle_target_tpr", "median"),
        median_retuned_gain=("retuned_tpr_gain", "median"),
    ).to_string())

    print("\n=== Diagnosis by Direction ===")
    direction_diag = result.groupby(
        ["source_dataset", "target_dataset", "diagnosis"]
    ).size().unstack(fill_value=0)
    print(direction_diag.to_string())

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose ranking failure vs. threshold drift in benchmark settings."
    )
    parser.add_argument(
        "--settings",
        default="artifacts/benchmark/settings.csv",
        help="Path to benchmark settings CSV.",
    )
    parser.add_argument(
        "--predictions",
        default="artifacts/benchmark/predictions.csv",
        help="Path to benchmark predictions CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/ranking_diagnosis",
        help="Directory for ranking diagnosis outputs.",
    )
    args = parser.parse_args()

    settings = pd.read_csv(args.settings)
    print("Loading prediction groups (this may take a moment for large files)...", flush=True)
    prediction_groups = _load_prediction_groups(args.predictions)
    print(f"Loaded {len(prediction_groups)} prediction groups.", flush=True)

    result = diagnose_ranking(settings, prediction_groups)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ranking_diagnosis.csv"
    result.to_csv(output_path, index=False)
    print(f"\nWrote {len(result)} ranking diagnosis rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
