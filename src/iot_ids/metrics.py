from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(labels: Iterable[int], scores: Iterable[float], target_fpr: float) -> float:
    """Choose a tie-safe threshold under the `score >= threshold` rule.

    The returned cutoff is the smallest threshold whose realized FPR does not
    exceed `target_fpr`. If a tied negative-score block would straddle the
    budget, the cutoff is moved strictly above that score so the tied negatives
    remain negative.
    """
    labels_array = np.asarray(list(labels))
    scores_array = np.asarray(list(scores), dtype=float)
    negative_scores = scores_array[labels_array == 0]
    if negative_scores.size == 0:
        raise ValueError("Cannot choose threshold without negative validation scores")
    clipped_target = float(np.clip(target_fpr, 0.0, 1.0))
    if clipped_target >= 1.0:
        return float(np.nextafter(np.min(negative_scores), -np.inf))

    unique_scores, counts = np.unique(np.sort(negative_scores), return_counts=True)
    strictly_above = negative_scores.size - np.cumsum(counts)
    realized_fpr = strictly_above / negative_scores.size
    candidate_idx = np.flatnonzero(realized_fpr <= clipped_target)
    if candidate_idx.size == 0:
        return float(np.nextafter(np.max(unique_scores), np.inf))
    chosen_idx = int(candidate_idx[0])
    chosen_score = float(unique_scores[chosen_idx])
    if chosen_idx + 1 < len(unique_scores):
        next_score = float(unique_scores[chosen_idx + 1])
        return float(chosen_score + ((next_score - chosen_score) / 2.0))
    return float(np.nextafter(chosen_score, np.inf))


def realized_fpr(labels: Iterable[int], scores: Iterable[float], threshold: float) -> float:
    labels_array = np.asarray(list(labels))
    scores_array = np.asarray(list(scores), dtype=float)
    negatives = labels_array == 0
    if negatives.sum() == 0:
        return float("nan")
    return float(((scores_array >= threshold) & negatives).sum() / negatives.sum())


def tpr_at_threshold(labels: Iterable[int], scores: Iterable[float], threshold: float) -> float:
    labels_array = np.asarray(list(labels))
    scores_array = np.asarray(list(scores), dtype=float)
    positives = labels_array == 1
    if positives.sum() == 0:
        return float("nan")
    return float(((scores_array >= threshold) & positives).sum() / positives.sum())


def compute_metrics(
    labels: Iterable[int],
    scores: Iterable[float],
    threshold: float,
) -> dict[str, float]:
    labels_array = np.asarray(list(labels))
    scores_array = np.asarray(list(scores), dtype=float)
    metrics = {
        "tpr_at_1pct_fpr": tpr_at_threshold(labels_array, scores_array, threshold),
        "realized_fpr": realized_fpr(labels_array, scores_array, threshold),
    }
    if len(np.unique(labels_array)) < 2:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(labels_array, scores_array))
        metrics["auprc"] = float(average_precision_score(labels_array, scores_array))
    return metrics


def grouped_bootstrap_metric(
    frame: pd.DataFrame,
    metric_name: str,
    threshold: float,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if "group_id" not in frame.columns:
        raise ValueError("Grouped bootstrap requires a group_id column")

    rng = np.random.default_rng(seed)
    working = frame[["group_id", "label", "score"]].copy()
    working["group_id"] = working["group_id"].astype(str)
    labels = working["label"].astype(int).to_numpy()
    scores = working["score"].astype(float).to_numpy()
    group_codes, _ = pd.factorize(working["group_id"], sort=True)
    grouped = [
        (
            group["label"].astype(int).to_numpy(),
            group["score"].astype(float).to_numpy(),
        )
        for _, group in working.groupby("group_id", sort=True)
    ]
    groups = np.arange(len(grouped))
    draws: list[float] = []
    for _ in range(n_boot):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        if metric_name == "auroc":
            sampled_counts = np.bincount(sampled_groups, minlength=len(groups))
            row_weights = sampled_counts[group_codes]
            if len(np.unique(labels)) < 2 or row_weights.sum() == 0:
                draws.append(float("nan"))
            else:
                draws.append(
                    float(roc_auc_score(labels, scores, sample_weight=row_weights))
                )
            continue
        sampled_labels = np.concatenate([grouped[idx][0] for idx in sampled_groups])
        sampled_scores = np.concatenate([grouped[idx][1] for idx in sampled_groups])
        metrics = compute_metrics(sampled_labels, sampled_scores, threshold)
        draws.append(metrics[metric_name])
    return tuple(np.nanquantile(draws, [0.025, 0.975]))


def grouped_bootstrap_threshold_intervals(
    frame: pd.DataFrame,
    threshold: float,
    n_boot: int,
    seed: int,
    interval: tuple[float, float] = (0.025, 0.975),
) -> dict[str, float]:
    if "group_id" not in frame.columns:
        raise ValueError("Grouped bootstrap requires a group_id column")

    working = frame[["group_id", "label", "score"]].copy()
    working["predicted_positive"] = working["score"].astype(float) >= float(threshold)
    working["positive"] = working["label"].astype(int) == 1
    working["negative"] = ~working["positive"]
    working["tp_hit"] = (working["predicted_positive"] & working["positive"]).astype(int)
    working["fp_hit"] = (working["predicted_positive"] & working["negative"]).astype(int)
    grouped = (
        working.groupby("group_id", sort=True)
        .agg(
            tp=("tp_hit", "sum"),
            pos=("positive", "sum"),
            fp=("fp_hit", "sum"),
            neg=("negative", "sum"),
        )
        .reset_index(drop=True)
    )
    if grouped.empty:
        return {
            "tpr_ci_lo": float("nan"),
            "tpr_ci_hi": float("nan"),
            "realized_fpr_ci_lo": float("nan"),
            "realized_fpr_ci_hi": float("nan"),
        }

    tp = grouped["tp"].to_numpy(dtype=float)
    pos = grouped["pos"].to_numpy(dtype=float)
    fp = grouped["fp"].to_numpy(dtype=float)
    neg = grouped["neg"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(grouped), size=(n_boot, len(grouped)))

    tp_draw = tp[sample_idx].sum(axis=1)
    pos_draw = pos[sample_idx].sum(axis=1)
    fp_draw = fp[sample_idx].sum(axis=1)
    neg_draw = neg[sample_idx].sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        tpr_draw = np.divide(
            tp_draw,
            pos_draw,
            out=np.full_like(tp_draw, np.nan, dtype=float),
            where=pos_draw > 0,
        )
        fpr_draw = np.divide(
            fp_draw,
            neg_draw,
            out=np.full_like(fp_draw, np.nan, dtype=float),
            where=neg_draw > 0,
        )

    lower, upper = interval
    return {
        "tpr_ci_lo": float(np.nanquantile(tpr_draw, lower)),
        "tpr_ci_hi": float(np.nanquantile(tpr_draw, upper)),
        "realized_fpr_ci_lo": float(np.nanquantile(fpr_draw, lower)),
        "realized_fpr_ci_hi": float(np.nanquantile(fpr_draw, upper)),
    }
