"""Operational Transfer Readiness Score (OTRS).

Combines existing unlabeled transfer monitors into a composite score
for deploy/abstain decisions. The combination rule is pre-specified:
a logistic combination trained with leave-one-direction-out CV on the
existing benchmark settings, or a simpler max-rule fallback.

This module does not introduce new estimators. It combines monitors
that already exist in the benchmark artifacts (alpha_hat, proxy_a_distance,
avg_wasserstein_1d, source_val_risk) into a single operational score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# Columns used as component scores for the composite readiness score.
MONITOR_COLUMNS = [
    "alpha_hat",
    "proxy_a_distance",
    "avg_wasserstein_1d",
    "risk_from_source_validation_tpr",
]


def _prepare_features(settings: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract and impute monitor features from settings."""
    working = settings.copy()
    if "risk_from_source_validation_tpr" not in working.columns:
        working["risk_from_source_validation_tpr"] = (
            -working["source_validation_tpr_at_1pct_fpr"].astype(float)
        )
    features = working[MONITOR_COLUMNS].astype(float).copy()
    # Impute NaN with column median (rare edge case).
    for col in features.columns:
        median = features[col].median()
        features[col] = features[col].fillna(median if not np.isnan(median) else 0.0)
    return features.to_numpy(), list(features.columns)


def compute_max_rule_otrs(settings: pd.DataFrame) -> pd.Series:
    """Simple max-rule composite: standardize each monitor, take the max.

    This is the non-learned fallback. No fitting required.
    """
    features, _ = _prepare_features(settings)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    return pd.Series(np.max(standardized, axis=1), index=settings.index, name="otrs_max_rule")


def compute_mean_rank_otrs(settings: pd.DataFrame) -> pd.Series:
    """Mean-rank aggregation: rank each monitor within the benchmark, average ranks.

    This is non-parametric, requires no fitting, and is robust to tiny N.
    The intuition is that a setting flagged as high-risk by multiple monitors
    (high rank across all) is more likely to be genuinely dangerous than one
    flagged by only a single monitor.
    """
    features, cols = _prepare_features(settings)
    n = len(settings)
    if n == 0:
        return pd.Series([], dtype=float, name="otrs_mean_rank")

    # Rank each column (high value = high risk = high rank).
    ranks = np.zeros_like(features)
    for j in range(features.shape[1]):
        col_vals = features[:, j]
        # scipy rankdata: average ties, ascending (1 = smallest).
        from scipy.stats import rankdata
        ranks[:, j] = rankdata(col_vals, method="average") / n

    # Mean rank across monitors.
    mean_ranks = ranks.mean(axis=1)
    return pd.Series(mean_ranks, index=settings.index, name="otrs_mean_rank")


def compute_vote_otrs(
    settings: pd.DataFrame,
    quantile_threshold: float = 0.75,
) -> pd.Series:
    """Vote-based aggregation: count how many monitors flag a setting as high-risk.

    A setting is flagged by a monitor if its value exceeds the quantile_threshold
    within that monitor's distribution. The composite score is the fraction of
    monitors that flag it.
    """
    features, cols = _prepare_features(settings)
    n = len(settings)
    if n == 0:
        return pd.Series([], dtype=float, name="otrs_vote")

    flags = np.zeros_like(features)
    for j in range(features.shape[1]):
        col_vals = features[:, j]
        threshold = np.nanquantile(col_vals, quantile_threshold)
        flags[:, j] = (col_vals >= threshold).astype(float)

    vote_fraction = flags.mean(axis=1)
    return pd.Series(vote_fraction, index=settings.index, name="otrs_vote")


def evaluate_readiness_score(
    settings: pd.DataFrame,
    score_column: str,
    severity_column: str = "operational_severity",
    catastrophic_column: str = "catastrophic_operational_failure",
) -> dict[str, float]:
    """Evaluate a readiness score against operational targets."""
    values = settings[score_column].astype(float)
    severity = settings[severity_column].astype(float)
    catastrophic = settings[catastrophic_column].astype(int)

    result: dict[str, float] = {}

    # Spearman correlation with operational severity.
    if values.nunique(dropna=True) >= 2 and severity.nunique(dropna=True) >= 2:
        result["spearman_vs_severity"] = float(values.corr(severity, method="spearman"))
    else:
        result["spearman_vs_severity"] = float("nan")

    # AUROC and AUPRC for catastrophic failure detection.
    if catastrophic.nunique() >= 2 and values.nunique(dropna=True) >= 2:
        result["catastrophic_auroc"] = float(roc_auc_score(catastrophic, values))
        result["catastrophic_auprc"] = float(average_precision_score(catastrophic, values))
    else:
        result["catastrophic_auroc"] = float("nan")
        result["catastrophic_auprc"] = float("nan")

    return result


def build_otrs_comparison(
    settings: pd.DataFrame,
    subsets: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build a full comparison table: individual monitors + OTRS variants.

    Parameters
    ----------
    settings : pd.DataFrame
        Full benchmark settings with operational metrics already computed.
    subsets : dict[str, pd.DataFrame] or None
        Named subsets to evaluate on. If None, uses {"all_natural": settings}.

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per (subset, score) combination.
    """
    if subsets is None:
        subsets = {"all_natural": settings}

    rows = []
    for subset_name, subset_df in subsets.items():
        working = subset_df.copy()
        if "risk_from_source_validation_tpr" not in working.columns:
            working["risk_from_source_validation_tpr"] = (
                -working["source_validation_tpr_at_1pct_fpr"].astype(float)
            )

        # Compute OTRS variants.
        working["otrs_max_rule"] = compute_max_rule_otrs(working)
        working["otrs_mean_rank"] = compute_mean_rank_otrs(working)
        working["otrs_vote"] = compute_vote_otrs(working)

        # Evaluate all scores.
        all_scores = MONITOR_COLUMNS + ["otrs_max_rule", "otrs_mean_rank", "otrs_vote"]
        for score_name in all_scores:
            if score_name not in working.columns:
                continue
            metrics = evaluate_readiness_score(working, score_name)
            rows.append(
                {
                    "subset": subset_name,
                    "score_name": score_name,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)
