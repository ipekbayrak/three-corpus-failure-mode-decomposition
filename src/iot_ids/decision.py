from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def relative_transfer_loss(target_tpr: float, source_tpr: float, eps: float = 1e-8) -> float:
    return float(1.0 - (target_tpr / max(source_tpr, eps)))


def add_operational_metrics(
    settings: pd.DataFrame,
    fpr_budget: float = 0.01,
    tpr_loss_threshold: float = 0.5,
    catastrophic_threshold: float = 1.0,
) -> pd.DataFrame:
    result = settings.copy()
    result["relative_tpr_drop"] = result["loss_rel"].clip(lower=0.0).astype(float)
    result["relative_tpr_change"] = (-result["loss_rel"]).astype(float)
    result["fpr_budget_multiple"] = (
        result["target_realized_fpr"].astype(float) / float(fpr_budget)
    )
    result["fpr_severity"] = (
        (result["fpr_budget_multiple"] - 1.0).clip(lower=0.0, upper=9.0) / 9.0
    ).astype(float)
    result["operational_severity"] = result[
        ["relative_tpr_drop", "fpr_severity"]
    ].max(axis=1)
    result["catastrophic_operational_failure"] = (
        result["operational_severity"].astype(float) >= float(catastrophic_threshold)
    ).astype(int)
    result["tpr_loss_unsafe"] = (result["loss_rel"].astype(float) > tpr_loss_threshold).astype(
        int
    )
    if "unsafe_setting" not in result.columns:
        result["unsafe_setting"] = result["tpr_loss_unsafe"]
    return result


def mark_unsafe(settings: pd.DataFrame, loss_threshold: float) -> pd.DataFrame:
    result = settings.copy()
    result["unsafe"] = (result["loss_rel"].astype(float) > loss_threshold).astype(int)
    return result


def summarise_monitor_performance(
    settings: pd.DataFrame,
    score_column: str,
    severity_column: str = "operational_severity",
    catastrophic_column: str = "catastrophic_operational_failure",
) -> dict[str, float]:
    values = settings[score_column].astype(float)
    severity = settings[severity_column].astype(float)
    catastrophic = settings[catastrophic_column].astype(int)
    if catastrophic.nunique() < 2:
        catastrophic_auroc = float("nan")
        catastrophic_auprc = float("nan")
    else:
        catastrophic_auroc = float(roc_auc_score(catastrophic, values))
        catastrophic_auprc = float(average_precision_score(catastrophic, values))
    if values.nunique(dropna=True) < 2 or severity.nunique(dropna=True) < 2:
        spearman = float("nan")
    else:
        spearman = float(values.corr(severity, method="spearman"))
    return {
        "spearman_vs_operational_severity": spearman,
        "catastrophic_auroc": catastrophic_auroc,
        "catastrophic_auprc": catastrophic_auprc,
    }


def deploy_abstain_curve(
    settings: pd.DataFrame,
    score_column: str,
    severity_column: str = "operational_severity",
    catastrophic_column: str = "catastrophic_operational_failure",
    quantiles: Iterable[float] = (0.1, 0.25, 0.5),
) -> pd.DataFrame:
    rows = []
    scored = settings.sort_values(score_column, ascending=False).reset_index(drop=True)
    for quantile in quantiles:
        abstain_n = int(np.ceil(len(scored) * quantile))
        abstained = scored.head(abstain_n)
        deployed = scored.iloc[abstain_n:]
        rows.append(
            {
                "score_column": score_column,
                "abstain_fraction": quantile,
                "coverage": float(len(deployed) / max(len(scored), 1)),
                "mean_operational_severity_abstained": float(
                    abstained[severity_column].astype(float).mean()
                ),
                "mean_operational_severity_deployed": float(
                    deployed[severity_column].astype(float).mean()
                ),
                "catastrophic_prevented": int(abstained[catastrophic_column].sum()),
                "catastrophic_remaining": int(deployed[catastrophic_column].sum()),
            }
        )
    return pd.DataFrame(rows)
