from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from iot_ids.decision import add_operational_metrics, relative_transfer_loss
from iot_ids.metrics import compute_metrics, threshold_at_fpr

TARGET_RETUNED_COLUMNS = [
    "setting_id",
    "source_dataset",
    "target_dataset",
    "model",
    "fixed_threshold",
    "fixed_target_tpr_at_1pct_fpr",
    "fixed_target_realized_fpr",
    "source_holdout_tpr_at_1pct_fpr",
    "oracle_target_threshold_at_1pct_fpr",
    "oracle_target_tpr_at_1pct_fpr",
    "oracle_target_realized_fpr",
    "threshold_shift",
    "retuned_tpr_gain",
    "fixed_relative_tpr_drop",
    "oracle_relative_tpr_drop",
]

FPR_BUDGET_COLUMNS = [
    "setting_id",
    "source_dataset",
    "target_dataset",
    "model",
    "fpr_budget",
    "threshold",
    "source_validation_tpr_at_budget",
    "source_holdout_tpr_at_budget",
    "target_tpr_at_budget",
    "target_realized_fpr",
    "loss_rel",
    "relative_tpr_drop",
    "relative_tpr_change",
    "fpr_budget_multiple",
    "fpr_severity",
    "operational_severity",
    "catastrophic_operational_failure",
]


def _natural_settings(settings: pd.DataFrame) -> pd.DataFrame:
    natural = settings.loc[settings["setting_type"].astype(str) == "natural_transfer"].copy()
    return natural.sort_values(["source_dataset", "target_dataset", "model"]).reset_index(drop=True)


def _prediction_groups(
    predictions: pd.DataFrame | Mapping[tuple[str, str], pd.DataFrame]
) -> Mapping[tuple[str, str], pd.DataFrame]:
    if isinstance(predictions, Mapping):
        return predictions
    working = predictions[["setting_id", "eval_role", "label", "score"]].copy()
    working["setting_id"] = working["setting_id"].astype(str)
    working["eval_role"] = working["eval_role"].astype(str)
    working["label"] = working["label"].astype(int)
    working["score"] = working["score"].astype(float)
    return {
        (str(setting_id), str(eval_role)): subset[["label", "score"]].reset_index(drop=True)
        for (setting_id, eval_role), subset in working.groupby(["setting_id", "eval_role"], sort=False)
    }


def build_target_retuned_oracle_baseline(
    settings: pd.DataFrame,
    predictions: pd.DataFrame,
    target_fpr: float = 0.01,
) -> pd.DataFrame:
    natural = _natural_settings(settings)
    prediction_groups = _prediction_groups(predictions)
    rows: list[dict[str, object]] = []

    for row in natural.itertuples(index=False):
        target = prediction_groups[(str(row.setting_id), "target_transfer")]
        oracle_threshold = threshold_at_fpr(
            target["label"].astype(int),
            target["score"].astype(float),
            target_fpr,
        )
        oracle_metrics = compute_metrics(
            target["label"].astype(int),
            target["score"].astype(float),
            oracle_threshold,
        )
        fixed_target_tpr = float(row.target_tpr_at_1pct_fpr)
        source_holdout_tpr = float(row.source_holdout_tpr_at_1pct_fpr)
        oracle_target_tpr = float(oracle_metrics["tpr_at_1pct_fpr"])
        rows.append(
            {
                "setting_id": str(row.setting_id),
                "source_dataset": str(row.source_dataset),
                "target_dataset": str(row.target_dataset),
                "model": str(row.model),
                "fixed_threshold": float(row.threshold),
                "fixed_target_tpr_at_1pct_fpr": fixed_target_tpr,
                "fixed_target_realized_fpr": float(row.target_realized_fpr),
                "source_holdout_tpr_at_1pct_fpr": source_holdout_tpr,
                "oracle_target_threshold_at_1pct_fpr": float(oracle_threshold),
                "oracle_target_tpr_at_1pct_fpr": oracle_target_tpr,
                "oracle_target_realized_fpr": float(oracle_metrics["realized_fpr"]),
                "threshold_shift": float(oracle_threshold - float(row.threshold)),
                "retuned_tpr_gain": float(oracle_target_tpr - fixed_target_tpr),
                "fixed_relative_tpr_drop": max(
                    relative_transfer_loss(fixed_target_tpr, source_holdout_tpr), 0.0
                ),
                "oracle_relative_tpr_drop": max(
                    relative_transfer_loss(oracle_target_tpr, source_holdout_tpr), 0.0
                ),
            }
        )

    result = pd.DataFrame(rows)[TARGET_RETUNED_COLUMNS]
    return result.sort_values(["source_dataset", "target_dataset", "model"]).reset_index(drop=True)


def build_fpr_budget_sensitivity(
    settings: pd.DataFrame,
    predictions: pd.DataFrame,
    budgets: Iterable[float],
) -> pd.DataFrame:
    natural = _natural_settings(settings)
    prediction_groups = _prediction_groups(predictions)
    rows: list[dict[str, object]] = []

    for row in natural.itertuples(index=False):
        setting_id = str(row.setting_id)
        source_validation = prediction_groups[(setting_id, "source_validation")]
        source_holdout = prediction_groups[(setting_id, "source_holdout")]
        target = prediction_groups[(setting_id, "target_transfer")]
        for budget in budgets:
            threshold = threshold_at_fpr(
                source_validation["label"].astype(int),
                source_validation["score"].astype(float),
                float(budget),
            )
            source_val_metrics = compute_metrics(
                source_validation["label"].astype(int),
                source_validation["score"].astype(float),
                threshold,
            )
            source_holdout_metrics = compute_metrics(
                source_holdout["label"].astype(int),
                source_holdout["score"].astype(float),
                threshold,
            )
            target_metrics = compute_metrics(
                target["label"].astype(int),
                target["score"].astype(float),
                threshold,
            )
            loss_rel = relative_transfer_loss(
                float(target_metrics["tpr_at_1pct_fpr"]),
                float(source_holdout_metrics["tpr_at_1pct_fpr"]),
            )
            rows.append(
                {
                    "setting_id": str(row.setting_id),
                    "source_dataset": str(row.source_dataset),
                    "target_dataset": str(row.target_dataset),
                    "model": str(row.model),
                    "fpr_budget": float(budget),
                    "threshold": float(threshold),
                    "source_validation_tpr_at_budget": float(
                        source_val_metrics["tpr_at_1pct_fpr"]
                    ),
                    "source_holdout_tpr_at_budget": float(
                        source_holdout_metrics["tpr_at_1pct_fpr"]
                    ),
                    "target_tpr_at_budget": float(target_metrics["tpr_at_1pct_fpr"]),
                    "target_realized_fpr": float(target_metrics["realized_fpr"]),
                    "loss_rel": float(loss_rel),
                }
            )

    result = pd.DataFrame(rows)
    result = add_operational_metrics(result, fpr_budget=0.01)
    # Recompute FPR-budget-scaled quantities against the active budget rather than the paper's fixed 1% budget.
    result["fpr_budget_multiple"] = (
        result["target_realized_fpr"].astype(float) / result["fpr_budget"].astype(float)
    )
    result["fpr_severity"] = (
        (result["fpr_budget_multiple"] - 1.0).clip(lower=0.0, upper=9.0) / 9.0
    ).astype(float)
    result["operational_severity"] = result[["relative_tpr_drop", "fpr_severity"]].max(axis=1)
    result["catastrophic_operational_failure"] = (
        result["operational_severity"].astype(float) >= 1.0
    ).astype(int)
    result = result[FPR_BUDGET_COLUMNS]
    return result.sort_values(["source_dataset", "target_dataset", "model", "fpr_budget"]).reset_index(
        drop=True
    )


def validate_acsac_diagnostics(
    target_retuned: pd.DataFrame,
    fpr_budget_sensitivity: pd.DataFrame,
    settings: pd.DataFrame,
    budgets: Iterable[float],
) -> None:
    natural = _natural_settings(settings)
    expected_settings = {
        (str(row.setting_id), str(row.source_dataset), str(row.target_dataset), str(row.model))
        for row in natural.itertuples(index=False)
    }
    observed_retuned = {
        (str(row.setting_id), str(row.source_dataset), str(row.target_dataset), str(row.model))
        for row in target_retuned.itertuples(index=False)
    }
    if observed_retuned != expected_settings:
        raise ValueError("Target-retuned oracle baseline does not match natural-transfer settings")

    budget_set = {float(value) for value in budgets}
    observed_budget_rows = {
        (
            str(row.setting_id),
            str(row.source_dataset),
            str(row.target_dataset),
            str(row.model),
            float(row.fpr_budget),
        )
        for row in fpr_budget_sensitivity.itertuples(index=False)
    }
    expected_budget_rows = {
        (*setting_key, budget) for setting_key in expected_settings for budget in budget_set
    }
    if observed_budget_rows != expected_budget_rows:
        raise ValueError("FPR budget sensitivity artifact does not match expected setting/budget rows")

    if len(target_retuned) != len(natural):
        raise ValueError("Target-retuned oracle baseline must contain exactly one row per natural setting")
    if len(fpr_budget_sensitivity) != len(natural) * len(budget_set):
        raise ValueError("FPR budget sensitivity must contain one row per natural setting and budget")
