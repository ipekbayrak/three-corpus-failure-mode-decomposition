from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

CHECKPOINT_LOGIC_VERSION = "separate_gates_v2"


def _safe_spearman(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    subset = frame[[x_col, y_col]].dropna()
    if len(subset) < 2:
        return float("nan")
    if subset[x_col].nunique(dropna=True) < 2 or subset[y_col].nunique(dropna=True) < 2:
        return float("nan")
    return float(subset[x_col].corr(subset[y_col], method="spearman"))


def _unsafe_auroc(frame: pd.DataFrame, score_col: str) -> float:
    subset = frame[[score_col, "unsafe_setting"]].dropna()
    if subset["unsafe_setting"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(subset["unsafe_setting"], subset[score_col]))


def _risk_frame(settings: pd.DataFrame) -> pd.DataFrame:
    return settings.assign(
        alpha_hat_risk=settings["alpha_hat"],
        source_validation_risk=-settings["source_validation_tpr_at_1pct_fpr"],
        proxy_a_distance_risk=settings["proxy_a_distance"],
        avg_wasserstein_1d_risk=settings["avg_wasserstein_1d"],
    )


def _alpha_hat_adds_value(settings: pd.DataFrame) -> bool:
    risk_frame = _risk_frame(settings)
    alpha_hat_spearman = _safe_spearman(risk_frame, "alpha_hat_risk", "loss_rel")
    alpha_hat_auroc = _unsafe_auroc(risk_frame, "alpha_hat_risk")
    baseline_spearmans = [
        _safe_spearman(risk_frame, "source_validation_risk", "loss_rel"),
        _safe_spearman(risk_frame, "proxy_a_distance_risk", "loss_rel"),
        _safe_spearman(risk_frame, "avg_wasserstein_1d_risk", "loss_rel"),
    ]
    baseline_aurocs = [
        _unsafe_auroc(risk_frame, "source_validation_risk"),
        _unsafe_auroc(risk_frame, "proxy_a_distance_risk"),
        _unsafe_auroc(risk_frame, "avg_wasserstein_1d_risk"),
    ]
    spearman_win = any(
        pd.notna(alpha_hat_spearman) and pd.notna(value) and alpha_hat_spearman > value
        for value in baseline_spearmans
    )
    auroc_win = any(
        pd.notna(alpha_hat_auroc) and pd.notna(value) and alpha_hat_auroc > value
        for value in baseline_aurocs
    )
    return bool(spearman_win or auroc_win)


def _has_viable_source_model(settings: pd.DataFrame) -> bool:
    natural = settings.loc[settings["setting_type"] == "natural_transfer"].copy()
    for _model_name, group in natural.groupby("model"):
        coverage = (
            group.groupby("source_dataset")["source_holdout_tpr_at_1pct_fpr"]
            .first()
            .reindex(["iot23", "ton_iot"])
        )
        if coverage.notna().all() and (coverage >= 0.60).all():
            return True
    return False


def extract_dataset_coverage(checkpoint: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if checkpoint.empty:
        raise ValueError("Checkpoint summary is empty")

    row = checkpoint.iloc[0]
    dataset_coverage: dict[str, tuple[float, float]] = {}
    for column in checkpoint.columns:
        if not column.endswith("_mapped_attack_coverage"):
            continue
        dataset = column[: -len("_mapped_attack_coverage")]
        other_share_col = f"{dataset}_other_attack_share"
        if other_share_col not in checkpoint.columns:
            raise ValueError(f"Missing paired other-share column for dataset {dataset}")
        dataset_coverage[dataset] = (float(row[column]), float(row[other_share_col]))

    if not dataset_coverage:
        raise ValueError("No dataset coverage fields found in checkpoint summary")
    return dataset_coverage


def summarize_pilot_checkpoint(
    settings: pd.DataFrame,
    dataset_coverage: dict[str, tuple[float, float]],
    run_id: str | None = None,
) -> pd.DataFrame:
    risk_frame = _risk_frame(settings)
    natural = risk_frame.loc[risk_frame["setting_type"] == "natural_transfer"].copy()
    controlled = risk_frame.loc[risk_frame["setting_type"] == "controlled_shift"].copy()

    mixed_oracle_spearman = _safe_spearman(risk_frame, "oracle_alpha", "loss_rel")
    natural_oracle_spearman = _safe_spearman(natural, "oracle_alpha", "loss_rel")
    controlled_oracle_spearman = _safe_spearman(controlled, "oracle_alpha", "loss_rel")
    alpha_hat_beats_baseline = _alpha_hat_adds_value(settings)
    viable_model = _has_viable_source_model(settings)
    min_mapped_coverage = min(coverage for coverage, _ in dataset_coverage.values())

    checkpoint = "STOP_THEORY_HEAVY_PATH"
    checkpoint_reason = "pilot criteria not met"
    oracle_pass = (
        pd.notna(natural_oracle_spearman)
        and natural_oracle_spearman >= 0.60
        and pd.notna(controlled_oracle_spearman)
        and controlled_oracle_spearman >= 0.50
    )
    if min_mapped_coverage < 0.80:
        checkpoint_reason = "mapped attack coverage below 0.80"
    elif len(natural) != 4:
        checkpoint_reason = "natural-transfer point count mismatch"
    elif len(controlled) != 10:
        checkpoint_reason = "controlled-shift point count mismatch"
    elif not viable_model:
        checkpoint_reason = "no model reaches source holdout TPR >= 0.60 on both source datasets"
    elif pd.isna(natural_oracle_spearman) or natural_oracle_spearman < 0.40:
        checkpoint_reason = "natural-transfer oracle alpha relation is too weak"
    elif pd.isna(controlled_oracle_spearman) or controlled_oracle_spearman < 0.50:
        checkpoint_reason = "controlled-shift oracle alpha relation is too weak"
    elif oracle_pass and alpha_hat_beats_baseline:
        checkpoint = "GO"
        checkpoint_reason = (
            "oracle alpha is structured in natural transfer and controlled shift, "
            "and alpha_hat adds value"
        )
    elif oracle_pass:
        checkpoint = "PIVOT_TO_FALLBACK"
        checkpoint_reason = (
            "oracle alpha is structured in natural transfer and controlled shift, "
            "but alpha_hat adds no value"
        )

    summary_row: dict[str, Any] = {
        "run_id": run_id
        or str(settings["run_id"].dropna().iloc[0] if "run_id" in settings.columns else "latest"),
        "checkpoint_logic_version": CHECKPOINT_LOGIC_VERSION,
        "mixed_oracle_alpha_spearman": mixed_oracle_spearman,
        "natural_transfer_oracle_alpha_spearman": natural_oracle_spearman,
        "controlled_shift_oracle_alpha_spearman": controlled_oracle_spearman,
        "alpha_hat_spearman": _safe_spearman(risk_frame, "alpha_hat_risk", "loss_rel"),
        "source_validation_risk_spearman": _safe_spearman(
            risk_frame,
            "source_validation_risk",
            "loss_rel",
        ),
        "proxy_a_distance_spearman": _safe_spearman(
            risk_frame,
            "proxy_a_distance_risk",
            "loss_rel",
        ),
        "avg_wasserstein_1d_spearman": _safe_spearman(
            risk_frame,
            "avg_wasserstein_1d_risk",
            "loss_rel",
        ),
        "alpha_hat_unsafe_auroc": _unsafe_auroc(risk_frame, "alpha_hat_risk"),
        "source_validation_risk_unsafe_auroc": _unsafe_auroc(
            risk_frame,
            "source_validation_risk",
        ),
        "proxy_a_distance_unsafe_auroc": _unsafe_auroc(
            risk_frame,
            "proxy_a_distance_risk",
        ),
        "avg_wasserstein_1d_unsafe_auroc": _unsafe_auroc(
            risk_frame,
            "avg_wasserstein_1d_risk",
        ),
        "natural_point_count": int(len(natural)),
        "controlled_point_count": int(len(controlled)),
        "alpha_hat_beats_baseline": int(alpha_hat_beats_baseline),
        "has_viable_source_model": int(viable_model),
        "checkpoint_reason": checkpoint_reason,
        "checkpoint": checkpoint,
    }
    for dataset in sorted(dataset_coverage):
        coverage, other_share = dataset_coverage[dataset]
        summary_row[f"{dataset}_mapped_attack_coverage"] = float(coverage)
        summary_row[f"{dataset}_other_attack_share"] = float(other_share)

    return pd.DataFrame([summary_row])
