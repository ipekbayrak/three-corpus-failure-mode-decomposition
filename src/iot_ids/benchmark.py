from __future__ import annotations

from collections.abc import Iterable
import hashlib

import pandas as pd

from iot_ids.metrics import grouped_bootstrap_metric, grouped_bootstrap_threshold_intervals

REQUIRED_BENCHMARK_SETTING_COLUMNS = [
    "setting_id",
    "setting_type",
    "source_dataset",
    "target_dataset",
    "model",
    "threshold",
    "source_holdout_tpr_at_1pct_fpr",
    "source_holdout_tpr_at_1pct_fpr_ci_lo",
    "source_holdout_tpr_at_1pct_fpr_ci_hi",
    "source_validation_tpr_at_1pct_fpr",
    "source_validation_tpr_at_1pct_fpr_ci_lo",
    "source_validation_tpr_at_1pct_fpr_ci_hi",
    "target_tpr_at_1pct_fpr",
    "target_tpr_at_1pct_fpr_ci_lo",
    "target_tpr_at_1pct_fpr_ci_hi",
    "target_realized_fpr",
    "target_realized_fpr_ci_lo",
    "target_realized_fpr_ci_hi",
    "target_auroc",
    "target_auroc_ci_lo",
    "target_auroc_ci_hi",
    "target_auprc",
    "loss_rel",
    "relative_tpr_drop",
    "relative_tpr_change",
    "fpr_budget_multiple",
    "fpr_severity",
    "operational_severity",
    "catastrophic_operational_failure",
    "tpr_loss_unsafe",
    "alpha_hat",
    "proxy_a_distance",
    "avg_wasserstein_1d",
    "source_mapped_attack_coverage",
    "source_other_attack_share",
    "target_mapped_attack_coverage",
    "target_other_attack_share",
    "oracle_alpha",
    "oracle_alpha_usable",
    "oracle_alpha_scope",
]

CROSS_DOMAIN_SUMMARY_COLUMNS = [
    "setting_id",
    "source_dataset",
    "target_dataset",
    "model",
    "threshold",
    "source_holdout_tpr_at_1pct_fpr",
    "source_holdout_tpr_at_1pct_fpr_ci_lo",
    "source_holdout_tpr_at_1pct_fpr_ci_hi",
    "source_validation_tpr_at_1pct_fpr",
    "source_validation_tpr_at_1pct_fpr_ci_lo",
    "source_validation_tpr_at_1pct_fpr_ci_hi",
    "target_tpr_at_1pct_fpr",
    "target_tpr_at_1pct_fpr_ci_lo",
    "target_tpr_at_1pct_fpr_ci_hi",
    "target_realized_fpr",
    "target_realized_fpr_ci_lo",
    "target_realized_fpr_ci_hi",
    "target_auroc",
    "target_auroc_ci_lo",
    "target_auroc_ci_hi",
    "target_auprc",
    "loss_rel",
    "relative_tpr_drop",
    "relative_tpr_change",
    "fpr_budget_multiple",
    "fpr_severity",
    "operational_severity",
    "catastrophic_operational_failure",
    "tpr_loss_unsafe",
    "alpha_hat",
    "proxy_a_distance",
    "avg_wasserstein_1d",
    "source_mapped_attack_coverage",
    "source_other_attack_share",
    "target_mapped_attack_coverage",
    "target_other_attack_share",
    "oracle_alpha",
    "oracle_alpha_usable",
    "oracle_alpha_scope",
]

REQUIRED_PREDICTION_COLUMNS = [
    "setting_id",
    "dataset",
    "record_id",
    "group_id",
    "label",
    "family_group",
    "score",
    "source_dataset",
    "target_dataset",
    "model",
    "threshold",
    "eval_role",
]

def _stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(base_seed + int(digest[:8], 16))


def build_within_domain_summary(settings: pd.DataFrame) -> pd.DataFrame:
    working = settings.copy()
    for column in [
        "source_holdout_tpr_at_1pct_fpr_ci_lo",
        "source_holdout_tpr_at_1pct_fpr_ci_hi",
        "source_validation_tpr_at_1pct_fpr_ci_lo",
        "source_validation_tpr_at_1pct_fpr_ci_hi",
    ]:
        if column not in working.columns:
            working[column] = float("nan")
    summary = (
        working[
            [
                "source_dataset",
                "model",
                "source_holdout_tpr_at_1pct_fpr",
                "source_holdout_tpr_at_1pct_fpr_ci_lo",
                "source_holdout_tpr_at_1pct_fpr_ci_hi",
                "source_validation_tpr_at_1pct_fpr",
                "source_validation_tpr_at_1pct_fpr_ci_lo",
                "source_validation_tpr_at_1pct_fpr_ci_hi",
            ]
        ]
        .drop_duplicates()
        .rename(columns={"source_dataset": "dataset"})
        .sort_values(["dataset", "model"])
        .reset_index(drop=True)
    )
    return summary


def build_cross_domain_summary(settings: pd.DataFrame) -> pd.DataFrame:
    working = settings.copy()
    for column in CROSS_DOMAIN_SUMMARY_COLUMNS:
        if column not in working.columns:
            working[column] = float("nan")
    summary = (
        working[CROSS_DOMAIN_SUMMARY_COLUMNS]
        .sort_values(["source_dataset", "target_dataset", "model"])
        .reset_index(drop=True)
    )
    return summary


def add_confidence_intervals(
    settings: pd.DataFrame,
    predictions: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    role_to_prefix = {
        "source_validation": "source_validation_tpr_at_1pct_fpr",
        "source_holdout": "source_holdout_tpr_at_1pct_fpr",
        "target_transfer": "target_tpr_at_1pct_fpr",
    }
    predictions_by_setting = {
        setting_id: frame.copy()
        for setting_id, frame in predictions.groupby("setting_id", sort=False)
    }

    rows = []
    for row in settings.itertuples(index=False):
        setting = dict(row._asdict())
        frame = predictions_by_setting[str(row.setting_id)]
        for eval_role, prefix in role_to_prefix.items():
            role_frame = frame.loc[frame["eval_role"] == eval_role].copy()
            seed_parts = (
                [str(row.source_dataset), str(row.model), eval_role]
                if eval_role != "target_transfer"
                else [str(row.setting_id), eval_role]
            )
            intervals = grouped_bootstrap_threshold_intervals(
                role_frame,
                threshold=float(row.threshold),
                n_boot=n_boot,
                seed=_stable_seed(seed, *seed_parts),
            )
            setting[f"{prefix}_ci_lo"] = intervals["tpr_ci_lo"]
            setting[f"{prefix}_ci_hi"] = intervals["tpr_ci_hi"]
            if eval_role == "target_transfer":
                setting["target_realized_fpr_ci_lo"] = intervals["realized_fpr_ci_lo"]
                setting["target_realized_fpr_ci_hi"] = intervals["realized_fpr_ci_hi"]
                auroc_ci_lo, auroc_ci_hi = grouped_bootstrap_metric(
                    role_frame,
                    "auroc",
                    threshold=float(row.threshold),
                    n_boot=n_boot,
                    seed=_stable_seed(seed, str(row.setting_id), eval_role, "auroc"),
                )
                setting["target_auroc_ci_lo"] = auroc_ci_lo
                setting["target_auroc_ci_hi"] = auroc_ci_hi
        rows.append(setting)
    return pd.DataFrame(rows)


def validate_benchmark_artifacts(
    settings: pd.DataFrame,
    predictions: pd.DataFrame,
    directions: Iterable[Iterable[str]],
    model_names: Iterable[str],
) -> None:
    missing_columns = [
        column for column in REQUIRED_BENCHMARK_SETTING_COLUMNS if column not in settings.columns
    ]
    if missing_columns:
        raise ValueError(f"Benchmark settings missing required columns: {missing_columns}")

    if (
        settings["setting_type"].nunique() != 1
        or settings["setting_type"].iloc[0] != "natural_transfer"
    ):
        raise ValueError("Benchmark settings must contain only natural-transfer rows")
    missing_prediction_columns = [
        column for column in REQUIRED_PREDICTION_COLUMNS if column not in predictions.columns
    ]
    if missing_prediction_columns:
        raise ValueError(
            f"Benchmark predictions missing required columns: {missing_prediction_columns}"
        )
    if predictions["setting_id"].nunique() != len(settings):
        raise ValueError("Benchmark predictions/settings setting_id counts do not match")
    expected_roles = {"source_validation", "source_holdout", "target_transfer"}
    observed_roles = set(predictions["eval_role"].astype(str))
    if observed_roles != expected_roles:
        raise ValueError(
            "Benchmark predictions must contain source_validation, source_holdout, "
            f"and target_transfer rows; observed {sorted(observed_roles)}"
        )

    expected_pairs = {
        (str(source), str(target), str(model))
        for source, target in directions
        for model in model_names
    }
    observed_pairs = {
        (str(row.source_dataset), str(row.target_dataset), str(row.model))
        for row in settings.itertuples(index=False)
    }
    if observed_pairs != expected_pairs:
        raise ValueError(
            "Benchmark settings do not match the configured direction/model matrix: "
            f"expected {len(expected_pairs)} rows, observed {len(observed_pairs)}"
        )

    if settings[["source_dataset", "target_dataset", "model", "threshold"]].isnull().any().any():
        raise ValueError("Benchmark settings contain null source/target/model/threshold values")

    ci_pairs = [
        ("source_validation_tpr_at_1pct_fpr_ci_lo", "source_validation_tpr_at_1pct_fpr", "source_validation_tpr_at_1pct_fpr_ci_hi"),
        ("source_holdout_tpr_at_1pct_fpr_ci_lo", "source_holdout_tpr_at_1pct_fpr", "source_holdout_tpr_at_1pct_fpr_ci_hi"),
        ("target_tpr_at_1pct_fpr_ci_lo", "target_tpr_at_1pct_fpr", "target_tpr_at_1pct_fpr_ci_hi"),
        ("target_realized_fpr_ci_lo", "target_realized_fpr", "target_realized_fpr_ci_hi"),
        ("target_auroc_ci_lo", "target_auroc", "target_auroc_ci_hi"),
    ]
    for lo_column, point_column, hi_column in ci_pairs:
        if not ((settings[lo_column] <= settings[point_column]) & (settings[point_column] <= settings[hi_column])).all():
            raise ValueError(
                f"Confidence interval ordering failed for {point_column}: expected lo <= point <= hi"
            )

    within = build_within_domain_summary(settings)
    expected_within = {
        (str(source), str(model)) for source, _target in directions for model in model_names
    }
    expected_within_count = len({dataset for dataset, _model in expected_within}) * len(
        set(model_names)
    )
    if len(within) != expected_within_count:
        raise ValueError(
            "Within-domain summary does not cover each dataset/model exactly once: "
            f"expected {expected_within_count}, observed {len(within)}"
        )

    low_coverage = (
        (settings["source_mapped_attack_coverage"] < 0.80)
        | (settings["target_mapped_attack_coverage"] < 0.80)
    )
    if not (settings.loc[low_coverage, "oracle_alpha_usable"] == 0).all():
        raise ValueError("Low-coverage settings must be marked oracle_alpha_usable = 0")
    if not (
        settings.loc[low_coverage, "oracle_alpha_scope"] == "stress_test_low_coverage"
    ).all():
        raise ValueError("Low-coverage settings must be marked as stress-test-low-coverage")
