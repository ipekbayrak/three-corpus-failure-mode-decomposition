from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

from iot_ids.decision import add_operational_metrics, relative_transfer_loss
from iot_ids.experiment import build_split_manifest, prepare_feature_frames
from iot_ids.features import FEATURE_COLUMNS
from iot_ids.metrics import compute_metrics, threshold_at_fpr
from iot_ids.modeling import (
    BoundedFeatureScaler,
    NATURAL_UNIT_INTERVAL_COLUMNS,
    SourceFeatureSanitizer,
    fit_estimator,
    make_model_core,
    score_positive_class,
)

CORAL_AUXILIARY_COLUMNS = [
    "setting_id",
    "auxiliary_method",
    "source_dataset",
    "target_dataset",
    "model",
    "threshold",
    "source_validation_tpr_at_1pct_fpr",
    "source_holdout_tpr_at_1pct_fpr",
    "source_holdout_realized_fpr",
    "source_holdout_auroc",
    "source_holdout_pauc_at_1pct",
    "target_tpr_at_1pct_fpr",
    "target_realized_fpr",
    "target_auroc",
    "target_pauc_at_1pct",
    "loss_rel",
    "relative_tpr_drop",
    "relative_tpr_change",
    "fpr_budget_multiple",
    "fpr_severity",
    "operational_severity",
    "catastrophic_operational_failure",
    "tpr_loss_unsafe",
    "target_adaptation_row_count",
    "target_adaptation_group_count",
    "target_evaluation_row_count",
    "target_evaluation_group_count",
    "target_split_seed",
]


@dataclass(frozen=True)
class CoralAuxiliaryOutputs:
    summary_csv: Path


@dataclass(frozen=True)
class TargetSplit:
    adaptation: pd.DataFrame
    evaluation: pd.DataFrame
    split_seed: int


class CoralAdapter:
    def __init__(self, regularization: float = 1.0) -> None:
        self.regularization = float(regularization)

    def fit(self, source_x: pd.DataFrame, target_x: pd.DataFrame) -> CoralAdapter:
        source = source_x.to_numpy(dtype=float)
        target = target_x.to_numpy(dtype=float)
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("CORAL expects 2D feature matrices")
        if source.shape[1] != target.shape[1]:
            raise ValueError("Source and target feature spaces must match for CORAL")
        self.feature_columns_ = list(source_x.columns)
        self.source_mean_ = source.mean(axis=0)
        self.target_mean_ = target.mean(axis=0)
        source_cov = self._covariance(source)
        target_cov = self._covariance(target)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            source_whitener = fractional_matrix_power(source_cov, -0.5)
            target_colorer = fractional_matrix_power(target_cov, 0.5)
            transform = np.real_if_close(source_whitener @ target_colorer)
        self.transform_matrix_ = np.nan_to_num(
            np.asarray(transform, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        centered = frame.loc[:, self.feature_columns_].to_numpy(dtype=float) - self.source_mean_
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            transformed = centered @ self.transform_matrix_ + self.target_mean_
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(transformed, columns=self.feature_columns_, index=frame.index)

    def _covariance(self, matrix: np.ndarray) -> np.ndarray:
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        denom = max(matrix.shape[0] - 1, 1)
        covariance = (centered.T @ centered) / denom
        covariance = covariance + (self.regularization * np.eye(matrix.shape[1], dtype=float))
        return np.nan_to_num(covariance, nan=0.0, posinf=self.regularization, neginf=0.0)


def compute_pauc(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores, max_fpr=max_fpr))
    except ValueError:
        return float("nan")


def _stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(base_seed + int(digest[:8], 16))


def _split_target_test_half_by_group(
    frame: pd.DataFrame,
    *,
    seed: int,
) -> TargetSplit:
    groups = sorted(
        frame["group_id"].astype(str).unique(),
        key=lambda value: (_stable_seed(seed, value), value),
    )
    if len(groups) < 2:
        raise ValueError("CORAL auxiliary requires at least two target-test groups")
    adaptation_groups, evaluation_groups = np.array_split(np.asarray(groups, dtype=object), 2)
    if len(adaptation_groups) == 0 or len(evaluation_groups) == 0:
        raise ValueError("CORAL auxiliary requires non-empty adaptation and evaluation halves")
    adaptation = frame.loc[
        frame["group_id"].astype(str).isin(set(adaptation_groups.tolist()))
    ].copy()
    evaluation = frame.loc[
        frame["group_id"].astype(str).isin(set(evaluation_groups.tolist()))
    ].copy()
    return TargetSplit(adaptation=adaptation, evaluation=evaluation, split_seed=seed)


def _postprocess_coral_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clipped = frame.copy()
    clipped = clipped.clip(lower=-12.0, upper=12.0)
    for column in NATURAL_UNIT_INTERVAL_COLUMNS:
        if column in clipped.columns:
            clipped[column] = clipped[column].clip(0.0, 1.0)
    return clipped.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _fit_core_estimator(
    model_name: str,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    seed: int,
) -> Any:
    estimator = make_model_core(model_name, seed, train_x.columns)
    if model_name in {"logistic_regression", "mlp"}:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"sklearn\.utils\.extmath",
            )
            estimator.fit(train_x, train_y)
        return estimator
    estimator.fit(train_x, train_y)
    return estimator


def _transform_frames_with_coral(
    source_train: pd.DataFrame,
    source_val: pd.DataFrame,
    source_holdout: pd.DataFrame,
    target_adaptation: pd.DataFrame,
    target_evaluation: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    sanitizer = SourceFeatureSanitizer(FEATURE_COLUMNS)
    sanitizer.fit(source_train)
    sanitized = {
        "source_train": sanitizer.transform(source_train),
        "source_val": sanitizer.transform(source_val),
        "source_holdout": sanitizer.transform(source_holdout),
        "target_adaptation": sanitizer.transform(target_adaptation),
        "target_evaluation": sanitizer.transform(target_evaluation),
    }
    scaler = BoundedFeatureScaler(FEATURE_COLUMNS)
    scaler.fit(sanitized["source_train"])
    scaled = {name: scaler.transform(frame) for name, frame in sanitized.items()}
    adapter = CoralAdapter().fit(scaled["source_train"], scaled["target_adaptation"])
    transformed = {
        name: _postprocess_coral_frame(adapter.transform(frame))
        for name, frame in scaled.items()
    }
    return transformed


def _prepare_model_inputs(
    model_name: str,
    transformed: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if model_name not in {"logistic_regression", "extra_trees", "hist_gradient_boosting", "mlp"}:
        raise ValueError(f"Unsupported CORAL auxiliary model: {model_name}")
    return transformed


def _evaluate_coral_setting(
    source_dataset: str,
    target_dataset: str,
    model_name: str,
    source_train: pd.DataFrame,
    source_val: pd.DataFrame,
    source_holdout: pd.DataFrame,
    target_split: TargetSplit,
    *,
    seed: int,
    target_fpr: float,
) -> dict[str, object]:
    transformed = _transform_frames_with_coral(
        source_train=source_train[FEATURE_COLUMNS],
        source_val=source_val[FEATURE_COLUMNS],
        source_holdout=source_holdout[FEATURE_COLUMNS],
        target_adaptation=target_split.adaptation[FEATURE_COLUMNS],
        target_evaluation=target_split.evaluation[FEATURE_COLUMNS],
    )
    model_inputs = _prepare_model_inputs(model_name, transformed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator = _fit_core_estimator(
            model_name,
            model_inputs["source_train"],
            source_train["label"].astype(int),
            seed,
        )

        source_val_scores = score_positive_class(
            estimator,
            model_inputs["source_val"],
            FEATURE_COLUMNS,
        )
        source_holdout_scores = score_positive_class(
            estimator,
            model_inputs["source_holdout"],
            FEATURE_COLUMNS,
        )
        target_eval_scores = score_positive_class(
            estimator,
            model_inputs["target_evaluation"],
            FEATURE_COLUMNS,
        )

    threshold = threshold_at_fpr(source_val["label"], source_val_scores, target_fpr)
    source_val_metrics = compute_metrics(source_val["label"], source_val_scores, threshold)
    source_holdout_metrics = compute_metrics(
        source_holdout["label"], source_holdout_scores, threshold
    )
    target_metrics = compute_metrics(
        target_split.evaluation["label"], target_eval_scores, threshold
    )
    source_holdout_pauc = compute_pauc(
        source_holdout["label"].astype(int).to_numpy(),
        np.asarray(source_holdout_scores, dtype=float),
        max_fpr=target_fpr,
    )
    target_pauc = compute_pauc(
        target_split.evaluation["label"].astype(int).to_numpy(),
        np.asarray(target_eval_scores, dtype=float),
        max_fpr=target_fpr,
    )
    source_holdout_tpr = float(source_holdout_metrics["tpr_at_1pct_fpr"])
    target_tpr = float(target_metrics["tpr_at_1pct_fpr"])
    row = {
        "setting_id": f"{source_dataset}__{target_dataset}__{model_name}__coral",
        "auxiliary_method": "coral",
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "model": model_name,
        "threshold": float(threshold),
        "source_validation_tpr_at_1pct_fpr": float(source_val_metrics["tpr_at_1pct_fpr"]),
        "source_holdout_tpr_at_1pct_fpr": source_holdout_tpr,
        "source_holdout_realized_fpr": float(source_holdout_metrics["realized_fpr"]),
        "source_holdout_auroc": float(source_holdout_metrics["auroc"]),
        "source_holdout_pauc_at_1pct": source_holdout_pauc,
        "target_tpr_at_1pct_fpr": target_tpr,
        "target_realized_fpr": float(target_metrics["realized_fpr"]),
        "target_auroc": float(target_metrics["auroc"]),
        "target_pauc_at_1pct": target_pauc,
        "loss_rel": float(relative_transfer_loss(target_tpr, source_holdout_tpr)),
        "target_adaptation_row_count": int(len(target_split.adaptation)),
        "target_adaptation_group_count": int(
            target_split.adaptation["group_id"].astype(str).nunique()
        ),
        "target_evaluation_row_count": int(len(target_split.evaluation)),
        "target_evaluation_group_count": int(
            target_split.evaluation["group_id"].astype(str).nunique()
        ),
        "target_split_seed": int(target_split.split_seed),
    }
    row = add_operational_metrics(pd.DataFrame([row]), fpr_budget=target_fpr).iloc[0].to_dict()
    return row


def _evaluate_source_only_matched_setting(
    source_dataset: str,
    target_dataset: str,
    model_name: str,
    source_train: pd.DataFrame,
    source_val: pd.DataFrame,
    source_holdout: pd.DataFrame,
    target_split: TargetSplit,
    *,
    seed: int,
    target_fpr: float,
) -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator = fit_estimator(
            model_name,
            source_train,
            FEATURE_COLUMNS,
            seed,
        )
        source_val_scores = score_positive_class(estimator, source_val, FEATURE_COLUMNS)
        source_holdout_scores = score_positive_class(estimator, source_holdout, FEATURE_COLUMNS)
        target_eval_scores = score_positive_class(
            estimator,
            target_split.evaluation,
            FEATURE_COLUMNS,
        )

    threshold = threshold_at_fpr(source_val["label"], source_val_scores, target_fpr)
    source_val_metrics = compute_metrics(source_val["label"], source_val_scores, threshold)
    source_holdout_metrics = compute_metrics(
        source_holdout["label"],
        source_holdout_scores,
        threshold,
    )
    target_metrics = compute_metrics(
        target_split.evaluation["label"],
        target_eval_scores,
        threshold,
    )
    source_holdout_pauc = compute_pauc(
        source_holdout["label"].astype(int).to_numpy(),
        np.asarray(source_holdout_scores, dtype=float),
        max_fpr=target_fpr,
    )
    target_pauc = compute_pauc(
        target_split.evaluation["label"].astype(int).to_numpy(),
        np.asarray(target_eval_scores, dtype=float),
        max_fpr=target_fpr,
    )
    source_holdout_tpr = float(source_holdout_metrics["tpr_at_1pct_fpr"])
    target_tpr = float(target_metrics["tpr_at_1pct_fpr"])
    row = {
        "setting_id": f"{source_dataset}__{target_dataset}__{model_name}__source_only_matched",
        "auxiliary_method": "source_only_matched",
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "model": model_name,
        "threshold": float(threshold),
        "source_validation_tpr_at_1pct_fpr": float(source_val_metrics["tpr_at_1pct_fpr"]),
        "source_holdout_tpr_at_1pct_fpr": source_holdout_tpr,
        "source_holdout_realized_fpr": float(source_holdout_metrics["realized_fpr"]),
        "source_holdout_auroc": float(source_holdout_metrics["auroc"]),
        "source_holdout_pauc_at_1pct": source_holdout_pauc,
        "target_tpr_at_1pct_fpr": target_tpr,
        "target_realized_fpr": float(target_metrics["realized_fpr"]),
        "target_auroc": float(target_metrics["auroc"]),
        "target_pauc_at_1pct": target_pauc,
        "loss_rel": float(relative_transfer_loss(target_tpr, source_holdout_tpr)),
        "target_adaptation_row_count": int(len(target_split.adaptation)),
        "target_adaptation_group_count": int(
            target_split.adaptation["group_id"].astype(str).nunique()
        ),
        "target_evaluation_row_count": int(len(target_split.evaluation)),
        "target_evaluation_group_count": int(
            target_split.evaluation["group_id"].astype(str).nunique()
        ),
        "target_split_seed": int(target_split.split_seed),
    }
    row = add_operational_metrics(pd.DataFrame([row]), fpr_budget=target_fpr).iloc[0].to_dict()
    return row


def run_coral_auxiliary(
    config: dict[str, Any],
    outputs: CoralAuxiliaryOutputs,
) -> dict[str, int]:
    coral_cfg = config["coral_auxiliary"]
    seed = int(config["project"]["seed"])
    benchmark_cfg = config.get("benchmark", {})
    target_fpr = float(
        coral_cfg.get(
            "target_fpr",
            benchmark_cfg.get("transfer_threshold_fpr", 0.01),
        )
    )
    models = list(
        coral_cfg.get(
            "models",
            benchmark_cfg.get(
                "models",
                [
                    "logistic_regression",
                    "extra_trees",
                    "hist_gradient_boosting",
                    "mlp",
                ],
            ),
        )
    )
    directions = [
        tuple(direction)
        for direction in benchmark_cfg.get(
            "directions",
            [
                ["iot23", "ton_iot"],
                ["iot23", "unsw_nb15"],
                ["ton_iot", "iot23"],
                ["ton_iot", "unsw_nb15"],
                ["unsw_nb15", "iot23"],
                ["unsw_nb15", "ton_iot"],
            ],
        )
    ]
    frames = prepare_feature_frames(config)
    split_frames, _ = build_split_manifest(frames, config)

    target_splits = {
        dataset_name: _split_target_test_half_by_group(
            splits.test,
            seed=_stable_seed(seed, "coral_target_split", dataset_name),
        )
        for dataset_name, splits in split_frames.items()
    }

    rows: list[dict[str, object]] = []
    for model_name in models:
        for source_dataset, target_dataset in directions:
            source_splits = split_frames[source_dataset]
            target_split = target_splits[target_dataset]
            source_only_row = _evaluate_source_only_matched_setting(
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                model_name=model_name,
                source_train=source_splits.train,
                source_val=source_splits.val,
                source_holdout=source_splits.test,
                target_split=target_split,
                seed=_stable_seed(
                    seed,
                    "source_only_matched_model",
                    source_dataset,
                    target_dataset,
                    model_name,
                ),
                target_fpr=target_fpr,
            )
            row = _evaluate_coral_setting(
                source_dataset=source_dataset,
                target_dataset=target_dataset,
                model_name=model_name,
                source_train=source_splits.train,
                source_val=source_splits.val,
                source_holdout=source_splits.test,
                target_split=target_split,
                seed=_stable_seed(seed, "coral_model", source_dataset, target_dataset, model_name),
                target_fpr=target_fpr,
            )
            rows.append(source_only_row)
            rows.append(row)

    summary = (
        pd.DataFrame(rows)[CORAL_AUXILIARY_COLUMNS]
        .sort_values(["source_dataset", "target_dataset", "model", "auxiliary_method"])
        .reset_index(drop=True)
    )
    outputs.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outputs.summary_csv, index=False)
    return {"summary_rows": int(len(summary))}
