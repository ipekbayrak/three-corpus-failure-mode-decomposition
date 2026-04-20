from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from iot_ids.metrics import realized_fpr, threshold_at_fpr, tpr_at_threshold

DEFAULT_RECALIBRATION_SEED = 20260413


@dataclass(frozen=True)
class RecalibrationSplit:
    calibration_benign: pd.DataFrame
    evaluation_benign: pd.DataFrame
    split_mode: str
    split_unit: str
    calibration_units: tuple[str, ...]
    evaluation_units: tuple[str, ...]
    disjoint: bool


def calibration_eval_score_ks_statistic(
    calibration_scores: np.ndarray | list[float],
    evaluation_scores: np.ndarray | list[float],
) -> float:
    calibration = np.sort(np.asarray(calibration_scores, dtype=float))
    evaluation = np.sort(np.asarray(evaluation_scores, dtype=float))
    if calibration.size == 0 or evaluation.size == 0:
        return float("nan")

    support = np.unique(np.concatenate([calibration, evaluation]))
    calibration_cdf = np.searchsorted(calibration, support, side="right") / calibration.size
    evaluation_cdf = np.searchsorted(evaluation, support, side="right") / evaluation.size
    return float(np.max(np.abs(calibration_cdf - evaluation_cdf)))


def _hash_key(value: str, seed: int) -> tuple[int, str]:
    payload = f"{seed}:{value}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16), value


def _partition_by_unit(
    frame: pd.DataFrame,
    unit_column: str,
    calibration_frac: float,
    seed: int,
) -> RecalibrationSplit | None:
    units = sorted(
        frame[unit_column].fillna("missing").astype(str).unique(),
        key=lambda value: _hash_key(value, seed),
    )
    if len(units) < 2:
        return None
    calibration_count = int(round(len(units) * calibration_frac))
    calibration_count = max(1, min(calibration_count, len(units) - 1))
    calibration_units = tuple(units[:calibration_count])
    evaluation_units = tuple(units[calibration_count:])
    calibration = frame.loc[
        frame[unit_column].fillna("missing").astype(str).isin(calibration_units)
    ].copy()
    evaluation = frame.loc[
        frame[unit_column].fillna("missing").astype(str).isin(evaluation_units)
    ].copy()
    if calibration.empty or evaluation.empty:
        return None
    return RecalibrationSplit(
        calibration_benign=calibration.reset_index(drop=True),
        evaluation_benign=evaluation.reset_index(drop=True),
        split_mode="deterministic_partition",
        split_unit=unit_column,
        calibration_units=calibration_units,
        evaluation_units=evaluation_units,
        disjoint=set(calibration_units).isdisjoint(set(evaluation_units)),
    )


def split_benign_calibration_eval(
    target_frame: pd.DataFrame,
    calibration_frac: float = 0.5,
    seed: int = DEFAULT_RECALIBRATION_SEED,
) -> RecalibrationSplit:
    benign = target_frame.loc[target_frame["label"].astype(int) == 0].copy()
    if benign.empty:
        return RecalibrationSplit(
            calibration_benign=benign,
            evaluation_benign=benign,
            split_mode="no_benign",
            split_unit="none",
            calibration_units=tuple(),
            evaluation_units=tuple(),
            disjoint=True,
        )

    if "group_id" in benign.columns:
        split = _partition_by_unit(benign, "group_id", calibration_frac, seed)
        if split is not None:
            return split

    row_unit_column = "record_id" if "record_id" in benign.columns else "__row_unit__"
    working = benign.copy()
    if row_unit_column == "__row_unit__":
        working[row_unit_column] = [f"row_{idx}" for idx in range(len(working))]
    split = _partition_by_unit(working, row_unit_column, calibration_frac, seed)
    if split is not None:
        return split

    # One benign row cannot support a disjoint evaluation split. Keep the
    # threshold calibration deterministic and report the held-out benign FPR as
    # missing rather than silently reusing the calibration sample.
    only_row_units = tuple(working[row_unit_column].astype(str).tolist())
    return RecalibrationSplit(
        calibration_benign=working.reset_index(drop=True),
        evaluation_benign=working.iloc[0:0].copy().reset_index(drop=True),
        split_mode="insufficient_benign_for_eval",
        split_unit=row_unit_column,
        calibration_units=only_row_units,
        evaluation_units=tuple(),
        disjoint=True,
    )


def run_unlabeled_recalibration(
    settings: pd.DataFrame,
    prediction_groups: Mapping[tuple[str, str], pd.DataFrame],
    target_fpr: float = 0.01,
    calibration_frac: float = 0.5,
    seed: int = DEFAULT_RECALIBRATION_SEED,
) -> pd.DataFrame:
    natural = settings[settings["setting_type"] == "natural_transfer"].copy()
    rows: list[dict[str, object]] = []

    for _, row in natural.iterrows():
        sid = str(row["setting_id"])
        target_key = (sid, "target_transfer")
        if target_key not in prediction_groups:
            continue

        target = prediction_groups[target_key].copy()
        split = split_benign_calibration_eval(
            target,
            calibration_frac=calibration_frac,
            seed=seed,
        )
        calibration_benign = split.calibration_benign
        evaluation_benign = split.evaluation_benign
        target_labels = target["label"].astype(int).to_numpy()
        target_scores = target["score"].astype(float).to_numpy()

        if calibration_benign.empty:
            continue

        calibration_scores = calibration_benign["score"].astype(float).to_numpy()
        recalibrated_threshold = threshold_at_fpr(
            np.zeros(len(calibration_scores), dtype=int),
            calibration_scores,
            target_fpr,
        )

        recal_tpr = tpr_at_threshold(target_labels, target_scores, recalibrated_threshold)
        recal_calibration_fpr = realized_fpr(
            np.zeros(len(calibration_scores), dtype=int),
            calibration_scores,
            recalibrated_threshold,
        )
        if evaluation_benign.empty:
            calibration_eval_score_ks = float("nan")
            calibration_eval_mean_score_delta = float("nan")
        else:
            evaluation_scores = evaluation_benign["score"].astype(float).to_numpy()
            calibration_eval_score_ks = calibration_eval_score_ks_statistic(
                calibration_scores,
                evaluation_scores,
            )
            calibration_eval_mean_score_delta = float(
                evaluation_scores.mean() - calibration_scores.mean()
            )
        if evaluation_benign.empty:
            recal_fpr = float("nan")
        else:
            recal_fpr = realized_fpr(
                np.zeros(len(evaluation_scores), dtype=int),
                evaluation_scores,
                recalibrated_threshold,
            )
        eval_fpr_exceeds_budget = int(
            np.isfinite(recal_fpr) and float(recal_fpr) > (target_fpr + 1e-12)
        )

        frozen_threshold = float(row["threshold"])
        frozen_tpr = float(row["target_tpr_at_1pct_fpr"])
        frozen_fpr = float(row["target_realized_fpr"])

        oracle_threshold = threshold_at_fpr(target_labels, target_scores, target_fpr)
        oracle_tpr = tpr_at_threshold(target_labels, target_scores, oracle_threshold)
        oracle_fpr = realized_fpr(target_labels, target_scores, oracle_threshold)

        if len(np.unique(target_labels)) >= 2:
            target_auroc = float(roc_auc_score(target_labels, target_scores))
        else:
            target_auroc = float("nan")

        source_holdout_tpr = float(row["source_holdout_tpr_at_1pct_fpr"])
        tpr_recovery = recal_tpr - frozen_tpr
        relative_recovery = tpr_recovery / max(source_holdout_tpr, 1e-8)

        if target_auroc <= 0.5:
            recal_diagnosis = "ranking_collapse_irrecoverable"
        elif target_auroc < 0.6:
            recal_diagnosis = "ranking_degraded_partial"
        elif recal_tpr > 0.5 * source_holdout_tpr:
            recal_diagnosis = "recalibration_effective"
        else:
            recal_diagnosis = "recalibration_insufficient"

        rows.append(
            {
                "setting_id": sid,
                "source_dataset": str(row["source_dataset"]),
                "target_dataset": str(row["target_dataset"]),
                "model": str(row["model"]),
                "target_auroc": target_auroc,
                "frozen_threshold": frozen_threshold,
                "frozen_tpr": frozen_tpr,
                "frozen_fpr": frozen_fpr,
                "recal_threshold": recalibrated_threshold,
                "recal_tpr": recal_tpr,
                "recal_fpr": recal_fpr,
                "recal_calibration_fpr": recal_calibration_fpr,
                "calibration_eval_score_ks": calibration_eval_score_ks,
                "calibration_eval_mean_score_delta": calibration_eval_mean_score_delta,
                "eval_fpr_exceeds_budget": eval_fpr_exceeds_budget,
                "oracle_threshold": oracle_threshold,
                "oracle_tpr": oracle_tpr,
                "oracle_fpr": oracle_fpr,
                "source_holdout_tpr": source_holdout_tpr,
                "tpr_recovery": tpr_recovery,
                "relative_recovery": relative_recovery,
                "recal_diagnosis": recal_diagnosis,
                "recalibration_split_mode": split.split_mode,
                "recalibration_split_unit": split.split_unit,
                "recalibration_eval_disjoint": int(split.disjoint),
                "n_target_total": int(len(target)),
                "n_target_benign": int((target["label"].astype(int) == 0).sum()),
                "n_target_attacks": int((target["label"].astype(int) == 1).sum()),
                "n_benign_calibration": int(len(calibration_benign)),
                "n_benign_evaluation": int(len(evaluation_benign)),
                "n_calibration_units": int(len(split.calibration_units)),
                "n_evaluation_units": int(len(split.evaluation_units)),
            }
        )

    return pd.DataFrame(rows)
