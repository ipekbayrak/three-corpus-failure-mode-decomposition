from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from iot_ids.data import (
    apply_ontology_mapping,
    load_dataset_specs,
    load_ontology_mapping,
    load_staged_dataset,
)
from iot_ids.decision import relative_transfer_loss
from iot_ids.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, build_feature_schema_v1
from iot_ids.metrics import compute_metrics, threshold_at_fpr
from iot_ids.modeling import fit_estimator, score_positive_class
from iot_ids.novelty import (
    KNNNoveltyProxy,
    average_wasserstein_1d,
    compute_oracle_alpha,
    mapping_coverage,
    proxy_a_distance,
    standardized_matrices,
)
from iot_ids.validate import require_columns, validate_no_group_overlap


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class FittedSourceContext:
    estimator: Any
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    threshold: float
    source_holdout_metrics: dict[str, float]
    source_val_metrics: dict[str, float]


def _hash_key(value: str, seed: int) -> tuple[int, str]:
    payload = f"{seed}:{value}".encode()
    hashed = int(hashlib.sha256(payload).hexdigest(), 16)
    return hashed, value


def split_by_group(
    frame: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> SplitFrames:
    groups = sorted(
        frame["group_id"].astype(str).unique(),
        key=lambda value: _hash_key(value, seed),
    )
    n_groups = len(groups)
    train_stop = max(1, int(round(n_groups * train_frac)))
    val_stop = max(train_stop + 1, int(round(n_groups * (train_frac + val_frac))))
    train_groups = set(groups[:train_stop])
    val_groups = set(groups[train_stop:val_stop])
    test_groups = set(groups[val_stop:])
    if not test_groups:
        test_groups = val_groups
        val_groups = set()

    train = frame.loc[frame["group_id"].astype(str).isin(train_groups)].copy()
    val = frame.loc[frame["group_id"].astype(str).isin(val_groups)].copy()
    test = frame.loc[frame["group_id"].astype(str).isin(test_groups)].copy()
    if val.empty:
        val = test.sample(frac=0.5, random_state=seed).copy()
        test = test.drop(val.index).copy()
    validate_no_group_overlap(train, val)
    validate_no_group_overlap(train, test)
    validate_no_group_overlap(val, test)
    return SplitFrames(train=train, val=val, test=test)


def prepare_feature_frames(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    specs = load_dataset_specs(config)
    mapping = load_ontology_mapping(
        config["ontology"]["mapping_csv"],
        config["ontology"].get("default_group", "other"),
    )
    prepared: dict[str, pd.DataFrame] = {}
    processed_root = Path(config["paths"]["processed_root"])
    processed_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for name, spec in specs.items():
        staged = load_staged_dataset(spec)
        mapped = apply_ontology_mapping(
            staged,
            mapping,
            config["ontology"].get("default_group", "other"),
        )
        featured = build_feature_schema_v1(mapped)
        featured["dataset"] = name
        featured["family_group"] = featured["family_group"].fillna("other").astype(str)
        prepared[name] = featured
        output_path = processed_root / f"{name}_{FEATURE_SCHEMA_VERSION}.csv"
        featured.to_csv(output_path, index=False)
        manifest_rows.append(
            pd.DataFrame(
                {
                    "dataset": [name] * len(featured),
                    "record_id": featured["record_id"],
                    "group_id": featured["group_id"],
                    "label": featured["label"],
                    "family_raw": featured["family_raw"],
                    "family_group": featured["family_group"],
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                }
            )
        )

    feature_manifest = pd.concat(manifest_rows, ignore_index=True)
    manifest_path = Path(config["paths"]["manifests_root"]) / "feature_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    feature_manifest.to_csv(manifest_path, index=False)
    return prepared


def build_split_manifest(
    frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[dict[str, SplitFrames], pd.DataFrame]:
    split_frames: dict[str, SplitFrames] = {}
    manifest_rows = []
    split_cfg = config["split"]
    for name, frame in frames.items():
        splits = split_by_group(
            frame,
            train_frac=float(split_cfg["train_frac"]),
            val_frac=float(split_cfg["val_frac"]),
            seed=int(config["project"]["seed"]),
        )
        split_frames[name] = splits
        for role, subset in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
            rows = subset[["record_id", "group_id", "label", "family_raw", "family_group"]].copy()
            rows["dataset"] = name
            rows["split_id"] = f"{name}_default"
            rows["fold_role"] = role
            manifest_rows.append(rows)
    manifest = pd.concat(manifest_rows, ignore_index=True)
    manifest_path = Path(config["paths"]["manifests_root"]) / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return split_frames, manifest


def _setting_row(
    setting_id: str,
    setting_type: str,
    source_name: str,
    target_name: str,
    model_name: str,
    threshold: float,
    source_holdout_tpr: float,
    target_metrics: dict[str, float],
    oracle_alpha: float,
    oracle_alpha_usable: int,
    oracle_alpha_scope: str,
    alpha_hat: float,
    selected_count: int,
    selected_by_fallback: bool,
    proxy_a: float,
    wasserstein: float,
    source_val_tpr: float,
    source_mapped_attack_coverage: float,
    source_other_attack_share: float,
    target_mapped_attack_coverage: float,
    target_other_attack_share: float,
    controlled_novelty_level: float | None,
    fpr_budget: float,
) -> dict[str, Any]:
    loss_rel = relative_transfer_loss(target_metrics["tpr_at_1pct_fpr"], source_holdout_tpr)
    relative_tpr_drop = max(loss_rel, 0.0)
    relative_tpr_change = -loss_rel
    fpr_budget_multiple = float(target_metrics["realized_fpr"]) / max(float(fpr_budget), 1e-12)
    fpr_severity = min(1.0, max(0.0, fpr_budget_multiple - 1.0) / 9.0)
    operational_severity = max(relative_tpr_drop, fpr_severity)
    return {
        "setting_id": setting_id,
        "setting_type": setting_type,
        "source_dataset": source_name,
        "target_dataset": target_name,
        "model": model_name,
        "threshold": threshold,
        "source_holdout_tpr_at_1pct_fpr": source_holdout_tpr,
        "target_tpr_at_1pct_fpr": target_metrics["tpr_at_1pct_fpr"],
        "target_realized_fpr": target_metrics["realized_fpr"],
        "target_auroc": target_metrics["auroc"],
        "target_auprc": target_metrics["auprc"],
        "oracle_alpha": oracle_alpha,
        "oracle_alpha_usable": oracle_alpha_usable,
        "oracle_alpha_scope": oracle_alpha_scope,
        "alpha_hat": alpha_hat,
        "alpha_hat_selected_count": selected_count,
        "alpha_hat_used_fallback": int(selected_by_fallback),
        "proxy_a_distance": proxy_a,
        "avg_wasserstein_1d": wasserstein,
        "source_validation_tpr_at_1pct_fpr": source_val_tpr,
        "source_mapped_attack_coverage": source_mapped_attack_coverage,
        "source_other_attack_share": source_other_attack_share,
        "target_mapped_attack_coverage": target_mapped_attack_coverage,
        "target_other_attack_share": target_other_attack_share,
        "mapped_attack_coverage": target_mapped_attack_coverage,
        "other_attack_share": target_other_attack_share,
        "unsafe_setting": int(loss_rel > 0.5),
        "tpr_loss_unsafe": int(loss_rel > 0.5),
        "relative_tpr_drop": relative_tpr_drop,
        "relative_tpr_change": relative_tpr_change,
        "fpr_budget_multiple": fpr_budget_multiple,
        "fpr_severity": fpr_severity,
        "operational_severity": operational_severity,
        "catastrophic_operational_failure": int(operational_severity >= 1.0),
        "controlled_novelty_level": controlled_novelty_level,
        "loss_rel": loss_rel,
    }


def _build_novelty_proxy(
    source_train: pd.DataFrame,
    source_val: pd.DataFrame,
    target_eval: pd.DataFrame,
    feature_columns: list[str],
    scores_target: np.ndarray,
    threshold: float,
    alpha_hat_cfg: dict[str, Any],
) -> dict[str, Any]:
    attack_bank = source_train.loc[source_train["label"] == 1, feature_columns]
    if attack_bank.empty or len(attack_bank) < 2:
        return {"alpha_hat": float("nan"), "selected_count": 0, "selected_by_fallback": False}
    scaler_train, _, scaled_target = standardized_matrices(
        source_train,
        source_val,
        target_eval,
        feature_columns,
    )
    scaled_attack_bank = scaler_train[source_train["label"].to_numpy() == 1]
    proxy = KNNNoveltyProxy(
        k_nn=int(alpha_hat_cfg["k_nn"]),
        distance_quantile=float(alpha_hat_cfg["distance_quantile"]),
    ).fit(scaled_attack_bank)
    q_src_val_pos = max(1e-6, float((source_val["score"] >= threshold).mean()))
    fallback_k = int(np.ceil(q_src_val_pos * len(target_eval)))
    summary = proxy.summarize(
        features=scaled_target,
        scores=scores_target,
        threshold=threshold,
        fallback_k=fallback_k,
    )
    return {
        "alpha_hat": summary.alpha_hat,
        "selected_count": summary.selected_count,
        "selected_by_fallback": summary.selected_by_fallback,
    }


def _fit_source_context(
    model_name: str,
    source_splits: SplitFrames,
    seed: int,
    target_fpr: float,
) -> FittedSourceContext:
    feature_columns = FEATURE_COLUMNS
    estimator = fit_estimator(model_name, source_splits.train, feature_columns, seed)
    source_val = source_splits.val.copy()
    source_test = source_splits.test.copy()
    source_val["score"] = score_positive_class(estimator, source_val, feature_columns)
    source_test["score"] = score_positive_class(estimator, source_test, feature_columns)

    threshold = threshold_at_fpr(source_val["label"], source_val["score"], target_fpr)
    source_holdout_metrics = compute_metrics(source_test["label"], source_test["score"], threshold)
    source_val_metrics = compute_metrics(source_val["label"], source_val["score"], threshold)
    return FittedSourceContext(
        estimator=estimator,
        train=source_splits.train.copy(),
        val=source_val,
        test=source_test,
        threshold=threshold,
        source_holdout_metrics=source_holdout_metrics,
        source_val_metrics=source_val_metrics,
    )


def _prediction_rows(
    frame: pd.DataFrame,
    setting_id: str,
    source_name: str,
    target_name: str,
    model_name: str,
    threshold: float,
    eval_role: str,
) -> pd.DataFrame:
    predictions = frame[
        ["dataset", "record_id", "group_id", "label", "family_group", "score"]
    ].copy()
    predictions["setting_id"] = setting_id
    predictions["source_dataset"] = source_name
    predictions["target_dataset"] = target_name
    predictions["model"] = model_name
    predictions["threshold"] = threshold
    predictions["eval_role"] = eval_role
    return predictions


def _evaluate_target_with_context(
    source_name: str,
    target_name: str,
    model_name: str,
    fitted_source: FittedSourceContext,
    target_eval: pd.DataFrame,
    seed: int,
    alpha_hat_cfg: dict[str, Any],
    setting_type: str,
    controlled_novelty_level: float | None = None,
    fpr_budget: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_columns = FEATURE_COLUMNS
    target_eval = target_eval.copy()
    target_eval["score"] = score_positive_class(
        fitted_source.estimator,
        target_eval,
        feature_columns,
    )
    target_metrics = compute_metrics(
        target_eval["label"],
        target_eval["score"],
        fitted_source.threshold,
    )

    source_attack_groups = fitted_source.train.loc[
        fitted_source.train["label"] == 1,
        "family_group",
    ]
    target_attack_groups = target_eval.loc[target_eval["label"] == 1, "family_group"]
    oracle_alpha = compute_oracle_alpha(source_attack_groups, target_attack_groups)
    source_mapped_attack_coverage, source_other_attack_share = mapping_coverage(
        source_attack_groups
    )
    target_mapped_attack_coverage, target_other_attack_share = mapping_coverage(
        target_attack_groups
    )
    oracle_alpha_usable = int(
        source_mapped_attack_coverage >= 0.80 and target_mapped_attack_coverage >= 0.80
    )
    oracle_alpha_scope = (
        "primary" if oracle_alpha_usable else "stress_test_low_coverage"
    )

    novelty = _build_novelty_proxy(
        source_train=fitted_source.train,
        source_val=fitted_source.val,
        target_eval=target_eval,
        feature_columns=feature_columns,
        scores_target=target_eval["score"].to_numpy(),
        threshold=fitted_source.threshold,
        alpha_hat_cfg=alpha_hat_cfg,
    )

    source_train_x, _, target_x = standardized_matrices(
        fitted_source.train,
        fitted_source.val,
        target_eval,
        feature_columns,
    )
    proxy_a = proxy_a_distance(source_train_x, target_x, seed)
    wasserstein = average_wasserstein_1d(source_train_x, target_x)

    setting_id = f"{source_name}__{target_name}__{model_name}__{setting_type}"
    predictions = pd.concat(
        [
            _prediction_rows(
                fitted_source.val,
                setting_id,
                source_name,
                target_name,
                model_name,
                fitted_source.threshold,
                "source_validation",
            ),
            _prediction_rows(
                fitted_source.test,
                setting_id,
                source_name,
                target_name,
                model_name,
                fitted_source.threshold,
                "source_holdout",
            ),
            _prediction_rows(
                target_eval,
                setting_id,
                source_name,
                target_name,
                model_name,
                fitted_source.threshold,
                "target_transfer",
            ),
        ],
        ignore_index=True,
    )

    setting_row = _setting_row(
        setting_id=setting_id,
        setting_type=setting_type,
        source_name=source_name,
        target_name=target_name,
        model_name=model_name,
        threshold=fitted_source.threshold,
        source_holdout_tpr=fitted_source.source_holdout_metrics["tpr_at_1pct_fpr"],
        target_metrics=target_metrics,
        oracle_alpha=oracle_alpha,
        oracle_alpha_usable=oracle_alpha_usable,
        oracle_alpha_scope=oracle_alpha_scope,
        alpha_hat=novelty["alpha_hat"],
        selected_count=novelty["selected_count"],
        selected_by_fallback=novelty["selected_by_fallback"],
        proxy_a=proxy_a,
        wasserstein=wasserstein,
        source_val_tpr=fitted_source.source_val_metrics["tpr_at_1pct_fpr"],
        source_mapped_attack_coverage=source_mapped_attack_coverage,
        source_other_attack_share=source_other_attack_share,
        target_mapped_attack_coverage=target_mapped_attack_coverage,
        target_other_attack_share=target_other_attack_share,
        controlled_novelty_level=controlled_novelty_level,
        fpr_budget=fpr_budget,
    )
    return predictions, setting_row


def _exclude_attack_groups(frame: pd.DataFrame, excluded_groups: set[str]) -> pd.DataFrame:
    attack_mask = (frame["label"] == 1) & frame["family_group"].astype(str).isin(excluded_groups)
    return frame.loc[~attack_mask].copy()


def _make_controlled_source_splits(
    source_splits: SplitFrames,
    excluded_train_groups: list[str],
) -> SplitFrames:
    excluded = {str(group) for group in excluded_train_groups}
    filtered_train = _exclude_attack_groups(source_splits.train, excluded)
    return SplitFrames(
        train=filtered_train,
        val=source_splits.val.copy(),
        test=source_splits.test.copy(),
    )


def _make_controlled_targets(
    source_name: str,
    source_splits: SplitFrames,
    novelty_schedule: list[float],
    seen_groups: list[str],
    novel_groups: list[str],
) -> list[tuple[float, str, pd.DataFrame]]:
    test_frame = source_splits.test.copy()
    benign = test_frame.loc[test_frame["label"] == 0].sort_values("record_id")
    attacks = test_frame.loc[
        (test_frame["label"] == 1) & (test_frame["family_group"].astype(str) != "other")
    ].copy()
    seen_attacks = attacks.loc[attacks["family_group"].astype(str).isin(seen_groups)].sort_values(
        "record_id"
    )
    novel_attacks = attacks.loc[
        attacks["family_group"].astype(str).isin(novel_groups)
    ].sort_values("record_id")
    if seen_attacks.empty or novel_attacks.empty:
        raise RuntimeError(
            "Controlled shift requires non-empty seen and novel attack pools "
            "after ontology filtering."
        )

    n_attack = min(len(seen_attacks), len(novel_attacks))
    targets = []
    for novelty_level in novelty_schedule:
        n_novel = int(round(n_attack * novelty_level))
        n_seen = n_attack - n_novel
        target = pd.concat(
            [benign, seen_attacks.head(n_seen), novel_attacks.head(n_novel)],
            ignore_index=True,
        )
        targets.append((novelty_level, f"{source_name}_controlled_{novelty_level:.2f}", target))
    return targets


def run_transfer_matrix(
    frames: dict[str, pd.DataFrame],
    split_frames: dict[str, SplitFrames],
    model_names: list[str],
    directions: list[list[str]],
    seed: int,
    target_fpr: float,
    alpha_hat_cfg: dict[str, Any],
    include_controlled_shift: bool = False,
    controlled_shift_dataset: str | None = None,
    controlled_shift_excluded_train_groups: list[str] | None = None,
    controlled_shift_seen_groups: list[str] | None = None,
    controlled_shift_novel_groups: list[str] | None = None,
    novelty_schedule: list[float] | None = None,
    progress_callback: Callable[[pd.DataFrame, dict[str, Any], int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    settings_rows: list[dict[str, Any]] = []
    total_settings = len(model_names) * len(directions)
    if include_controlled_shift and controlled_shift_dataset:
        total_settings += len(model_names) * len(novelty_schedule or [])
    completed_settings = 0

    for model_name in model_names:
        fitted_by_source: dict[str, FittedSourceContext] = {}
        for source_name, target_name in directions:
            require_columns(
                frames[source_name],
                FEATURE_COLUMNS + ["label"],
                f"{source_name} features",
            )
            require_columns(
                frames[target_name],
                FEATURE_COLUMNS + ["label"],
                f"{target_name} features",
            )
            if source_name not in fitted_by_source:
                fitted_by_source[source_name] = _fit_source_context(
                    model_name=model_name,
                    source_splits=split_frames[source_name],
                    seed=seed,
                    target_fpr=target_fpr,
                )
            fitted_source = fitted_by_source[source_name]
            preds, row = _evaluate_target_with_context(
                source_name=source_name,
                target_name=target_name,
                model_name=model_name,
                fitted_source=fitted_source,
                target_eval=split_frames[target_name].test,
                seed=seed,
                alpha_hat_cfg=alpha_hat_cfg,
                setting_type="natural_transfer",
                fpr_budget=target_fpr,
            )
            prediction_rows.append(preds)
            settings_rows.append(row)
            completed_settings += 1
            print(
                f"[{completed_settings}/{total_settings}] completed "
                f"{row['setting_id']}",
                flush=True,
            )
            if progress_callback is not None:
                progress_callback(preds, row, completed_settings, total_settings)

        if include_controlled_shift and controlled_shift_dataset:
            source_splits = _make_controlled_source_splits(
                split_frames[controlled_shift_dataset],
                controlled_shift_excluded_train_groups or [],
            )
            fitted_source = _fit_source_context(
                model_name=model_name,
                source_splits=source_splits,
                seed=seed,
                target_fpr=target_fpr,
            )
            for novelty_level, target_name, target_frame in _make_controlled_targets(
                controlled_shift_dataset,
                source_splits,
                novelty_schedule or [],
                controlled_shift_seen_groups or [],
                controlled_shift_novel_groups or [],
            ):
                preds, row = _evaluate_target_with_context(
                    source_name=controlled_shift_dataset,
                    target_name=target_name,
                    model_name=model_name,
                    fitted_source=fitted_source,
                    target_eval=target_frame,
                    seed=seed,
                    alpha_hat_cfg=alpha_hat_cfg,
                    setting_type="controlled_shift",
                    controlled_novelty_level=novelty_level,
                    fpr_budget=target_fpr,
                )
                prediction_rows.append(preds)
                settings_rows.append(row)
                completed_settings += 1
                print(
                    f"[{completed_settings}/{total_settings}] completed "
                    f"{row['setting_id']}",
                    flush=True,
                )
                if progress_callback is not None:
                    progress_callback(preds, row, completed_settings, total_settings)

    predictions = (
        pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    )
    settings = pd.DataFrame(settings_rows)
    return predictions, settings
