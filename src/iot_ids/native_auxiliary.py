from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from iot_ids.data import (
    apply_ontology_mapping,
    load_dataset_specs,
    load_ontology_mapping,
    load_staged_dataset,
)
from iot_ids.experiment import SplitFrames, split_by_group
from iot_ids.features import FEATURE_COLUMNS, build_feature_schema_v1
from iot_ids.metrics import compute_metrics, threshold_at_fpr
from iot_ids.modeling import fit_estimator, score_positive_class

NATIVE_METADATA_COLUMNS = [
    "record_id",
    "group_id",
    "label",
    "family_raw",
    "family_group",
    "dataset",
]

PROVENANCE_COLUMNS = {
    "record_id",
    "group_id",
    "label",
    "family_raw",
    "family_group",
    "dataset",
    "feature_schema_version",
    "source_file",
    "source_split",
    "capture_id",
    "id",
}

RAW_IDENTITY_COLUMNS = {"src_ip", "dst_ip", "src_port", "dst_port"}


@dataclass(frozen=True)
class NativeAuxiliaryOutputs:
    summary_csv: Path
    feature_manifest_csv: Path
    split_audit_csv: Path
    journal_table: Path
    acsac_table: Path
    staged_root: Path


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _ip_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for value in series.fillna("").astype(str):
        try:
            parsed = ipaddress.ip_address(value)
            rows.append(
                {
                    f"{prefix}_is_private": float(parsed.is_private),
                    f"{prefix}_is_ipv6": float(parsed.version == 6),
                    f"{prefix}_is_multicast": float(parsed.is_multicast),
                }
            )
        except ValueError:
            rows.append(
                {
                    f"{prefix}_is_private": 0.0,
                    f"{prefix}_is_ipv6": 0.0,
                    f"{prefix}_is_multicast": 0.0,
                }
            )
    return pd.DataFrame(rows, index=series.index)


def _port_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    values = pd.to_numeric(series, errors="coerce")
    return pd.DataFrame(
        {
            f"{prefix}_port_well_known": ((values >= 0) & (values <= 1023))
            .fillna(False)
            .astype(float),
            f"{prefix}_port_registered": ((values >= 1024) & (values <= 49151))
            .fillna(False)
            .astype(float),
            f"{prefix}_port_dynamic": ((values >= 49152) & (values <= 65535))
            .fillna(False)
            .astype(float),
            f"{prefix}_port_missing": values.isna().astype(float),
        },
        index=series.index,
    )


def _native_extra_features(
    raw_frame: pd.DataFrame,
    max_categorical_cardinality: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_frame = pd.DataFrame(index=raw_frame.index)
    manifest_rows: list[dict[str, object]] = []

    for column in raw_frame.columns:
        if column in PROVENANCE_COLUMNS:
            continue
        if column in RAW_IDENTITY_COLUMNS:
            continue
        if column in FEATURE_COLUMNS:
            continue

        series = raw_frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any() and series.dtype.kind in {"i", "u", "f"}:
            feature_name = f"native_num__{column}"
            feature_frame[feature_name] = numeric.fillna(0.0).astype(float)
            manifest_rows.append(
                {
                    "raw_column": column,
                    "feature_name": feature_name,
                    "feature_origin": "native_numeric",
                }
            )
            continue

        if series.dtype == object or str(series.dtype).startswith("string"):
            cardinality = int(series.fillna("missing").astype(str).nunique())
            if 1 < cardinality <= max_categorical_cardinality:
                values = series.fillna("missing").astype(str)
                for category in sorted(values.unique()):
                    safe_category = _safe_name(category)
                    feature_name = f"native_cat__{column}__{safe_category}"
                    feature_frame[feature_name] = (values == category).astype(float)
                    manifest_rows.append(
                        {
                            "raw_column": column,
                            "feature_name": feature_name,
                            "feature_origin": "native_categorical",
                        }
                    )

    for raw_column, prefix in [("src_ip", "src"), ("dst_ip", "dst")]:
        if raw_column in raw_frame.columns:
            derived = _ip_features(raw_frame[raw_column], prefix)
            feature_frame = pd.concat([feature_frame, derived], axis=1)
            for feature_name in derived.columns:
                manifest_rows.append(
                    {
                        "raw_column": raw_column,
                        "feature_name": feature_name,
                        "feature_origin": "endpoint_summary",
                    }
                )

    for raw_column, prefix in [("src_port", "src"), ("dst_port", "dst")]:
        if raw_column in raw_frame.columns:
            derived = _port_features(raw_frame[raw_column], prefix)
            feature_frame = pd.concat([feature_frame, derived], axis=1)
            for feature_name in derived.columns:
                manifest_rows.append(
                    {
                        "raw_column": raw_column,
                        "feature_name": feature_name,
                        "feature_origin": "endpoint_summary",
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    if manifest.empty:
        manifest = pd.DataFrame(columns=["raw_column", "feature_name", "feature_origin"])
    return feature_frame, manifest


def build_native_augmented_frame(
    raw_frame: pd.DataFrame,
    featured_frame: pd.DataFrame,
    *,
    max_categorical_cardinality: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    native_extra, manifest = _native_extra_features(
        raw_frame,
        max_categorical_cardinality=max_categorical_cardinality,
    )
    combined = pd.concat(
        [
            featured_frame[NATIVE_METADATA_COLUMNS + FEATURE_COLUMNS].reset_index(drop=True),
            native_extra.reset_index(drop=True),
        ],
        axis=1,
    )
    return combined, manifest


def build_native_frames(
    config: dict[str, Any],
    *,
    max_categorical_cardinality: int = 16,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    specs = load_dataset_specs(config)
    mapping = load_ontology_mapping(
        config["ontology"]["mapping_csv"],
        config["ontology"].get("default_group", "other"),
    )
    frames: dict[str, pd.DataFrame] = {}
    manifest_rows: list[pd.DataFrame] = []

    for dataset_name, spec in specs.items():
        raw = load_staged_dataset(spec)
        mapped = apply_ontology_mapping(
            raw,
            mapping,
            config["ontology"].get("default_group", "other"),
        )
        featured = build_feature_schema_v1(mapped)
        featured["dataset"] = dataset_name
        featured["family_group"] = featured["family_group"].fillna("other").astype(str)
        native_frame, manifest = build_native_augmented_frame(
            mapped.assign(dataset=dataset_name),
            featured,
            max_categorical_cardinality=max_categorical_cardinality,
        )
        native_frame["dataset"] = dataset_name
        frames[dataset_name] = native_frame
        shared_manifest = pd.DataFrame(
            {
                "raw_column": ["shared_schema"] * len(FEATURE_COLUMNS),
                "feature_name": FEATURE_COLUMNS,
                "feature_origin": ["shared_canonical"] * len(FEATURE_COLUMNS),
            }
        )
        manifest = manifest.copy()
        manifest = pd.concat([shared_manifest, manifest], ignore_index=True)
        manifest["dataset"] = dataset_name
        manifest_rows.append(manifest)

    manifest_frame = pd.concat(manifest_rows, ignore_index=True)
    return frames, manifest_frame


def _split_manifest_lookup(split_manifest: pd.DataFrame) -> dict[tuple[str, str], str]:
    return {
        (str(row.dataset), str(row.record_id)): str(row.fold_role)
        for row in split_manifest.itertuples(index=False)
    }


def apply_split_manifest(
    frame: pd.DataFrame,
    split_manifest: pd.DataFrame | None,
    dataset_name: str,
    seed: int,
) -> SplitFrames:
    if split_manifest is None or split_manifest.empty:
        return split_by_group(frame, train_frac=0.6, val_frac=0.2, seed=seed)

    lookup = _split_manifest_lookup(split_manifest)
    working = frame.copy()
    working["fold_role"] = working["record_id"].map(
        lambda value: lookup.get((dataset_name, str(value)), "")
    )
    if (working["fold_role"] == "").any():
        raise ValueError(f"Split manifest missing rows for dataset {dataset_name}")
    return SplitFrames(
        train=working.loc[working["fold_role"] == "train"].drop(columns=["fold_role"]).copy(),
        val=working.loc[working["fold_role"] == "val"].drop(columns=["fold_role"]).copy(),
        test=working.loc[working["fold_role"] == "test"].drop(columns=["fold_role"]).copy(),
    )


def evaluate_within_domain_representation(
    frame: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    seed: int,
    target_fpr: float,
    split_frames: SplitFrames,
) -> dict[str, float]:
    estimator = fit_estimator(model_name, split_frames.train, feature_columns, seed)
    source_val = split_frames.val.copy()
    source_test = split_frames.test.copy()
    source_val["score"] = score_positive_class(estimator, source_val, feature_columns)
    source_test["score"] = score_positive_class(estimator, source_test, feature_columns)
    threshold = threshold_at_fpr(source_val["label"], source_val["score"], target_fpr)
    metrics = compute_metrics(source_test["label"], source_test["score"], threshold)
    return {
        "threshold": threshold,
        "source_holdout_tpr_at_1pct_fpr": float(metrics["tpr_at_1pct_fpr"]),
        "source_holdout_realized_fpr": float(metrics["realized_fpr"]),
        "source_holdout_auroc": float(metrics["auroc"]),
        "source_holdout_auprc": float(metrics["auprc"]),
    }


def run_native_auxiliary(
    config: dict[str, Any],
    outputs: NativeAuxiliaryOutputs,
) -> dict[str, int]:
    aux_cfg = config["native_auxiliary"]
    seed = int(config["project"]["seed"])
    target_fpr = float(aux_cfg.get("target_fpr", 0.01))
    models = list(aux_cfg.get("models", ["hist_gradient_boosting"]))
    split_manifest_path = Path(aux_cfg.get("split_manifest_csv", ""))
    split_manifest = (
        pd.read_csv(split_manifest_path)
        if str(split_manifest_path) and split_manifest_path.exists()
        else None
    )

    native_frames, manifest = build_native_frames(
        config,
        max_categorical_cardinality=int(aux_cfg.get("max_categorical_cardinality", 16)),
    )

    rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    outputs.staged_root.mkdir(parents=True, exist_ok=True)

    for dataset_name, native_frame in native_frames.items():
        native_frame.to_csv(
            outputs.staged_root / f"{dataset_name}_native_features.csv", index=False
        )
        shared_frame = native_frame[NATIVE_METADATA_COLUMNS + FEATURE_COLUMNS].copy()
        for representation_name, representation_frame in [
            ("shared_16", shared_frame),
            ("native_aux", native_frame),
        ]:
            split_frames = apply_split_manifest(
                representation_frame,
                split_manifest,
                dataset_name,
                seed=seed,
            )
            for role, subset in [
                ("train", split_frames.train),
                ("val", split_frames.val),
                ("test", split_frames.test),
            ]:
                split_rows.append(
                    {
                        "dataset": dataset_name,
                        "representation": representation_name,
                        "split_role": role,
                        "n_rows": len(subset),
                        "n_groups": subset["group_id"].astype(str).nunique(),
                    }
                )
            feature_columns = [
                column
                for column in representation_frame.columns
                if column not in NATIVE_METADATA_COLUMNS
            ]
            for model_name in models:
                metrics = evaluate_within_domain_representation(
                    representation_frame,
                    feature_columns,
                    model_name,
                    seed=seed,
                    target_fpr=target_fpr,
                    split_frames=split_frames,
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "representation": representation_name,
                        "model": model_name,
                        "n_features": len(feature_columns),
                        **metrics,
                    }
                )

    summary = (
        pd.DataFrame(rows)
        .sort_values(["dataset", "model", "representation"])
        .reset_index(drop=True)
    )
    feature_manifest = manifest.sort_values(
        ["dataset", "feature_origin", "feature_name"]
    ).reset_index(drop=True)
    split_audit = (
        pd.DataFrame(split_rows)
        .sort_values(["dataset", "representation", "split_role"])
        .reset_index(drop=True)
    )

    outputs.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outputs.summary_csv, index=False)
    feature_manifest.to_csv(outputs.feature_manifest_csv, index=False)
    split_audit.to_csv(outputs.split_audit_csv, index=False)

    _write_auxiliary_table(
        summary,
        outputs.journal_table,
        label="tab:native-feature-aux",
        font_size=r"\footnotesize",
        resize_to_textwidth=True,
    )
    _write_auxiliary_table(
        summary,
        outputs.acsac_table,
        label="tab:acsac-native-feature-aux",
        font_size=r"\scriptsize",
        resize_to_textwidth=True,
        column_float=True,
    )

    return {
        "summary_rows": int(len(summary)),
        "feature_rows": int(len(feature_manifest)),
        "split_rows": int(len(split_audit)),
    }


def _write_auxiliary_table(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    label: str,
    font_size: str,
    resize_to_textwidth: bool,
    column_float: bool = False,
) -> None:
    rows: list[dict[str, object]] = []
    for (dataset, model), subset in summary.groupby(["dataset", "model"], sort=True):
        shared = subset.loc[subset["representation"] == "shared_16"].iloc[0]
        native = subset.loc[subset["representation"] == "native_aux"].iloc[0]
        delta_auroc = float(
            native["source_holdout_auroc"] - shared["source_holdout_auroc"]
        )
        delta_tpr = float(
            native["source_holdout_tpr_at_1pct_fpr"]
            - shared["source_holdout_tpr_at_1pct_fpr"]
        )
        rows.append(
            {
                "Dataset": dataset,
                "Model": model,
                "Shared feat.": int(shared["n_features"]),
                "Native feat.": int(native["n_features"]),
                "Shared AUROC": f"{float(shared['source_holdout_auroc']):.3f}",
                "Native AUROC": f"{float(native['source_holdout_auroc']):.3f}",
                r"$\Delta$ AUROC": f"{delta_auroc:.3f}",
                r"$\Delta$ TPR@1\%": f"{delta_tpr:.3f}",
            }
        )
    frame = pd.DataFrame(rows)
    frame["Dataset"] = frame["Dataset"].map(
        {"iot23": "IoT-23", "ton_iot": "ToN-IoT", "unsw_nb15": "UNSW-NB15"}
    )
    frame["Model"] = (
        frame["Model"]
        .map(
            {
                "hist_gradient_boosting": "HGB",
                "ft_transformer": "FTT",
                "logistic_regression": "LR",
                "extra_trees": "ET",
                "mlp": "MLP",
            }
        )
        .fillna(frame["Model"])
    )
    lines = [
        "% Generated by src/iot_ids/native_auxiliary.py",
        r"\begin{table}[t]" if column_float else r"\begin{table*}[t]",
        r"\centering",
        font_size,
        r"\setlength{\tabcolsep}{2pt}",
        (
            r"\caption{Auxiliary within-domain information-retention audit. "
            r"The canonical shared-16 representation is compared with a "
            r"dataset-native augmented representation that adds admissible "
            r"local fields and summarized endpoint features. HGB is shown "
            r"as one representative strong shared-schema model; this "
            r"HGB-only audit does not establish the same retention behavior "
            r"for LR / ET / MLP. These native features are dataset-specific "
            r"and are not used in the cross-dataset transfer benchmark.}"
        ),
        rf"\label{{{label}}}",
    ]
    if resize_to_textwidth:
        target_width = r"\columnwidth" if column_float else r"\textwidth"
        lines.append(rf"\resizebox{{{target_width}}}{{!}}{{%")
    lines.extend(
        [
            r"\begin{tabular}{@{}llcccccc@{}}",
            r"\toprule",
            (
                r"Dataset & Model & \begin{tabular}[c]{@{}c@{}}Shared\\feat.\end{tabular} "
                r"& \begin{tabular}[c]{@{}c@{}}Native\\feat.\end{tabular} "
                r"& \begin{tabular}[c]{@{}c@{}}Shared\\AUROC\end{tabular} "
                r"& \begin{tabular}[c]{@{}c@{}}Native\\AUROC\end{tabular} "
                r"& \begin{tabular}[c]{@{}c@{}}$\Delta$\\AUROC\end{tabular} "
                r"& \begin{tabular}[c]{@{}c@{}}$\Delta$\\TPR@1\%\end{tabular} \\"
            ),
            r"\midrule",
        ]
    )
    for row in frame.itertuples(index=False):
        lines.append(" & ".join(map(str, row)) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if resize_to_textwidth:
        lines.append(r"}")
    lines.extend([r"\end{table}" if column_float else r"\end{table*}", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
