from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

RAW_REQUIRED_COLUMNS = [
    "record_id",
    "group_id",
    "label",
    "family_raw",
    "proto",
    "duration_s",
    "src_bytes",
    "dst_bytes",
    "src_pkts",
    "dst_pkts",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_csv: Path
    column_map: dict[str, str]
    version: str = "unset"


def _normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    lowered = str(value).strip().lower()
    if lowered in {"1", "attack", "attacker", "malicious", "malware", "true"}:
        return 1
    if lowered in {"0", "benign", "normal", "false"}:
        return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def _canonicalize_columns(frame: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    renamed = frame.rename(columns={raw: canonical for canonical, raw in column_map.items()})
    missing = [column for column in RAW_REQUIRED_COLUMNS if column not in renamed.columns]
    if missing:
        raise ValueError(f"Missing required mapped columns: {missing}")
    return renamed


def load_staged_dataset(spec: DatasetSpec) -> pd.DataFrame:
    if not spec.raw_csv.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {spec.raw_csv}")

    raw = pd.read_csv(spec.raw_csv)
    frame = _canonicalize_columns(raw, spec.column_map).copy()
    frame["dataset"] = spec.name
    frame["record_id"] = frame["record_id"].astype(str)
    frame["group_id"] = frame["group_id"].astype(str)
    frame["family_raw"] = frame["family_raw"].fillna("unknown").astype(str)
    frame["label"] = frame["label"].map(_normalize_label)
    for column in ["duration_s", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["proto"] = frame["proto"].fillna("other").astype(str)
    return frame


def load_dataset_specs(config: dict[str, Any]) -> dict[str, DatasetSpec]:
    datasets = {}
    for name, spec in config["datasets"].items():
        datasets[name] = DatasetSpec(
            name=name,
            raw_csv=Path(spec["raw_csv"]),
            column_map=spec["column_map"],
            version=str(spec.get("version", "unset")),
        )
    return datasets


def load_ontology_mapping(mapping_path: str | Path, default_group: str = "other") -> pd.DataFrame:
    path = Path(mapping_path)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "dataset",
                "raw_label",
                "mapped_group",
                "mapping_source",
                "mapping_confidence",
                "notes",
            ]
        )
    mapping = pd.read_csv(path)
    required = {"dataset", "raw_label", "mapped_group"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"Ontology mapping missing columns: {sorted(missing)}")
    mapping["mapped_group"] = mapping["mapped_group"].fillna(default_group).astype(str)
    return mapping


def apply_ontology_mapping(
    frame: pd.DataFrame,
    mapping: pd.DataFrame,
    default_group: str = "other",
) -> pd.DataFrame:
    if mapping.empty:
        result = frame.copy()
        result["family_group"] = result["family_raw"].str.strip().str.lower()
        return result

    lookup = mapping.rename(columns={"raw_label": "family_raw"}).copy()
    lookup["_family_raw_norm"] = lookup["family_raw"].astype(str).str.strip().str.lower()
    source = frame.copy()
    source["_family_raw_norm"] = source["family_raw"].astype(str).str.strip().str.lower()
    merged = source.merge(
        lookup[["dataset", "_family_raw_norm", "mapped_group"]],
        on=["dataset", "_family_raw_norm"],
        how="left",
    )
    merged["family_group"] = merged["mapped_group"].fillna(default_group).astype(str)
    return merged.drop(columns=["mapped_group", "_family_raw_norm"])
