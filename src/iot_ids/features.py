from __future__ import annotations

import numpy as np
import pandas as pd

PRIMITIVE_COLUMNS = [
    "duration_s",
    "src_bytes",
    "dst_bytes",
    "src_pkts",
    "dst_pkts",
    "proto",
]

FEATURE_SCHEMA_VERSION = "feature_schema_v1"
FEATURE_COLUMNS = [
    "log_duration_s",
    "log_src_bytes",
    "log_dst_bytes",
    "log_src_pkts",
    "log_dst_pkts",
    "log_total_bytes",
    "log_total_pkts",
    "log_bytes_per_s",
    "log_pkts_per_s",
    "src_byte_frac",
    "src_pkt_frac",
    "bytes_per_pkt",
    "proto_tcp",
    "proto_udp",
    "proto_icmp",
    "proto_other",
]

FEATURE_APPENDIX_ROWS = [
    {
        "feature_name": "log_duration_s",
        "definition": "log1p(max(duration_s, 0))",
        "derived_from": "duration_s",
    },
    {
        "feature_name": "log_src_bytes",
        "definition": "log1p(max(src_bytes, 0))",
        "derived_from": "src_bytes",
    },
    {
        "feature_name": "log_dst_bytes",
        "definition": "log1p(max(dst_bytes, 0))",
        "derived_from": "dst_bytes",
    },
    {
        "feature_name": "log_src_pkts",
        "definition": "log1p(max(src_pkts, 0))",
        "derived_from": "src_pkts",
    },
    {
        "feature_name": "log_dst_pkts",
        "definition": "log1p(max(dst_pkts, 0))",
        "derived_from": "dst_pkts",
    },
    {
        "feature_name": "log_total_bytes",
        "definition": "log1p(src_bytes + dst_bytes)",
        "derived_from": "src_bytes, dst_bytes",
    },
    {
        "feature_name": "log_total_pkts",
        "definition": "log1p(src_pkts + dst_pkts)",
        "derived_from": "src_pkts, dst_pkts",
    },
    {
        "feature_name": "log_bytes_per_s",
        "definition": "log1p((src_bytes + dst_bytes) / duration_s); 0 when duration_s = 0",
        "derived_from": "duration_s, src_bytes, dst_bytes",
    },
    {
        "feature_name": "log_pkts_per_s",
        "definition": "log1p((src_pkts + dst_pkts) / duration_s); 0 when duration_s = 0",
        "derived_from": "duration_s, src_pkts, dst_pkts",
    },
    {
        "feature_name": "src_byte_frac",
        "definition": "src_bytes / (src_bytes + dst_bytes); 0.5 when total_bytes = 0",
        "derived_from": "src_bytes, dst_bytes",
    },
    {
        "feature_name": "src_pkt_frac",
        "definition": "src_pkts / (src_pkts + dst_pkts); 0.5 when total_pkts = 0",
        "derived_from": "src_pkts, dst_pkts",
    },
    {
        "feature_name": "bytes_per_pkt",
        "definition": "(src_bytes + dst_bytes) / (src_pkts + dst_pkts); 0 when total_pkts = 0",
        "derived_from": "src_bytes, dst_bytes, src_pkts, dst_pkts",
    },
    {
        "feature_name": "proto_tcp",
        "definition": "1[proto_norm = tcp]",
        "derived_from": "proto",
    },
    {
        "feature_name": "proto_udp",
        "definition": "1[proto_norm = udp]",
        "derived_from": "proto",
    },
    {
        "feature_name": "proto_icmp",
        "definition": "1[proto_norm = icmp]",
        "derived_from": "proto",
    },
    {
        "feature_name": "proto_other",
        "definition": "1[proto_norm = other]",
        "derived_from": "proto",
    },
]


def _safe_log1p(values: pd.Series) -> pd.Series:
    return np.log1p(values.clip(lower=0.0))


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _normalize_proto(value: str) -> str:
    lowered = str(value).strip().lower()
    if lowered in {"tcp", "6"}:
        return "tcp"
    if lowered in {"udp", "17"}:
        return "udp"
    if lowered in {"icmp", "1"}:
        return "icmp"
    return "other"


def build_feature_schema_v1(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in PRIMITIVE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing primitive columns for feature schema: {missing}")

    result = frame.copy()
    result["proto_norm"] = result["proto"].map(_normalize_proto)

    total_bytes = result["src_bytes"] + result["dst_bytes"]
    total_pkts = result["src_pkts"] + result["dst_pkts"]
    bytes_per_s = _safe_divide(total_bytes, result["duration_s"])
    pkts_per_s = _safe_divide(total_pkts, result["duration_s"])
    bytes_per_pkt = _safe_divide(total_bytes, total_pkts)

    result["log_duration_s"] = _safe_log1p(result["duration_s"])
    result["log_src_bytes"] = _safe_log1p(result["src_bytes"])
    result["log_dst_bytes"] = _safe_log1p(result["dst_bytes"])
    result["log_src_pkts"] = _safe_log1p(result["src_pkts"])
    result["log_dst_pkts"] = _safe_log1p(result["dst_pkts"])
    result["log_total_bytes"] = _safe_log1p(total_bytes)
    result["log_total_pkts"] = _safe_log1p(total_pkts)
    result["log_bytes_per_s"] = _safe_log1p(bytes_per_s.fillna(0.0))
    result["log_pkts_per_s"] = _safe_log1p(pkts_per_s.fillna(0.0))
    result["src_byte_frac"] = _safe_divide(result["src_bytes"], total_bytes).fillna(0.5)
    result["src_pkt_frac"] = _safe_divide(result["src_pkts"], total_pkts).fillna(0.5)
    result["bytes_per_pkt"] = bytes_per_pkt.fillna(0.0)

    for proto_name in ["tcp", "udp", "icmp", "other"]:
        result[f"proto_{proto_name}"] = (result["proto_norm"] == proto_name).astype(float)

    result["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    return result
