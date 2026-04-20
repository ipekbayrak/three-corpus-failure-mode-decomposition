from __future__ import annotations

import io
import shutil
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

STAGED_COLUMNS = [
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

SCHEMA_APPENDIX_MAPPING_RULES = {
    "iot23": {
        "record_id": "uid; fallback capture_id:row_index",
        "group_id": (
            "capture_id|id.orig_h|id.orig_p|id.resp_h|id.resp_p|proto; "
            "fallback record_id"
        ),
        "label": "label; fallback family_raw-derived binary label",
        "family_raw": "detailed-label or multi-label; fallback label",
        "proto": "proto; fallback proto_tcp/proto_udp/proto_icmp indicators",
        "duration_s": "duration or dur",
        "src_bytes": "orig_bytes or src_bytes or sbytes",
        "dst_bytes": "resp_bytes or dst_bytes or dbytes",
        "src_pkts": "orig_pkts or src_pkts or spkts",
        "dst_pkts": "resp_pkts or dst_pkts or dpkts",
    },
    "ton_iot": {
        "record_id": "synthetic ton_iot:row_index",
        "group_id": "src_ip|src_port|dst_ip|dst_port|proto",
        "label": "label",
        "family_raw": "type",
        "proto": "proto",
        "duration_s": "duration",
        "src_bytes": "src_bytes",
        "dst_bytes": "dst_bytes",
        "src_pkts": "src_pkts",
        "dst_pkts": "dst_pkts",
    },
    "unsw_nb15": {
        "record_id": "source_split:id",
        "group_id": "record_id",
        "label": "label",
        "family_raw": "attack_cat",
        "proto": "proto",
        "duration_s": "dur",
        "src_bytes": "sbytes",
        "dst_bytes": "dbytes",
        "src_pkts": "spkts",
        "dst_pkts": "dpkts",
    },
}


def _normalize_binary_label(value: object) -> int:
    lowered = str(value).strip().lower()
    if lowered in {"0", "false", "normal", "benign", "-", "background"}:
        return 0
    if lowered in {"1", "true", "attack", "malicious", "anomaly"}:
        return 1
    try:
        return int(float(lowered) > 0.0)
    except ValueError:
        return 0 if "benign" in lowered or "normal" in lowered else 1


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result


def _append_chunks(
    chunks: list[pd.DataFrame],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.unlink(missing_ok=True)
    first = True
    for chunk in chunks:
        _write_chunk(chunk, output_csv, first)
        first = False


def _write_chunk(chunk: pd.DataFrame, output_csv: Path, first: bool) -> None:
    staged = chunk[STAGED_COLUMNS + [c for c in chunk.columns if c not in STAGED_COLUMNS]]
    staged.to_csv(output_csv, index=False, mode="w" if first else "a", header=first)


def stage_ton_iot(source_csv: Path, output_csv: Path, chunksize: int = 200_000) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.unlink(missing_ok=True)
    first = True
    row_offset = 0
    usecols = [
        "src_ip",
        "src_port",
        "dst_ip",
        "dst_port",
        "proto",
        "duration",
        "src_bytes",
        "dst_bytes",
        "src_pkts",
        "dst_pkts",
        "label",
        "type",
    ]
    for chunk in pd.read_csv(source_csv, usecols=usecols, chunksize=chunksize):
        chunk = chunk.rename(columns={"type": "family_raw", "duration": "duration_s"})
        chunk = _coerce_numeric(
            chunk,
            ["duration_s", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts"],
        )
        chunk["label"] = chunk["label"].map(_normalize_binary_label)
        chunk["family_raw"] = chunk["family_raw"].fillna("unknown").astype(str)
        chunk["proto"] = chunk["proto"].fillna("other").astype(str)
        chunk["record_id"] = [f"ton_iot:{i}" for i in range(row_offset, row_offset + len(chunk))]
        chunk["group_id"] = (
            chunk["src_ip"].astype(str)
            + "|"
            + chunk["src_port"].astype(str)
            + "|"
            + chunk["dst_ip"].astype(str)
            + "|"
            + chunk["dst_port"].astype(str)
            + "|"
            + chunk["proto"].astype(str)
        )
        chunk["source_file"] = source_csv.name
        row_offset += len(chunk)
        _write_chunk(chunk, output_csv, first)
        first = False
    return row_offset


def stage_unsw_nb15(
    train_csv: Path,
    test_csv: Path,
    output_csv: Path,
    chunksize: int = 200_000,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.unlink(missing_ok=True)
    first = True
    total_rows = 0
    usecols = ["id", "dur", "proto", "spkts", "dpkts", "sbytes", "dbytes", "attack_cat", "label"]
    for split_name, source_csv in [("train", train_csv), ("test", test_csv)]:
        for chunk in pd.read_csv(source_csv, usecols=usecols, chunksize=chunksize):
            chunk = chunk.rename(
                columns={
                    "dur": "duration_s",
                    "spkts": "src_pkts",
                    "dpkts": "dst_pkts",
                    "sbytes": "src_bytes",
                    "dbytes": "dst_bytes",
                    "attack_cat": "family_raw",
                }
            )
            chunk = _coerce_numeric(
                chunk,
                ["duration_s", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts"],
            )
            chunk["label"] = chunk["label"].map(_normalize_binary_label)
            chunk["family_raw"] = chunk["family_raw"].fillna("unknown").astype(str)
            chunk["proto"] = chunk["proto"].fillna("other").astype(str)
            chunk["record_id"] = split_name + ":" + chunk["id"].astype(str)
            chunk["group_id"] = chunk["record_id"]
            chunk["source_split"] = split_name
            total_rows += len(chunk)
            _write_chunk(chunk, output_csv, first)
            first = False
    return total_rows


def _find_iot23_sources(root: Path) -> list[Path]:
    patterns = [
        "*conn*.csv",
        "*conn*.log",
        "*conn*.labeled",
        "*conn*.txt",
        "*.csv",
    ]
    files = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    unique = []
    seen = set()
    for path in sorted(files):
        if path.is_file() and path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _candidate_column(row_columns: list[str], names: list[str]) -> str | None:
    normalized = {column.strip().lower(): column for column in row_columns}
    for name in names:
        if name in normalized:
            return normalized[name]
    return None


def _delimiter_from_sample(sample: str) -> str:
    counts = {
        "|": sample.count("|"),
        "\t": sample.count("\t"),
        ",": sample.count(","),
    }
    return max(counts, key=counts.get)


def _load_tabular_with_sniff(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        sample = handle.read(4096)
    delimiter = _delimiter_from_sample(sample)
    return pd.read_csv(path, sep=delimiter, engine="python")


def _resolve_iot23_archive(archive_path: Path, extract_root: Path) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    if any(extract_root.iterdir()):
        return extract_root

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_root)
        return extract_root

    with tarfile.open(archive_path) as tf:
        tf.extractall(extract_root)
    return extract_root


def _transform_iot23_frame(
    frame: pd.DataFrame,
    source_name: str,
    capture_id: str,
    total_rows: int,
) -> tuple[pd.DataFrame | None, int]:
    columns = list(frame.columns)
    proto_col = _candidate_column(columns, ["proto"])
    duration_col = _candidate_column(columns, ["duration", "dur"])
    src_bytes_col = _candidate_column(columns, ["orig_bytes", "src_bytes", "sbytes"])
    dst_bytes_col = _candidate_column(columns, ["resp_bytes", "dst_bytes", "dbytes"])
    src_pkts_col = _candidate_column(columns, ["orig_pkts", "src_pkts", "spkts"])
    dst_pkts_col = _candidate_column(columns, ["resp_pkts", "dst_pkts", "dpkts"])
    uid_col = _candidate_column(columns, ["uid"])
    label_col = _candidate_column(columns, ["label"])
    detailed_col = _candidate_column(
        columns,
        ["detailed-label", "detailed_label", "multi-label", "multi_label"],
    )
    src_ip_col = _candidate_column(columns, ["id.orig_h", "src_ip"])
    src_port_col = _candidate_column(columns, ["id.orig_p", "src_port"])
    dst_ip_col = _candidate_column(columns, ["id.resp_h", "dst_ip"])
    dst_port_col = _candidate_column(columns, ["id.resp_p", "dst_port"])
    proto_tcp_col = _candidate_column(columns, ["proto_tcp"])
    proto_udp_col = _candidate_column(columns, ["proto_udp"])
    proto_icmp_col = _candidate_column(columns, ["proto_icmp"])

    required = [duration_col, src_bytes_col, dst_bytes_col, src_pkts_col, dst_pkts_col]
    if any(column is None for column in required):
        return None, total_rows

    family_col = detailed_col or label_col
    if family_col is None:
        return None, total_rows

    if proto_col is not None:
        proto_series = frame[proto_col]
    else:
        proto_series = pd.Series("other", index=frame.index, dtype="object")
        if proto_tcp_col is not None:
            proto_series = proto_series.mask(
                pd.to_numeric(frame[proto_tcp_col], errors="coerce").fillna(0) > 0,
                "tcp",
            )
        if proto_udp_col is not None:
            proto_series = proto_series.mask(
                pd.to_numeric(frame[proto_udp_col], errors="coerce").fillna(0) > 0,
                "udp",
            )
        if proto_icmp_col is not None:
            proto_series = proto_series.mask(
                pd.to_numeric(frame[proto_icmp_col], errors="coerce").fillna(0) > 0,
                "icmp",
            )

    subset = pd.DataFrame(
        {
            "proto": proto_series,
            "duration_s": frame[duration_col],
            "src_bytes": frame[src_bytes_col],
            "dst_bytes": frame[dst_bytes_col],
            "src_pkts": frame[src_pkts_col],
            "dst_pkts": frame[dst_pkts_col],
            "family_raw": frame[family_col].fillna("unknown"),
        }
    )
    if label_col is not None:
        subset["label"] = frame[label_col].map(_normalize_binary_label)
    else:
        subset["label"] = subset["family_raw"].map(_normalize_binary_label)

    subset = _coerce_numeric(
        subset,
        ["duration_s", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts"],
    )
    subset["proto"] = subset["proto"].fillna("other").astype(str)
    subset["family_raw"] = subset["family_raw"].fillna("unknown").astype(str)

    if uid_col is not None:
        subset["record_id"] = capture_id + ":" + frame[uid_col].astype(str)
    else:
        subset["record_id"] = [
            f"{capture_id}:{i}" for i in range(total_rows, total_rows + len(subset))
        ]

    if all(
        column is not None for column in [src_ip_col, src_port_col, dst_ip_col, dst_port_col]
    ):
        subset["group_id"] = (
            capture_id
            + "|"
            + frame[src_ip_col].astype(str)
            + "|"
            + frame[src_port_col].astype(str)
            + "|"
            + frame[dst_ip_col].astype(str)
            + "|"
            + frame[dst_port_col].astype(str)
            + "|"
            + subset["proto"].astype(str)
        )
    else:
        subset["group_id"] = subset["record_id"]

    subset["capture_id"] = capture_id
    subset["source_file"] = source_name
    return subset, total_rows + len(subset)


def _iot23_capture_id(root: Path, source_file: Path) -> str:
    relative = source_file.relative_to(root)
    if relative.parent == Path("."):
        return relative.stem
    return str(relative.parent).replace("/", "::")


def stage_iot23(source_path: Path, output_csv: Path) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.unlink(missing_ok=True)
    first = True
    total_rows = 0
    if source_path.is_file() and tarfile.is_tarfile(source_path):
        with tarfile.open(source_path, "r:*") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.isfile() and "conn.log.labeled" in member.name
            ]
            if not members:
                raise FileNotFoundError(
                    f"No candidate IoT-23 conn.log.labeled files found in archive {source_path}"
                )
            for member in members:
                preview = archive.extractfile(member)
                if preview is None:
                    continue
                sample = preview.read(4096).decode("utf-8", errors="ignore")
                delimiter = _delimiter_from_sample(sample)
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                text_stream = io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
                for chunk in pd.read_csv(
                    text_stream,
                    sep=delimiter,
                    engine="python",
                    chunksize=200_000,
                ):
                    capture_id = str(Path(member.name).parent).replace("/", "::")
                    subset, total_rows = _transform_iot23_frame(
                        chunk,
                        source_name=member.name,
                        capture_id=capture_id,
                        total_rows=total_rows,
                    )
                    if subset is None:
                        continue
                    _write_chunk(subset, output_csv, first)
                    first = False
    elif source_path.is_file() and zipfile.is_zipfile(source_path):
        with zipfile.ZipFile(source_path) as archive:
            members = [name for name in archive.namelist() if "conn.log.labeled" in name]
            if not members:
                raise FileNotFoundError(
                    f"No candidate IoT-23 conn.log.labeled files found in archive {source_path}"
                )
            for member_name in members:
                with archive.open(member_name) as preview:
                    sample = preview.read(4096).decode("utf-8", errors="ignore")
                delimiter = _delimiter_from_sample(sample)
                with archive.open(member_name) as stream:
                    text_stream = io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
                    for chunk in pd.read_csv(
                        text_stream,
                        sep=delimiter,
                        engine="python",
                        chunksize=200_000,
                    ):
                        capture_id = str(Path(member_name).parent).replace("/", "::")
                        subset, total_rows = _transform_iot23_frame(
                            chunk,
                            source_name=member_name,
                            capture_id=capture_id,
                            total_rows=total_rows,
                        )
                        if subset is None:
                            continue
                        _write_chunk(subset, output_csv, first)
                        first = False
    else:
        root = source_path
        conn_files = _find_iot23_sources(root)
        if not conn_files:
            raise FileNotFoundError(f"No candidate IoT-23 connection files found under {root}")
        for conn_file in conn_files:
            frame = _load_tabular_with_sniff(conn_file)
            capture_id = _iot23_capture_id(root, conn_file)
            subset, total_rows = _transform_iot23_frame(
                frame,
                source_name=str(conn_file.relative_to(root)),
                capture_id=capture_id,
                total_rows=total_rows,
            )
            if subset is None:
                continue
            _write_chunk(subset, output_csv, first)
            first = False

    if first:
        raise RuntimeError(
            f"Unable to stage IoT-23 from {source_path}; no compatible conn files were found"
        )
    return total_rows


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
