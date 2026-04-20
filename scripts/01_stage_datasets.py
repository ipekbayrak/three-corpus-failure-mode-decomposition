#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from iot_ids.staging import stage_iot23, stage_ton_iot, stage_unsw_nb15

IOT23_CANDIDATES = [
    Path("/Users/mustafa/.cache/kagglehub/datasets/srifqi/iot-23-cleaned-data"),
    Path("/Users/mustafa/.cache/kagglehub/datasets/astralfate/iot23-dataset"),
    Path("/Users/mustafa/.cache/kagglehub/datasets/engraqeel/iot23preprocesseddata"),
    Path("/Users/mustafa/.cache/kagglehub/datasets/surajsooraj26/iot-23"),
]
TON_SOURCE = Path(
    "/Users/mustafa/.cache/kagglehub/datasets/arnobbhowmik/ton-iot-network-dataset/versions/1/train_test_network.csv"
)
UNSW_TRAIN = Path("/Users/mustafa/Downloads/UNSW_NB15_training-set.csv")
UNSW_TEST = Path("/Users/mustafa/Downloads/UNSW_NB15_testing-set.csv")


def _resolve_iot23_source() -> Path:
    for cache_root in IOT23_CANDIDATES:
        archive = cache_root / "1.archive"
        if archive.exists():
            return archive
        versions = sorted(cache_root.glob("versions/*"))
        if versions:
            return versions[-1]
    raise FileNotFoundError(
        "IoT-23 source not found under any known Kaggle cache root: "
        + ", ".join(str(path) for path in IOT23_CANDIDATES)
    )


def _print_summary(name: str, path: Path) -> None:
    frame = pd.read_csv(path, nrows=5)
    print(f"{name}: {path}")
    print(frame.to_string(index=False))


def main() -> int:
    iot23_output = Path("artifacts/raw/iot23/staged.csv")
    ton_output = Path("artifacts/raw/ton_iot/staged.csv")
    unsw_output = Path("artifacts/raw/unsw_nb15/staged.csv")

    iot23_rows = stage_iot23(_resolve_iot23_source(), iot23_output)
    ton_rows = stage_ton_iot(TON_SOURCE, ton_output)
    unsw_rows = stage_unsw_nb15(UNSW_TRAIN, UNSW_TEST, unsw_output)

    print(f"IoT-23 staged rows: {iot23_rows}")
    print(f"ToN-IoT staged rows: {ton_rows}")
    print(f"UNSW-NB15 staged rows: {unsw_rows}")
    _print_summary("IoT-23 sample", iot23_output)
    _print_summary("ToN-IoT sample", ton_output)
    _print_summary("UNSW-NB15 sample", unsw_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
