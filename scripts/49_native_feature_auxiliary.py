#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from iot_ids.config import load_config
from iot_ids.native_auxiliary import NativeAuxiliaryOutputs, run_native_auxiliary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the auxiliary native-feature information-retention study."
    )
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["native_auxiliary"]
    outputs = NativeAuxiliaryOutputs(
        summary_csv=Path(cfg["outputs"]["summary_csv"]),
        feature_manifest_csv=Path(cfg["outputs"]["feature_manifest_csv"]),
        split_audit_csv=Path(cfg["outputs"]["split_audit_csv"]),
        journal_table=Path(cfg["outputs"]["journal_table"]),
        acsac_table=Path(cfg["outputs"]["acsac_table"]),
        staged_root=Path(cfg["outputs"]["staged_root"]),
    )
    summary = run_native_auxiliary(config, outputs)
    print(
        "Built native auxiliary artifacts: "
        f"summary={summary['summary_rows']}, "
        f"features={summary['feature_rows']}, "
        f"splits={summary['split_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
