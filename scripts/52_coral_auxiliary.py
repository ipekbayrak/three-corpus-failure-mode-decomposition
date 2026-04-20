#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from iot_ids.config import load_config
from iot_ids.coral_auxiliary import CoralAuxiliaryOutputs, run_coral_auxiliary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the auxiliary CORAL domain-adaptation benchmark."
    )
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["coral_auxiliary"]
    outputs = CoralAuxiliaryOutputs(
        summary_csv=Path(cfg["outputs"]["summary_csv"]),
    )
    summary = run_coral_auxiliary(config, outputs)
    print(
        "Built auxiliary CORAL artifacts: "
        f"summary={summary['summary_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
