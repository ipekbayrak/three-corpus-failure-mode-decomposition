#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iot_ids.config import load_config
from iot_ids.pilot_checkpoint import extract_dataset_coverage, summarize_pilot_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute the pilot checkpoint from artifacts.")
    parser.add_argument("--config", required=True, help="Path to pilot YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    outputs = config["pilot"]["outputs"]
    settings_path = Path(outputs["settings_csv"])
    checkpoint_path = Path(outputs["checkpoint_csv"])
    if not settings_path.exists():
        raise FileNotFoundError(f"Pilot settings artifact not found: {settings_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Existing pilot checkpoint not found: {checkpoint_path}")

    settings = pd.read_csv(settings_path)
    previous_checkpoint = pd.read_csv(checkpoint_path)
    dataset_coverage = extract_dataset_coverage(previous_checkpoint)
    if settings.empty:
        raise RuntimeError("Pilot settings are empty; cannot recompute checkpoint")

    if "run_id" in settings.columns and settings["run_id"].notna().any():
        run_id = str(settings["run_id"].dropna().iloc[0])
    elif "run_id" in previous_checkpoint.columns and previous_checkpoint["run_id"].notna().any():
        run_id = str(previous_checkpoint["run_id"].dropna().iloc[0])
    else:
        run_id = "latest"

    summary = summarize_pilot_checkpoint(settings, dataset_coverage, run_id=run_id)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(checkpoint_path, index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
