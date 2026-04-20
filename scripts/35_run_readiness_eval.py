#!/usr/bin/env python3
"""Evaluate the Operational Transfer Readiness Score (OTRS) on benchmark settings."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iot_ids.decision import add_operational_metrics
from iot_ids.readiness import build_otrs_comparison


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate composite transfer readiness scores on benchmark settings."
    )
    parser.add_argument(
        "--settings",
        required=True,
        help="Path to benchmark settings CSV (artifacts/benchmark/settings.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/readiness",
        help="Directory to write readiness evaluation outputs.",
    )
    args = parser.parse_args()

    settings = pd.read_csv(args.settings)
    settings = add_operational_metrics(settings)

    # Build subsets.
    natural = settings.loc[settings["setting_type"] == "natural_transfer"].copy()
    primary_iot = natural.loc[natural["oracle_alpha_usable"] == 1].copy()

    subsets = {
        "all_natural": natural,
        "primary_iot": primary_iot,
    }

    comparison = build_otrs_comparison(natural, subsets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "readiness_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"Wrote readiness comparison to {comparison_path}")
    print()
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
