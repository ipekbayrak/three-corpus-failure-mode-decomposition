#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iot_ids.claim_package import (
    ClaimPackageArtifacts,
    build_claim_decision,
    build_claim_summary_markdown,
    build_monitor_comparison,
    build_primary_oracle_context,
    write_claim_package,
)
from iot_ids.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the full-data claim-ready diagnostic package from canonical artifacts."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    claim_cfg = config["claim_package"]

    settings_path = Path(claim_cfg["benchmark_settings"])
    within_path = Path(claim_cfg["benchmark_within_domain_summary"])
    cross_path = Path(claim_cfg["benchmark_cross_domain_summary"])
    decision_eval_path = Path(claim_cfg["decision_eval_audit"])

    settings = pd.read_csv(settings_path)
    within_summary = pd.read_csv(within_path)
    cross_summary = pd.read_csv(cross_path)

    monitor_comparison = build_monitor_comparison(settings)
    primary_context = build_primary_oracle_context(settings)
    decision = build_claim_decision(settings, monitor_comparison)
    summary_markdown = build_claim_summary_markdown(
        settings_path=settings_path,
        within_path=within_path,
        cross_path=cross_path,
        decision_eval_path=decision_eval_path,
        within_summary=within_summary,
        cross_summary=cross_summary,
        comparison=monitor_comparison,
        primary_context=primary_context,
        decision=decision,
    )

    outputs = claim_cfg["outputs"]
    artifacts = ClaimPackageArtifacts(
        monitor_comparison_csv=Path(outputs["monitor_comparison_csv"]),
        primary_oracle_context_csv=Path(outputs["primary_oracle_context_csv"]),
        claim_decision_json=Path(outputs["claim_decision_json"]),
        claim_summary_md=Path(outputs["claim_summary_md"]),
    )
    write_claim_package(
        artifacts=artifacts,
        monitor_comparison=monitor_comparison,
        primary_context=primary_context,
        decision=decision,
        summary_markdown=summary_markdown,
    )
    print(f"Wrote claim package to {artifacts.monitor_comparison_csv.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
