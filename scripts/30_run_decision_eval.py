#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iot_ids.config import load_config
from iot_ids.decision import add_operational_metrics, deploy_abstain_curve, summarise_monitor_performance


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deploy or abstain decisions.")
    parser.add_argument("--config", required=True, help="Path to decision YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    settings_path = Path(config["decision"]["benchmark_settings_csv"])
    if not settings_path.exists():
        raise FileNotFoundError(f"Benchmark settings not found: {settings_path}")

    settings = pd.read_csv(settings_path)
    decision_cfg = config["decision"]
    settings = add_operational_metrics(settings)

    score_specs = [
        ("alpha_hat", "alpha_hat"),
        ("risk_from_source_validation_tpr", "source_validation_tpr_at_1pct_fpr"),
        ("proxy_a_distance", "proxy_a_distance"),
        ("avg_wasserstein_1d", "avg_wasserstein_1d"),
    ]
    settings["risk_from_source_validation_tpr"] = (
        -settings["source_validation_tpr_at_1pct_fpr"].astype(float)
    )

    summary_rows = []
    curve_rows = []
    for score_name, score_column in score_specs:
        working_column = (
            score_name if score_name == "risk_from_source_validation_tpr" else score_column
        )
        metrics = summarise_monitor_performance(settings, working_column)
        metrics["score_column"] = score_name
        summary_rows.append(metrics)
        curve_rows.append(deploy_abstain_curve(settings, working_column))

    summary = pd.DataFrame(summary_rows)
    curve = pd.concat(curve_rows, ignore_index=True)
    output_path = Path(decision_cfg["outputs"]["decision_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    curve.to_csv(output_path.with_name("decision_abstain_curve.csv"), index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
