#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from iot_ids.acsac_diagnostics import (
    build_fpr_budget_sensitivity,
    build_target_retuned_oracle_baseline,
    validate_acsac_diagnostics,
)
from iot_ids.config import load_config


def _load_prediction_groups(path: str, chunksize: int = 250000) -> dict[tuple[str, str], pd.DataFrame]:
    grouped_chunks: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    for chunk in pd.read_csv(
        path,
        usecols=["setting_id", "eval_role", "label", "score"],
        chunksize=chunksize,
    ):
        chunk["setting_id"] = chunk["setting_id"].astype(str)
        chunk["eval_role"] = chunk["eval_role"].astype(str)
        chunk["label"] = chunk["label"].astype(int)
        chunk["score"] = chunk["score"].astype(float)
        for (setting_id, eval_role), subset in chunk.groupby(["setting_id", "eval_role"], sort=False):
            grouped_chunks[(str(setting_id), str(eval_role))].append(
                subset[["label", "score"]].reset_index(drop=True)
            )
    return {
        key: pd.concat(parts, ignore_index=True)
        for key, parts in grouped_chunks.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build ACSAC diagnostic artifacts from frozen benchmark predictions."
    )
    parser.add_argument("--config", required=True, help="Path to render YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    diag_cfg = config["acsac_diagnostics"]
    settings = pd.read_csv(
        diag_cfg["inputs"]["benchmark_settings"],
        usecols=[
            "setting_id",
            "setting_type",
            "source_dataset",
            "target_dataset",
            "model",
            "threshold",
            "source_holdout_tpr_at_1pct_fpr",
            "target_tpr_at_1pct_fpr",
            "target_realized_fpr",
        ],
    )
    prediction_groups = _load_prediction_groups(diag_cfg["inputs"]["benchmark_predictions"])
    budgets = [float(value) for value in diag_cfg.get("budgets", [0.005, 0.01, 0.05])]

    target_retuned = build_target_retuned_oracle_baseline(
        settings,
        prediction_groups,
        target_fpr=float(diag_cfg.get("target_fpr", 0.01)),
    )
    budget_sensitivity = build_fpr_budget_sensitivity(settings, prediction_groups, budgets)
    validate_acsac_diagnostics(target_retuned, budget_sensitivity, settings, budgets)

    target_retuned_path = Path(diag_cfg["outputs"]["target_retuned_oracle_baseline_csv"])
    budget_sensitivity_path = Path(diag_cfg["outputs"]["fpr_budget_sensitivity_csv"])
    target_retuned_path.parent.mkdir(parents=True, exist_ok=True)
    budget_sensitivity_path.parent.mkdir(parents=True, exist_ok=True)
    target_retuned.to_csv(target_retuned_path, index=False)
    budget_sensitivity.to_csv(budget_sensitivity_path, index=False)
    print(f"Wrote {len(target_retuned)} target-retuned rows to {target_retuned_path}")
    print(f"Wrote {len(budget_sensitivity)} FPR-budget rows to {budget_sensitivity_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
