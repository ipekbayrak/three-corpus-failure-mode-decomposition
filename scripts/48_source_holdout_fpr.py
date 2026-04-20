#!/usr/bin/env python3
"""Compute source-holdout realized FPR and add to within-domain summary.

The reviewer calls the absence of source-holdout FPR 'fatal for the central
claim.' The data exists in predictions — we just never reported it.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from iot_ids.metrics import realized_fpr, grouped_bootstrap_threshold_intervals


def _load_source_holdout_predictions(
    path: str, chunksize: int = 500000
) -> dict[str, pd.DataFrame]:
    """Load source_holdout predictions grouped by setting_id."""
    grouped_chunks: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for chunk in pd.read_csv(
        path,
        usecols=["setting_id", "eval_role", "label", "score", "group_id"],
        chunksize=chunksize,
    ):
        sh = chunk[chunk["eval_role"] == "source_holdout"]
        if len(sh) == 0:
            continue
        for sid, sub in sh.groupby("setting_id"):
            grouped_chunks[str(sid)].append(
                sub[["label", "score", "group_id"]].reset_index(drop=True)
            )
    return {
        key: pd.concat(parts, ignore_index=True)
        for key, parts in grouped_chunks.items()
    }


def compute_source_holdout_fpr(
    settings: pd.DataFrame,
    predictions_path: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute source-holdout realized FPR for each unique source-model pair."""
    natural = settings[settings["setting_type"] == "natural_transfer"].copy()
    holdout_preds = _load_source_holdout_predictions(predictions_path)

    # Each source-model pair has the same threshold across all targets.
    seen: dict[tuple[str, str], dict] = {}
    rows = []

    for _, row in natural.iterrows():
        source = str(row["source_dataset"])
        model = str(row["model"])
        key = (source, model)
        if key in seen:
            continue

        sid = str(row["setting_id"])
        threshold = float(row["threshold"])

        if sid not in holdout_preds:
            seen[key] = {}
            continue

        preds = holdout_preds[sid]
        labels = preds["label"].astype(int)
        scores = preds["score"].astype(float)

        holdout_fpr = realized_fpr(labels, scores, threshold)

        # Bootstrap CI for FPR.
        ci = grouped_bootstrap_threshold_intervals(
            preds, threshold, n_boot=n_boot, seed=seed
        )

        seen[key] = {
            "dataset": source,
            "model": model,
            "threshold": threshold,
            "source_holdout_realized_fpr": holdout_fpr,
            "source_holdout_realized_fpr_ci_lo": ci["realized_fpr_ci_lo"],
            "source_holdout_realized_fpr_ci_hi": ci["realized_fpr_ci_hi"],
        }
        rows.append(seen[key])

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute source-holdout realized FPR for within-domain table."
    )
    parser.add_argument("--settings", default="artifacts/benchmark/settings.csv")
    parser.add_argument("--predictions", default="artifacts/benchmark/predictions.csv")
    parser.add_argument("--output", default="artifacts/benchmark/source_holdout_fpr.csv")
    args = parser.parse_args()

    settings = pd.read_csv(args.settings)
    result = compute_source_holdout_fpr(settings, args.predictions)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("=== Source-Holdout Realized FPR ===")
    print(result.to_string(index=False))
    print(f"\nWrote {len(result)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
