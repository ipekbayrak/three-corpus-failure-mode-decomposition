#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from iot_ids.benchmark import (
    add_confidence_intervals,
    build_cross_domain_summary,
    build_within_domain_summary,
    validate_benchmark_artifacts,
)
from iot_ids.decision import add_operational_metrics, relative_transfer_loss
from iot_ids.metrics import compute_metrics, threshold_at_fpr


def _recompute_settings(
    settings: pd.DataFrame,
    predictions: pd.DataFrame,
    target_fpr: float,
) -> pd.DataFrame:
    prediction_groups = {
        (str(setting_id), str(eval_role)): frame.copy()
        for (setting_id, eval_role), frame in predictions.groupby(
            ["setting_id", "eval_role"], sort=False
        )
    }
    rows: list[dict[str, object]] = []
    for row in settings.itertuples(index=False):
        setting = dict(row._asdict())
        setting_id = str(row.setting_id)
        source_validation = prediction_groups[(setting_id, "source_validation")]
        source_holdout = prediction_groups[(setting_id, "source_holdout")]
        target_transfer = prediction_groups[(setting_id, "target_transfer")]

        threshold = threshold_at_fpr(
            source_validation["label"].astype(int),
            source_validation["score"].astype(float),
            target_fpr,
        )
        source_validation_metrics = compute_metrics(
            source_validation["label"].astype(int),
            source_validation["score"].astype(float),
            threshold,
        )
        source_holdout_metrics = compute_metrics(
            source_holdout["label"].astype(int),
            source_holdout["score"].astype(float),
            threshold,
        )
        target_metrics = compute_metrics(
            target_transfer["label"].astype(int),
            target_transfer["score"].astype(float),
            threshold,
        )

        setting["threshold"] = float(threshold)
        setting["source_validation_tpr_at_1pct_fpr"] = float(
            source_validation_metrics["tpr_at_1pct_fpr"]
        )
        setting["source_holdout_tpr_at_1pct_fpr"] = float(
            source_holdout_metrics["tpr_at_1pct_fpr"]
        )
        setting["target_tpr_at_1pct_fpr"] = float(target_metrics["tpr_at_1pct_fpr"])
        setting["target_realized_fpr"] = float(target_metrics["realized_fpr"])
        setting["target_auroc"] = float(target_metrics["auroc"])
        setting["target_auprc"] = float(target_metrics["auprc"])
        setting["loss_rel"] = float(
            relative_transfer_loss(
                float(target_metrics["tpr_at_1pct_fpr"]),
                float(source_holdout_metrics["tpr_at_1pct_fpr"]),
            )
        )
        rows.append(setting)

    corrected = pd.DataFrame(rows)
    corrected = add_operational_metrics(corrected, fpr_budget=target_fpr)
    corrected = add_confidence_intervals(
        corrected,
        predictions,
        n_boot=2000,
        seed=20260414,
    )
    return corrected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute threshold-dependent benchmark artifacts from frozen predictions."
    )
    parser.add_argument("--settings", default="artifacts/benchmark/settings.csv")
    parser.add_argument("--predictions", default="artifacts/benchmark/predictions.csv")
    parser.add_argument(
        "--output-settings",
        default="artifacts/benchmark/settings.csv",
    )
    parser.add_argument(
        "--output-predictions",
        default="artifacts/benchmark/predictions.csv",
    )
    parser.add_argument(
        "--output-within",
        default="artifacts/benchmark/within_domain_summary.csv",
    )
    parser.add_argument(
        "--output-cross",
        default="artifacts/benchmark/cross_domain_summary.csv",
    )
    parser.add_argument("--target-fpr", type=float, default=0.01)
    args = parser.parse_args()

    settings = pd.read_csv(args.settings)
    predictions = pd.read_csv(args.predictions)

    corrected_settings = _recompute_settings(settings, predictions, target_fpr=args.target_fpr)
    threshold_map = corrected_settings.set_index("setting_id")["threshold"].astype(float)
    corrected_predictions = predictions.copy()
    corrected_predictions["threshold"] = (
        corrected_predictions["setting_id"].astype(str).map(threshold_map).astype(float)
    )

    within_domain_summary = build_within_domain_summary(corrected_settings)
    cross_domain_summary = build_cross_domain_summary(corrected_settings)
    directions = list(
        corrected_settings[["source_dataset", "target_dataset"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    model_names = corrected_settings["model"].astype(str).drop_duplicates().tolist()
    validate_benchmark_artifacts(
        corrected_settings,
        corrected_predictions,
        directions=directions,
        model_names=model_names,
    )

    Path(args.output_settings).parent.mkdir(parents=True, exist_ok=True)
    corrected_settings.to_csv(args.output_settings, index=False)
    corrected_predictions.to_csv(args.output_predictions, index=False)
    within_domain_summary.to_csv(args.output_within, index=False)
    cross_domain_summary.to_csv(args.output_cross, index=False)

    print(f"Wrote corrected settings to {args.output_settings}")
    print(f"Wrote corrected predictions to {args.output_predictions}")
    print(f"Wrote within-domain summary to {args.output_within}")
    print(f"Wrote cross-domain summary to {args.output_cross}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
