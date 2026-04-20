#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from iot_ids.benchmark import (
    add_confidence_intervals,
    build_cross_domain_summary,
    build_within_domain_summary,
    validate_benchmark_artifacts,
)
from iot_ids.config import load_config, stable_hash
from iot_ids.experiment import (
    SplitFrames,
    build_split_manifest,
    prepare_feature_frames,
    run_transfer_matrix,
)


def _cap_frame_rows(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()

    label_strata = frame["label"].astype(str)
    family = frame["family_group"].fillna("other").astype(str)
    strata = label_strata + "|" + family
    counts = strata.value_counts()
    rare_fallback = np.where(frame["label"].to_numpy() == 1, "rare_pos", "rare_neg")
    strata = pd.Series(
        np.where(strata.map(counts).to_numpy() > 1, strata.to_numpy(), rare_fallback),
        index=frame.index,
    )
    selected_idx, _ = train_test_split(
        frame.index.to_numpy(),
        train_size=max_rows,
        random_state=seed,
        stratify=strata.to_numpy(),
    )
    return frame.loc[selected_idx].copy()


def _cap_split_frames(
    split_frames: dict[str, SplitFrames],
    caps: dict[str, int],
    seed: int,
) -> dict[str, SplitFrames]:
    capped: dict[str, SplitFrames] = {}
    for offset, (dataset, splits) in enumerate(split_frames.items()):
        capped[dataset] = SplitFrames(
            train=_cap_frame_rows(splits.train, int(caps["train"]), seed + offset * 10 + 1),
            val=_cap_frame_rows(splits.val, int(caps["val"]), seed + offset * 10 + 2),
            test=_cap_frame_rows(splits.test, int(caps["test"]), seed + offset * 10 + 3),
        )
    return capped


def _partial_path(path: Path) -> Path:
    return path.with_suffix(".partial.csv")


def _append_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _write_partial_summaries(
    settings_partial_path: Path,
    within_partial_path: Path,
    cross_partial_path: Path,
) -> None:
    settings = pd.read_csv(settings_partial_path)
    build_within_domain_summary(settings).to_csv(within_partial_path, index=False)
    build_cross_domain_summary(settings).to_csv(cross_partial_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full transfer benchmark.")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = f"benchmark_{stable_hash(config)}"
    frames = prepare_feature_frames(config)
    split_frames, _ = build_split_manifest(frames, config)

    benchmark_cfg = config["benchmark"]
    if "split_row_caps" in benchmark_cfg:
        split_frames = _cap_split_frames(
            split_frames,
            dict(benchmark_cfg["split_row_caps"]),
            int(config["project"]["seed"]),
        )
    directions = list(benchmark_cfg["directions"])
    model_names = list(benchmark_cfg["models"])
    predictions_path = Path(benchmark_cfg["outputs"]["predictions_csv"])
    settings_path = Path(benchmark_cfg["outputs"]["settings_csv"])
    within_domain_summary_path = Path(benchmark_cfg["outputs"]["within_domain_summary_csv"])
    cross_domain_summary_path = Path(benchmark_cfg["outputs"]["cross_domain_summary_csv"])
    predictions_partial_path = _partial_path(predictions_path)
    settings_partial_path = _partial_path(settings_path)
    within_partial_path = _partial_path(within_domain_summary_path)
    cross_partial_path = _partial_path(cross_domain_summary_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    within_domain_summary_path.parent.mkdir(parents=True, exist_ok=True)
    cross_domain_summary_path.parent.mkdir(parents=True, exist_ok=True)
    for partial_path in [
        predictions_partial_path,
        settings_partial_path,
        within_partial_path,
        cross_partial_path,
    ]:
        if partial_path.exists():
            partial_path.unlink()

    def progress_callback(
        setting_predictions: pd.DataFrame,
        setting_row: dict[str, object],
        completed: int,
        total: int,
    ) -> None:
        _append_csv(predictions_partial_path, setting_predictions.assign(run_id=run_id))
        _append_csv(settings_partial_path, pd.DataFrame([{**setting_row, "run_id": run_id}]))
        _write_partial_summaries(
            settings_partial_path,
            within_partial_path,
            cross_partial_path,
        )
        print(
            f"flushed partial benchmark artifacts after setting {completed}/{total}",
            flush=True,
        )

    predictions, settings = run_transfer_matrix(
        frames=frames,
        split_frames=split_frames,
        model_names=model_names,
        directions=directions,
        seed=int(config["project"]["seed"]),
        target_fpr=float(benchmark_cfg["transfer_threshold_fpr"]),
        alpha_hat_cfg={"k_nn": 10, "distance_quantile": 0.95},
        include_controlled_shift=False,
        progress_callback=progress_callback,
    )
    bootstrap_cfg = benchmark_cfg.get("bootstrap", {})
    print("computing grouped bootstrap confidence intervals", flush=True)
    settings = add_confidence_intervals(
        settings,
        predictions,
        n_boot=int(bootstrap_cfg.get("n_boot", 2000)),
        seed=int(bootstrap_cfg.get("seed", 20260414)),
    )
    settings["run_id"] = run_id
    predictions["run_id"] = run_id
    validate_benchmark_artifacts(settings, predictions, directions, model_names)
    within_domain_summary = build_within_domain_summary(settings)
    cross_domain_summary = build_cross_domain_summary(settings)

    predictions.to_csv(predictions_path, index=False)
    settings.to_csv(settings_path, index=False)
    within_domain_summary.to_csv(within_domain_summary_path, index=False)
    cross_domain_summary.to_csv(cross_domain_summary_path, index=False)
    for partial_path in [
        predictions_partial_path,
        settings_partial_path,
        within_partial_path,
        cross_partial_path,
    ]:
        if partial_path.exists():
            partial_path.unlink()
    print(f"Wrote benchmark settings to {settings_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
