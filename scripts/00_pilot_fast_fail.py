#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

from iot_ids.config import load_config, stable_hash
from iot_ids.experiment import (
    SplitFrames,
    build_split_manifest,
    prepare_feature_frames,
    run_transfer_matrix,
)
from iot_ids.novelty import mapping_coverage
from iot_ids.pilot_checkpoint import summarize_pilot_checkpoint


def _dataset_mapping_summary(
    frames: dict[str, pd.DataFrame],
    datasets: list[str],
) -> dict[str, tuple[float, float]]:
    summary = {}
    for dataset in datasets:
        attack_groups = frames[dataset].loc[frames[dataset]["label"] == 1, "family_group"]
        summary[dataset] = mapping_coverage(attack_groups)
    return summary


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


def _render_hero_figure(settings: pd.DataFrame, output_path: Path) -> None:
    plot_frame = settings[["setting_type", "model", "oracle_alpha", "loss_rel"]].dropna()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = {"natural_transfer": "tab:blue", "controlled_shift": "tab:orange"}
    markers = {"logistic_regression": "o", "hist_gradient_boosting": "s"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in plot_frame.iterrows():
        ax.scatter(
            row["oracle_alpha"],
            row["loss_rel"],
            color=colors.get(str(row["setting_type"]), "tab:gray"),
            marker=markers.get(str(row["model"]), "o"),
            s=70,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xlabel("Oracle novelty alpha")
    ax.set_ylabel("Relative transfer loss")
    ax.set_title("Fast-fail pilot: novelty vs transfer loss")
    ax.grid(alpha=0.25)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Natural transfer",
            markerfacecolor="tab:blue",
            markeredgecolor="black",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Controlled shift",
            markerfacecolor="tab:orange",
            markeredgecolor="black",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            label="Logistic regression",
            linestyle="None",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="black",
            label="HistGradientBoosting",
            linestyle="None",
            markersize=7,
        ),
    ]
    ax.legend(handles=legend_handles, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fast-fail pilot.")
    parser.add_argument("--config", required=True, help="Path to pilot YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    run_id = f"pilot_{stable_hash(config)}"

    frames = prepare_feature_frames(config)
    split_frames, _ = build_split_manifest(frames, config)
    pilot_cfg = config["pilot"]
    if "split_row_caps" in pilot_cfg:
        split_frames = _cap_split_frames(split_frames, dict(pilot_cfg["split_row_caps"]), seed)
    datasets_in_pilot = sorted(
        {dataset for direction in pilot_cfg["transfer_directions"] for dataset in direction}
    )
    dataset_coverage = _dataset_mapping_summary(frames, datasets_in_pilot)

    predictions, settings = run_transfer_matrix(
        frames=frames,
        split_frames=split_frames,
        model_names=list(pilot_cfg["models"]),
        directions=list(pilot_cfg["transfer_directions"]),
        seed=seed,
        target_fpr=float(pilot_cfg["threshold_fpr"]),
        alpha_hat_cfg=dict(pilot_cfg["alpha_hat"]),
        include_controlled_shift=bool(pilot_cfg["controlled_shift"]["enabled"]),
        controlled_shift_dataset=str(pilot_cfg["controlled_shift"]["source_dataset"]),
        controlled_shift_excluded_train_groups=list(
            pilot_cfg["controlled_shift"]["excluded_train_groups"]
        ),
        controlled_shift_seen_groups=list(pilot_cfg["controlled_shift"]["seen_groups"]),
        controlled_shift_novel_groups=list(pilot_cfg["controlled_shift"]["novel_groups"]),
        novelty_schedule=list(pilot_cfg["controlled_shift"]["novelty_schedule"]),
    )
    if settings.empty:
        raise RuntimeError(
            "Pilot produced no settings; check dataset staging and ontology coverage."
        )

    settings["run_id"] = run_id
    predictions["run_id"] = run_id
    settings_path = Path(pilot_cfg["outputs"]["settings_csv"])
    results_path = Path(pilot_cfg["outputs"]["results_csv"])
    checkpoint_path = Path(pilot_cfg["outputs"]["checkpoint_csv"])
    hero_path = Path(pilot_cfg["outputs"]["hero_png"])
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    settings.to_csv(settings_path, index=False)
    predictions.to_csv(results_path, index=False)
    _render_hero_figure(settings, hero_path)

    summary = summarize_pilot_checkpoint(settings, dataset_coverage, run_id=run_id)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(checkpoint_path, index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
