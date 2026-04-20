#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from iot_ids.config import load_config
from iot_ids.experiment import SplitFrames, split_by_group
from iot_ids.features import FEATURE_SCHEMA_VERSION
from iot_ids.novelty import compute_oracle_alpha


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


def _load_processed_features(dataset: str, config: dict[str, object]) -> pd.DataFrame:
    processed_root = Path(str(config["paths"]["processed_root"]))
    path = processed_root / f"{dataset}_{FEATURE_SCHEMA_VERSION}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed feature file not found: {path}")
    return pd.read_csv(path)


def _reconstruct_source_train_groups(
    settings: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, set[str]]:
    seed = int(config["project"]["seed"])
    split_cfg = config["split"]
    pilot_cfg = config["pilot"]
    source_datasets = sorted(settings["source_dataset"].astype(str).unique())
    split_frames: dict[str, SplitFrames] = {}

    for dataset in source_datasets:
        frame = _load_processed_features(dataset, config)
        split_frames[dataset] = split_by_group(
            frame,
            train_frac=float(split_cfg["train_frac"]),
            val_frac=float(split_cfg["val_frac"]),
            seed=seed,
        )

    if "split_row_caps" in pilot_cfg:
        split_frames = _cap_split_frames(split_frames, dict(pilot_cfg["split_row_caps"]), seed)

    train_groups = {}
    for dataset, splits in split_frames.items():
        train_groups[dataset] = set(
            splits.train.loc[
                (splits.train["label"] == 1)
                & (splits.train["family_group"].astype(str) != "other"),
                "family_group",
            ]
            .astype(str)
            .unique()
        )
    return train_groups


def _safe_spearman(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    subset = frame[[x_col, y_col]].dropna()
    if len(subset) < 2:
        return float("nan")
    return float(subset[x_col].corr(subset[y_col], method="spearman"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose natural-transfer pilot settings.")
    parser.add_argument("--config", required=True, help="Path to pilot YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    settings_path = Path("artifacts/pilot/pilot_settings.csv")
    results_path = Path("artifacts/pilot/pilot_results.csv")
    diagnosis_path = Path("artifacts/pilot/natural_transfer_diagnosis.csv")
    summary_path = Path("artifacts/pilot/natural_transfer_summary.csv")

    settings = pd.read_csv(settings_path)
    results = pd.read_csv(results_path)

    natural_settings = settings.loc[settings["setting_type"] == "natural_transfer"].copy()
    if len(natural_settings) != 4:
        raise RuntimeError(f"Expected 4 natural-transfer settings, found {len(natural_settings)}")

    natural_results = results.loc[
        results["setting_id"].isin(natural_settings["setting_id"].astype(str))
    ].copy()
    source_train_groups = _reconstruct_source_train_groups(natural_settings, config)

    diagnosis_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for _, setting in natural_settings.sort_values(
        ["source_dataset", "target_dataset", "model"]
    ).iterrows():
        setting_id = str(setting["setting_id"])
        setting_results = natural_results.loc[natural_results["setting_id"] == setting_id].copy()
        threshold = float(setting_results["threshold"].iloc[0])
        attacks = setting_results.loc[setting_results["label"] == 1].copy()
        total_target_attack_count = int(len(attacks))
        if total_target_attack_count == 0:
            raise RuntimeError(f"No target attacks found for natural-transfer setting {setting_id}")

        source_attack_groups = source_train_groups[str(setting["source_dataset"])]
        target_attack_groups = attacks["family_group"].astype(str)
        oracle_alpha_ex_other = compute_oracle_alpha(
            source_attack_groups,
            target_attack_groups,
            excluded_groups=("other",),
        )
        other_attack_share = float((target_attack_groups == "other").mean())

        group_counts = (
            attacks.groupby("family_group", dropna=False)
            .size()
            .rename("attack_count_by_family_group")
            .reset_index()
            .sort_values(["attack_count_by_family_group", "family_group"], ascending=[False, True])
        )
        for group_row in group_counts.itertuples(index=False):
            group_name = str(group_row.family_group)
            attack_count = int(group_row.attack_count_by_family_group)
            group_attacks = attacks.loc[attacks["family_group"].astype(str) == group_name].copy()
            group_tpr = float((group_attacks["score"] >= threshold).mean())
            diagnosis_rows.append(
                {
                    "setting_id": setting_id,
                    "source_dataset": setting["source_dataset"],
                    "target_dataset": setting["target_dataset"],
                    "model": setting["model"],
                    "family_group": group_name,
                    "total_target_attack_count": total_target_attack_count,
                    "attack_count_by_family_group": attack_count,
                    "attack_mass_share_by_family_group": float(
                        attack_count / total_target_attack_count
                    ),
                    "other_share_among_target_attacks": other_attack_share,
                    "group_tpr_at_threshold": group_tpr,
                    "oracle_alpha_ex_other": oracle_alpha_ex_other,
                    "threshold": threshold,
                }
            )

        summary_rows.append(
            {
                "setting_id": setting_id,
                "source_dataset": setting["source_dataset"],
                "target_dataset": setting["target_dataset"],
                "model": setting["model"],
                "original_oracle_alpha": float(setting["oracle_alpha"]),
                "recomputed_oracle_alpha_ex_other": oracle_alpha_ex_other,
                "loss_rel": float(setting["loss_rel"]),
                "other_attack_share": other_attack_share,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["source_dataset", "target_dataset", "model"]
    )
    if len(summary) != 4:
        raise RuntimeError(
            f"Expected natural_transfer_summary.csv to have 4 rows, found {len(summary)}"
        )

    pooled_original = _safe_spearman(summary, "original_oracle_alpha", "loss_rel")
    pooled_ex_other = _safe_spearman(summary, "recomputed_oracle_alpha_ex_other", "loss_rel")
    summary["pooled_spearman_original"] = pooled_original
    summary["pooled_spearman_ex_other"] = pooled_ex_other

    diagnosis = pd.DataFrame(diagnosis_rows).sort_values(
        [
            "source_dataset",
            "target_dataset",
            "model",
            "attack_count_by_family_group",
            "family_group",
        ],
        ascending=[True, True, True, False, True],
    )

    diagnosis_path.parent.mkdir(parents=True, exist_ok=True)
    diagnosis.to_csv(diagnosis_path, index=False)
    summary.to_csv(summary_path, index=False)

    if not diagnosis_path.exists() or not summary_path.exists():
        raise RuntimeError("Expected diagnosis outputs were not created")

    below_threshold = pd.notna(pooled_ex_other) and pooled_ex_other < 0.40
    print(f"Natural-transfer settings analyzed: {len(summary)}")
    print(f"Pooled original oracle_alpha vs loss_rel Spearman: {pooled_original}")
    print(f"Pooled recomputed oracle_alpha_ex_other vs loss_rel Spearman: {pooled_ex_other}")
    print(f"Below 0.40 after recompute: {'yes' if below_threshold else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
