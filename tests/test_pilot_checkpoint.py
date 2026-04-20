from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from iot_ids.pilot_checkpoint import summarize_pilot_checkpoint


def _base_settings() -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "run_id": "pilot_test",
            "setting_id": "iot23__ton_iot__logistic_regression__natural_transfer",
            "setting_type": "natural_transfer",
            "source_dataset": "iot23",
            "target_dataset": "ton_iot",
            "model": "logistic_regression",
            "source_holdout_tpr_at_1pct_fpr": 0.70,
            "source_validation_tpr_at_1pct_fpr": 0.80,
            "oracle_alpha": 1.0,
            "alpha_hat": 0.80,
            "proxy_a_distance": 0.20,
            "avg_wasserstein_1d": 0.40,
            "loss_rel": 0.80,
            "unsafe_setting": 1,
        },
        {
            "run_id": "pilot_test",
            "setting_id": "iot23__ton_iot__hist_gradient_boosting__natural_transfer",
            "setting_type": "natural_transfer",
            "source_dataset": "iot23",
            "target_dataset": "ton_iot",
            "model": "hist_gradient_boosting",
            "source_holdout_tpr_at_1pct_fpr": 0.72,
            "source_validation_tpr_at_1pct_fpr": 0.80,
            "oracle_alpha": 1.0,
            "alpha_hat": 0.90,
            "proxy_a_distance": 0.10,
            "avg_wasserstein_1d": 0.35,
            "loss_rel": 0.90,
            "unsafe_setting": 1,
        },
        {
            "run_id": "pilot_test",
            "setting_id": "ton_iot__iot23__logistic_regression__natural_transfer",
            "setting_type": "natural_transfer",
            "source_dataset": "ton_iot",
            "target_dataset": "iot23",
            "model": "logistic_regression",
            "source_holdout_tpr_at_1pct_fpr": 0.71,
            "source_validation_tpr_at_1pct_fpr": 0.80,
            "oracle_alpha": 0.0,
            "alpha_hat": 0.60,
            "proxy_a_distance": 0.90,
            "avg_wasserstein_1d": 0.12,
            "loss_rel": 0.60,
            "unsafe_setting": 1,
        },
        {
            "run_id": "pilot_test",
            "setting_id": "ton_iot__iot23__hist_gradient_boosting__natural_transfer",
            "setting_type": "natural_transfer",
            "source_dataset": "ton_iot",
            "target_dataset": "iot23",
            "model": "hist_gradient_boosting",
            "source_holdout_tpr_at_1pct_fpr": 0.73,
            "source_validation_tpr_at_1pct_fpr": 0.80,
            "oracle_alpha": 0.0,
            "alpha_hat": 0.70,
            "proxy_a_distance": 0.80,
            "avg_wasserstein_1d": 0.14,
            "loss_rel": 0.70,
            "unsafe_setting": 1,
        },
    ]
    novelty_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    for model in ["logistic_regression", "hist_gradient_boosting"]:
        for level, loss, alpha_hat, proxy in zip(
            novelty_levels,
            [0.05, 0.10, 0.15, 0.20, 0.25],
            [0.05, 0.10, 0.15, 0.20, 0.25],
            [1.00, 0.80, 0.60, 0.40, 0.20],
        ):
            rows.append(
                {
                    "run_id": "pilot_test",
                    "setting_id": f"iot23__iot23__{model}__controlled_{level}",
                    "setting_type": "controlled_shift",
                    "source_dataset": "iot23",
                    "target_dataset": "iot23",
                    "model": model,
                    "source_holdout_tpr_at_1pct_fpr": 0.70,
                    "source_validation_tpr_at_1pct_fpr": 0.80,
                    "oracle_alpha": level,
                    "alpha_hat": alpha_hat,
                    "proxy_a_distance": proxy,
                    "avg_wasserstein_1d": 0.10 + level / 10.0,
                    "loss_rel": loss,
                    "unsafe_setting": 0,
                }
            )
    return pd.DataFrame(rows)


class PilotCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_coverage = {
            "iot23": (0.98, 0.02),
            "ton_iot": (0.87, 0.13),
        }

    def test_go_when_natural_and_controlled_pass_separately(self) -> None:
        summary = summarize_pilot_checkpoint(_base_settings(), self.dataset_coverage)
        row = summary.iloc[0]
        self.assertEqual(row["checkpoint"], "GO")
        self.assertLess(row["mixed_oracle_alpha_spearman"], 0.60)
        self.assertGreaterEqual(row["natural_transfer_oracle_alpha_spearman"], 0.60)
        self.assertGreaterEqual(row["controlled_shift_oracle_alpha_spearman"], 0.50)

    def test_stop_when_natural_transfer_signal_is_too_weak(self) -> None:
        settings = _base_settings()
        natural_mask = settings["setting_type"] == "natural_transfer"
        settings.loc[natural_mask, "loss_rel"] = [0.80, 0.10, 0.20, 0.90]
        settings.loc[natural_mask, "unsafe_setting"] = (
            settings.loc[natural_mask, "loss_rel"] > 0.50
        ).astype(int)

        summary = summarize_pilot_checkpoint(settings, self.dataset_coverage)
        row = summary.iloc[0]
        self.assertEqual(row["checkpoint"], "STOP_THEORY_HEAVY_PATH")
        self.assertEqual(
            row["checkpoint_reason"],
            "natural-transfer oracle alpha relation is too weak",
        )

    def test_stop_when_controlled_shift_signal_is_too_weak(self) -> None:
        settings = _base_settings()
        controlled_mask = settings["setting_type"] == "controlled_shift"
        settings.loc[controlled_mask, "loss_rel"] = 0.20
        settings.loc[controlled_mask, "unsafe_setting"] = 0

        summary = summarize_pilot_checkpoint(settings, self.dataset_coverage)
        row = summary.iloc[0]
        self.assertEqual(row["checkpoint"], "STOP_THEORY_HEAVY_PATH")
        self.assertEqual(
            row["checkpoint_reason"],
            "controlled-shift oracle alpha relation is too weak",
        )

    def test_pivot_when_alpha_hat_does_not_add_value(self) -> None:
        settings = _base_settings()
        settings["alpha_hat"] = 1.0 - settings["loss_rel"]
        settings["proxy_a_distance"] = settings["loss_rel"]
        settings["avg_wasserstein_1d"] = settings["loss_rel"] * 0.9

        summary = summarize_pilot_checkpoint(settings, self.dataset_coverage)
        row = summary.iloc[0]
        self.assertEqual(row["checkpoint"], "PIVOT_TO_FALLBACK")
        self.assertEqual(row["alpha_hat_beats_baseline"], 0)

    def test_hard_failures_precede_oracle_gating(self) -> None:
        coverage_fail = summarize_pilot_checkpoint(
            _base_settings(),
            {"iot23": (0.79, 0.21), "ton_iot": (0.87, 0.13)},
        )
        self.assertEqual(
            coverage_fail.iloc[0]["checkpoint_reason"],
            "mapped attack coverage below 0.80",
        )

        count_fail = summarize_pilot_checkpoint(
            _base_settings().loc[lambda df: df["setting_type"] != "controlled_shift"].copy(),
            self.dataset_coverage,
        )
        self.assertEqual(
            count_fail.iloc[0]["checkpoint_reason"],
            "controlled-shift point count mismatch",
        )

    def test_checkpoint_recompute_script_rewrites_artifact(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            settings_path = tmp_root / "pilot_settings.csv"
            checkpoint_path = tmp_root / "pilot_checkpoint.csv"
            config_path = tmp_root / "pilot.yaml"

            _base_settings().to_csv(settings_path, index=False)
            pd.DataFrame(
                [
                    {
                        "run_id": "pilot_test",
                        "iot23_mapped_attack_coverage": 0.98,
                        "iot23_other_attack_share": 0.02,
                        "ton_iot_mapped_attack_coverage": 0.87,
                        "ton_iot_other_attack_share": 0.13,
                    }
                ]
            ).to_csv(checkpoint_path, index=False)
            config_path.write_text(
                "\n".join(
                    [
                        "pilot:",
                        "  outputs:",
                        f"    settings_csv: {settings_path}",
                        f"    checkpoint_csv: {checkpoint_path}",
                    ]
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts/03_recompute_pilot_checkpoint.py"),
                    "--config",
                    str(config_path),
                ],
                check=True,
                cwd=repo_root,
                text=True,
                capture_output=True,
            )

            summary = pd.read_csv(checkpoint_path)
            row = summary.iloc[0]
            self.assertEqual(row["checkpoint"], "GO")
            self.assertIn("natural_transfer_oracle_alpha_spearman", summary.columns)
            self.assertIn("checkpoint_logic_version", summary.columns)
            self.assertIn("GO", result.stdout)


if __name__ == "__main__":
    unittest.main()
