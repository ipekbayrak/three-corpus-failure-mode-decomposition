from __future__ import annotations

import unittest

import pandas as pd

from iot_ids.benchmark import (
    add_confidence_intervals,
    build_cross_domain_summary,
    build_within_domain_summary,
    validate_benchmark_artifacts,
)


def _benchmark_settings_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    directions = [
        ("iot23", "ton_iot"),
        ("iot23", "unsw_nb15"),
        ("ton_iot", "iot23"),
        ("ton_iot", "unsw_nb15"),
        ("unsw_nb15", "iot23"),
        ("unsw_nb15", "ton_iot"),
    ]
    models = [
        "logistic_regression",
        "extra_trees",
        "hist_gradient_boosting",
        "mlp",
        "ft_transformer",
    ]
    dataset_coverage = {
        "iot23": (0.98, 0.02),
        "ton_iot": (0.87, 0.13),
        "unsw_nb15": (0.00, 1.00),
    }

    for source, target in directions:
        for model_idx, model in enumerate(models):
            source_cov, source_other = dataset_coverage[source]
            target_cov, target_other = dataset_coverage[target]
            oracle_usable = int(source_cov >= 0.80 and target_cov >= 0.80)
            rows.append(
                {
                    "setting_id": f"{source}__{target}__{model}__natural_transfer",
                    "setting_type": "natural_transfer",
                    "source_dataset": source,
                    "target_dataset": target,
                    "model": model,
                    "threshold": 0.99 - model_idx / 100.0,
                    "source_holdout_tpr_at_1pct_fpr": 0.70 + model_idx / 100.0,
                    "source_validation_tpr_at_1pct_fpr": 0.72 + model_idx / 100.0,
                    "target_tpr_at_1pct_fpr": 0.40 + model_idx / 100.0,
                    "target_realized_fpr": 0.01 + model_idx / 1000.0,
                    "target_realized_fpr_ci_lo": 0.009 + model_idx / 1000.0,
                    "target_realized_fpr_ci_hi": 0.011 + model_idx / 1000.0,
                    "target_auroc": 0.80 + model_idx / 100.0,
                    "target_auroc_ci_lo": 0.79 + model_idx / 100.0,
                    "target_auroc_ci_hi": 0.81 + model_idx / 100.0,
                    "target_auprc": 0.75 + model_idx / 100.0,
                    "loss_rel": 0.30 + model_idx / 100.0,
                    "relative_tpr_drop": 0.30 + model_idx / 100.0,
                    "relative_tpr_change": -(0.30 + model_idx / 100.0),
                    "fpr_budget_multiple": 1.0 + model_idx / 10.0,
                    "fpr_severity": min(1.0, (1.0 + model_idx / 10.0 - 1.0) / 9.0),
                    "operational_severity": 0.30 + model_idx / 100.0,
                    "catastrophic_operational_failure": 0,
                    "tpr_loss_unsafe": 0,
                    "oracle_alpha": 0.25,
                    "alpha_hat": 0.35 + model_idx / 100.0,
                    "proxy_a_distance": 0.15 + model_idx / 100.0,
                    "avg_wasserstein_1d": 0.05 + model_idx / 100.0,
                    "source_holdout_tpr_at_1pct_fpr_ci_lo": 0.69 + model_idx / 100.0,
                    "source_holdout_tpr_at_1pct_fpr_ci_hi": 0.71 + model_idx / 100.0,
                    "source_validation_tpr_at_1pct_fpr_ci_lo": 0.71 + model_idx / 100.0,
                    "source_validation_tpr_at_1pct_fpr_ci_hi": 0.73 + model_idx / 100.0,
                    "target_tpr_at_1pct_fpr_ci_lo": 0.39 + model_idx / 100.0,
                    "target_tpr_at_1pct_fpr_ci_hi": 0.41 + model_idx / 100.0,
                    "source_mapped_attack_coverage": source_cov,
                    "source_other_attack_share": source_other,
                    "target_mapped_attack_coverage": target_cov,
                    "target_other_attack_share": target_other,
                    "oracle_alpha_usable": oracle_usable,
                    "oracle_alpha_scope": (
                        "primary" if oracle_usable else "stress_test_low_coverage"
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    assert len(frame) == len(directions) * len(models)
    return frame


def _benchmark_predictions_frame(settings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for setting_id in settings["setting_id"]:
        setting = settings.loc[settings["setting_id"] == setting_id].iloc[0]
        for eval_role, dataset in [
            ("source_validation", setting["source_dataset"]),
            ("source_holdout", setting["source_dataset"]),
            ("target_transfer", setting["target_dataset"]),
        ]:
            for idx in range(2):
                rows.append(
                    {
                        "setting_id": setting_id,
                        "dataset": dataset,
                        "record_id": f"{setting_id}__{eval_role}__{idx}",
                        "group_id": f"{setting_id}__{eval_role}__g{idx}",
                        "family_group": "service_disruption",
                        "score": 0.5 + idx / 10.0,
                        "label": idx % 2,
                        "source_dataset": setting["source_dataset"],
                        "target_dataset": setting["target_dataset"],
                        "model": setting["model"],
                        "threshold": setting["threshold"],
                        "eval_role": eval_role,
                    }
                )
    return pd.DataFrame(rows)


class BenchmarkArtifactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = _benchmark_settings_frame()
        self.predictions = _benchmark_predictions_frame(self.settings)
        self.directions = [
            ["iot23", "ton_iot"],
            ["iot23", "unsw_nb15"],
            ["ton_iot", "iot23"],
            ["ton_iot", "unsw_nb15"],
            ["unsw_nb15", "iot23"],
            ["unsw_nb15", "ton_iot"],
        ]
        self.models = [
            "logistic_regression",
            "extra_trees",
            "hist_gradient_boosting",
            "mlp",
            "ft_transformer",
        ]

    def test_build_benchmark_summaries(self) -> None:
        within = build_within_domain_summary(self.settings)
        cross = build_cross_domain_summary(self.settings)
        self.assertEqual(len(within), 15)
        self.assertEqual(len(cross), 30)
        self.assertEqual(
            sorted(within["dataset"].unique().tolist()),
            ["iot23", "ton_iot", "unsw_nb15"],
        )

    def test_validate_benchmark_artifacts(self) -> None:
        validate_benchmark_artifacts(self.settings, self.predictions, self.directions, self.models)

    def test_add_confidence_intervals_populates_target_auroc_interval(self) -> None:
        settings = pd.DataFrame(
            [
                {
                    "setting_id": "iot23__ton_iot__lr__natural_transfer",
                    "setting_type": "natural_transfer",
                    "source_dataset": "iot23",
                    "target_dataset": "ton_iot",
                    "model": "logistic_regression",
                    "threshold": 0.5,
                    "source_holdout_tpr_at_1pct_fpr": 1.0,
                    "source_validation_tpr_at_1pct_fpr": 1.0,
                    "target_tpr_at_1pct_fpr": 1.0,
                    "target_realized_fpr": 0.0,
                    "target_auroc": 1.0,
                    "target_auprc": 1.0,
                    "loss_rel": 0.0,
                    "relative_tpr_drop": 0.0,
                    "relative_tpr_change": 0.0,
                    "fpr_budget_multiple": 0.0,
                    "fpr_severity": 0.0,
                    "operational_severity": 0.0,
                    "catastrophic_operational_failure": 0,
                    "tpr_loss_unsafe": 0,
                    "alpha_hat": 0.0,
                    "proxy_a_distance": 0.0,
                    "avg_wasserstein_1d": 0.0,
                    "source_mapped_attack_coverage": 1.0,
                    "source_other_attack_share": 0.0,
                    "target_mapped_attack_coverage": 1.0,
                    "target_other_attack_share": 0.0,
                    "oracle_alpha": 0.0,
                    "oracle_alpha_usable": 1,
                    "oracle_alpha_scope": "primary",
                }
            ]
        )
        predictions = pd.DataFrame(
            {
                "setting_id": ["iot23__ton_iot__lr__natural_transfer"] * 12,
                "dataset": ["iot23"] * 4 + ["iot23"] * 4 + ["ton_iot"] * 4,
                "record_id": [f"r{i}" for i in range(12)],
                "group_id": ["a", "a", "b", "b"] * 3,
                "label": [0, 1, 0, 1] * 3,
                "family_group": ["service_disruption"] * 12,
                "score": [0.1, 0.9, 0.2, 0.8] * 3,
                "source_dataset": ["iot23"] * 12,
                "target_dataset": ["ton_iot"] * 12,
                "model": ["logistic_regression"] * 12,
                "threshold": [0.5] * 12,
                "eval_role": (
                    ["source_validation"] * 4
                    + ["source_holdout"] * 4
                    + ["target_transfer"] * 4
                ),
            }
        )

        enriched = add_confidence_intervals(settings, predictions, n_boot=20, seed=7)

        self.assertIn("target_auroc_ci_lo", enriched.columns)
        self.assertIn("target_auroc_ci_hi", enriched.columns)
        self.assertLessEqual(
            float(enriched.loc[0, "target_auroc_ci_lo"]),
            float(enriched.loc[0, "target_auroc"]),
        )
        self.assertLessEqual(
            float(enriched.loc[0, "target_auroc"]),
            float(enriched.loc[0, "target_auroc_ci_hi"]),
        )

    def test_validate_rejects_controlled_shift_rows(self) -> None:
        bad = self.settings.copy()
        bad.loc[0, "setting_type"] = "controlled_shift"
        with self.assertRaisesRegex(ValueError, "natural-transfer"):
            validate_benchmark_artifacts(bad, self.predictions, self.directions, self.models)

    def test_validate_rejects_low_coverage_flag_mismatch(self) -> None:
        bad = self.settings.copy()
        unsw_mask = bad["target_dataset"] == "unsw_nb15"
        bad.loc[unsw_mask, "oracle_alpha_usable"] = 1
        with self.assertRaisesRegex(ValueError, "oracle_alpha_usable"):
            validate_benchmark_artifacts(bad, self.predictions, self.directions, self.models)


if __name__ == "__main__":
    unittest.main()
