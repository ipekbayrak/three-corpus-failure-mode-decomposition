from __future__ import annotations

import unittest

import pandas as pd

from iot_ids.acsac_diagnostics import (
    build_fpr_budget_sensitivity,
    build_target_retuned_oracle_baseline,
    validate_acsac_diagnostics,
)


class AcsacDiagnosticsTest(unittest.TestCase):
    def test_diagnostics_use_target_for_oracle_retune_and_source_validation_for_budgets(
        self,
    ) -> None:
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
                    "target_tpr_at_1pct_fpr": 1.0,
                    "target_realized_fpr": 1.0,
                }
            ]
        )
        predictions = pd.DataFrame(
            [
                *[
                    {
                        "setting_id": "iot23__ton_iot__lr__natural_transfer",
                        "dataset": "iot23",
                        "record_id": f"sv_{idx}",
                        "group_id": f"sv_{idx}",
                        "label": 0 if idx < 5 else 1,
                        "family_group": "other" if idx < 5 else "service_disruption",
                        "score": score,
                        "source_dataset": "iot23",
                        "target_dataset": "ton_iot",
                        "model": "logistic_regression",
                        "threshold": 0.5,
                        "eval_role": "source_validation",
                    }
                    for idx, score in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
                ],
                *[
                    {
                        "setting_id": "iot23__ton_iot__lr__natural_transfer",
                        "dataset": "iot23",
                        "record_id": f"sh_{idx}",
                        "group_id": f"sh_{idx}",
                        "label": 0 if idx < 2 else 1,
                        "family_group": "other" if idx < 2 else "service_disruption",
                        "score": score,
                        "source_dataset": "iot23",
                        "target_dataset": "ton_iot",
                        "model": "logistic_regression",
                        "threshold": 0.5,
                        "eval_role": "source_holdout",
                    }
                    for idx, score in enumerate([0.1, 0.2, 0.6, 0.7])
                ],
                *[
                    {
                        "setting_id": "iot23__ton_iot__lr__natural_transfer",
                        "dataset": "ton_iot",
                        "record_id": f"tt_{idx}",
                        "group_id": f"tt_{idx}",
                        "label": 0 if idx < 5 else 1,
                        "family_group": "other" if idx < 5 else "service_disruption",
                        "score": score,
                        "source_dataset": "iot23",
                        "target_dataset": "ton_iot",
                        "model": "logistic_regression",
                        "threshold": 0.5,
                        "eval_role": "target_transfer",
                    }
                    for idx, score in enumerate([0.7, 0.8, 0.9, 0.95, 0.99, 0.96, 0.97])
                ],
            ]
        )

        target_retuned = build_target_retuned_oracle_baseline(settings, predictions, target_fpr=0.01)
        budget_sensitivity = build_fpr_budget_sensitivity(settings, predictions, [0.005, 0.01, 0.05])

        self.assertAlmostEqual(
            float(target_retuned["oracle_target_threshold_at_1pct_fpr"].iloc[0]),
            0.99,
        )
        self.assertTrue((budget_sensitivity["threshold"].astype(float) > 0.5).all())

    def test_diagnostics_cover_current_canonical_benchmark(self) -> None:
        settings = pd.read_csv(
            "artifacts/benchmark/settings.csv",
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
        budgets = [0.005, 0.01, 0.05]
        target_retuned = pd.read_csv("artifacts/acsac/target_retuned_oracle_baseline.csv")
        budget_sensitivity = pd.read_csv("artifacts/acsac/fpr_budget_sensitivity.csv")
        validate_acsac_diagnostics(target_retuned, budget_sensitivity, settings, budgets)
        self.assertEqual(len(target_retuned), 24)
        self.assertEqual(len(budget_sensitivity), 72)
        self.assertEqual(sorted(budget_sensitivity["fpr_budget"].unique().tolist()), budgets)


if __name__ == "__main__":
    unittest.main()
