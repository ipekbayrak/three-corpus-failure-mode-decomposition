from __future__ import annotations

import unittest

import pandas as pd

from iot_ids.manuscript_assets import _source_competence_frame


class CompetenceRuleTest(unittest.TestCase):
    def test_competence_rule_is_metric_derived_not_name_derived(self) -> None:
        ranking = pd.DataFrame(
            [
                {
                    "source_dataset": "custom_a",
                    "target_dataset": "custom_b",
                    "model": "surprise_model",
                    "source_holdout_auroc": 0.95,
                    "source_holdout_tpr": 0.09,
                },
                {
                    "source_dataset": "custom_c",
                    "target_dataset": "custom_b",
                    "model": "surprise_model",
                    "source_holdout_auroc": 0.95,
                    "source_holdout_tpr": 0.30,
                },
            ]
        )
        source_holdout_fpr = pd.DataFrame(
            [
                {
                    "dataset": "custom_a",
                    "model": "surprise_model",
                    "threshold": 0.90,
                    "source_holdout_realized_fpr": 0.01,
                    "source_holdout_realized_fpr_ci_lo": 0.009,
                    "source_holdout_realized_fpr_ci_hi": 0.011,
                },
                {
                    "dataset": "custom_c",
                    "model": "surprise_model",
                    "threshold": 0.90,
                    "source_holdout_realized_fpr": 0.01,
                    "source_holdout_realized_fpr_ci_lo": 0.009,
                    "source_holdout_realized_fpr_ci_hi": 0.011,
                },
            ]
        )

        competence = _source_competence_frame(ranking, source_holdout_fpr)

        low_tpr_row = competence.loc[competence["source_dataset"] == "custom_a"].iloc[0]
        good_row = competence.loc[competence["source_dataset"] == "custom_c"].iloc[0]
        self.assertEqual(int(low_tpr_row["is_competent"]), 0)
        self.assertEqual(low_tpr_row["competence_reason"], "source_holdout_tpr_below_floor")
        self.assertEqual(int(good_row["is_competent"]), 1)

    def test_competence_rule_flags_saturated_threshold_budget_violation(self) -> None:
        ranking = pd.DataFrame(
            [
                {
                    "source_dataset": "iot23",
                    "target_dataset": "ton_iot",
                    "model": "ft_transformer",
                    "source_holdout_auroc": 0.80,
                    "source_holdout_tpr": 0.50,
                }
            ]
        )
        source_holdout_fpr = pd.DataFrame(
            [
                {
                    "dataset": "iot23",
                    "model": "ft_transformer",
                    "threshold": 1.0,
                    "source_holdout_realized_fpr": 0.02,
                    "source_holdout_realized_fpr_ci_lo": 0.01,
                    "source_holdout_realized_fpr_ci_hi": 0.03,
                }
            ]
        )

        competence = _source_competence_frame(ranking, source_holdout_fpr)
        row = competence.iloc[0]
        self.assertEqual(int(row["threshold_saturated_top"]), 1)
        self.assertEqual(int(row["is_competent"]), 0)
        self.assertEqual(
            row["competence_reason"],
            "threshold_saturated_with_budget_violation",
        )


if __name__ == "__main__":
    unittest.main()
