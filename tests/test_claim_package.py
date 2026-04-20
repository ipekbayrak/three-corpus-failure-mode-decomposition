from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from iot_ids.claim_package import (
    build_claim_decision,
    build_monitor_comparison,
    build_primary_oracle_context,
)


class ClaimPackageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = pd.read_csv(Path("artifacts/benchmark/settings.csv"))

    def test_build_monitor_comparison_from_canonical_settings(self) -> None:
        comparison = build_monitor_comparison(self.settings)

        self.assertEqual(len(comparison), 8)
        self.assertEqual(
            comparison["subset"].tolist(),
            [
                "all_natural",
                "all_natural",
                "all_natural",
                "all_natural",
                "primary_iot",
                "primary_iot",
                "primary_iot",
                "primary_iot",
            ],
        )
        self.assertIn("risk_from_source_validation_tpr", comparison["score_name"].tolist())
        self.assertEqual(
            comparison.loc[comparison["subset"] == "all_natural", "n_settings"].unique().tolist(),
            [24],
        )
        self.assertEqual(
            comparison.loc[comparison["subset"] == "primary_iot", "n_settings"].unique().tolist(),
            [8],
        )
        for column in [
            "spearman_vs_operational_severity",
            "catastrophic_auroc",
            "catastrophic_auprc",
        ]:
            self.assertIn(column, comparison.columns)

    def test_build_primary_oracle_context_from_canonical_settings(self) -> None:
        primary_context = build_primary_oracle_context(self.settings)

        self.assertEqual(len(primary_context), 8)
        self.assertTrue((primary_context["source_dataset"].isin(["iot23", "ton_iot"])).all())
        self.assertTrue((primary_context["target_dataset"].isin(["iot23", "ton_iot"])).all())
        self.assertIn("relative_tpr_drop", primary_context.columns)
        self.assertIn("catastrophic_operational_failure", primary_context.columns)

    def test_claim_decision_regression_from_canonical_settings(self) -> None:
        comparison = build_monitor_comparison(self.settings)
        decision = build_claim_decision(self.settings, comparison)

        self.assertFalse(decision["deployable_alpha_hat_main_claim"])
        self.assertEqual(decision["alpha_hat_placement"], "main_negative_result")
        self.assertFalse(decision["proxy_a_promoted_to_main_claim"])
        self.assertEqual(
            decision["strongest_audited_monitor_all_natural"],
            "risk_from_source_validation_tpr",
        )
        self.assertEqual(
            decision["alpha_hat_all_natural_ranks"],
            {
                "catastrophic_auroc": 2,
                "catastrophic_auprc": 2,
                "operational_severity_spearman": 3,
            },
        )
        self.assertIn("did not outperform", decision["decision_reason"])
        self.assertTrue(decision["strongest_unlabeled_baseline_all_natural"])
        self.assertIn(
            "proxy_a_distance",
            decision["strongest_unlabeled_baseline_primary_iot"],
        )


if __name__ == "__main__":
    unittest.main()
