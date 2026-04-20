from __future__ import annotations

import unittest

import pandas as pd

from iot_ids.decision import add_operational_metrics, summarise_monitor_performance


class DecisionTest(unittest.TestCase):
    def test_add_operational_metrics(self) -> None:
        frame = pd.DataFrame(
            {
                "loss_rel": [0.8, -0.2],
                "target_realized_fpr": [0.01, 0.25],
            }
        )
        enriched = add_operational_metrics(frame)
        self.assertEqual(enriched["relative_tpr_drop"].tolist(), [0.8, 0.0])
        self.assertEqual(enriched["tpr_loss_unsafe"].tolist(), [1, 0])
        self.assertEqual(enriched["catastrophic_operational_failure"].tolist(), [0, 1])

    def test_monitor_summary_uses_operational_targets(self) -> None:
        frame = pd.DataFrame(
            {
                "score": [0.1, 0.2, 0.8, 0.9],
                "operational_severity": [0.0, 0.2, 0.8, 1.0],
                "catastrophic_operational_failure": [0, 0, 1, 1],
            }
        )
        summary = summarise_monitor_performance(frame, "score")
        self.assertIn("spearman_vs_operational_severity", summary)
        self.assertIn("catastrophic_auroc", summary)
        self.assertIn("catastrophic_auprc", summary)


if __name__ == "__main__":
    unittest.main()
