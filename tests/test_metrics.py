from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from iot_ids.metrics import (
    compute_metrics,
    grouped_bootstrap_metric,
    grouped_bootstrap_threshold_intervals,
    realized_fpr,
    threshold_at_fpr,
)


class MetricsTest(unittest.TestCase):
    def test_threshold_and_metrics(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
        threshold = threshold_at_fpr(labels, scores, target_fpr=0.01)
        metrics = compute_metrics(labels, scores, threshold)
        self.assertGreaterEqual(metrics["tpr_at_1pct_fpr"], 0.0)
        self.assertLessEqual(metrics["realized_fpr"], 1.0)

    def test_threshold_tie_safe_under_score_ge_rule(self) -> None:
        labels = np.array([0, 0, 0, 0, 0, 1, 1])
        scores = np.array([0.1, 1.0, 1.0, 1.0, 1.0, 1.1, 0.9])
        threshold = threshold_at_fpr(labels, scores, target_fpr=0.2)
        self.assertGreater(threshold, 1.0)
        self.assertLessEqual(realized_fpr(labels, scores, threshold), 0.2)

    def test_threshold_matches_budget_in_untied_case(self) -> None:
        labels = np.array([0] * 100 + [1, 1])
        scores = np.array(list(range(100)) + [150.0, 151.0], dtype=float)
        threshold = threshold_at_fpr(labels, scores, target_fpr=0.01)
        self.assertLessEqual(realized_fpr(labels, scores, threshold), 0.01)
        self.assertAlmostEqual(realized_fpr(labels, scores, threshold), 0.01)

    def test_grouped_bootstrap_metric(self) -> None:
        frame = pd.DataFrame(
            {
                "group_id": ["a", "a", "b", "b", "c", "c"],
                "label": [0, 1, 0, 1, 0, 1],
                "score": [0.1, 0.8, 0.2, 0.7, 0.3, 0.9],
            }
        )
        lo, hi = grouped_bootstrap_metric(
            frame,
            "tpr_at_1pct_fpr",
            threshold=0.5,
            n_boot=20,
            seed=3,
        )
        self.assertLessEqual(lo, hi)

    def test_grouped_bootstrap_metric_supports_auroc(self) -> None:
        frame = pd.DataFrame(
            {
                "group_id": ["a", "a", "b", "b", "c", "c"],
                "label": [0, 1, 0, 1, 0, 1],
                "score": [0.1, 0.8, 0.2, 0.7, 0.3, 0.9],
            }
        )

        lo, hi = grouped_bootstrap_metric(
            frame,
            "auroc",
            threshold=0.5,
            n_boot=20,
            seed=3,
        )

        self.assertLessEqual(lo, 1.0)
        self.assertLessEqual(lo, hi)
        self.assertLessEqual(hi, 1.0)

    def test_grouped_bootstrap_threshold_intervals(self) -> None:
        frame = pd.DataFrame(
            {
                "group_id": ["a", "a", "b", "b", "c", "c"],
                "label": [0, 1, 0, 1, 0, 1],
                "score": [0.1, 0.8, 0.2, 0.7, 0.3, 0.9],
            }
        )
        intervals = grouped_bootstrap_threshold_intervals(
            frame,
            threshold=0.5,
            n_boot=20,
            seed=3,
        )
        self.assertLessEqual(intervals["tpr_ci_lo"], intervals["tpr_ci_hi"])
        self.assertLessEqual(
            intervals["realized_fpr_ci_lo"],
            intervals["realized_fpr_ci_hi"],
        )


if __name__ == "__main__":
    unittest.main()
