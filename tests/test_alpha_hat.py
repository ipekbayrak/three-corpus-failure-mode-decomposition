from __future__ import annotations

import unittest

import numpy as np

from iot_ids.novelty import KNNNoveltyProxy, compute_oracle_alpha, mapping_coverage


class AlphaHatTest(unittest.TestCase):
    def test_oracle_alpha_fraction(self) -> None:
        alpha = compute_oracle_alpha(["a", "b"], ["a", "c", "c", "b"])
        self.assertAlmostEqual(alpha, 0.5)

    def test_oracle_alpha_excludes_other(self) -> None:
        alpha = compute_oracle_alpha(["seen", "other"], ["other", "novel", "seen"])
        self.assertAlmostEqual(alpha, 0.5)

    def test_mapping_coverage(self) -> None:
        coverage, other_share = mapping_coverage(["service_disruption", "other", "other"])
        self.assertAlmostEqual(coverage, 1.0 / 3.0)
        self.assertAlmostEqual(other_share, 2.0 / 3.0)

    def test_knn_proxy_fallback_and_summary(self) -> None:
        rng = np.random.default_rng(0)
        source_attacks = rng.normal(loc=0.0, scale=0.3, size=(20, 4))
        target = rng.normal(loc=2.0, scale=0.3, size=(10, 4))
        proxy = KNNNoveltyProxy(k_nn=3, distance_quantile=0.9).fit(source_attacks)
        scores = np.zeros(10)
        summary = proxy.summarize(target, scores, threshold=10.0, fallback_k=3)
        self.assertTrue(summary.selected_by_fallback)
        self.assertEqual(summary.selected_count, 3)
        self.assertGreaterEqual(summary.alpha_hat, 0.0)


if __name__ == "__main__":
    unittest.main()
