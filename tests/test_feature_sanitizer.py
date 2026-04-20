from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from iot_ids.features import FEATURE_COLUMNS
from iot_ids.modeling import (
    NATURAL_UNIT_INTERVAL_COLUMNS,
    BoundedFeatureScaler,
    SourceFeatureSanitizer,
    fit_estimator,
    score_positive_class,
)
from iot_ids.ft_transformer import torch_is_available
from iot_ids.novelty import standardized_matrices


def _feature_frame(n_rows: int, extreme_tail: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(
        {
            "log_duration_s": rng.normal(1.0, 0.3, n_rows),
            "log_src_bytes": rng.normal(4.0, 0.8, n_rows),
            "log_dst_bytes": rng.normal(3.0, 0.7, n_rows),
            "log_src_pkts": rng.normal(2.5, 0.4, n_rows),
            "log_dst_pkts": rng.normal(2.0, 0.4, n_rows),
            "log_total_bytes": rng.normal(4.5, 0.8, n_rows),
            "log_total_pkts": rng.normal(3.0, 0.5, n_rows),
            "log_bytes_per_s": rng.normal(1.0, 0.5, n_rows),
            "log_pkts_per_s": rng.normal(0.8, 0.5, n_rows),
            "src_byte_frac": rng.uniform(-0.25, 1.25, n_rows),
            "src_pkt_frac": rng.uniform(-0.25, 1.25, n_rows),
            "bytes_per_pkt": rng.lognormal(mean=2.0, sigma=1.0, size=n_rows),
            "proto_tcp": (rng.integers(0, 4, n_rows) == 0).astype(float),
            "proto_udp": (rng.integers(0, 4, n_rows) == 1).astype(float),
            "proto_icmp": (rng.integers(0, 4, n_rows) == 2).astype(float),
            "proto_other": (rng.integers(0, 4, n_rows) == 3).astype(float),
        }
    )
    if extreme_tail:
        frame.loc[n_rows - 1, "bytes_per_pkt"] = 1e12
        frame.loc[0, "log_total_bytes"] = np.inf
        frame.loc[1, "log_total_pkts"] = -np.inf
        frame.loc[2, "log_src_bytes"] = np.nan
    return frame


class FeatureSanitizerTest(unittest.TestCase):
    def test_sanitizer_compresses_extreme_tail_and_enforces_finite_outputs(self) -> None:
        train = _feature_frame(128, extreme_tail=True)
        eval_frame = _feature_frame(16, extreme_tail=True)
        sanitizer = SourceFeatureSanitizer(FEATURE_COLUMNS).fit(train)

        transformed = sanitizer.transform(eval_frame)
        repeated = sanitizer.transform(eval_frame)

        self.assertEqual(list(transformed.columns), FEATURE_COLUMNS)
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())
        self.assertLess(
            float(transformed["bytes_per_pkt"].max()),
            float(np.log1p(1e12)),
        )
        self.assertLessEqual(
            float(transformed["bytes_per_pkt"].max()),
            float(sanitizer.upper_bounds_["bytes_per_pkt"]),
        )
        pd.testing.assert_frame_equal(transformed, repeated)
        for column in NATURAL_UNIT_INTERVAL_COLUMNS:
            self.assertTrue(((transformed[column] >= 0.0) & (transformed[column] <= 1.0)).all())

    def test_bounded_feature_scaler_limits_continuous_magnitude(self) -> None:
        train = _feature_frame(128, extreme_tail=True)
        eval_frame = _feature_frame(16, extreme_tail=True)
        sanitized_train = SourceFeatureSanitizer(FEATURE_COLUMNS).fit(train).transform(train)
        sanitized_eval = SourceFeatureSanitizer(FEATURE_COLUMNS).fit(train).transform(eval_frame)
        scaler = BoundedFeatureScaler(FEATURE_COLUMNS, clip_value=8.0).fit(sanitized_train)

        transformed = scaler.transform(sanitized_eval)

        self.assertTrue(np.isfinite(transformed.to_numpy()).all())
        self.assertLessEqual(float(np.abs(transformed.to_numpy()).max()), 8.0)
        for column in NATURAL_UNIT_INTERVAL_COLUMNS:
            self.assertTrue(((transformed[column] >= 0.0) & (transformed[column] <= 1.0)).all())

    def test_standardized_matrices_stay_finite_on_extreme_inputs(self) -> None:
        source_train = _feature_frame(96, extreme_tail=True)
        source_eval = _feature_frame(48, extreme_tail=True)
        target_eval = _feature_frame(40, extreme_tail=True)

        train_x, source_x, target_x = standardized_matrices(
            source_train=source_train,
            source_eval=source_eval,
            target_eval=target_eval,
            feature_columns=FEATURE_COLUMNS,
        )
        train_x_2, source_x_2, target_x_2 = standardized_matrices(
            source_train=source_train,
            source_eval=source_eval,
            target_eval=target_eval,
            feature_columns=FEATURE_COLUMNS,
        )

        self.assertEqual(train_x.shape, (96, len(FEATURE_COLUMNS)))
        self.assertEqual(source_x.shape, (48, len(FEATURE_COLUMNS)))
        self.assertEqual(target_x.shape, (40, len(FEATURE_COLUMNS)))
        self.assertTrue(np.isfinite(train_x).all())
        self.assertTrue(np.isfinite(source_x).all())
        self.assertTrue(np.isfinite(target_x).all())
        self.assertLessEqual(float(np.abs(train_x).max()), 8.0)
        np.testing.assert_allclose(train_x, train_x_2)
        np.testing.assert_allclose(source_x, source_x_2)
        np.testing.assert_allclose(target_x, target_x_2)

    def test_mlp_dense_path_smoke_with_extreme_bytes_per_pkt(self) -> None:
        train = _feature_frame(160, extreme_tail=True)
        train["label"] = ([0, 1] * 80)
        eval_frame = _feature_frame(32, extreme_tail=True)

        estimator = fit_estimator("mlp", train, FEATURE_COLUMNS, seed=17)
        scores = score_positive_class(estimator, eval_frame, FEATURE_COLUMNS)

        self.assertEqual(scores.shape, (32,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0.0) & (scores <= 1.0)).all())

    @unittest.skipUnless(torch_is_available(), "torch is required for ft_transformer")
    def test_ft_transformer_smoke_and_determinism(self) -> None:
        train = _feature_frame(192, extreme_tail=True)
        train["group_id"] = [f"g{idx // 4}" for idx in range(len(train))]
        train["label"] = ([0, 1, 0, 1] * 48)
        eval_frame = _feature_frame(40, extreme_tail=True)

        estimator_1 = fit_estimator("ft_transformer", train, FEATURE_COLUMNS, seed=23)
        estimator_2 = fit_estimator("ft_transformer", train, FEATURE_COLUMNS, seed=23)
        scores_1 = score_positive_class(estimator_1, eval_frame, FEATURE_COLUMNS)
        scores_2 = score_positive_class(estimator_2, eval_frame, FEATURE_COLUMNS)

        self.assertEqual(scores_1.shape, (40,))
        self.assertTrue(np.isfinite(scores_1).all())
        self.assertTrue(((scores_1 >= 0.0) & (scores_1 <= 1.0)).all())
        np.testing.assert_allclose(scores_1, scores_2, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
