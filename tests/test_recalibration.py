from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from iot_ids.metrics import realized_fpr
from iot_ids.recalibration import (
    DEFAULT_RECALIBRATION_SEED,
    calibration_eval_score_ks_statistic,
    run_unlabeled_recalibration,
    split_benign_calibration_eval,
)


class RecalibrationTest(unittest.TestCase):
    def _load_recalibration_script(self):
        module_path = Path("scripts/46_unlabeled_recalibration.py")
        spec = importlib.util.spec_from_file_location("unlabeled_recalibration_script", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_calibration_eval_score_ks_statistic(self) -> None:
        self.assertAlmostEqual(
            calibration_eval_score_ks_statistic([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]),
            0.0,
        )
        self.assertAlmostEqual(
            calibration_eval_score_ks_statistic([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            1.0,
        )

    def test_recalibration_reports_held_out_benign_eval_fpr(self) -> None:
        seed = DEFAULT_RECALIBRATION_SEED
        group_ids = ["g0", "g1", "g2", "g3"]
        dummy_target = pd.DataFrame(
            {
                "record_id": [f"dummy_{idx}" for idx in range(len(group_ids))],
                "group_id": group_ids,
                "label": [0, 0, 0, 0],
                "score": [0.0, 0.0, 0.0, 0.0],
            }
        )
        split_probe = split_benign_calibration_eval(dummy_target, calibration_frac=0.5, seed=seed)
        calibration_groups = set(split_probe.calibration_units)
        evaluation_groups = set(split_probe.evaluation_units)
        self.assertTrue(calibration_groups)
        self.assertTrue(evaluation_groups)
        self.assertTrue(calibration_groups.isdisjoint(evaluation_groups))

        rows: list[dict[str, object]] = []
        for group_id in group_ids:
            benign_score = 0.05 if group_id in calibration_groups else 0.95
            for row_idx in range(2):
                rows.append(
                    {
                        "record_id": f"{group_id}_{row_idx}",
                        "group_id": group_id,
                        "label": 0,
                        "score": benign_score,
                    }
                )
        rows.extend(
            [
                {
                    "record_id": "attack_0",
                    "group_id": "attack_group",
                    "label": 1,
                    "score": 0.99,
                },
                {
                    "record_id": "attack_1",
                    "group_id": "attack_group",
                    "label": 1,
                    "score": 0.97,
                },
            ]
        )
        target_frame = pd.DataFrame(rows)
        settings = pd.DataFrame(
            [
                {
                    "setting_id": "s1",
                    "setting_type": "natural_transfer",
                    "source_dataset": "iot23",
                    "target_dataset": "ton_iot",
                    "model": "logistic_regression",
                    "threshold": 0.50,
                    "target_tpr_at_1pct_fpr": 0.25,
                    "target_realized_fpr": 0.50,
                    "source_holdout_tpr_at_1pct_fpr": 0.90,
                }
            ]
        )

        result = run_unlabeled_recalibration(
            settings,
            {("s1", "target_transfer"): target_frame},
            target_fpr=0.01,
            calibration_frac=0.5,
            seed=seed,
        )

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        split = split_benign_calibration_eval(target_frame, calibration_frac=0.5, seed=seed)
        expected_eval_fpr = realized_fpr(
            [0] * len(split.evaluation_benign),
            split.evaluation_benign["score"],
            float(row["recal_threshold"]),
        )
        pooled_benign = target_frame.loc[target_frame["label"] == 0]
        pooled_fpr = realized_fpr(
            [0] * len(pooled_benign),
            pooled_benign["score"],
            float(row["recal_threshold"]),
        )

        self.assertEqual(row["recalibration_split_unit"], "group_id")
        self.assertEqual(int(row["recalibration_eval_disjoint"]), 1)
        self.assertEqual(int(row["n_target_benign"]), 8)
        self.assertEqual(int(row["n_benign_calibration"]), 4)
        self.assertEqual(int(row["n_benign_evaluation"]), 4)
        self.assertAlmostEqual(float(row["recal_calibration_fpr"]), 0.0)
        self.assertAlmostEqual(float(row["recal_fpr"]), expected_eval_fpr)
        self.assertAlmostEqual(float(row["calibration_eval_score_ks"]), 1.0)
        self.assertAlmostEqual(float(row["calibration_eval_mean_score_delta"]), 0.9)
        self.assertEqual(int(row["eval_fpr_exceeds_budget"]), 1)
        self.assertNotAlmostEqual(float(row["recal_fpr"]), pooled_fpr)

    def test_acsac_table_prints_eval_fpr_to_four_decimals(self) -> None:
        module = self._load_recalibration_script()
        result = pd.DataFrame(
            [
                {
                    "source_dataset": "iot23",
                    "target_dataset": "ton_iot",
                    "model": "logistic_regression",
                    "target_auroc": 0.6362,
                    "frozen_tpr": 0.5004,
                    "recal_tpr": 0.0084,
                    "recal_fpr": 0.0099,
                },
                {
                    "source_dataset": "iot23",
                    "target_dataset": "ton_iot",
                    "model": "extra_trees",
                    "target_auroc": 0.6331,
                    "frozen_tpr": 0.0001,
                    "recal_tpr": 0.0714,
                    "recal_fpr": 0.0100,
                },
                {
                    "source_dataset": "iot23",
                    "target_dataset": "ton_iot",
                    "model": "hist_gradient_boosting",
                    "target_auroc": 0.6172,
                    "frozen_tpr": 0.0102,
                    "recal_tpr": 0.0661,
                    "recal_fpr": 0.0104,
                },
                {
                    "source_dataset": "iot23",
                    "target_dataset": "ton_iot",
                    "model": "mlp",
                    "target_auroc": 0.5590,
                    "frozen_tpr": 0.1522,
                    "recal_tpr": 0.0003,
                    "recal_fpr": 0.0116,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "recalibration.tex"
            module.write_acsac_table(result, output_path)
            table = output_path.read_text(encoding="utf-8")

        self.assertIn("printed to four decimals", table)
        self.assertIn("& 0.636 & 0.500 & 0.008 & 0.0099", table)
        self.assertIn("& 0.633 & 0.000 & 0.071 & 0.0100", table)
        self.assertIn("& 0.617 & 0.010 & 0.066 & 0.0104", table)
        self.assertIn("& 0.559 & 0.152 & 0.000 & 0.0116", table)
        self.assertIn(r"\emph{Mean (4)} & -- & 0.611 & 0.166 & 0.037 & 0.0105 \\", table)


if __name__ == "__main__":
    unittest.main()
