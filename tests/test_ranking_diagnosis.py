from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    module_path = Path("scripts/43_ranking_diagnosis.py")
    spec = importlib.util.spec_from_file_location("ranking_diagnosis_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RankingDiagnosisTest(unittest.TestCase):
    def test_compute_pauc_uses_standardized_low_fpr_auc(self) -> None:
        module = _load_module()

        perfect = module.compute_pauc(
            np.array([0, 0, 1, 1]),
            np.array([0.10, 0.20, 0.70, 0.80]),
            max_fpr=0.01,
        )
        chance = module.compute_pauc(
            np.array([0, 0, 1, 1]),
            np.array([0.10, 0.40, 0.20, 0.30]),
            max_fpr=0.01,
        )

        self.assertAlmostEqual(perfect, 1.0)
        self.assertAlmostEqual(chance, 0.49748743718592964)

    def test_noncollapsed_low_tpr_loss_label_and_realized_fpr_column(self) -> None:
        module = _load_module()
        settings = pd.DataFrame(
            [
                {
                    "setting_id": "iot23__ton_iot__mlp__natural_transfer",
                    "setting_type": "natural_transfer",
                    "source_dataset": "iot23",
                    "target_dataset": "ton_iot",
                    "model": "mlp",
                    "target_tpr_at_1pct_fpr": 0.75,
                    "target_realized_fpr": 0.33,
                    "target_auroc_ci_lo": 0.70,
                    "target_auroc_ci_hi": 0.95,
                    "source_holdout_tpr_at_1pct_fpr": 1.0,
                    "threshold": 0.50,
                }
            ]
        )
        prediction_groups = {
            ("iot23__ton_iot__mlp__natural_transfer", "target_transfer"): pd.DataFrame(
                {
                    "label": [0, 0, 1, 1],
                    "score": [0.10, 0.20, 0.70, 0.80],
                }
            ),
            ("iot23__ton_iot__mlp__natural_transfer", "source_holdout"): pd.DataFrame(
                {
                    "label": [0, 0, 1, 1],
                    "score": [0.10, 0.20, 0.90, 1.00],
                }
            ),
        }

        result = module.diagnose_ranking(settings, prediction_groups)

        self.assertEqual(len(result), 1)
        self.assertIn("fixed_target_realized_fpr", result.columns)
        self.assertIn("ranking_orientation", result.columns)
        self.assertIn("target_auroc_ci_lo", result.columns)
        self.assertIn("target_auroc_ci_hi", result.columns)
        self.assertIn("source_holdout_benign_score_median", result.columns)
        self.assertIn("source_holdout_attack_score_median", result.columns)
        self.assertIn("target_benign_score_median", result.columns)
        self.assertIn("target_attack_score_median", result.columns)
        self.assertIn("source_holdout_median_score_gap", result.columns)
        self.assertIn("target_median_score_gap", result.columns)
        self.assertEqual(result.loc[0, "diagnosis"], "noncollapsed_low_tpr_loss")
        self.assertEqual(result.loc[0, "ranking_orientation"], "positive")
        self.assertAlmostEqual(float(result.loc[0, "fixed_target_realized_fpr"]), 0.33)
        self.assertAlmostEqual(float(result.loc[0, "target_auroc_ci_lo"]), 0.70)
        self.assertAlmostEqual(float(result.loc[0, "target_auroc_ci_hi"]), 0.95)
        self.assertAlmostEqual(float(result.loc[0, "source_holdout_benign_score_median"]), 0.15)
        self.assertAlmostEqual(float(result.loc[0, "source_holdout_attack_score_median"]), 0.95)
        self.assertAlmostEqual(float(result.loc[0, "target_benign_score_median"]), 0.15)
        self.assertAlmostEqual(float(result.loc[0, "target_attack_score_median"]), 0.75)
        self.assertAlmostEqual(float(result.loc[0, "source_holdout_median_score_gap"]), 0.80)
        self.assertAlmostEqual(float(result.loc[0, "target_median_score_gap"]), 0.60)

    def test_inverted_orientation_is_marked_as_collapse(self) -> None:
        module = _load_module()
        settings = pd.DataFrame(
            [
                {
                    "setting_id": "iot23__ton_iot__lr__natural_transfer",
                    "setting_type": "natural_transfer",
                    "source_dataset": "iot23",
                    "target_dataset": "ton_iot",
                    "model": "lr",
                    "target_tpr_at_1pct_fpr": 0.0,
                    "target_realized_fpr": 0.01,
                    "source_holdout_tpr_at_1pct_fpr": 1.0,
                    "threshold": 0.50,
                }
            ]
        )
        prediction_groups = {
            ("iot23__ton_iot__lr__natural_transfer", "target_transfer"): pd.DataFrame(
                {
                    "label": [0, 0, 1, 1],
                    "score": [0.90, 0.80, 0.20, 0.10],
                }
            )
        }

        result = module.diagnose_ranking(settings, prediction_groups)

        self.assertEqual(result.loc[0, "diagnosis"], "ranking_collapse")
        self.assertEqual(result.loc[0, "ranking_orientation"], "inverted")

    def test_at_chance_orientation_is_marked_separately(self) -> None:
        module = _load_module()
        settings = pd.DataFrame(
            [
                {
                    "setting_id": "ton_iot__iot23__mlp__natural_transfer",
                    "setting_type": "natural_transfer",
                    "source_dataset": "ton_iot",
                    "target_dataset": "iot23",
                    "model": "mlp",
                    "target_tpr_at_1pct_fpr": 0.0,
                    "target_realized_fpr": 0.01,
                    "source_holdout_tpr_at_1pct_fpr": 1.0,
                    "threshold": 0.50,
                }
            ]
        )
        prediction_groups = {
            ("ton_iot__iot23__mlp__natural_transfer", "target_transfer"): pd.DataFrame(
                {
                    "label": [0, 0, 1, 1],
                    "score": [0.90, 0.10, 0.20, 0.80],
                }
            )
        }

        result = module.diagnose_ranking(settings, prediction_groups)

        self.assertEqual(result.loc[0, "diagnosis"], "ranking_collapse")
        self.assertEqual(result.loc[0, "ranking_orientation"], "at_chance")

    def test_score_gap_fields_capture_source_separation_and_target_inversion(self) -> None:
        module = _load_module()
        settings = pd.DataFrame(
            [
                {
                    "setting_id": "unsw_nb15__ton_iot__lr__natural_transfer",
                    "setting_type": "natural_transfer",
                    "source_dataset": "unsw_nb15",
                    "target_dataset": "ton_iot",
                    "model": "lr",
                    "target_tpr_at_1pct_fpr": 0.0,
                    "target_realized_fpr": 0.02,
                    "target_auroc_ci_lo": 0.20,
                    "target_auroc_ci_hi": 0.40,
                    "source_holdout_tpr_at_1pct_fpr": 1.0,
                    "threshold": 0.50,
                }
            ]
        )
        prediction_groups = {
            ("unsw_nb15__ton_iot__lr__natural_transfer", "target_transfer"): pd.DataFrame(
                {
                    "label": [0, 0, 1, 1],
                    "score": [0.80, 0.70, 0.20, 0.10],
                }
            ),
            ("unsw_nb15__ton_iot__lr__natural_transfer", "source_holdout"): pd.DataFrame(
                {
                    "label": [0, 0, 1, 1],
                    "score": [0.10, 0.20, 0.80, 0.90],
                }
            ),
        }

        result = module.diagnose_ranking(settings, prediction_groups)

        self.assertAlmostEqual(float(result.loc[0, "source_holdout_benign_score_median"]), 0.15)
        self.assertAlmostEqual(float(result.loc[0, "source_holdout_attack_score_median"]), 0.85)
        self.assertAlmostEqual(float(result.loc[0, "target_benign_score_median"]), 0.75)
        self.assertAlmostEqual(float(result.loc[0, "target_attack_score_median"]), 0.15)
        self.assertAlmostEqual(float(result.loc[0, "source_holdout_median_score_gap"]), 0.70)
        self.assertAlmostEqual(float(result.loc[0, "target_median_score_gap"]), -0.60)
        self.assertGreater(float(result.loc[0, "source_holdout_median_score_gap"]), 0.0)
        self.assertLessEqual(float(result.loc[0, "target_median_score_gap"]), 0.0)
        self.assertEqual(result.loc[0, "ranking_orientation"], "inverted")


if __name__ == "__main__":
    unittest.main()
