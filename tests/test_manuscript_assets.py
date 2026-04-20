from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from iot_ids.manuscript_assets import (
    AcsacOutputs,
    FROZEN_FPR_BUDGET,
    ManuscriptOutputs,
    ManuscriptSources,
    SATURATED_THRESHOLD_TOL,
    SATURATED_THRESHOLD_TOP,
    SOURCE_COMPETENCE_FPR_CEILING,
    SOURCE_COMPETENCE_TPR_FLOOR,
    write_acsac_assets,
    write_manuscript_assets,
)


class ManuscriptAssetsTest(unittest.TestCase):
    def test_build_manuscript_assets_from_canonical_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outputs = ManuscriptOutputs(
                within_table=root / "within_domain.tex",
                cross_table=root / "cross_domain.tex",
                monitor_table=root / "monitor_comparison.tex",
                primary_table=root / "primary_oracle_context.tex",
                ranking_diagnosis_table=root / "ranking_diagnosis.tex",
                budget_robustness_table=root / "budget_robustness.tex",
                oracle_retuned_summary_table=root / "oracle_retuned_summary.tex",
                coral_auxiliary_table=root / "coral_auxiliary.tex",
                manuscript_numbers=root / "manuscript_numbers.tex",
                transfer_tpr_figure_pdf=root / "transfer_tpr.pdf",
                transfer_tpr_figure_png=root / "transfer_tpr.png",
                transfer_fpr_figure_pdf=root / "transfer_fpr.pdf",
                transfer_fpr_figure_png=root / "transfer_fpr.png",
                oracle_figure_pdf=root / "oracle_novelty_primary.pdf",
                oracle_figure_png=root / "oracle_novelty_primary.png",
                monitor_figure_pdf=root / "monitor_comparison.pdf",
                monitor_figure_png=root / "monitor_comparison.png",
            )
            sources = ManuscriptSources(
                within_domain_summary=Path("artifacts/benchmark/within_domain_summary.csv"),
                cross_domain_summary=Path("artifacts/benchmark/cross_domain_summary.csv"),
                monitor_comparison=Path("artifacts/claims/full_data_monitor_comparison.csv"),
                primary_oracle_context=Path(
                    "artifacts/claims/full_data_primary_oracle_context.csv"
                ),
                claim_decision=Path("artifacts/claims/full_data_claim_decision.json"),
                target_retuned_oracle_baseline=Path(
                    "artifacts/acsac/target_retuned_oracle_baseline.csv"
                ),
                fpr_budget_sensitivity=Path("artifacts/acsac/fpr_budget_sensitivity.csv"),
                ranking_diagnosis=Path("artifacts/ranking_diagnosis/ranking_diagnosis.csv"),
                source_holdout_fpr=Path("artifacts/benchmark/source_holdout_fpr.csv"),
                recalibration=Path("artifacts/recalibration/unlabeled_recalibration.csv"),
                direction_covariate_shift=Path(
                    "artifacts/covariate_shift/direction_covariate_shift.csv"
                ),
                coverage_sensitivity=Path(
                    "artifacts/coverage_sensitivity/coverage_sensitivity.csv"
                ),
                coral_auxiliary=Path("artifacts/da/coral_auxiliary.csv"),
            )

            summary = write_manuscript_assets(sources, outputs)

            self.assertEqual(
                summary["within_rows"],
                len(pd.read_csv("artifacts/benchmark/within_domain_summary.csv")),
            )
            self.assertEqual(
                summary["cross_rows"],
                len(pd.read_csv("artifacts/benchmark/cross_domain_summary.csv")),
            )
            self.assertEqual(
                summary["monitor_rows"],
                len(pd.read_csv("artifacts/claims/full_data_monitor_comparison.csv")),
            )
            self.assertEqual(
                summary["primary_rows"],
                len(pd.read_csv("artifacts/claims/full_data_primary_oracle_context.csv")),
            )

            for path in [
                outputs.within_table,
                outputs.cross_table,
                outputs.monitor_table,
                outputs.primary_table,
                outputs.ranking_diagnosis_table,
                outputs.budget_robustness_table,
                outputs.oracle_retuned_summary_table,
                outputs.coral_auxiliary_table,
                outputs.manuscript_numbers,
                outputs.transfer_tpr_figure_pdf,
                outputs.transfer_tpr_figure_png,
                outputs.transfer_fpr_figure_pdf,
                outputs.transfer_fpr_figure_png,
                outputs.oracle_figure_pdf,
                outputs.oracle_figure_png,
                outputs.monitor_figure_pdf,
                outputs.monitor_figure_png,
            ]:
                self.assertTrue(path.exists(), msg=f"missing output: {path}")

            within_table = outputs.within_table.read_text(encoding="utf-8")
            ranking_table = outputs.ranking_diagnosis_table.read_text(encoding="utf-8")
            oracle_retuned_table = outputs.oracle_retuned_summary_table.read_text(
                encoding="utf-8"
            )
            coral_table = outputs.coral_auxiliary_table.read_text(encoding="utf-8")
            self.assertIn(r"ET$\dagger$", within_table)
            self.assertNotIn("Transfer OK", ranking_table)
            self.assertIn("Non-collapsed, low TPR loss", ranking_table)
            self.assertIn(r"Median paired $\Delta$TPR", oracle_retuned_table)
            self.assertIn(
                r"runs far above the target-domain \FrozenFprTarget{} FPR budget",
                oracle_retuned_table,
            )
            self.assertIn(r"that corrected cutoff can collapse TPR", oracle_retuned_table)
            self.assertIn("Auxiliary unlabeled CORAL comparison", coral_table)
            self.assertIn("Source-only", coral_table)
            self.assertIn("CORAL", coral_table)
            self.assertIn("Rank fail./18", coral_table)
            self.assertIn(r"Med. tgt pAUC@1\%", coral_table)
            self.assertIn(r"Med. FPR$\times$", coral_table)
            self.assertIn("both rows are evaluated on the same held-out target half", coral_table)
            self.assertIn(
                r"leaves low-FPR pAUC unchanged and worsens FPR$\times$/Cat.",
                coral_table,
            )

            numbers = outputs.manuscript_numbers.read_text(encoding="utf-8")
            self.assertIn(r"\newcommand{\NaturalSettingCount}{", numbers)
            self.assertIn(r"\newcommand{\PrimaryIotSettingCount}{", numbers)
            self.assertIn(r"\newcommand{\CompetentSourcePairCount}{", numbers)
            self.assertIn(r"\newcommand{\RecalibrationEvalFprMax}{", numbers)
            self.assertIn(r"\newcommand{\RecalibrationEvalFprExceedanceCount}{", numbers)
            self.assertIn(r"\newcommand{\RecalibrationPartitionScoreKsMedian}{", numbers)
            self.assertIn(r"\newcommand{\RecalibrationPartitionScoreKsMax}{", numbers)
            self.assertIn(r"\newcommand{\AllNaturalMedianRetunedTprGain}", numbers)
            self.assertIn(r"\newcommand{\BudgetFivePctMedianTargetTpr}", numbers)
            self.assertIn(r"\newcommand{\DirectionOracleSpearman}{", numbers)
            self.assertIn(r"\newcommand{\DirectionBenignKsFprSeveritySpearman}{", numbers)
            self.assertIn(r"\newcommand{\CoverageSweepStableFloorMin}{", numbers)
            self.assertIn(
                rf"\newcommand{{\ModelCount}}{{{pd.read_csv('artifacts/benchmark/within_domain_summary.csv')['model'].nunique()}}}",
                numbers,
            )
            ranking_frame = pd.read_csv("artifacts/ranking_diagnosis/ranking_diagnosis.csv")
            full_failure_count = int((ranking_frame["target_auroc"] < 0.60).sum())
            full_collapse_count = int((ranking_frame["target_auroc"] <= 0.50).sum())
            full_inversion_count = int((ranking_frame["target_auroc"] < 0.50).sum())
            self.assertIn(rf"\newcommand{{\FullRankingFailureCount}}{{{full_failure_count}}}", numbers)
            self.assertIn(rf"\newcommand{{\FullCollapseCount}}{{{full_collapse_count}}}", numbers)
            self.assertIn(rf"\newcommand{{\FullInversionCount}}{{{full_inversion_count}}}", numbers)
            self.assertIn(r"\newcommand{\CompetentRankingFailureCount}{", numbers)
            self.assertIn(r"\newcommand{\CompetentCollapseCount}{", numbers)
            self.assertIn(r"\newcommand{\CompetentInversionCount}{", numbers)
            self.assertIn(r"\newcommand{\CompetentRobustInversionCount}{", numbers)
            self.assertIn(r"\newcommand{\CompetentSourcePositiveMedianGapCount}{", numbers)
            self.assertIn(r"\newcommand{\CompetentInversionMedianCrossoverCount}{", numbers)
            self.assertIn(r"\newcommand{\FullRobustInversionCount}{", numbers)
            self.assertIn(r"\newcommand{\FullSourcePositiveMedianGapCount}{", numbers)
            self.assertIn(r"\newcommand{\FullInversionMedianCrossoverCount}{", numbers)
            self.assertIn(r"\newcommand{\CompetentIotOnlySettingCount}{", numbers)
            self.assertIn(r"\newcommand{\CompetentIotOnlyRankingFailureCount}{", numbers)
            self.assertIn(r"\newcommand{\CompetentIotOnlyNonCollapsedCount}{", numbers)
            self.assertIn(r"\newcommand{\CompetentUnswStressTestSettingCount}{", numbers)
            self.assertIn(r"\newcommand{\CompetentUnswStressTestRankingFailureCount}{", numbers)
            self.assertIn(r"\newcommand{\CompetentMinTargetAuroc}{", numbers)
            self.assertIn(r"\newcommand{\CompetentMaxTargetAuroc}{", numbers)
            self.assertIn(r"\newcommand{\CompetentTargetPaucMedian}{", numbers)
            self.assertIn(r"\newcommand{\CompetentTargetPaucMin}{", numbers)
            self.assertIn(r"\newcommand{\CompetentTargetPaucMax}{", numbers)
            self.assertIn(r"\newcommand{\CompetentSourcePaucMedian}{", numbers)
            self.assertIn(r"\newcommand{\BudgetHalfPctCompetentMedianSourceTpr}{", numbers)
            self.assertIn(r"\newcommand{\BudgetHalfPctCompetentMedianTargetTpr}{", numbers)
            self.assertIn(r"\newcommand{\BudgetHalfPctCompetentMedianTargetRealizedFpr}{", numbers)
            self.assertIn(r"\newcommand{\BudgetHalfPctCompetentMedianBudgetMultiple}{", numbers)
            self.assertIn(r"\newcommand{\BudgetHalfPctCompetentCatastrophicCount}{", numbers)
            self.assertIn(r"\newcommand{\BudgetOnePctCompetentMedianSourceTpr}{", numbers)
            self.assertIn(r"\newcommand{\BudgetOnePctCompetentMedianTargetTpr}{", numbers)
            self.assertIn(r"\newcommand{\BudgetOnePctCompetentMedianTargetRealizedFpr}{", numbers)
            self.assertIn(r"\newcommand{\BudgetOnePctCompetentMedianBudgetMultiple}{", numbers)
            self.assertIn(r"\newcommand{\BudgetOnePctCompetentCatastrophicCount}{", numbers)
            self.assertIn(r"\newcommand{\BudgetFivePctCompetentMedianSourceTpr}{", numbers)
            self.assertIn(r"\newcommand{\BudgetFivePctCompetentMedianTargetTpr}{", numbers)
            self.assertIn(r"\newcommand{\BudgetFivePctCompetentMedianTargetRealizedFpr}{", numbers)
            self.assertIn(r"\newcommand{\BudgetFivePctCompetentMedianBudgetMultiple}{", numbers)
            self.assertIn(r"\newcommand{\BudgetFivePctCompetentCatastrophicCount}{", numbers)
            self.assertIn(r"\newcommand{\BudgetHalfPctCompetentDominantFprxDirection}{", numbers)
            self.assertIn(r"\newcommand{\BudgetOnePctCompetentDominantFprxDirection}{", numbers)
            self.assertIn(r"\newcommand{\BudgetFivePctCompetentDominantFprxDirections}{", numbers)
            self.assertIn(r"\newcommand{\BudgetFivePctCompetentCatastrophicShare}{", numbers)
            self.assertIn(r"\newcommand{\CoralCompetentMedianTargetAuroc}{", numbers)
            self.assertIn(r"\newcommand{\CoralCompetentRankingFailureCount}{", numbers)
            self.assertIn(r"\newcommand{\CoralCompetentMedianTargetPauc}{", numbers)
            self.assertIn(r"\newcommand{\CoralCompetentMedianBudgetMultiple}{", numbers)
            self.assertIn(r"\newcommand{\CoralCompetentCatastrophicCount}{", numbers)
            self.assertIn(r"\newcommand{\CoralCompetentCatastrophicShare}{", numbers)
            ranking_table = outputs.ranking_diagnosis_table.read_text(encoding="utf-8")
            self.assertIn("inverted-orientation failures", ranking_table)
            self.assertNotIn("target AUROC below~0.60", ranking_table)

            source_holdout_fpr = pd.read_csv("artifacts/benchmark/source_holdout_fpr.csv").rename(
                columns={"dataset": "source_dataset"}
            )
            ranking_source = ranking_frame[
                ["source_dataset", "model", "source_holdout_tpr"]
            ].drop_duplicates()
            competence = ranking_source.merge(
                source_holdout_fpr[
                    ["source_dataset", "model", "threshold", "source_holdout_realized_fpr"]
                ],
                on=["source_dataset", "model"],
                how="left",
                validate="one_to_one",
            )
            competence["threshold_saturated_top"] = (
                competence["threshold"].astype(float)
                >= (SATURATED_THRESHOLD_TOP - SATURATED_THRESHOLD_TOL)
            ).astype(int)
            competence["low_tpr"] = (
                competence["source_holdout_tpr"].astype(float) < SOURCE_COMPETENCE_TPR_FLOOR
            ).astype(int)
            competence["high_fpr"] = (
                competence["source_holdout_realized_fpr"].astype(float)
                > SOURCE_COMPETENCE_FPR_CEILING
            ).astype(int)
            competence["saturated_budget_violation"] = (
                competence["threshold_saturated_top"].astype(bool)
                & (
                    competence["source_holdout_realized_fpr"].astype(float)
                    > FROZEN_FPR_BUDGET
                )
            ).astype(int)
            competence["is_competent"] = (
                ~(
                    competence["low_tpr"].astype(bool)
                    | competence["high_fpr"].astype(bool)
                    | competence["saturated_budget_violation"].astype(bool)
                )
            ).astype(int)
            competent_pairs = {
                (row.source_dataset, row.model)
                for row in competence.loc[competence["is_competent"] == 1].itertuples()
            }
            competent_ranking = ranking_frame.loc[
                ranking_frame.apply(
                    lambda row: (row["source_dataset"], row["model"]) in competent_pairs,
                    axis=1,
                )
            ].copy()
            full_robust_inversion_count = int(
                (
                    (ranking_frame["target_auroc"].astype(float) < 0.50)
                    & (ranking_frame["target_auroc_ci_hi"].astype(float) < 0.50)
                ).sum()
            )
            full_source_positive_median_gap_count = int(
                (ranking_frame["source_holdout_median_score_gap"].astype(float) > 0.0).sum()
            )
            full_inversion_median_crossover_count = int(
                (
                    (ranking_frame["target_auroc"].astype(float) < 0.50)
                    & (ranking_frame["target_median_score_gap"].astype(float) <= 0.0)
                ).sum()
            )
            competent_robust_inversion_count = int(
                (
                    (competent_ranking["target_auroc"].astype(float) < 0.50)
                    & (competent_ranking["target_auroc_ci_hi"].astype(float) < 0.50)
                ).sum()
            )
            competent_source_positive_median_gap_count = int(
                (competent_ranking["source_holdout_median_score_gap"].astype(float) > 0.0).sum()
            )
            competent_inversion_median_crossover_count = int(
                (
                    (competent_ranking["target_auroc"].astype(float) < 0.50)
                    & (competent_ranking["target_median_score_gap"].astype(float) <= 0.0)
                ).sum()
            )
            competent_iot_only = competent_ranking.loc[
                (
                    (competent_ranking["source_dataset"] == "iot23")
                    & (competent_ranking["target_dataset"] == "ton_iot")
                )
                | (
                    (competent_ranking["source_dataset"] == "ton_iot")
                    & (competent_ranking["target_dataset"] == "iot23")
                )
            ].copy()
            competent_unsw_related = competent_ranking.loc[
                (competent_ranking["source_dataset"] == "unsw_nb15")
                | (competent_ranking["target_dataset"] == "unsw_nb15")
            ].copy()
            self.assertIn(
                rf"\newcommand{{\CompetentIotOnlySettingCount}}{{{len(competent_iot_only)}}}",
                numbers,
            )
            self.assertIn(
                rf"\newcommand{{\CompetentIotOnlyRankingFailureCount}}{{{int((competent_iot_only['target_auroc'] < 0.60).sum())}}}",
                numbers,
            )
            self.assertIn(
                rf"\newcommand{{\CompetentIotOnlyNonCollapsedCount}}{{{int((competent_iot_only['target_auroc'] >= 0.60).sum())}}}",
                numbers,
            )
            self.assertIn(
                rf"\newcommand{{\CompetentUnswStressTestSettingCount}}{{{len(competent_unsw_related)}}}",
                numbers,
            )
            self.assertIn(
                rf"\newcommand{{\CompetentUnswStressTestRankingFailureCount}}{{{int((competent_unsw_related['target_auroc'] < 0.60).sum())}}}",
                numbers,
            )
            self.assertIn(
                rf"\newcommand{{\CompetentRobustInversionCount}}{{{competent_robust_inversion_count}}}",
                numbers,
            )
            self.assertIn(
                rf"\newcommand{{\CompetentSourcePositiveMedianGapCount}}{{{competent_source_positive_median_gap_count}}}",
                numbers,
            )
            self.assertIn(
                rf"\newcommand{{\CompetentInversionMedianCrossoverCount}}{{{competent_inversion_median_crossover_count}}}",
                numbers,
            )
            self.assertIn(
                rf"\newcommand{{\FullRobustInversionCount}}{{{full_robust_inversion_count}}}",
                numbers,
            )
            self.assertIn(
                rf"\newcommand{{\FullSourcePositiveMedianGapCount}}{{{full_source_positive_median_gap_count}}}",
                numbers,
            )
            self.assertIn(
                rf"\newcommand{{\FullInversionMedianCrossoverCount}}{{{full_inversion_median_crossover_count}}}",
                numbers,
            )
            budget_frame = pd.read_csv("artifacts/acsac/fpr_budget_sensitivity.csv")
            competent_budget = budget_frame.loc[
                budget_frame.apply(
                    lambda row: (row["source_dataset"], row["model"]) in competent_pairs,
                    axis=1,
                )
            ].copy()
            budget_name_map = {
                0.005: "BudgetHalfPct",
                0.01: "BudgetOnePct",
                0.05: "BudgetFivePct",
            }
            for budget, stem in budget_name_map.items():
                subset = competent_budget.loc[
                    competent_budget["fpr_budget"].astype(float).sub(budget).abs() < 1e-12
                ].copy()
                self.assertIn(
                    rf"\newcommand{{\{stem}CompetentMedianSourceTpr}}{{{subset['source_holdout_tpr_at_budget'].median():.3f}}}",
                    numbers,
                )
                self.assertIn(
                    rf"\newcommand{{\{stem}CompetentMedianTargetTpr}}{{{subset['target_tpr_at_budget'].median():.3f}}}",
                    numbers,
                )
                self.assertIn(
                    rf"\newcommand{{\{stem}CompetentMedianTargetRealizedFpr}}{{{subset['target_realized_fpr'].median():.3f}}}",
                    numbers,
                )
                self.assertIn(
                    rf"\newcommand{{\{stem}CompetentMedianBudgetMultiple}}{{{subset['fpr_budget_multiple'].median():.3f}}}",
                    numbers,
                )
                self.assertIn(
                    rf"\newcommand{{\{stem}CompetentCatastrophicCount}}{{{int(subset['catastrophic_operational_failure'].sum())}}}",
                    numbers,
                )
            journal_budget_table = outputs.budget_robustness_table.read_text(encoding="utf-8")
            self.assertIn("Src holdout med. TPR", journal_budget_table)
            self.assertIn("Tgt med. TPR", journal_budget_table)
            self.assertIn(r"Med. realized FPR", journal_budget_table)
            self.assertIn(r"Med. FPR$\times$", journal_budget_table)
            self.assertIn("Cat./18", journal_budget_table)
            self.assertIn("Source-selected operating-budget robustness on the 18 competent settings.", journal_budget_table)
            self.assertIn(r"FPR$\times$ denotes the median realized target FPR divided by the selected source-side budget.", journal_budget_table)

    def test_build_acsac_assets_from_canonical_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outputs = AcsacOutputs(
                source_table=root / "source_quality.tex",
                within_table=root / "within_domain.tex",
                cross_table=root / "cross_domain_summary.tex",
                monitor_table=root / "monitor_comparison.tex",
                oracle_table=root / "oracle_direction.tex",
                collapse_table=root / "ranking_collapse.tex",
                budget_robustness_table=root / "budget_robustness.tex",
                oracle_retuned_summary_table=root / "oracle_retuned_summary.tex",
                coral_auxiliary_table=root / "coral_auxiliary.tex",
                transfer_tpr_figure_pdf=root / "transfer_tpr.pdf",
                transfer_tpr_figure_png=root / "transfer_tpr.png",
                transfer_fpr_figure_pdf=root / "transfer_fpr.pdf",
                transfer_fpr_figure_png=root / "transfer_fpr.png",
                oracle_figure_pdf=root / "oracle_novelty_primary.pdf",
                oracle_figure_png=root / "oracle_novelty_primary.png",
            )
            sources = ManuscriptSources(
                within_domain_summary=Path("artifacts/benchmark/within_domain_summary.csv"),
                cross_domain_summary=Path("artifacts/benchmark/cross_domain_summary.csv"),
                monitor_comparison=Path("artifacts/claims/full_data_monitor_comparison.csv"),
                primary_oracle_context=Path(
                    "artifacts/claims/full_data_primary_oracle_context.csv"
                ),
                claim_decision=Path("artifacts/claims/full_data_claim_decision.json"),
                target_retuned_oracle_baseline=Path(
                    "artifacts/acsac/target_retuned_oracle_baseline.csv"
                ),
                fpr_budget_sensitivity=Path("artifacts/acsac/fpr_budget_sensitivity.csv"),
                ranking_diagnosis=Path("artifacts/ranking_diagnosis/ranking_diagnosis.csv"),
                source_holdout_fpr=Path("artifacts/benchmark/source_holdout_fpr.csv"),
                recalibration=Path("artifacts/recalibration/unlabeled_recalibration.csv"),
                direction_covariate_shift=Path(
                    "artifacts/covariate_shift/direction_covariate_shift.csv"
                ),
                coverage_sensitivity=Path(
                    "artifacts/coverage_sensitivity/coverage_sensitivity.csv"
                ),
                coral_auxiliary=Path("artifacts/da/coral_auxiliary.csv"),
            )

            summary = write_acsac_assets(sources, outputs)

            self.assertEqual(
                summary["within_rows"],
                len(pd.read_csv("artifacts/benchmark/within_domain_summary.csv")),
            )
            self.assertEqual(
                summary["cross_rows"],
                len(pd.read_csv("artifacts/benchmark/cross_domain_summary.csv")),
            )
            self.assertEqual(
                summary["monitor_rows"],
                len(
                    pd.read_csv("artifacts/claims/full_data_monitor_comparison.csv").loc[
                        lambda frame: frame["subset"] == "all_natural"
                    ]
                ),
            )
            self.assertEqual(
                summary["primary_rows"],
                len(pd.read_csv("artifacts/claims/full_data_primary_oracle_context.csv")),
            )
            self.assertEqual(summary["direction_rows"], 6)

            for path in [
                outputs.source_table,
                outputs.within_table,
                outputs.cross_table,
                outputs.monitor_table,
                outputs.oracle_table,
                outputs.collapse_table,
                outputs.budget_robustness_table,
                outputs.oracle_retuned_summary_table,
                outputs.coral_auxiliary_table,
                outputs.transfer_tpr_figure_pdf,
                outputs.transfer_tpr_figure_png,
                outputs.transfer_fpr_figure_pdf,
                outputs.transfer_fpr_figure_png,
                outputs.oracle_figure_pdf,
                outputs.oracle_figure_png,
            ]:
                self.assertTrue(path.exists(), msg=f"missing ACSAC output: {path}")

            cross_table = outputs.cross_table.read_text(encoding="utf-8")
            oracle_table = outputs.oracle_table.read_text(encoding="utf-8")
            self.assertNotIn("→", cross_table)
            self.assertIn(r"I23$\to$TON", cross_table)
            self.assertIn("Cat./", oracle_table)
            self.assertNotIn("two degenerate", oracle_table)
            self.assertIn(
                r"\label{tab:acsac-oracle-retuned}",
                outputs.oracle_retuned_summary_table.read_text(encoding="utf-8"),
            )
            collapse_table = outputs.collapse_table.read_text(encoding="utf-8")
            self.assertIn(r"Ranking failure ($\dagger$: target AUROC~$<$~0.60)", collapse_table)
            self.assertIn(r"$\ddagger$: target AUROC~$<$~0.50 under the transferred score", collapse_table)
            self.assertIn(r"Tgt AUROC [95\% CI]", collapse_table)
            self.assertIn(r"Src/Tgt pAUC@1\%", collapse_table)
            self.assertIn(r"grouped-bootstrap 95\% percentile CIs", collapse_table)
            self.assertIn(r"descriptive intervals rather than multiplicity-adjusted inference", collapse_table)
            self.assertIn(r"The final column reports source-holdout / target standardized pAUC@1\%", collapse_table)
            self.assertIn(r"0.5 denotes chance-level low-FPR ranking", collapse_table)
            self.assertRegex(collapse_table, r"0\.\d{3} \[0\.\d{3}, 0\.\d{3}\]")
            self.assertRegex(collapse_table, r"0\.\d{3} / 0\.\d{3}")
            self.assertNotIn(r"Ranking collapse ($\dagger$: target AUROC~$<$~0.60)", collapse_table)
            budget_table = outputs.budget_robustness_table.read_text(encoding="utf-8")
            self.assertIn("Src med. TPR", budget_table)
            self.assertIn("Tgt med. TPR", budget_table)
            self.assertIn(r"Med. FPR$\times$", budget_table)
            self.assertIn("Cat./18", budget_table)
            self.assertIn("Source-selected operating-budget robustness on the 18 competent settings.", budget_table)
            self.assertIn(r"FPR$\times$ denotes the median realized target FPR divided by the selected source-side budget.", budget_table)
            self.assertRegex(budget_table, r"0\.5\\% & 0\.\d{3} & 0\.\d{3} & \d+\.\d{3} & \d+ \\\\")
            coral_table = outputs.coral_auxiliary_table.read_text(encoding="utf-8")
            self.assertIn("Auxiliary unlabeled CORAL comparison", coral_table)
            self.assertIn("Source-only", coral_table)
            self.assertIn("CORAL", coral_table)
            self.assertIn(r"Med. tgt AUROC", coral_table)
            self.assertIn(r"Med. tgt pAUC@1\%", coral_table)
            self.assertIn("Cat./18", coral_table)
            self.assertIn("both rows are evaluated on the same held-out target half", coral_table)


if __name__ == "__main__":
    unittest.main()
