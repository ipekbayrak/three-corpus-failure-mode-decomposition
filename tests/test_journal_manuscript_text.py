from __future__ import annotations

import unittest
from pathlib import Path


class JournalManuscriptTextTest(unittest.TestCase):
    def test_journal_manuscript_inlines_shared_schema_and_keeps_only_oracle_appendix(self) -> None:
        manuscript = Path("paper/manuscript.tex").read_text(encoding="utf-8")

        self.assertIn(
            "\\title{Decomposing Fixed-Threshold Transfer in Tabular Flow-Level NIDS}",
            manuscript,
        )
        self.assertNotIn(
            "\\title{Fixed-Threshold Tabular Flow-Level NIDS under Domain Shift: Ranking Failure, Covariate Drift, and the Limits of Target-Free Monitoring}",
            manuscript,
        )
        self.assertNotIn("Appendix~\\ref{app:shared-schema}", manuscript)
        self.assertNotIn("\\label{app:shared-schema}", manuscript)
        self.assertNotIn("\\section{Shared Feature Schema}", manuscript)
        self.assertIn("\\input{tables/shared_feature_schema.tex}", manuscript)
        self.assertIn(
            "Table~\\ref{tab:shared-feature-schema-quantitative} lists the 12 quantitative derived features. The protocol field is one-hot encoded into four canonical indicators: proto\\_tcp, proto\\_udp, proto\\_icmp, and proto\\_other.",
            manuscript,
        )
        self.assertIn("\\appendix", manuscript)
        self.assertIn("\\section{Diagnostic Oracle Retuning}", manuscript)
        self.assertIn("\\label{app:oracle-retuning}", manuscript)
        self.assertIn("\\FullRankingFailureCount{}", manuscript)
        self.assertIn("\\FullCollapseCount{}", manuscript)
        self.assertIn("inverted target score orientation", manuscript)
        self.assertIn(
            "Cross-dataset evaluations of network intrusion detection systems (NIDS) routinely show degradation under domain shift, but they seldom distinguish threshold miscalibration from loss of ranking quality.",
            manuscript,
        )
        self.assertIn("This paper addresses the decomposition gap in a three-corpus benchmark.", manuscript)
        self.assertIn(
            "The paper contributes three empirical findings. First, the dominant failure mode in this three-corpus benchmark is ranking failure",
            manuscript,
        )
        self.assertIn(
            "This benchmark-wide result includes the Tier~1 stress-test directions involving UNSW-NB15, while the claim-bearing novelty analysis remains restricted to the \\PrimaryIotSettingCount{} IoT-only settings.",
            manuscript,
        )
        self.assertIn(
            "The benchmark-wide count is driven by the \\CompetentUnswStressTestSettingCount{} UNSW-related stress-test directions; the \\CompetentIotOnlySettingCount{} IoT-only competent transfers split \\CompetentIotOnlyRankingFailureCount{} failures and \\CompetentIotOnlyNonCollapsedCount{} rows above the 0.60 failure floor.",
            manuscript,
        )
        self.assertIn("No setting in this benchmark lies exactly at the AUROC~=~0.50 boundary.", manuscript)
        self.assertIn(
            "On point estimates, \\FullInversionCount{} of those failing settings fall strictly below chance, indicating inverted target score orientation under the transferred rule.",
            manuscript,
        )
        self.assertIn(
            "On point estimates, \\FullCollapseCount{} of those settings are collapse rows; because no setting lies exactly at the AUROC~=~0.50 boundary, all \\FullCollapseCount{} are below-chance inversions under the transferred score orientation.",
            manuscript,
        )
        self.assertIn(
            "A score-space view is consistent with that interpretation: all \\FullSourcePositiveMedianGapCount{} source-holdout rows keep attack median scores above benign medians, while \\FullInversionMedianCrossoverCount{} of the \\FullInversionCount{} point-estimate inversion rows show target benign medians meeting or exceeding target attack medians.",
            manuscript,
        )
        self.assertNotIn(
            "\\FullCollapseCount{} settings are at or below chance and \\FullInversionCount{} fall below chance",
            manuscript,
        )
        self.assertNotIn(
            "all \\FullInversionCount{} collapse rows fall strictly below chance",
            manuscript,
        )
        self.assertNotIn("\\UnswTargetCollapseCount", manuscript)
        self.assertNotIn("\\TonToIotCollapseCount", manuscript)
        self.assertNotIn(
            "\\FullCollapseCount{} of \\NaturalSettingCount{} settings have target AUROC below~0.6",
            manuscript,
        )
        self.assertNotIn("Panel~A", manuscript)
        self.assertNotIn("Panel~B", manuscript)
        self.assertNotIn("\\label{fig:transfer-benchmark}", manuscript)
        self.assertIn("Figures~\\ref{fig:transfer-tpr} and~\\ref{fig:transfer-fpr}", manuscript)
        self.assertIn("\\includegraphics[width=\\textwidth]{figures/transfer_tpr.pdf}", manuscript)
        self.assertIn("\\includegraphics[width=\\textwidth]{figures/transfer_fpr.pdf}", manuscript)
        self.assertIn(
            "Taken together, these results support a stricter diagnostic framework for evaluating benchmark-bounded cross-dataset tabular flow-level NIDS deployment",
            manuscript,
        )
        self.assertIn(
            "the same frozen source-threshold transfer protocol is additionally audited at 0.5\\%, 1\\%, and 5\\% source-validation FPR budgets, transferring each source-selected threshold unchanged to source holdout and target.",
            manuscript,
        )
        self.assertIn(
            "As one auxiliary unlabeled adaptation check, CORAL is fit using source-train features and a deterministic 50/50 target-test adaptation half split by group, then evaluated on the disjoint target-evaluation half under the same frozen \\FrozenFprTarget{} source-validation threshold rule.",
            manuscript,
        )
        self.assertIn("\\subsection{Operating-Budget Robustness}", manuscript)
        self.assertIn("\\label{sec:budget-robustness}", manuscript)
        self.assertIn("\\input{tables/budget_robustness.tex}", manuscript)
        self.assertIn(
            "Table~\\ref{tab:budget-robustness} shows that loosening the source-selected budget makes the transferred threshold more aggressive",
            manuscript,
        )
        self.assertIn(
            "The largest FPR$\\times$ medians are driven by \\BudgetHalfPctCompetentDominantFprxDirection{} at 0.5\\%, \\BudgetOnePctCompetentDominantFprxDirection{} at 1\\%, and \\BudgetFivePctCompetentDominantFprxDirections{} at 5\\%.",
            manuscript,
        )
        self.assertIn(
            "neither a 0.5\\%/1\\%/5\\% source-side budget sweep nor an auxiliary CORAL comparison improves the operational conclusion at the deployment budget.",
            manuscript,
        )
        self.assertIn(
            "Even at 5\\%, \\BudgetFivePctCompetentCatastrophicCount{} of \\CompetentSettingCount{} competent settings remain catastrophic, still \\BudgetFivePctCompetentCatastrophicShare{} of the competent set.",
            manuscript,
        )
        self.assertIn(
            "\\subsection{Auxiliary CORAL Check}",
            manuscript,
        )
        self.assertIn(
            "\\label{sec:coral-auxiliary}",
            manuscript,
        )
        self.assertIn(
            "\\input{tables/coral_auxiliary.tex}",
            manuscript,
        )
        self.assertIn(
            "Table~\\ref{tab:coral-auxiliary} is a split-matched auxiliary comparison on the same competent-set view and held-out target-evaluation halves.",
            manuscript,
        )
        self.assertIn(
            "In that matched view, CORAL raises median target AUROC but leaves target pAUC@1\\% at \\CoralCompetentMedianTargetPauc{} while worsening median FPR$\\times$ to \\CoralCompetentMedianBudgetMultiple{} and increasing catastrophic settings to \\CoralCompetentCatastrophicCount{}.",
            manuscript,
        )
        self.assertIn(
            "The auxiliary adaptation check therefore supports the paper's diagnosis: better whole-ROC ranking does not recover low-FPR deployability here.",
            manuscript,
        )
        self.assertIn(
            "This conclusion is not a single-threshold artifact: even at 5\\%, competent-set median realized target FPR remains \\BudgetFivePctCompetentMedianBudgetMultiple{}x the selected budget and \\BudgetFivePctCompetentCatastrophicCount{} of \\CompetentSettingCount{} settings remain catastrophic.",
            manuscript,
        )
        self.assertIn(
            "This central result is robust to the 0.5\\%/1\\%/5\\% source-side budget sweep",
            manuscript,
        )
        self.assertIn(
            "A split-matched auxiliary CORAL check raises whole-ROC AUROC but leaves low-FPR ranking unchanged and worsens off-budget/catastrophic behavior.",
            manuscript,
        )
        self.assertIn(
            "The auxiliary audit is also HGB-only, so it should be read as a representative strong-model diagnostic rather than evidence that LR / ET / MLP retain the same amount of dataset-native signal.",
            manuscript,
        )


if __name__ == "__main__":
    unittest.main()
