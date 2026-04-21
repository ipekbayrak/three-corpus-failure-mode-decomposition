from __future__ import annotations

import unittest
from pathlib import Path


class AcsacManuscriptTextTest(unittest.TestCase):
    def test_acsac_manuscript_resolves_reviewer_visible_contradictions(self) -> None:
        manuscript = Path("paper/acsac/manuscript.tex").read_text(encoding="utf-8")

        self.assertIn(
            "\\title{Decomposing Fixed-Threshold Transfer in Tabular Flow-Level NIDS}",
            manuscript,
        )
        self.assertNotIn(
            "\\title{Ranking Failure Dominates Fixed-Threshold Tabular Flow-Level NIDS Transfer: A Negative Result}",
            manuscript,
        )
        self.assertIn(
            "Cross-dataset evaluations of network intrusion detection systems (NIDS) routinely show degradation under domain shift, but they seldom distinguish threshold miscalibration from loss of ranking quality.",
            manuscript,
        )
        self.assertNotIn("Appendix~\\ref{app:shared-schema}", manuscript)
        self.assertNotIn("\\label{app:shared-schema}", manuscript)
        self.assertNotIn("\\appendices", manuscript)
        self.assertIn("\\input{tables/shared_feature_schema.tex}", manuscript)
        self.assertIn(
            "Table~\\ref{tab:shared-feature-schema-quantitative} lists the 12 quantitative derived features. The protocol field is one-hot encoded into four canonical indicators: proto\\_tcp, proto\\_udp, proto\\_icmp, and proto\\_other.",
            manuscript,
        )
        self.assertNotIn("while staying within the target FPR budget", manuscript)
        self.assertNotIn("sampling variance pushes that held-out estimate above budget", manuscript)
        self.assertIn(
            "held-out benign Eval FPR exceeds the nominal budget in \\RecalibrationEvalFprExceedanceCount{} settings",
            manuscript,
        )
        self.assertIn("A score-space audit of the calibration-versus-evaluation benign partitions", manuscript)
        self.assertIn(
            "limited two-direction IoT contrast in which attack novelty and benign covariate shift appear to align with different failure dimensions",
            manuscript,
        )
        self.assertNotIn("the resulting failure has two operationally distinct forms", manuscript)
        self.assertIn(
            "the resulting target behavior can fail in three operationally distinct ways under the transferred score rule",
            manuscript,
        )
        self.assertIn("This paper answers the question directly in a three-corpus benchmark.", manuscript)
        self.assertNotIn(
            "ranking-collapse versus threshold-drift decomposition",
            manuscript,
        )
        self.assertIn(
            "distinguishes threshold drift from ranking failure and then separates degraded ranking from collapse, including the below-chance orientation-inversion case",
            manuscript,
        )
        self.assertIn(
            "\\emph{Threshold drift} means the ranking remains usable, but the source-calibrated operating point falls in the wrong region of the target score distribution",
            manuscript,
        )
        self.assertIn(
            "A stricter collapse case is \\emph{orientation inversion}: target AUROC falls below chance, indicating that the transferred score direction is reversed relative to the target labels.",
            manuscript,
        )
        self.assertIn(
            "Inversion is therefore a distinct subtype of collapse within ranking failure, not a benign-only recalibration problem or a label-free repair path.",
            manuscript,
        )
        self.assertIn(
            "The TON$\\to$I23 direction is not a case where oracle retuning arbitrarily destroys a good detector",
            manuscript,
        )
        self.assertIn(
            "restoring the \\FrozenFprTarget{} FPR budget requires a much higher cutoff than the target ranking can support",
            manuscript,
        )
        self.assertIn("ranking failure", manuscript)
        self.assertIn("inverted target score orientation", manuscript)
        self.assertNotIn("non-collapsed by the coarse AUROC taxonomy", manuscript)
        self.assertNotIn("0.19--0.64", manuscript)
        self.assertIn(
            "Source models with AUROC 0.80--1.00 drop to target AUROC \\CompetentMinTargetAuroc{}--\\CompetentMaxTargetAuroc{}.",
            manuscript,
        )
        self.assertIn(
            "Table~\\ref{tab:collapse} now reports grouped-bootstrap 95\\% CIs for target AUROC together with the source-holdout / target pAUC@1\\% pair for every competent setting; even these near-boundary rows remain below the paper's acceptable-ranking threshold on their point estimates.",
            manuscript,
        )
        self.assertNotIn(
            "The same table shows that even the three rows above the 0.60 failure floor remain at or near the 0.5 chance baseline in target pAUC@1\\%, despite stronger source-holdout low-FPR ranking.",
            manuscript,
        )
        self.assertIn(
            "These point-estimate failures are carried by the \\CompetentUnswStressTestSettingCount{} UNSW-related stress-test transfers, which all fail, whereas the \\CompetentIotOnlySettingCount{} IoT-only competent settings split \\CompetentIotOnlyRankingFailureCount{} failures and \\CompetentIotOnlyNonCollapsedCount{} rows above the 0.60 failure floor.",
            manuscript,
        )
        self.assertIn(
            "the paper also reports pAUC@1\\%, defined here as the standardized partial AUROC over FPR~$\\leq$~\\FrozenFprTarget{} under the transferred score orientation",
            manuscript,
        )
        self.assertIn(
            "on this scale, 0.5 is chance-level low-FPR ranking and 1.0 is perfect low-FPR ranking",
            manuscript,
        )
        self.assertIn(
            "the same frozen source-threshold transfer protocol is additionally audited at 0.5\\%, 1\\%, and 5\\% source-validation FPR budgets, transferring each source-selected threshold unchanged to source holdout and target. As one auxiliary unlabeled adaptation check, CORAL is fit using source-train features and a deterministic 50/50 target-test adaptation half split by group, then evaluated on the disjoint target-evaluation half under the same frozen \\FrozenFprTarget{} source-validation threshold rule.",
            manuscript,
        )
        self.assertIn("The low-FPR view is benchmark-wide, not just a survivor check.", manuscript)
        self.assertIn(
            "Across all \\CompetentSettingCount{} competent settings, target pAUC@1\\% ranges from \\CompetentTargetPaucMin{} to \\CompetentTargetPaucMax{} with median \\CompetentTargetPaucMedian{}, while the median source-holdout pAUC@1\\% is \\CompetentSourcePaucMedian{}.",
            manuscript,
        )
        self.assertIn(
            "That target distribution sits effectively on the chance-level 0.5 baseline, far below source-holdout performance.",
            manuscript,
        )
        self.assertIn(
            "This operational picture does not change even where near-boundary target-AUROC intervals approach the 0.60 floor: benign-only recalibration and oracle retuning still fail to recover useful sensitivity at the \\FrozenFprTarget{} budget.",
            manuscript,
        )
        self.assertIn(
            "Across the \\CompetentNonCollapsedCount{} competent settings above the 0.60 failure floor, target pAUC@1\\% ranges only from \\CompetentNonCollapsedTargetPaucMin{} to \\CompetentNonCollapsedTargetPaucMax{}, while the median source-holdout pAUC@1\\% for the same settings is \\CompetentNonCollapsedSourcePaucMedian{}.",
            manuscript,
        )
        self.assertIn("\\input{tables/budget_robustness.tex}", manuscript)
        self.assertIn(
            "Table~\\ref{tab:acsac-budget-robustness} extends that audit across the 0.5\\%/1\\%/5\\% source-side budget sweep.",
            manuscript,
        )
        self.assertIn(
            "Looser source budgets raise competent-set median target TPR from \\BudgetHalfPctCompetentMedianTargetTpr{} to \\BudgetFivePctCompetentMedianTargetTpr{}",
            manuscript,
        )
        self.assertIn(
            "The largest FPR$\\times$ medians are driven by \\BudgetHalfPctCompetentDominantFprxDirection{} at 0.5\\%, \\BudgetOnePctCompetentDominantFprxDirection{} at 1\\%, and \\BudgetFivePctCompetentDominantFprxDirections{} at 5\\%.",
            manuscript,
        )
        self.assertIn(
            "Even at 5\\%, \\BudgetFivePctCompetentCatastrophicCount{} settings remain catastrophic, still \\BudgetFivePctCompetentCatastrophicShare{} of the competent set.",
            manuscript,
        )
        self.assertIn(
            "The deployment-facing conclusion therefore does not reverse across this source-side budget sweep.",
            manuscript,
        )
        self.assertIn("\\input{tables/coral_auxiliary.tex}", manuscript)
        self.assertIn(
            "Table~\\ref{tab:acsac-coral-auxiliary} is a split-matched auxiliary comparison on the same \\CompetentSettingCount{} competent settings and held-out target-evaluation halves.",
            manuscript,
        )
        self.assertIn(
            "In that matched view, CORAL raises median target AUROC but leaves target pAUC@1\\% at \\CoralCompetentMedianTargetPauc{} while worsening median FPR$\\times$ to \\CoralCompetentMedianBudgetMultiple{} and increasing catastrophic settings to \\CoralCompetentCatastrophicCount{}.",
            manuscript,
        )
        self.assertIn(
            "This auxiliary adaptation check therefore supports the paper's diagnosis: improving whole-ROC ranking does not recover low-FPR deployability here.",
            manuscript,
        )
        self.assertIn(
            "On point estimates, \\CompetentInversionCount{} of those failing settings fall strictly below chance, indicating inverted score orientation under the transferred rule; the remaining failing setting is degraded but not inverted.",
            manuscript,
        )
        self.assertIn(
            "Of these failing settings, \\CompetentCollapseCount{} are point-estimate collapse rows. Because no setting lies exactly at the AUROC~=~0.50 boundary, all \\CompetentCollapseCount{} are below-chance inversions under the transferred score orientation. The remaining failing setting is degraded but not inverted.",
            manuscript,
        )
        self.assertIn(
            "Of these point-estimate inversion rows, \\CompetentRobustInversionCount{} still have 95\\% AUROC upper bounds below 0.50.",
            manuscript,
        )
        self.assertIn(
            "A score-space view is consistent with that interpretation: all \\CompetentSourcePositiveMedianGapCount{} competent source-holdout rows keep attack median scores above benign medians, while \\CompetentInversionMedianCrossoverCount{} of the \\CompetentInversionCount{} point-estimate inversion rows show target benign medians meeting or exceeding target attack medians.",
            manuscript,
        )
        self.assertIn(
            "These below-chance rows should not be read as evidence of a deployable sign-flip fix",
            manuscript,
        )
        self.assertIn("source-only deployment setting", manuscript)
        self.assertIn("No setting in this benchmark lies exactly at the AUROC~=~0.50 boundary.", manuscript)
        self.assertNotIn(
            "\\CompetentCollapseCount{} are at or below chance and \\CompetentInversionCount{} fall below chance",
            manuscript,
        )
        self.assertNotIn(
            "\\CompetentInversionCount{} of the \\CompetentRankingFailureCount{} failing competent settings fall strictly below chance",
            manuscript,
        )
        self.assertNotIn(
            "The dominant failure is ranking collapse: in \\CompetentCollapseCount{} of \\CompetentSettingCount{} settings",
            manuscript,
        )
        self.assertNotIn(
            "Even these three survivors remain ``poor'' under the Hosmer--Lemeshow guideline",
            manuscript,
        )
        self.assertIn("Combined low-FPR evidence points to ranking failure, usually via inversion.", manuscript)
        self.assertNotIn("Ranking failure dominates this three-corpus benchmark, usually via inversion.", manuscript)
        self.assertNotIn("Ranking failure dominates, usually via inversion.", manuscript)
        self.assertIn(
            "use the already-defined pAUC@1\\% low-FPR ranking check both in the competent-setting benchmark-wide summary and per setting in Table~\\ref{tab:collapse}",
            manuscript,
        )
        self.assertIn(
            "The \\CompetentRankingFailureCount{}/\\CompetentSettingCount{} figure is therefore a point-estimate summary of the coarse AUROC taxonomy, not the sole basis for the deployment-facing conclusion.",
            manuscript,
        )
        self.assertIn(
            "Under the coarse AUROC taxonomy, that appears as \\CompetentRankingFailureCount{}/\\CompetentSettingCount{} competent settings.",
            manuscript,
        )
        self.assertIn(
            "On point estimates, \\CompetentInversionCount{} of the failing competent settings fall below chance under the transferred score orientation, while the remaining failing setting is degraded but not inverted.",
            manuscript,
        )
        self.assertIn(
            "A score-space view supports that reading: all \\CompetentSourcePositiveMedianGapCount{} competent source-holdout rows keep attack median scores above benign medians, while \\CompetentInversionMedianCrossoverCount{} of the \\CompetentInversionCount{} point-estimate inversion rows show target benign medians meeting or exceeding target attack medians, consistent with score-order collapse or reversal under transfer rather than a label-free repair path.",
            manuscript,
        )
        self.assertIn("using the staged group\\_id as the split and grouped-bootstrap unit", manuscript)
        self.assertIn("IoT-23 uses capture\\_id concatenated with the 5-tuple and protocol", manuscript)
        self.assertIn("ToN-IoT uses the 5-tuple and protocol", manuscript)
        self.assertIn("UNSW-NB15 uses record\\_id", manuscript)
        self.assertIn("all 4 IoT model rows in Tier~2", manuscript)
        self.assertIn("competence-filtered Cat./3 directional summary", manuscript)
        self.assertNotIn(
            "Before deploying a source-trained IDS on a new network, practitioners should evaluate the model's score distribution on a target benign sample.",
            manuscript,
        )
        self.assertNotIn(
            "Large differences in the benign score CDF between source and target may indicate representational shift beyond what threshold adjustment can fix.",
            manuscript,
        )
        self.assertIn(
            "As a debugging heuristic, practitioners may inspect how the benign score distribution shifts between source data and a small target-benign sample, but this paper does not validate benign score-CDF drift as a standalone warning signal.",
            manuscript,
        )
        self.assertIn(
            "A benign-only sample also cannot quantify ranking quality; for that, a small labeled target sample or a domain-adaptation evaluation~\\cite{dinids2021,sun2016return} is needed.",
            manuscript,
        )
        self.assertIn(
            "$\\hat{\\alpha}$ only sees attack-side novelty and is blind to benign shift, and its zero-positive fallback ties the estimate to the transferred detector's own score ranking under shift.",
            manuscript,
        )
        self.assertIn(
            "These point-estimate failures are carried by the \\CompetentUnswStressTestSettingCount{} UNSW-related stress-test transfers, which all fail, whereas the \\CompetentIotOnlySettingCount{} IoT-only competent settings split \\CompetentIotOnlyRankingFailureCount{} failures and \\CompetentIotOnlyNonCollapsedCount{} rows above the 0.60 failure floor.",
            manuscript,
        )
        self.assertIn(
            "Even the three competent settings above the 0.60 failure floor---I23$\\to$TON / LR, I23$\\to$TON / HGB, and TON$\\to$I23 / MLP---remain weak after recalibration",
            manuscript,
        )
        self.assertIn(
            "Benign-only threshold reselection does not rescue detection (mean recalibrated TPR 0.013 across all \\NaturalSettingCount{} settings), and neither a 0.5\\%/1\\%/5\\% source-side budget sweep nor an auxiliary CORAL comparison improves the operational conclusion at the deployment budget.",
            manuscript,
        )
        self.assertIn(
            "When target AUROC survives only in the 0.617--0.636 range, the rows remain above the 0.60 failure floor but their low-FPR discrimination is still too weak to support useful detection at that budget.",
            manuscript,
        )
        self.assertIn(
            "The paired median $\\Delta$TPR is negative in five directions, with the remaining direction at 0.000, and the direction-level retuned TPRs remain low.",
            manuscript,
        )
        self.assertNotIn(
            "The paired median $\\Delta$TPR is non-positive in five of the six directions, and the direction-level retuned TPRs remain low.",
            manuscript,
        )
        self.assertNotIn("three competent non-collapsed settings", manuscript)
        self.assertNotIn("non-collapsed rows", manuscript)
        self.assertNotIn("Panel~A", manuscript)
        self.assertNotIn("Panel~B", manuscript)
        self.assertNotIn("\\label{fig:transfer}", manuscript)
        self.assertIn("Figures~\\ref{fig:transfer-tpr} and~\\ref{fig:transfer-fpr}", manuscript)
        self.assertIn("\\includegraphics[width=\\columnwidth]{figures/transfer_tpr.pdf}", manuscript)
        self.assertIn("\\includegraphics[width=\\columnwidth]{figures/transfer_fpr.pdf}", manuscript)
        self.assertIn(
            "This conclusion is robust to the 0.5\\%/1\\%/5\\% source-side budget sweep: even at 5\\%, competent-set median realized target FPR remains \\BudgetFivePctCompetentMedianBudgetMultiple{}x the selected budget and \\BudgetFivePctCompetentCatastrophicCount{} settings remain catastrophic, still \\BudgetFivePctCompetentCatastrophicShare{} of the competent set.",
            manuscript,
        )
        self.assertIn(
            "A split-matched auxiliary CORAL check raises whole-ROC AUROC but leaves low-FPR ranking unchanged and worsens off-budget/catastrophic behavior, reinforcing rather than weakening the main deployment conclusion.",
            manuscript,
        )
        self.assertNotIn(
            "These results motivate future work on lightweight unsupervised domain adaptation for IDS deployment across networks.",
            manuscript,
        )


if __name__ == "__main__":
    unittest.main()
