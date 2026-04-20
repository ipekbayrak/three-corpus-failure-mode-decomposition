# Evaluation Protocol

## Primary Metric
- `TPR@1% FPR`

## Secondary Metrics
- `AUROC`
- `AUPRC`

## Thresholding
- Choose the operating threshold on source validation negatives only.
- Predictions are defined by `score >= threshold`.
- Use the tie-safe threshold rule: pick the smallest cutoff whose realized FPR under that prediction rule does not exceed the budget.
- If a tied block of negative scores straddles the budget, move the cutoff strictly above that tied score so those tied negatives remain negative.
- Freeze the threshold.
- Transfer unchanged to target data.
- No target retuning.

## Splits
- Deterministic seeds and manifests are required.
- Use group-aware splitting where groups approximate captures or sessions.
- Target groups must not appear in source training splits when the design requires disjointness.

## Inference
- No cross-domain paired significance tests on unmatched units.
- Prefer confidence intervals and effect sizes over p-value clutter.
- For group-dependent settings, uncertainty must respect grouping rather than treat overlapping windows as i.i.d.

## Deployable vs Oracle
- Deployable analyses may not use target labels at transfer time.
- Oracle analyses must be labeled as oracle-only.

## Recalibration Baseline
- The benign-only recalibration baseline may use target benign traffic only for threshold selection.
- Split target benign traffic deterministically into disjoint calibration and evaluation partitions, using `group_id` when available.
- Select the recalibrated threshold on the benign calibration split only.
- Report realized recalibrated FPR on the held-out benign evaluation split only.
- Target attack labels may be used only to evaluate realized recalibrated TPR after the threshold has already been fixed.
