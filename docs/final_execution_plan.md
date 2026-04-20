# Final Execution Plan

## A. Working Paper Thesis

This paper tests a narrow empirical claim in shared-feature tabular IDS transfer: when a source-trained IDS is deployed cross-domain with its threshold fixed on source validation only, transfer failure at `TPR@1% FPR` is strongly structured by attack novelty and overlap. The scientific quantity is an oracle novelty measure `alpha` defined with target labels and a pre-registered attack ontology. The deployable question is whether an unlabeled proxy `alpha_hat` can identify unsafe transfer settings without target labels. The paper depends on whether novelty explains degradation cleanly, whether a frozen deployable proxy adds decision value, and whether a controlled-shift experiment lands on the same relationship as the natural cross-dataset transfers.

## B. Fast-Fail Pilot Protocol

### Purpose
- Test common-feature viability.
- Test whether oracle `alpha` visibly tracks transfer degradation.
- Test whether frozen `alpha_hat` shows usable no-label signal.
- Test whether controlled shift reinforces the same relationship.

### Pilot Scope
- Datasets:
  - `IoT-23`
  - `ToN-IoT`
  - `UNSW-NB15` as stress-test only
- Models:
  - `LogisticRegression`
  - `HistGradientBoostingClassifier`
- Natural transfer directions:
  - `IoT-23 -> ToN-IoT`
  - `ToN-IoT -> IoT-23`
  - `IoT-23 -> UNSW-NB15`
- Controlled-shift points:
  - `IoT-23` family or ontology novelty sweep on a fixed source split
- Primary metric:
  - `TPR@1% FPR`
- Secondary:
  - `AUROC`
  - `AUPRC`
- No calibration theme.
- No Theorem 3 work in the pilot.

### Pre-Registered Common Feature Schema `feature_schema_v1`
Primitive fields that must exist in all three datasets:
- `duration_s`
- `src_bytes`
- `dst_bytes`
- `src_pkts`
- `dst_pkts`
- `proto`

Frozen derived features:
- `log_duration_s`
- `log_src_bytes`
- `log_dst_bytes`
- `log_src_pkts`
- `log_dst_pkts`
- `log_total_bytes`
- `log_total_pkts`
- `log_bytes_per_s`
- `log_pkts_per_s`
- `src_byte_frac`
- `src_pkt_frac`
- `bytes_per_pkt`
- `proto_tcp`
- `proto_udp`
- `proto_icmp`
- `proto_other`

If any dataset cannot support these primitives without dataset-specific hacks, stop and narrow the project immediately.

### Hero Figure
- Pooled, normalized scatter.
- X-axis: oracle novelty `alpha`.
- Y-axis: relative transfer loss:
  - `loss_rel = 1 - TPR_target@1% / max(TPR_source_holdout@1%, eps)`
- Point types:
  - natural transfer settings
  - controlled-shift settings
- Encoding:
  - color = natural vs controlled shift
  - marker = model
- Appendix:
  - per-model versions if needed

Why pooled normalized:
- pooled provides enough points to see structure,
- normalized removes source and model scale effects,
- per-model alone is too small for the hero claim.

### Pilot Go / Pivot / Stop
- GO:
  - common schema is viable on all three datasets,
  - at least one pilot model has non-trivial within-domain sanity on `IoT-23` and `ToN-IoT`,
  - oracle `alpha` and normalized transfer loss form a clear monotone relationship,
  - controlled-shift points lie on the same trend rather than a separate cloud,
  - `alpha_hat` shows at least one credible deployment benefit relative to generic unlabeled baselines.
- PIVOT TO FALLBACK:
  - oracle `alpha` relation is strong,
  - `alpha_hat` is weak, noisy, or adds no decision value,
  - the paper becomes an oracle-novelty transfer paper with deployment caution.
- STOP THEORY-HEAVY PATH:
  - common feature intersection is not viable,
  - or oracle `alpha` does not explain degradation cleanly after leakage checks,
  - then drop novelty-theorem ambitions and reduce or stop.

## C. Minimal Architecture Overview

### Lean Stack
- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- PyYAML
- no notebooks as source of truth
- no custom deep sequence models
- no heavy orchestration

### Core Execution Path
1. load raw staged dataset
2. map to `feature_schema_v1`
3. build deterministic splits and manifests
4. train source model
5. lock source threshold at `1% FPR` using source validation only
6. evaluate transfer
7. compute oracle `alpha`
8. compute frozen deployable `alpha_hat`
9. compare against generic unlabeled baselines
10. render figures and tables from artifacts

### Data Contracts
- `datasets.yaml`
  - dataset name, version, raw root, required files, loader name
- `split_manifest.csv`
  - `dataset, split_id, fold_role, group_id, family_label_raw, family_label_mapped, binary_label, source_allowed`
- `feature_manifest.csv`
  - `dataset, record_id, group_id, feature_schema_version, ...features...`
- `ontology_mapping.csv`
  - `dataset, raw_label, mapped_group, mapping_source, mapping_confidence, notes`
- `oracle_alpha_inputs.csv`
  - `source_dataset, target_dataset, target_group, target_attack_mass, seen_in_source`
- `alpha_hat_inputs.csv`
  - `run_id, target_record_id, source_score, predicted_positive_flag, selected_by_fallback_flag, knn_distance, outlier_flag`
- `run_manifest.json`
  - config hash, code version, dataset versions, schema version
- `results_manifest.csv`
  - figure or table id to artifact path mapping
- `claims.csv`
  - manuscript claim id to artifact row or column reference

### Frozen Default `alpha_hat`
- Train the source IDS.
- Lock threshold at `1% FPR` using source validation only.
- Collect source attack examples in the common feature space.
- Fit no extra learned novelty model.
- Use `k`-NN distance to the source-attack bank as the attack-novelty score.
- On target:
  - take predicted positives under the frozen source threshold,
  - if zero positives, use the top-`K` target instances by source model score,
  - `K` is fixed from the source-side positive prediction rate on source validation.
- Define primary `alpha_hat` as outlier rate among selected target instances, where outlier = `k`-NN distance above the source-validation positive `95th` percentile.
- Fixed defaults:
  - `k_nn = 10`
  - percentile = `95`
  - `K = ceil(q_src_val_pos * n_target)`

Why this replacement:
- fewer assumptions than one-class modeling,
- easier to audit,
- no estimator shopping,
- built from sklearn primitives only.

## D. Milestone Plan in Dependency Order

1. Freeze the scientific spec:
   - feature schema
   - ontology rules
   - `alpha`
   - `alpha_hat`
   - pilot matrix
2. Run the fast-fail pilot.
3. Checkpoint:
   - GO / PIVOT / STOP
4. If GO or PIVOT: build the minimum full benchmark:
   - 3 datasets
   - 6 ordered transfer directions
   - 4 models
5. Build the decision-task evaluation.
6. Only then consider Theorem 3 or certificate work:
   - best 1–2 models only
   - drop immediately if weak
7. Only after evidence stabilizes:
   - paper figures and tables
   - manuscript rewrite
   - supplement and provenance checks

## E. Exact Experiment Matrix

| ID | Experiment | Purpose | Oracle / Deployable | Models | Primary Metric | Main Output | De-Scope Trigger |
|---|---|---|---|---|---|---|---|
| E0 | Common-feature viability | Verify `feature_schema_v1` is real | Neither | N/A | extraction success + sanity stats | feature viability table | any dataset needs custom feature branch |
| E1 | Within-domain sanity on all 3 datasets | Confirm source models are meaningful before transfer | Deployable | LR, ExtraTrees, HGB, MLP | `TPR@1% FPR` | within-domain baseline table | common schema yields trivial performance |
| E2 | Natural cross-domain transfers | Core benchmark | Deployable | LR, ExtraTrees, HGB, MLP | `TPR@1% FPR` | transfer summary table | leakage or split ambiguity |
| E3 | Oracle novelty vs transfer loss | Test Theorem 2 center claim | Oracle | same as E2 | correlation vs `loss_rel` | hero figure | no visible monotone structure |
| E4 | Observed vs predicted degradation | Test whether novelty predicts transfer behavior | Oracle + Deployable | same as E2 | `R^2`, rank correlation | predicted vs observed plot | predictions collapse to baselines |
| E5 | `alpha_hat` vs unlabeled shift baselines | Test no-label monitoring value | Deployable | same as E2 | unsafe-deployment AUROC/AUPRC | baseline comparison table | no incremental value |
| E6 | Deploy / abstain decision task | Convert `alpha_hat` into deployment value | Deployable | same as E2 | catastrophic failures avoided, regret, abstention burden | decision figure or table | no better than generic shift baseline |
| E7 | Controlled novelty sweep on IoT-23 | Reinforce hero relationship | Oracle + Deployable | LR, HGB | `loss_rel` relationship | hero points + optional side figure | points do not align with natural-transfer trend |
| E8 | Theorem 3 / certificate | Keep only if non-vacuous and additive | Deployable if useful, otherwise appendix | best 1–2 | non-vacuity + added decision value | optional appendix or main figure | loose, decorative, or redundant |

## F. Theory-to-Experiment Mapping

- Theorem 1: transfer decomposition
  - maps to E2 + E4
  - supporting structure only
- Theorem 2: Price of Novelty
  - maps to E3 + E7
  - center of the paper
- Theorem 3: deployment-style certificate
  - maps to E8 only
  - keep only if empirically useful
  - minimum viable paper does not require it

## G. Alpha vs Alpha_Hat Specification

### Oracle `alpha`
- scientific quantity
- label-using
- primary definition:
  - fraction of target attack mass belonging to ontology groups unseen in the source training attack set
- ontology:
  - small, hand-audited, coarse mapping
  - ambiguous raw labels go to `other`
  - `other` share must be reported
- natural transfer:
  - use mapped cross-dataset ontology
- controlled shift:
  - construct held-out groups so resulting points express the same novelty quantity

### Deployable `alpha_hat`
- no target labels
- frozen after the pilot
- primary estimator:
  - source-attack-bank `k`-NN outlier rate among target instances flagged by the frozen source IDS
- zero-positive fallback:
  - top-`K` target instances by source score
  - `K` fixed from source validation only
- no estimator shopping after pilot freeze

### Baselines for `alpha_hat`
- source validation performance
- oracle attack-family or ontology overlap percentage
- proxy `A`-distance from source-vs-target domain classifier
- simple distribution distance:
  - default = average 1D Wasserstein across standardized features
- optional appendix only:
  - score-distribution drift

### Success Criterion for `alpha_hat`
`alpha_hat` does not need to win everywhere. It survives if it shows clear incremental value in at least one of:
- predicting target degradation,
- stability across settings,
- improving no-target-label decisions.

If it shows none of these, the deployable claim is dropped.

## H. Decision-Task Specification

### Primary Decision Task
- deploy / abstain
- unit = a candidate source-model deployment on a target dataset or controlled-shift setting
- decision input = `alpha_hat` or baseline monitor, with no target labels

### Unsafe Target Definition
- primary:
  - `loss_rel > 0.5`
- sensitivity appendix:
  - absolute `TPR@1% FPR < 0.4`

### Decision Metrics
- unsafe deployments prevented
- catastrophic failures avoided
- abstention burden / coverage
- regret vs oracle deploy policy
- AUROC/AUPRC for unsafe-setting detection

## I. Controlled-Shift Specification

- Primary controlled shift:
  - `IoT-23` only
  - fixed source split
  - construct target mixtures with increasing withheld family or ontology novelty mass
  - no threshold retuning
  - use the same `loss_rel` definition as natural transfers
- Role:
  - not a side figure
  - controlled-shift points feed directly into the hero figure
- Priority:
  - higher than Theorem 3
  - if this experiment is messy, cut Theorem 3 first

## J. Negative-Result Fallback Paper

If `alpha_hat` fails but oracle `alpha` works:
- the paper becomes:
  - a fixed-threshold transfer benchmark across `IoT-23`, `ToN-IoT`, and `UNSW-NB15`,
  - an oracle novelty-overlap analysis explaining transfer degradation,
  - a cautionary negative result on unlabeled transfer-risk estimation.
- main contributions:
  1. reproducible cross-domain low-FPR IDS benchmark,
  2. evidence that oracle novelty explains much of transfer failure,
  3. negative result on deployable no-label transfer risk estimation.

This is the honest strong-Q2 fallback.

Minimum viable paper if Theorem 3 is dropped:
- Theorem 1 supportive
- Theorem 2 central
- no certificate in the main text

What would still be only strong Q2:
- clean benchmark and good hygiene,
- oracle novelty relation only moderate or dataset-specific,
- `alpha_hat` weak or inconsistent,
- hero figure fragmented by model or domain,
- controlled shift supportive but not decisive.

What would make this a credible Q1 shot:
- hero figure shows a clean monotone novelty-loss structure across natural and controlled points,
- `alpha_hat` beats or materially complements generic unlabeled baselines on the decision task,
- relationship holds across at least two model classes,
- no target retuning, no leakage, no overclaiming.

## K. Risks / Likely Failure Modes

- common feature intersection is too weak or semantically inconsistent,
- ontology mapping is noisy or too subjective,
- attack labels do not map cleanly across datasets,
- within-domain performance under the common schema is too weak,
- `alpha_hat` depends too much on source attack sample size,
- zero-positive fallback becomes dominant,
- controlled-shift construction does not line up with natural transfer,
- theorem work becomes decorative,
- manuscript drifts toward infrastructure or theorem count rather than the hero relation.

## L. De-Scoping Rules

Cut immediately if needed, in this order:
1. Theorem 3 / certification
2. extra baselines beyond the required set
3. extra decision tasks beyond deploy / abstain
4. controlled shift on more than one dataset
5. per-model main-text figures
6. appendix-only `TPR@0.5% FPR`

Do not cut:
- fixed `1% FPR` source-only thresholding,
- 6 ordered transfer directions,
- oracle `alpha`,
- frozen `alpha_hat`,
- decision task,
- leakage checks,
- provenance.

## M. Done When Criteria

The project is done when:
- the pilot produced an explicit GO / PIVOT / STOP decision,
- all 6 ordered transfers are run for the 4 required models,
- all tables and figures are script-generated from artifacts,
- the hero figure is final and defensible,
- `alpha` and `alpha_hat` are clearly separated,
- the decision-task evaluation is complete,
- controlled shift either reinforces the hero relation or the paper is reframed honestly,
- Theorem 3 is either justified or demoted,
- every manuscript number is traceable to an artifact,
- final claims stay inside the shared-feature tabular IDS setting actually tested.
