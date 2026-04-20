# Progress Log

## 2026-04-13 — Phase 0: Audit Frozen
- Confirmed the repo was effectively a greenfield rebuild with one draft file and no reusable experiment stack.
- Preserved the legacy draft under `paper/legacy_draft.txt`.
- Wrote the rebuild audit and final execution plan to freeze the scientific scope before coding.

## 2026-04-13 — Phase 1: Lean Pilot Scaffold Started
- Created the pilot-first repo skeleton, config files, and artifact directories.
- Started the minimal Python package and CLI surface for feature-schema validation, frozen-threshold transfer, novelty estimation, and provenance checks.
- Deferred broad benchmark and theorem-heavy work until the pilot can run on staged data.

## 2026-04-13 — Phase 1: Lean Pilot Scaffold Verified
- Added a shared experiment runner plus thin CLI entrypoints for pilot, feature build, benchmark, decision evaluation, rendering, and provenance validation.
- Added unit tests for feature construction, deterministic splits, novelty proxy behavior, metric computation, and provenance checks.
- Built a Python 3.9 virtual environment, installed the lean dependency set, and passed both unit tests and lint.
- Confirmed the pilot and feature-build CLIs fail cleanly on missing staged dataset files, which is the current external blocker.

## 2026-04-13 — Phase 1: Datasets Staged and Validated
- Staged `IoT-23` from the cached `srifqi/iot-23-cleaned-data` Kaggle mirror after rejecting the 47 GB full-mirror path as the wrong critical path for pilot work.
- Restaged `ToN-IoT` and `UNSW-NB15` into the canonical raw artifact locations with the shared staged-column contract.
- Verified staged row counts: `IoT-23=1,885,004`, `ToN-IoT=211,043`, `UNSW-NB15=257,673`.
- Ran the common-feature build successfully and wrote all three processed files under `artifacts/processed/feature_schema_v1`.

## 2026-04-13 — Fast-Fail Pilot Executed
- Added a pilot-only coarse ontology for `IoT-23` and `ToN-IoT`, then replaced the empty controlled-shift path with an explicit seen-vs-novel attack construction on `IoT-23`.
- Hardened the pilot runner to emit a pooled hero plot, setting-level monitoring metrics, and a checkpoint artifact with frozen `GO / PIVOT / STOP` logic.
- Ran the capped pilot (`train=100000`, `val=25000`, `test=25000` per dataset split) and wrote `pilot_results.csv`, `pilot_settings.csv`, `pilot_checkpoint.csv`, and `pilot_hero.png`.
- Pilot outcome: `STOP_THEORY_HEAVY_PATH` because pooled oracle novelty vs transfer loss remained weak (`Spearman=0.232`), despite stronger controlled-shift structure (`Spearman=0.615`) and a strong `alpha_hat` monitor on this capped pilot.

## 2026-04-14 — Natural-Transfer Diagnostic Executed
- Generated `natural_transfer_diagnosis.csv` and `natural_transfer_summary.csv` from the existing pilot artifacts without rerunning training.
- Recomputed natural-transfer oracle novelty excluding `other` at the per-setting level and measured the pooled natural-transfer Spearman separately from the mixed pilot checkpoint.
- Result: the natural-transfer pooled Spearman was strong and unchanged under the ex-`other` recomputation (`0.894 -> 0.894`), so the earlier weak pooled pilot checkpoint came from mixing natural and controlled settings rather than an `other`-denominator artifact.

## 2026-04-14 — Pilot Checkpoint Logic V2 Applied
- Refactored the pilot checkpoint to gate natural transfer and controlled shift separately, and added a checkpoint-only recompute script so the summary can be regenerated from existing pilot artifacts without retraining.
- Recomputed `artifacts/pilot/pilot_checkpoint.csv` with `checkpoint_logic_version=separate_gates_v2`; the pilot now adjudicates as `GO` because natural-transfer Spearman is `0.894`, controlled-shift Spearman is `0.615`, and `alpha_hat` still beats at least one unlabeled baseline.

## 2026-04-14 — Full Benchmark Pipeline Hardened
- Renamed the benchmark raw-output artifact to `predictions.csv`, kept `settings.csv` as the canonical one-row-per-setting artifact, and added derived `within_domain_summary.csv` and `cross_domain_summary.csv` outputs from benchmark settings.
- Extended benchmark setting rows with threshold, source/target coverage fields, and `oracle_alpha_usable` / `oracle_alpha_scope` so low-coverage `UNSW-NB15` directions remain in the transfer matrix but are excluded from claim-bearing oracle-novelty analysis.
- Added source-fit reuse so each `(source_dataset, model)` is trained once per benchmark run instead of once per target direction.
- Recomputed the pilot checkpoint from existing artifacts only and confirmed it remains `GO`.
- Ran the capped six-direction benchmark smoke config successfully: `24` settings rows, `12` within-domain rows, `24` cross-domain rows, all four required models, all six directions, and all `UNSW-NB15` directions flagged `oracle_alpha_usable = 0`.
- Attempted the canonical full-data benchmark on `configs/benchmark.yaml`, but it did not finish within the current turn and produced no benchmark artifacts; the remaining blocker is full-data model runtime and numerical instability in the dense model path rather than the benchmark artifact pipeline itself.

## 2026-04-14 — Canonical Benchmark Stabilized and Completed
- Added a source-fitted sanitizer and bounded scaler so dense models and drift metrics consume the same clipped feature space, with `bytes_per_pkt` log-compressed and standardized continuous features clipped after source-only fitting.
- Replaced the unstable `mlp` optimizer path with a smaller `lbfgs` configuration, scoped warning suppression to known `sklearn.utils.extmath` runtime chatter, and verified the full-data `iot23` source `mlp` fit now completes in about `14s` with finite scores and no runtime-warning flood.
- Fixed the benchmark runner so it flushes partial `predictions/settings/summary` artifacts during long runs instead of writing only at the end.
- Completed the canonical full-data `6 directions × 4 models` benchmark and wrote `artifacts/benchmark/predictions.csv`, `settings.csv`, `within_domain_summary.csv`, and `cross_domain_summary.csv` with the expected `24 / 12 / 24` row counts.
- Ran full-data decision evaluation on the canonical benchmark settings; current monitor ranking by AUROC is `proxy_a_distance (0.752) > avg_wasserstein_1d (0.685) > alpha_hat (0.546) > source_validation_tpr_at_1pct_fpr (0.391)`.
- Regenerated paper-facing artifact tables from canonical benchmark outputs and passed provenance validation against the refreshed artifact manifest and claims file.

## 2026-04-14 — Full-Data Claim Package Frozen
- Added an artifact-only claim-package builder that recomputes deployable-monitor comparisons directly from canonical `artifacts/benchmark/settings.csv`, with two fixed subsets: `all_natural` and `primary_iot`.
- Wrote `full_data_monitor_comparison.csv`, `full_data_primary_oracle_context.csv`, `full_data_claim_decision.json`, and `full_data_claim_summary.md` under `artifacts/claims`.
- Frozen decision: `alpha_hat` does not support a deployable main claim on the full-data benchmark and remains a main-text negative result; `proxy_a_distance` stays a comparator rather than a replacement contribution.
- Refreshed the render/provenance path so the new claim-package artifacts are included in `results_manifest.csv` and `claims.csv`, then reran provenance validation successfully.

## 2026-04-14 — Standalone Fallback Manuscript Rebuilt
- Added a manuscript-asset builder that reads only the canonical benchmark and claim-package artifacts, then writes the paper tables, figures, and generated numeric macros under `paper/`.
- Rewrote `paper/manuscript.tex` from scratch around the fallback evidence: fixed-threshold transfer benchmark as the main result, oracle novelty limited to the claim-bearing IoT subset, and `alpha_hat` reported as a negative main-text result.
- Integrated the new paper assets into `results_manifest.csv` and `claims.csv`, reran provenance validation successfully, and compiled the standalone manuscript PDF with no appendix or supplement dependency.

## 2026-04-14 — Internet of Things Submission Port and Editorial Tightening
- Ported the standalone manuscript into Elsevier's single-column `elsarticle` preprint format for *Internet of Things*, moved references into `paper/references.bib`, and kept the frozen evidence spine unchanged.
- Tightened the abstract, section transitions, captions, and limitations wording; removed remaining macro-spacing joins in the compiled PDF; and kept the negative `alpha_hat` result explicit without promoting stronger baselines into new contributions.
- Regenerated the main-paper tables and figures from canonical artifacts only, improved in-paper readability with grouped tables and larger figure text, registered `paper/manuscript.pdf` in provenance, and revalidated the final submission package successfully.

## 2026-04-14 — Final Pre-Submission Revision Applied
- Removed the remaining self-referential framing from the manuscript introduction, discussion, and conclusion so the paper now reads as a standalone benchmark study rather than a draft-history narrative.
- Added implementation-faithful definitions for oracle novelty, `\hat{\alpha}`, proxy-A distance, average one-dimensional Wasserstein distance, and the unsafe-setting rule directly in the Study Design section.
- Recompiled the Elsevier manuscript, reran the manuscript-body grep checks and PDF macro-join checks successfully, and confirmed that the cross-domain table, oracle-context table, and the three main figures remain readable in the compiled paper.

## 2026-04-14 — Operational-Safety Revision Completed
- Reran the canonical full benchmark with richer predictions, explicit `eval_role` tagging, grouped-bootstrap confidence intervals, and derived operational metrics including relative TPR drop, FPR budget multiple, operational severity, and catastrophic operational failure.
- Rebuilt the claim package and manuscript assets around operational severity and catastrophic failure rather than the legacy TPR-only unsafe flag, then regenerated the paper tables, figures, numeric macros, and compiled manuscript PDF from the updated artifacts.
- The revised operational target changed the monitor story: `\hat{\alpha}` remained part of the audited monitor comparison under the new target, but its full-benchmark catastrophic AUROC and severity association still remained too weak for a deployable claim, so it stays out of the contribution set and remains a negative deployable result.

## 2026-04-14 — ACSAC Submission Package Added
- Added a second, anonymous ACSAC manuscript track that reads from the same frozen benchmark and claim artifacts as the journal manuscript, without reopening experiments or changing the evidence spine.
- Extended the manuscript asset builder to emit ACSAC-specific compressed tables and figures, including a direction-level transfer summary table, a compact IoT-only oracle-direction table, and an all-natural monitor table.
- Compiled the anonymous IEEE-style ACSAC manuscript successfully, refreshed the artifact manifest and claims manifest to include the new conference assets and PDF, and revalidated provenance successfully.

## 2026-04-14 — ACSAC Dense Editorial Pass Completed
- Expanded the anonymous ACSAC manuscript from a short note into a six-page conference paper with a real Related Work section, fuller benchmark design, stronger benchmark interpretation, and an explicit Limitations section, while keeping the frozen evidence and asset set unchanged.
- Kept the benchmark-first story intact, preserved the bounded IoT-only oracle novelty reading, and retained `\hat{\alpha}` as a negative result under the operational-severity target rather than a deployable contribution.
- Rebuilt manuscript assets from canonical artifacts, reran provenance registration and validation, and rechecked the ACSAC PDF for anonymity, removed storylines, and in-page readability.

## 2026-04-14 — ACSAC Full-Paper Recovery and Submission Freeze
- Locked the supported build path so the canonical manuscript PDFs are only `paper/manuscript.pdf` for the journal track and `paper/acsac/manuscript.pdf` for the ACSAC track, with root-level `manuscript.*` outputs treated as invalid leftovers and removed by the build workflow.
- Expanded the ACSAC manuscript further around the same frozen asset set: clearer benchmark motivation, stronger protocol boundaries, more explicit directional interpretation in Results, and a sharper discussion of what the benchmark changes for deployment-style IDS evaluation.
- Rebuilt the ACSAC package through the guarded `make acsac` path, confirmed that the stale mixed root PDF no longer survives the workflow, and kept the conference PDF anonymous and provenance-clean.

## 2026-04-14 — ACSAC Frozen-Prediction Diagnostic Pass
- Added an artifact-only ACSAC diagnostic layer derived from canonical `predictions.csv` and `settings.csv`, producing a target-retuned oracle comparison and a three-budget source-threshold sensitivity sweep without rerunning the benchmark.
- Extended the manuscript asset builder to consume those diagnostics, emit a compact ACSAC retuned-threshold table, and generate new prose macros for the diagnostic and budget-robustness paragraphs.
- Rewrote the ACSAC manuscript around the stronger interpretation: explicit RQ framing, plain-English lead-in sentences before the formula blocks, a target-retuned diagnostic result showing that enforcing the target-side 1\% budget reduces median TPR further, and a short budget-sensitivity paragraph that keeps the benchmark-first story intact.
