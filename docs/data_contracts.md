# Data Contracts

## Raw Dataset Contract
Each dataset is staged as a tabular file declared in config. The file may use dataset-specific raw columns, but config must map them into the primitive fields required by `feature_schema_v1`.

Required primitive fields after mapping:
- `record_id`
- `group_id`
- `label`
- `family_raw`
- `proto`
- `duration_s`
- `src_bytes`
- `dst_bytes`
- `src_pkts`
- `dst_pkts`

Optional:
- `timestamp`

## Processed Feature Contract
Processed feature files are written under `artifacts/processed/feature_schema_v1/` as CSV files.

Required columns:
- `dataset`
- `record_id`
- `group_id`
- `label`
- `family_raw`
- `family_group`
- `feature_schema_version`
- every feature in `feature_schema_v1`

## Benchmark Predictions Contract
File: `artifacts/benchmark/predictions.csv`

Columns:
- `dataset`
- `record_id`
- `group_id`
- `label`
- `family_group`
- `score`
- `setting_id`
- `source_dataset`
- `target_dataset`
- `model`
- `threshold`
- `run_id`

## Benchmark Settings Contract
File: `artifacts/benchmark/settings.csv`

One row per natural-transfer setting.

Required columns:
- `setting_id`
- `setting_type`
- `source_dataset`
- `target_dataset`
- `model`
- `threshold`
- `source_holdout_tpr_at_1pct_fpr`
- `source_validation_tpr_at_1pct_fpr`
- `target_tpr_at_1pct_fpr`
- `target_realized_fpr`
- `target_auroc`
- `target_auprc`
- `loss_rel`
- `oracle_alpha`
- `oracle_alpha_usable`
- `oracle_alpha_scope`
- `alpha_hat`
- `proxy_a_distance`
- `avg_wasserstein_1d`
- `source_mapped_attack_coverage`
- `source_other_attack_share`
- `target_mapped_attack_coverage`
- `target_other_attack_share`

## Benchmark Summaries Contract
Files:
- `artifacts/benchmark/within_domain_summary.csv`
- `artifacts/benchmark/cross_domain_summary.csv`

`within_domain_summary.csv` columns:
- `dataset`
- `model`
- `source_holdout_tpr_at_1pct_fpr`
- `source_validation_tpr_at_1pct_fpr`

`cross_domain_summary.csv` contains one row per benchmark setting and is a claim-ready subset of `settings.csv`.

## Split Manifest Contract
File: `artifacts/manifests/split_manifest.csv`

Columns:
- `dataset`
- `split_id`
- `fold_role`
- `record_id`
- `group_id`
- `label`
- `family_raw`
- `family_group`

## Oracle Alpha Inputs
File: `artifacts/manifests/oracle_alpha_inputs.csv`

Columns:
- `source_dataset`
- `target_dataset`
- `target_group`
- `target_attack_mass`
- `seen_in_source`

## Deployable Alpha_Hat Inputs
File: `artifacts/manifests/alpha_hat_inputs.csv`

Columns:
- `run_id`
- `target_record_id`
- `source_score`
- `predicted_positive_flag`
- `selected_by_fallback_flag`
- `knn_distance`
- `outlier_flag`

## Run Metadata
File: `artifacts/manifests/run_manifest.json`

Keys:
- `run_id`
- `config_hash`
- `code_version`
- `dataset_versions`
- `feature_schema_version`
- `seed`

## Result Provenance
File: `artifacts/manifests/results_manifest.csv`

Columns:
- `artifact_id`
- `artifact_type`
- `path`
- `producer_script`
- `run_id`

File: `artifacts/claims/claims.csv`

Columns:
- `claim_id`
- `artifact_id`
- `source_path`
- `locator`
- `description`
