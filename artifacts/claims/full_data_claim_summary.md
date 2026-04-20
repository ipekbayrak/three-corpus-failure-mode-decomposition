# Full-Data Claim Summary

## Inputs
- benchmark settings: `artifacts/benchmark/settings.csv`
- within-domain summary: `artifacts/benchmark/within_domain_summary.csv`
- cross-domain summary: `artifacts/benchmark/cross_domain_summary.csv`
- decision audit artifact: `artifacts/decision/decision_eval.csv` (audit only; not used for claim adjudication)

## Canonical Benchmark Facts
- natural-transfer settings: 24
- primary oracle-novelty settings: 8
- within-domain summary rows: 12
- cross-domain summary rows: 24
- primary oracle context rows: 8

## Deployable Monitor Comparison
### all_natural
| score_name | spearman_vs_operational_severity | catastrophic_auroc | catastrophic_auprc |
|---|---:|---:|---:|
| alpha_hat | -0.184 | 0.403 | 0.696 |
| proxy_a_distance | -0.228 | 0.349 | 0.665 |
| avg_wasserstein_1d | -0.176 | 0.382 | 0.675 |
| risk_from_source_validation_tpr | -0.067 | 0.492 | 0.787 |

### primary_iot
| score_name | spearman_vs_operational_severity | catastrophic_auroc | catastrophic_auprc |
|---|---:|---:|---:|
| alpha_hat | -0.078 | 0.500 | 0.800 |
| proxy_a_distance | 0.071 | 0.500 | 0.750 |
| avg_wasserstein_1d | 0.071 | 0.500 | 0.750 |
| risk_from_source_validation_tpr | -0.296 | 0.333 | 0.758 |

- On `all_natural`, `alpha_hat` ranks 2/2/3 across catastrophic AUROC / catastrophic AUPRC / severity association, so it remains non-dominant as well as non-deployable on the full benchmark.
- The headline monitor target is operational severity, with catastrophic operational failure used as a secondary binary outcome.
- The strongest audited full-benchmark comparator is `risk_from_source_validation_tpr`, but it also remains too weak in absolute terms for a deployable monitoring claim.
- The strongest generic unlabeled comparators remain `avg_wasserstein_1d` on `all_natural` and `proxy_a_distance|avg_wasserstein_1d` on `primary_iot`, but none of them changes the paper's center.

## Primary Oracle Context
- The primary oracle context is limited to the IoT-only claim-bearing rows where `oracle_alpha_scope == "primary"`.
- These rows are emitted in `full_data_primary_oracle_context.csv` for direct manuscript use.

## Claim Decision
- deployable `alpha_hat` main claim: `False`
- placement: `main_negative_result`
- proxy-A promoted to main claim: `False`
- reason: alpha_hat did not outperform the strongest audited comparator on the full benchmark and its full-benchmark catastrophic AUROC and severity association remained too weak for a deployable claim.

## Manuscript Instruction
- Keep `alpha_hat` out of the contribution list and describe it as a weak monitor result rather than a deployable win.
- Remove any main-contribution wording that implies `alpha_hat` is operationally reliable on full data.
- State that `alpha_hat` remained non-deployable on full data and did not beat the strongest audited comparator on the full benchmark.
- Do not replace the paper thesis with `proxy_a_distance`; keep it as a stronger comparator, not a new contribution.
