from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SUBSET_ORDER = ["all_natural", "primary_iot"]
SCORE_ORDER = [
    "alpha_hat",
    "proxy_a_distance",
    "avg_wasserstein_1d",
    "risk_from_source_validation_tpr",
]
GENERIC_UNLABELED_BASELINES = ["proxy_a_distance", "avg_wasserstein_1d"]
PRIMARY_METRIC_COLUMNS = [
    "catastrophic_auroc",
    "catastrophic_auprc",
    "spearman_vs_operational_severity",
]
MIN_DEPLOYABLE_ALL_NATURAL_CATASTROPHIC_AUROC = 0.65
MIN_DEPLOYABLE_ALL_NATURAL_SEVERITY_SPEARMAN = 0.10

SCORE_SPECS = [
    {
        "score_name": "alpha_hat",
        "source_column": "alpha_hat",
        "risk_orientation": "native_higher_is_more_unsafe",
        "negate": False,
    },
    {
        "score_name": "proxy_a_distance",
        "source_column": "proxy_a_distance",
        "risk_orientation": "native_higher_is_more_unsafe",
        "negate": False,
    },
    {
        "score_name": "avg_wasserstein_1d",
        "source_column": "avg_wasserstein_1d",
        "risk_orientation": "native_higher_is_more_unsafe",
        "negate": False,
    },
    {
        "score_name": "risk_from_source_validation_tpr",
        "source_column": "source_validation_tpr_at_1pct_fpr",
        "risk_orientation": "negated_source_validation_tpr_higher_is_more_unsafe",
        "negate": True,
    },
]


@dataclass(frozen=True)
class ClaimPackageArtifacts:
    monitor_comparison_csv: Path
    primary_oracle_context_csv: Path
    claim_decision_json: Path
    claim_summary_md: Path


def _safe_spearman(values: pd.Series, target: pd.Series) -> float:
    if values.nunique(dropna=True) < 2 or target.nunique(dropna=True) < 2:
        return float("nan")
    return float(values.corr(target, method="spearman"))


def _safe_unsafe_metrics(values: pd.Series, unsafe: pd.Series) -> tuple[float, float]:
    if unsafe.nunique(dropna=True) < 2 or values.nunique(dropna=True) < 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(unsafe, values)),
        float(average_precision_score(unsafe, values)),
    )


def _subset_frames(settings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    natural = settings.loc[settings["setting_type"] == "natural_transfer"].copy()
    primary = natural.loc[natural["oracle_alpha_scope"] == "primary"].copy()
    return {
        "all_natural": natural.reset_index(drop=True),
        "primary_iot": primary.reset_index(drop=True),
    }


def build_monitor_comparison(settings: pd.DataFrame) -> pd.DataFrame:
    subset_frames = _subset_frames(settings)
    rows: list[dict[str, Any]] = []

    for subset_name in SUBSET_ORDER:
        subset = subset_frames[subset_name]
        operational_severity = subset["operational_severity"].astype(float)
        catastrophic = subset["catastrophic_operational_failure"].astype(int)
        loss_rel = subset["loss_rel"].astype(float)
        unsafe = subset["tpr_loss_unsafe"].astype(int)
        for spec in SCORE_SPECS:
            raw = subset[spec["source_column"]].astype(float)
            values = -raw if spec["negate"] else raw
            severity_spearman = _safe_spearman(values, operational_severity)
            catastrophic_auroc, catastrophic_auprc = _safe_unsafe_metrics(values, catastrophic)
            loss_spearman = _safe_spearman(values, loss_rel)
            unsafe_auroc, unsafe_auprc = _safe_unsafe_metrics(values, unsafe)
            rows.append(
                {
                    "subset": subset_name,
                    "n_settings": len(subset),
                    "n_catastrophic": int(catastrophic.sum()),
                    "n_unsafe": int(unsafe.sum()),
                    "score_name": spec["score_name"],
                    "risk_orientation": spec["risk_orientation"],
                    "spearman_vs_operational_severity": severity_spearman,
                    "catastrophic_auroc": catastrophic_auroc,
                    "catastrophic_auprc": catastrophic_auprc,
                    "unsafe_auroc": unsafe_auroc,
                    "unsafe_auprc": unsafe_auprc,
                    "spearman_vs_loss_rel": loss_spearman,
                }
            )

    comparison = pd.DataFrame(rows)
    for subset_name in SUBSET_ORDER:
        mask = comparison["subset"] == subset_name
        for metric_column, rank_column in [
            ("catastrophic_auroc", "rank_catastrophic_auroc"),
            ("catastrophic_auprc", "rank_catastrophic_auprc"),
            ("spearman_vs_operational_severity", "rank_spearman_vs_operational_severity"),
        ]:
            comparison.loc[mask, rank_column] = (
                comparison.loc[mask, metric_column]
                .rank(method="min", ascending=False, na_option="bottom")
                .astype("Int64")
            )

        alpha_row = comparison.loc[
            mask & (comparison["score_name"] == "alpha_hat"),
            [
                "catastrophic_auroc",
                "catastrophic_auprc",
                "spearman_vs_operational_severity",
            ],
        ].iloc[0]
        for metric_column, beats_column in [
            ("catastrophic_auroc", "beats_alpha_hat_on_catastrophic_auroc"),
            ("catastrophic_auprc", "beats_alpha_hat_on_catastrophic_auprc"),
            (
                "spearman_vs_operational_severity",
                "beats_alpha_hat_on_operational_severity_spearman",
            ),
        ]:
            alpha_value = float(alpha_row[metric_column])
            comparison.loc[mask, beats_column] = (
                comparison.loc[mask, metric_column].astype(float) > alpha_value
            ).astype(int)

    for beats_column in [
        "beats_alpha_hat_on_catastrophic_auroc",
        "beats_alpha_hat_on_catastrophic_auprc",
        "beats_alpha_hat_on_operational_severity_spearman",
    ]:
        comparison[beats_column] = comparison[beats_column].astype("Int64")

    return comparison[
        [
            "subset",
            "n_settings",
            "n_catastrophic",
            "n_unsafe",
            "score_name",
            "risk_orientation",
            "spearman_vs_operational_severity",
            "catastrophic_auroc",
            "catastrophic_auprc",
            "spearman_vs_loss_rel",
            "unsafe_auroc",
            "unsafe_auprc",
            "rank_catastrophic_auroc",
            "rank_catastrophic_auprc",
            "rank_spearman_vs_operational_severity",
            "beats_alpha_hat_on_catastrophic_auroc",
            "beats_alpha_hat_on_catastrophic_auprc",
            "beats_alpha_hat_on_operational_severity_spearman",
        ]
    ].sort_values(
        by=["subset", "score_name"],
        key=lambda column: column.map(
            {name: index for index, name in enumerate(SUBSET_ORDER + SCORE_ORDER)}
        ).fillna(len(SUBSET_ORDER) + len(SCORE_ORDER)),
    ).reset_index(drop=True)


def build_primary_oracle_context(settings: pd.DataFrame) -> pd.DataFrame:
    primary = settings.loc[settings["oracle_alpha_scope"] == "primary"].copy()
    return primary[
        [
            "setting_id",
            "source_dataset",
            "target_dataset",
            "model",
            "source_holdout_tpr_at_1pct_fpr",
            "target_tpr_at_1pct_fpr",
            "target_realized_fpr",
            "relative_tpr_drop",
            "loss_rel",
            "oracle_alpha",
            "alpha_hat",
            "proxy_a_distance",
            "avg_wasserstein_1d",
            "catastrophic_operational_failure",
            "unsafe_setting",
        ]
    ].sort_values(["source_dataset", "target_dataset", "model"]).reset_index(drop=True)


def _strict_alpha_hat_pass(comparison: pd.DataFrame) -> bool:
    alpha_row = comparison.loc[
        (comparison["subset"] == "all_natural") & (comparison["score_name"] == "alpha_hat")
    ].iloc[0]
    subset_rows = comparison.loc[comparison["subset"] == "all_natural"]
    strict_best = False
    for metric_column in PRIMARY_METRIC_COLUMNS:
        metric_values = subset_rows[metric_column].astype(float)
        if metric_values.isna().all():
            continue
        best_value = float(metric_values.max())
        alpha_value = float(alpha_row[metric_column])
        if np.isclose(alpha_value, best_value) and (metric_values == best_value).sum() == 1:
            strict_best = True
            break
    if not strict_best:
        return False
    return bool(
        float(alpha_row["catastrophic_auroc"]) >= MIN_DEPLOYABLE_ALL_NATURAL_CATASTROPHIC_AUROC
        and float(alpha_row["spearman_vs_operational_severity"])
        >= MIN_DEPLOYABLE_ALL_NATURAL_SEVERITY_SPEARMAN
    )


def _strongest_generic_unlabeled_baseline(comparison: pd.DataFrame, subset: str) -> str:
    subset_rows = comparison.loc[
        (comparison["subset"] == subset)
        & (comparison["score_name"].isin(GENERIC_UNLABELED_BASELINES))
    ].copy()
    subset_rows["rank_sum"] = (
        subset_rows["rank_catastrophic_auroc"].astype(int)
        + subset_rows["rank_catastrophic_auprc"].astype(int)
        + subset_rows["rank_spearman_vs_operational_severity"].astype(int)
    )
    best_rank_sum = int(subset_rows["rank_sum"].min())
    strongest = subset_rows.loc[subset_rows["rank_sum"] == best_rank_sum, "score_name"].tolist()
    ordered = [name for name in GENERIC_UNLABELED_BASELINES if name in strongest]
    return "|".join(ordered)


def _strongest_audited_monitor(comparison: pd.DataFrame, subset: str) -> str:
    subset_rows = comparison.loc[comparison["subset"] == subset].copy()
    subset_rows["rank_sum"] = (
        subset_rows["rank_catastrophic_auroc"].astype(int)
        + subset_rows["rank_catastrophic_auprc"].astype(int)
        + subset_rows["rank_spearman_vs_operational_severity"].astype(int)
    )
    best_rank_sum = int(subset_rows["rank_sum"].min())
    strongest = subset_rows.loc[subset_rows["rank_sum"] == best_rank_sum, "score_name"].tolist()
    ordered = [name for name in SCORE_ORDER if name in strongest]
    return "|".join(ordered)


def _rank_snapshot(comparison: pd.DataFrame, subset: str, score_name: str) -> dict[str, int]:
    row = comparison.loc[
        (comparison["subset"] == subset) & (comparison["score_name"] == score_name)
    ].iloc[0]
    return {
        "catastrophic_auroc": int(row["rank_catastrophic_auroc"]),
        "catastrophic_auprc": int(row["rank_catastrophic_auprc"]),
        "operational_severity_spearman": int(row["rank_spearman_vs_operational_severity"]),
    }


def _row_metrics(frame: pd.DataFrame, score_name: str) -> dict[str, float]:
    row = frame.loc[frame["score_name"] == score_name].iloc[0]
    return {
        "spearman_vs_operational_severity": float(row["spearman_vs_operational_severity"]),
        "catastrophic_auroc": float(row["catastrophic_auroc"]),
        "catastrophic_auprc": float(row["catastrophic_auprc"]),
    }


def build_claim_decision(settings: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, Any]:
    deployable_alpha_hat_main_claim = _strict_alpha_hat_pass(comparison)
    strongest_all = _strongest_generic_unlabeled_baseline(comparison, "all_natural")
    strongest_primary = _strongest_generic_unlabeled_baseline(comparison, "primary_iot")
    strongest_audited_all = _strongest_audited_monitor(comparison, "all_natural")
    alpha_hat_all_ranks = _rank_snapshot(comparison, "all_natural", "alpha_hat")
    run_ids = sorted(settings["run_id"].dropna().astype(str).unique().tolist())
    canonical_run_id = run_ids[0] if len(run_ids) == 1 else "|".join(run_ids)

    if strongest_audited_all == "alpha_hat":
        decision_reason = (
            "alpha_hat ranked first under the revised operational audit on the "
            "full benchmark, but its full-benchmark catastrophic AUROC and "
            "severity association still remained too weak for a deployable claim."
        )
    else:
        decision_reason = (
            "alpha_hat did not outperform the strongest audited comparator on the "
            "full benchmark and its full-benchmark catastrophic AUROC and "
            "severity association remained too weak for a deployable claim."
        )

    evidence_snapshot = {
        "all_natural": {
            "alpha_hat": _row_metrics(
                comparison.loc[comparison["subset"] == "all_natural"],
                "alpha_hat",
            ),
            "strongest_generic_unlabeled_baseline": strongest_all,
        },
        "primary_iot": {
            "alpha_hat": _row_metrics(
                comparison.loc[comparison["subset"] == "primary_iot"],
                "alpha_hat",
            ),
            "strongest_generic_unlabeled_baseline": strongest_primary,
        },
    }

    return {
        "canonical_run_id": canonical_run_id,
        "n_all_natural": int((settings["setting_type"] == "natural_transfer").sum()),
        "n_primary_iot": int((settings["oracle_alpha_scope"] == "primary").sum()),
        "deployable_alpha_hat_main_claim": bool(deployable_alpha_hat_main_claim),
        "alpha_hat_placement": "main_negative_result",
        "strongest_unlabeled_baseline_all_natural": strongest_all,
        "strongest_unlabeled_baseline_primary_iot": strongest_primary,
        "strongest_audited_monitor_all_natural": strongest_audited_all,
        "proxy_a_promoted_to_main_claim": False,
        "decision_reason": decision_reason,
        "alpha_hat_all_natural_ranks": alpha_hat_all_ranks,
        "evidence_snapshot": evidence_snapshot,
    }


def build_claim_summary_markdown(
    settings_path: Path,
    within_path: Path,
    cross_path: Path,
    decision_eval_path: Path,
    within_summary: pd.DataFrame,
    cross_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    primary_context: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    all_rows = comparison.loc[comparison["subset"] == "all_natural"].copy()
    primary_rows = comparison.loc[comparison["subset"] == "primary_iot"].copy()

    def render_subset_table(frame: pd.DataFrame) -> str:
        rendered = frame[
            [
                "score_name",
                "spearman_vs_operational_severity",
                "catastrophic_auroc",
                "catastrophic_auprc",
            ]
        ].copy()
        for column in [
            "spearman_vs_operational_severity",
            "catastrophic_auroc",
            "catastrophic_auprc",
        ]:
            rendered[column] = rendered[column].map(lambda value: f"{float(value):.3f}")
        header = (
            "| score_name | spearman_vs_operational_severity | "
            "catastrophic_auroc | catastrophic_auprc |"
        )
        divider = "|---|---:|---:|---:|"
        rows = [
            (
                f"| {row.score_name} | {row.spearman_vs_operational_severity} | "
                f"{row.catastrophic_auroc} | {row.catastrophic_auprc} |"
            )
            for row in rendered.itertuples(index=False)
        ]
        return "\n".join([header, divider, *rows])

    return "\n".join(
        [
            "# Full-Data Claim Summary",
            "",
            "## Inputs",
            f"- benchmark settings: `{settings_path}`",
            f"- within-domain summary: `{within_path}`",
            f"- cross-domain summary: `{cross_path}`",
            (
                f"- decision audit artifact: `{decision_eval_path}` (audit only; "
                "not used for claim adjudication)"
            ),
            "",
            "## Canonical Benchmark Facts",
            f"- natural-transfer settings: {decision['n_all_natural']}",
            f"- primary oracle-novelty settings: {decision['n_primary_iot']}",
            f"- within-domain summary rows: {len(within_summary)}",
            f"- cross-domain summary rows: {len(cross_summary)}",
            f"- primary oracle context rows: {len(primary_context)}",
            "",
            "## Deployable Monitor Comparison",
            "### all_natural",
            render_subset_table(all_rows),
            "",
            "### primary_iot",
            render_subset_table(primary_rows),
            "",
            (
                "- On `all_natural`, `alpha_hat` ranks "
                f"{decision['alpha_hat_all_natural_ranks']['catastrophic_auroc']}/"
                f"{decision['alpha_hat_all_natural_ranks']['catastrophic_auprc']}/"
                f"{decision['alpha_hat_all_natural_ranks']['operational_severity_spearman']}"
                " across catastrophic AUROC / catastrophic AUPRC / severity association, "
                "so it remains non-dominant as well as non-deployable on the full benchmark."
            ),
            (
                "- The headline monitor target is operational severity, with "
                "catastrophic operational failure used as a secondary binary outcome."
            ),
            (
                "- The strongest audited full-benchmark comparator is "
                f"`{decision['strongest_audited_monitor_all_natural']}`, but it also "
                "remains too weak in absolute terms for a deployable monitoring claim."
            ),
            (
                "- The strongest generic unlabeled comparators remain "
                f"`{decision['strongest_unlabeled_baseline_all_natural']}` on `all_natural` "
                f"and `{decision['strongest_unlabeled_baseline_primary_iot']}` on `primary_iot`, "
                "but none of them changes the paper's center."
            ),
            "",
            "## Primary Oracle Context",
            (
                "- The primary oracle context is limited to the IoT-only claim-bearing rows "
                "where `oracle_alpha_scope == \"primary\"`."
            ),
            (
                "- These rows are emitted in `full_data_primary_oracle_context.csv` "
                "for direct manuscript use."
            ),
            "",
            "## Claim Decision",
            f"- deployable `alpha_hat` main claim: `{decision['deployable_alpha_hat_main_claim']}`",
            f"- placement: `{decision['alpha_hat_placement']}`",
            f"- proxy-A promoted to main claim: `{decision['proxy_a_promoted_to_main_claim']}`",
            f"- reason: {decision['decision_reason']}",
            "",
            "## Manuscript Instruction",
            (
                "- Keep `alpha_hat` out of the contribution list and describe it as a "
                "weak monitor result rather than a deployable win."
            ),
            (
                "- Remove any main-contribution wording that implies `alpha_hat` is "
                "operationally reliable on full data."
            ),
            (
                "- State that `alpha_hat` remained non-deployable on full data and did not "
                "beat the strongest audited comparator on the full benchmark."
            ),
            (
                "- Do not replace the paper thesis with `proxy_a_distance`; keep it as a "
                "stronger comparator, not a new contribution."
            ),
        ]
    )


def write_claim_package(
    artifacts: ClaimPackageArtifacts,
    monitor_comparison: pd.DataFrame,
    primary_context: pd.DataFrame,
    decision: dict[str, Any],
    summary_markdown: str,
) -> None:
    artifacts.monitor_comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    monitor_comparison.to_csv(artifacts.monitor_comparison_csv, index=False)
    primary_context.to_csv(artifacts.primary_oracle_context_csv, index=False)
    artifacts.claim_decision_json.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts.claim_summary_md.write_text(summary_markdown + "\n", encoding="utf-8")
