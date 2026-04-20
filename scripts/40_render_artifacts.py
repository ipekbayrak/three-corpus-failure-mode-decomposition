#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from iot_ids.config import load_config


def _write_manifest_entry(
    rows: list[dict[str, str]],
    artifact_id: str,
    artifact_type: str,
    path: Path,
    script: str,
) -> None:
    rows.append(
        {
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "path": str(path),
            "producer_script": script,
            "run_id": "latest",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render pilot and benchmark artifacts.")
    parser.add_argument("--config", required=True, help="Path to render YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_rows: list[dict[str, str]] = []
    claims_rows: list[dict[str, str]] = []

    pilot_settings = Path(config["render"]["pilot_settings"])
    figures_root = Path(config["paths"]["figures_root"])
    tables_root = Path(config["paths"]["tables_root"])
    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)

    if pilot_settings.exists():
        settings = pd.read_csv(pilot_settings)
        fig_path = figures_root / "hero_novelty_vs_loss.png"
        plt.figure(figsize=(7, 5))
        for (setting_type, model), subset in settings.groupby(["setting_type", "model"]):
            plt.scatter(
                subset["oracle_alpha"],
                subset["loss_rel"],
                label=f"{setting_type}:{model}",
                alpha=0.8,
            )
        plt.xlabel("Oracle novelty alpha")
        plt.ylabel("Relative transfer loss")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        _write_manifest_entry(
            manifest_rows,
            "hero_figure",
            "figure",
            fig_path,
            "scripts/40_render_artifacts.py",
        )
        claims_rows.append(
            {
                "claim_id": "hero_figure_path",
                "artifact_id": "hero_figure",
                "source_path": str(fig_path),
                "locator": "file",
                "description": "Hero figure for oracle novelty vs relative transfer loss",
            }
        )

        summary_path = tables_root / "pilot_settings_summary.csv"
        settings.to_csv(summary_path, index=False)
        _write_manifest_entry(
            manifest_rows,
            "pilot_settings_summary",
            "table",
            summary_path,
            "scripts/40_render_artifacts.py",
        )

    benchmark_predictions = Path(config["render"]["benchmark_predictions"])
    if benchmark_predictions.exists():
        benchmark = pd.read_csv(benchmark_predictions)
        benchmark_summary = (
            benchmark.groupby(["source_dataset", "target_dataset", "model"], as_index=False)
            .agg(
                {
                    "score": "mean",
                    "label": "mean",
                }
            )
            .rename(columns={"score": "mean_score", "label": "attack_rate"})
        )
        summary_path = tables_root / "benchmark_prediction_summary.csv"
        benchmark_summary.to_csv(summary_path, index=False)
        _write_manifest_entry(
            manifest_rows,
            "benchmark_prediction_summary",
            "table",
            summary_path,
            "scripts/40_render_artifacts.py",
        )

    benchmark_within = Path(config["render"]["benchmark_within_domain_summary"])
    if benchmark_within.exists():
        table_path = tables_root / "within_domain_summary.csv"
        pd.read_csv(benchmark_within).to_csv(table_path, index=False)
        _write_manifest_entry(
            manifest_rows,
            "within_domain_summary",
            "table",
            table_path,
            "scripts/40_render_artifacts.py",
        )
        claims_rows.append(
            {
                "claim_id": "within_domain_summary_path",
                "artifact_id": "within_domain_summary",
                "source_path": str(table_path),
                "locator": "file",
                "description": "Canonical within-domain benchmark summary table",
            }
        )

    benchmark_cross = Path(config["render"]["benchmark_cross_domain_summary"])
    if benchmark_cross.exists():
        table_path = tables_root / "cross_domain_summary.csv"
        pd.read_csv(benchmark_cross).to_csv(table_path, index=False)
        _write_manifest_entry(
            manifest_rows,
            "cross_domain_summary",
            "table",
            table_path,
            "scripts/40_render_artifacts.py",
        )
        claims_rows.append(
            {
                "claim_id": "cross_domain_summary_path",
                "artifact_id": "cross_domain_summary",
                "source_path": str(table_path),
                "locator": "file",
                "description": "Canonical cross-domain benchmark summary table",
            }
        )

    source_holdout_fpr = Path(config["render"].get("source_holdout_fpr", ""))
    if str(source_holdout_fpr) and source_holdout_fpr.exists():
        table_path = tables_root / "source_holdout_fpr.csv"
        pd.read_csv(source_holdout_fpr).to_csv(table_path, index=False)
        _write_manifest_entry(
            manifest_rows,
            "source_holdout_fpr",
            "table",
            table_path,
            "scripts/40_render_artifacts.py",
        )
        claims_rows.append(
            {
                "claim_id": "source_holdout_fpr_path",
                "artifact_id": "source_holdout_fpr",
                "source_path": str(table_path),
                "locator": "file",
                "description": "Source-holdout realized FPR table used by the competence audit",
            }
        )

    ranking_diagnosis = Path(config["render"].get("ranking_diagnosis", ""))
    if str(ranking_diagnosis) and ranking_diagnosis.exists():
        table_path = tables_root / "ranking_diagnosis.csv"
        pd.read_csv(ranking_diagnosis).to_csv(table_path, index=False)
        _write_manifest_entry(
            manifest_rows,
            "ranking_diagnosis",
            "table",
            table_path,
            "scripts/40_render_artifacts.py",
        )
        claims_rows.append(
            {
                "claim_id": "ranking_diagnosis_path",
                "artifact_id": "ranking_diagnosis",
                "source_path": str(table_path),
                "locator": "file",
                "description": "Ranking diagnosis table with supporting low-FPR pAUC fields",
            }
        )

    decision_metrics = Path(config["render"]["decision_metrics"])
    if decision_metrics.exists():
        decision = pd.read_csv(decision_metrics)
        table_path = tables_root / "decision_summary.csv"
        decision.to_csv(table_path, index=False)
        _write_manifest_entry(
            manifest_rows,
            "decision_summary",
            "table",
            table_path,
            "scripts/40_render_artifacts.py",
        )
        claims_rows.append(
            {
                "claim_id": "decision_summary_path",
                "artifact_id": "decision_summary",
                "source_path": str(table_path),
                "locator": "file",
                "description": "Decision evaluation summary table",
            }
        )

    claim_cfg = config.get("claim_package", {})
    claim_outputs = claim_cfg.get("outputs", {})
    if claim_outputs:
        claim_artifacts = [
            (
                "full_data_monitor_comparison",
                "table",
                Path(claim_outputs["monitor_comparison_csv"]),
                "Full-data deployable monitor comparison table",
            ),
            (
                "full_data_primary_oracle_context",
                "table",
                Path(claim_outputs["primary_oracle_context_csv"]),
                "Full-data primary oracle context table",
            ),
            (
                "full_data_claim_decision",
                "json",
                Path(claim_outputs["claim_decision_json"]),
                "Full-data alpha_hat claim decision JSON",
            ),
            (
                "full_data_claim_summary",
                "note",
                Path(claim_outputs["claim_summary_md"]),
                "Full-data alpha_hat claim summary memo",
            ),
        ]
        for artifact_id, artifact_type, path, description in claim_artifacts:
            if path.exists():
                _write_manifest_entry(
                    manifest_rows,
                    artifact_id,
                    artifact_type,
                    path,
                    "scripts/40_render_artifacts.py",
                )
                claims_rows.append(
                    {
                        "claim_id": f"{artifact_id}_path",
                        "artifact_id": artifact_id,
                        "source_path": str(path),
                        "locator": "file",
                        "description": description,
                    }
                )

    acsac_diag_cfg = config.get("acsac_diagnostics", {})
    acsac_diag_outputs = acsac_diag_cfg.get("outputs", {})
    if acsac_diag_outputs:
        diagnostic_artifacts = [
            (
                "acsac_target_retuned_oracle_baseline",
                "table",
                Path(acsac_diag_outputs["target_retuned_oracle_baseline_csv"]),
                "ACSAC target-retuned oracle diagnostic derived from frozen benchmark predictions",
            ),
            (
                "acsac_fpr_budget_sensitivity",
                "table",
                Path(acsac_diag_outputs["fpr_budget_sensitivity_csv"]),
                "ACSAC source-selected FPR-budget sensitivity diagnostic derived from frozen benchmark predictions",
            ),
        ]
        for artifact_id, artifact_type, path, description in diagnostic_artifacts:
            if path.exists():
                _write_manifest_entry(
                    manifest_rows,
                    artifact_id,
                    artifact_type,
                    path,
                    "scripts/40_render_artifacts.py",
                )
                claims_rows.append(
                    {
                        "claim_id": f"{artifact_id}_path",
                        "artifact_id": artifact_id,
                        "source_path": str(path),
                        "locator": "file",
                        "description": description,
                    }
                )

    native_aux_cfg = config.get("native_auxiliary", {})
    native_outputs = native_aux_cfg.get("outputs", {})
    if native_outputs:
        native_summary = Path(native_outputs["summary_csv"])
        if native_summary.exists():
            _write_manifest_entry(
                manifest_rows,
                "native_auxiliary_summary",
                "table",
                native_summary,
                "scripts/49_native_feature_auxiliary.py",
            )
            claims_rows.append(
                {
                    "claim_id": "native_auxiliary_summary_path",
                    "artifact_id": "native_auxiliary_summary",
                    "source_path": str(native_summary),
                    "locator": "file",
                    "description": "Auxiliary within-domain native-feature information-retention summary",
                }
            )

    coral_aux_cfg = config.get("coral_auxiliary", {})
    coral_outputs = coral_aux_cfg.get("outputs", {})
    if coral_outputs:
        coral_summary = Path(coral_outputs["summary_csv"])
        if coral_summary.exists():
            _write_manifest_entry(
                manifest_rows,
                "coral_auxiliary_summary",
                "table",
                coral_summary,
                "scripts/52_coral_auxiliary.py",
            )
            claims_rows.append(
                {
                    "claim_id": "coral_auxiliary_summary_path",
                    "artifact_id": "coral_auxiliary_summary",
                    "source_path": str(coral_summary),
                    "locator": "file",
                    "description": "Auxiliary unlabeled CORAL transfer summary on the canonical three-corpus benchmark",
                }
            )

    schema_cfg = config.get("schema_appendix_assets", {})
    schema_outputs = schema_cfg.get("outputs", {})
    if schema_outputs:
        schema_artifacts = [
            (
                "schema_feature_schema_csv",
                "table",
                Path(schema_outputs["feature_schema_csv"]),
                "Generated shared feature schema artifact",
            ),
            (
                "schema_raw_mapping_csv",
                "table",
                Path(schema_outputs["raw_mapping_csv"]),
                "Generated dataset-to-primitive mapping artifact",
            ),
            (
                "schema_discarded_columns_csv",
                "table",
                Path(schema_outputs["discarded_columns_csv"]),
                "Generated discarded source-columns artifact",
            ),
            (
                "paper_shared_feature_schema_table",
                "table",
                Path(schema_outputs["journal_feature_table"]),
                "Journal shared feature schema inline table",
            ),
            (
                "acsac_shared_feature_schema_table",
                "table",
                Path(schema_outputs["acsac_feature_table"]),
                "ACSAC shared feature schema inline table",
            ),
        ]
        for artifact_id, artifact_type, path, description in schema_artifacts:
            if path.exists():
                _write_manifest_entry(
                    manifest_rows,
                    artifact_id,
                    artifact_type,
                    path,
                    "scripts/40_render_artifacts.py",
                )
                claims_rows.append(
                    {
                        "claim_id": f"{artifact_id}_path",
                        "artifact_id": artifact_id,
                        "source_path": str(path),
                        "locator": "file",
                        "description": description,
                    }
                )

    recal_cfg = config.get("recalibration_assets", {})
    recal_outputs = recal_cfg.get("outputs", {})
    if recal_outputs:
        recal_artifacts = [
            (
                "recalibration_csv",
                "table",
                Path(recal_outputs["recalibration_csv"]),
                "Benign-only recalibration audit with disjoint calibration and evaluation splits",
            ),
            (
                "acsac_recalibration_table",
                "table",
                Path(recal_outputs["acsac_table"]),
                "ACSAC recalibration table generated from the recalibration audit",
            ),
        ]
        for artifact_id, artifact_type, path, description in recal_artifacts:
            if path.exists():
                _write_manifest_entry(
                    manifest_rows,
                    artifact_id,
                    artifact_type,
                    path,
                    "scripts/40_render_artifacts.py",
                )
                claims_rows.append(
                    {
                        "claim_id": f"{artifact_id}_path",
                        "artifact_id": artifact_id,
                        "source_path": str(path),
                        "locator": "file",
                        "description": description,
                    }
                )

    manuscript_cfg = config.get("manuscript_assets", {})
    manuscript_outputs = manuscript_cfg.get("outputs", {})
    if manuscript_outputs:
        manuscript_artifacts = [
            (
                "paper_within_domain_table",
                "table",
                Path(manuscript_outputs["within_table"]),
                "Within-domain manuscript table generated from canonical benchmark artifacts",
            ),
            (
                "paper_cross_domain_table",
                "table",
                Path(manuscript_outputs["cross_table"]),
                "Cross-domain manuscript table generated from canonical benchmark artifacts",
            ),
            (
                "paper_monitor_comparison_table",
                "table",
                Path(manuscript_outputs["monitor_table"]),
                "Deployable monitor comparison manuscript table generated from canonical artifacts",
            ),
            (
                "paper_primary_oracle_context_table",
                "table",
                Path(manuscript_outputs["primary_table"]),
                "Primary oracle-context manuscript table generated from canonical artifacts",
            ),
            (
                "paper_manuscript_numbers",
                "tex",
                Path(manuscript_outputs["manuscript_numbers"]),
                "Generated LaTeX numeric macros used by manuscript prose",
            ),
            (
                "paper_manuscript_pdf",
                "pdf",
                Path(manuscript_outputs["manuscript_pdf"]),
                "Compiled manuscript PDF for the current canonical submission package",
            ),
            (
                "paper_transfer_tpr_figure_pdf",
                "figure",
                Path(manuscript_outputs["transfer_tpr_figure_pdf"]),
                "Cross-domain transfer target-TPR manuscript figure (PDF)",
            ),
            (
                "paper_transfer_tpr_figure_png",
                "figure",
                Path(manuscript_outputs["transfer_tpr_figure_png"]),
                "Cross-domain transfer target-TPR manuscript figure (PNG)",
            ),
            (
                "paper_transfer_fpr_figure_pdf",
                "figure",
                Path(manuscript_outputs["transfer_fpr_figure_pdf"]),
                "Cross-domain transfer realized-FPR manuscript figure (PDF)",
            ),
            (
                "paper_transfer_fpr_figure_png",
                "figure",
                Path(manuscript_outputs["transfer_fpr_figure_png"]),
                "Cross-domain transfer realized-FPR manuscript figure (PNG)",
            ),
            (
                "paper_oracle_novelty_figure_pdf",
                "figure",
                Path(manuscript_outputs["oracle_figure_pdf"]),
                "Primary oracle novelty manuscript figure (PDF)",
            ),
            (
                "paper_oracle_novelty_figure_png",
                "figure",
                Path(manuscript_outputs["oracle_figure_png"]),
                "Primary oracle novelty manuscript figure (PNG)",
            ),
            (
                "paper_monitor_comparison_figure_pdf",
                "figure",
                Path(manuscript_outputs["monitor_figure_pdf"]),
                "Deployable monitor comparison manuscript figure (PDF)",
            ),
            (
                "paper_monitor_comparison_figure_png",
                "figure",
                Path(manuscript_outputs["monitor_figure_png"]),
                "Deployable monitor comparison manuscript figure (PNG)",
            ),
        ]
        for artifact_id, artifact_type, path, description in manuscript_artifacts:
            if path.exists():
                _write_manifest_entry(
                    manifest_rows,
                    artifact_id,
                    artifact_type,
                    path,
                    "scripts/40_render_artifacts.py",
                )
                claims_rows.append(
                    {
                        "claim_id": f"{artifact_id}_path",
                        "artifact_id": artifact_id,
                        "source_path": str(path),
                        "locator": "file",
                        "description": description,
                    }
                )

    acsac_cfg = config.get("acsac_assets", {})
    acsac_outputs = acsac_cfg.get("outputs", {})
    if acsac_outputs:
        acsac_artifacts = [
            (
                "acsac_within_domain_table",
                "table",
                Path(acsac_outputs["within_table"]),
                "ACSAC within-domain manuscript table generated from canonical benchmark artifacts",
            ),
            (
                "acsac_cross_domain_summary_table",
                "table",
                Path(acsac_outputs["cross_table"]),
                "ACSAC cross-domain summary table generated from canonical benchmark artifacts",
            ),
            (
                "acsac_monitor_comparison_table",
                "table",
                Path(acsac_outputs["monitor_table"]),
                "ACSAC monitor comparison table generated from canonical artifacts",
            ),
            (
                "acsac_oracle_direction_table",
                "table",
                Path(acsac_outputs["oracle_table"]),
                "ACSAC oracle-direction table generated from canonical artifacts",
            ),
            (
                "acsac_transfer_tpr_figure_pdf",
                "figure",
                Path(acsac_outputs["transfer_tpr_figure_pdf"]),
                "ACSAC transfer target-TPR figure (PDF)",
            ),
            (
                "acsac_transfer_tpr_figure_png",
                "figure",
                Path(acsac_outputs["transfer_tpr_figure_png"]),
                "ACSAC transfer target-TPR figure (PNG)",
            ),
            (
                "acsac_transfer_fpr_figure_pdf",
                "figure",
                Path(acsac_outputs["transfer_fpr_figure_pdf"]),
                "ACSAC transfer realized-FPR figure (PDF)",
            ),
            (
                "acsac_transfer_fpr_figure_png",
                "figure",
                Path(acsac_outputs["transfer_fpr_figure_png"]),
                "ACSAC transfer realized-FPR figure (PNG)",
            ),
            (
                "acsac_oracle_novelty_figure_pdf",
                "figure",
                Path(acsac_outputs["oracle_figure_pdf"]),
                "ACSAC oracle novelty figure (PDF)",
            ),
            (
                "acsac_oracle_novelty_figure_png",
                "figure",
                Path(acsac_outputs["oracle_figure_png"]),
                "ACSAC oracle novelty figure (PNG)",
            ),
            (
                "acsac_manuscript_pdf",
                "pdf",
                Path(acsac_outputs["manuscript_pdf"]),
                "Compiled anonymous ACSAC manuscript PDF",
            ),
        ]
        for artifact_id, artifact_type, path, description in acsac_artifacts:
            if path.exists():
                _write_manifest_entry(
                    manifest_rows,
                    artifact_id,
                    artifact_type,
                    path,
                    "scripts/40_render_artifacts.py",
                )
                claims_rows.append(
                    {
                        "claim_id": f"{artifact_id}_path",
                        "artifact_id": artifact_id,
                        "source_path": str(path),
                        "locator": "file",
                        "description": description,
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = Path(config["render"]["results_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    claims = pd.DataFrame(claims_rows)
    claims_path = Path(config["render"]["claims_csv"])
    claims_path.parent.mkdir(parents=True, exist_ok=True)
    claims.to_csv(claims_path, index=False)
    print(f"Wrote artifact manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
