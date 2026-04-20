#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from iot_ids.config import load_config
from iot_ids.manuscript_assets import (
    AcsacOutputs,
    ManuscriptOutputs,
    ManuscriptSources,
    write_acsac_assets,
    write_manuscript_assets,
)
from iot_ids.schema_appendix import SchemaAppendixOutputs, write_schema_appendix_assets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build manuscript-ready tables, figures, and numeric macros."
    )
    parser.add_argument("--config", required=True, help="Path to render YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)

    schema_cfg = config.get("schema_appendix_assets")
    if schema_cfg:
        schema_outputs = SchemaAppendixOutputs(
            feature_schema_csv=Path(schema_cfg["outputs"]["feature_schema_csv"]),
            raw_mapping_csv=Path(schema_cfg["outputs"]["raw_mapping_csv"]),
            discarded_columns_csv=Path(schema_cfg["outputs"]["discarded_columns_csv"]),
            journal_feature_table=Path(schema_cfg["outputs"]["journal_feature_table"]),
            acsac_feature_table=Path(schema_cfg["outputs"]["acsac_feature_table"]),
        )
        schema_summary = write_schema_appendix_assets(config, schema_outputs)
        print(
            "Built schema assets: "
            f"features={schema_summary['feature_rows']}, "
            f"mappings={schema_summary['mapping_rows']}, "
            f"discarded={schema_summary['discarded_rows']}"
        )

    def build_sources(cfg: dict[str, object]) -> ManuscriptSources:
        return ManuscriptSources(
            within_domain_summary=Path(cfg["inputs"]["within_domain_summary"]),
            cross_domain_summary=Path(cfg["inputs"]["cross_domain_summary"]),
            monitor_comparison=Path(cfg["inputs"]["monitor_comparison"]),
            primary_oracle_context=Path(cfg["inputs"]["primary_oracle_context"]),
            claim_decision=Path(cfg["inputs"]["claim_decision"]),
            target_retuned_oracle_baseline=Path(cfg["inputs"]["target_retuned_oracle_baseline"]),
            fpr_budget_sensitivity=Path(cfg["inputs"]["fpr_budget_sensitivity"]),
            ranking_diagnosis=Path(cfg["inputs"]["ranking_diagnosis"]),
            source_holdout_fpr=Path(cfg["inputs"]["source_holdout_fpr"]),
            recalibration=Path(cfg["inputs"]["recalibration"]),
            direction_covariate_shift=Path(cfg["inputs"]["direction_covariate_shift"]),
            coverage_sensitivity=Path(cfg["inputs"]["coverage_sensitivity"]),
            coral_auxiliary=Path(cfg["inputs"]["coral_auxiliary"]),
        )

    journal_cfg = config["manuscript_assets"]
    journal_sources = build_sources(journal_cfg)
    journal_outputs = ManuscriptOutputs(
        within_table=Path(journal_cfg["outputs"]["within_table"]),
        cross_table=Path(journal_cfg["outputs"]["cross_table"]),
        monitor_table=Path(journal_cfg["outputs"]["monitor_table"]),
        primary_table=Path(journal_cfg["outputs"]["primary_table"]),
        ranking_diagnosis_table=Path(journal_cfg["outputs"]["ranking_diagnosis_table"]),
        budget_robustness_table=Path(journal_cfg["outputs"]["budget_robustness_table"]),
        oracle_retuned_summary_table=Path(
            journal_cfg["outputs"]["oracle_retuned_summary_table"]
        ),
        coral_auxiliary_table=Path(journal_cfg["outputs"]["coral_auxiliary_table"]),
        manuscript_numbers=Path(journal_cfg["outputs"]["manuscript_numbers"]),
        transfer_tpr_figure_pdf=Path(journal_cfg["outputs"]["transfer_tpr_figure_pdf"]),
        transfer_tpr_figure_png=Path(journal_cfg["outputs"]["transfer_tpr_figure_png"]),
        transfer_fpr_figure_pdf=Path(journal_cfg["outputs"]["transfer_fpr_figure_pdf"]),
        transfer_fpr_figure_png=Path(journal_cfg["outputs"]["transfer_fpr_figure_png"]),
        oracle_figure_pdf=Path(journal_cfg["outputs"]["oracle_figure_pdf"]),
        oracle_figure_png=Path(journal_cfg["outputs"]["oracle_figure_png"]),
        monitor_figure_pdf=Path(journal_cfg["outputs"]["monitor_figure_pdf"]),
        monitor_figure_png=Path(journal_cfg["outputs"]["monitor_figure_png"]),
    )

    journal_summary = write_manuscript_assets(journal_sources, journal_outputs)
    print(
        "Built journal manuscript assets: "
        f"within={journal_summary['within_rows']}, "
        f"cross={journal_summary['cross_rows']}, "
        f"monitor={journal_summary['monitor_rows']}, "
        f"primary={journal_summary['primary_rows']}"
    )

    acsac_cfg = config.get("acsac_assets")
    if acsac_cfg:
        acsac_sources = build_sources(acsac_cfg)
        acsac_outputs = AcsacOutputs(
            source_table=Path(acsac_cfg["outputs"]["source_table"]),
            within_table=Path(acsac_cfg["outputs"]["within_table"]),
            cross_table=Path(acsac_cfg["outputs"]["cross_table"]),
            monitor_table=Path(acsac_cfg["outputs"]["monitor_table"]),
            oracle_table=Path(acsac_cfg["outputs"]["oracle_table"]),
            collapse_table=Path(acsac_cfg["outputs"]["collapse_table"]),
            budget_robustness_table=Path(acsac_cfg["outputs"]["budget_robustness_table"]),
            oracle_retuned_summary_table=Path(
                acsac_cfg["outputs"]["oracle_retuned_summary_table"]
            ),
            coral_auxiliary_table=Path(acsac_cfg["outputs"]["coral_auxiliary_table"]),
            transfer_tpr_figure_pdf=Path(acsac_cfg["outputs"]["transfer_tpr_figure_pdf"]),
            transfer_tpr_figure_png=Path(acsac_cfg["outputs"]["transfer_tpr_figure_png"]),
            transfer_fpr_figure_pdf=Path(acsac_cfg["outputs"]["transfer_fpr_figure_pdf"]),
            transfer_fpr_figure_png=Path(acsac_cfg["outputs"]["transfer_fpr_figure_png"]),
            oracle_figure_pdf=Path(acsac_cfg["outputs"]["oracle_figure_pdf"]),
            oracle_figure_png=Path(acsac_cfg["outputs"]["oracle_figure_png"]),
        )
        acsac_summary = write_acsac_assets(acsac_sources, acsac_outputs)
        print(
            "Built ACSAC manuscript assets: "
            f"within={acsac_summary['within_rows']}, "
            f"cross={acsac_summary['cross_rows']}, "
            f"monitor={acsac_summary['monitor_rows']}, "
            f"primary={acsac_summary['primary_rows']}, "
            f"directions={acsac_summary['direction_rows']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
