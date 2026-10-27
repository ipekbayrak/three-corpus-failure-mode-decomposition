from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from iot_ids.data import RAW_REQUIRED_COLUMNS
from iot_ids.features import FEATURE_APPENDIX_ROWS, FEATURE_COLUMNS
from iot_ids.schema_appendix import SchemaAppendixOutputs, write_schema_appendix_assets


def _write_staged_csv(path: Path, extra_columns: dict[str, object]) -> None:
    frame = pd.DataFrame(
        [
            {
                "record_id": "r0",
                "group_id": "g0",
                "label": 0,
                "family_raw": "benign",
                "proto": "tcp",
                "duration_s": 1.0,
                "src_bytes": 10.0,
                "dst_bytes": 5.0,
                "src_pkts": 2.0,
                "dst_pkts": 1.0,
                **extra_columns,
            }
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _normalize_latex(text: str) -> str:
    normalized = text
    for token in [r"\begin{tabular}[t]{@{}l@{}}", r"\end{tabular}", r"\\"]:
        normalized = normalized.replace(token, "")
    return "".join(normalized.split())


class SchemaAppendixTest(unittest.TestCase):
    def test_schema_appendix_emits_features_mappings_and_discards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            iot23_csv = root / "iot23.csv"
            ton_csv = root / "ton_iot.csv"
            unsw_csv = root / "unsw_nb15.csv"
            _write_staged_csv(iot23_csv, {"capture_id": "cap_0", "source_file": "iot23.csv"})
            _write_staged_csv(ton_csv, {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2"})
            _write_staged_csv(unsw_csv, {"source_split": "train", "id": 17})

            config = {
                "datasets": {
                    "iot23": {
                        "raw_csv": str(iot23_csv),
                        "column_map": {column: column for column in RAW_REQUIRED_COLUMNS},
                    },
                    "ton_iot": {
                        "raw_csv": str(ton_csv),
                        "column_map": {column: column for column in RAW_REQUIRED_COLUMNS},
                    },
                    "unsw_nb15": {
                        "raw_csv": str(unsw_csv),
                        "column_map": {column: column for column in RAW_REQUIRED_COLUMNS},
                    },
                }
            }
            outputs = SchemaAppendixOutputs(
                feature_schema_csv=root / "shared_feature_schema.csv",
                raw_mapping_csv=root / "raw_to_primitive_mapping.csv",
                discarded_columns_csv=root / "discarded_source_columns.csv",
                journal_feature_table=root / "journal_shared_feature_schema.tex",
                acsac_feature_table=root / "acsac_shared_feature_schema.tex",
            )

            summary = write_schema_appendix_assets(config, outputs)

            self.assertEqual(summary["feature_rows"], len(FEATURE_COLUMNS))
            self.assertEqual(summary["mapping_rows"], len(RAW_REQUIRED_COLUMNS))
            self.assertEqual(summary["discarded_rows"], 3)

            feature_frame = pd.read_csv(outputs.feature_schema_csv)
            mapping_frame = pd.read_csv(outputs.raw_mapping_csv)
            discarded_frame = pd.read_csv(outputs.discarded_columns_csv)

            self.assertEqual(feature_frame["feature_name"].tolist(), FEATURE_COLUMNS)
            self.assertEqual(mapping_frame["primitive_field"].tolist(), RAW_REQUIRED_COLUMNS)
            self.assertEqual(
                discarded_frame["dataset"].tolist(),
                ["IoT-23", "ToN-IoT", "UNSW-NB15"],
            )
            discarded_lookup = dict(
                zip(
                    discarded_frame["dataset"].tolist(),
                    discarded_frame["discarded_source_columns"].tolist(),
                )
            )
            self.assertIn("capture_id", discarded_lookup["IoT-23"])
            self.assertIn("src_ip", discarded_lookup["ToN-IoT"])
            self.assertIn("source_split", discarded_lookup["UNSW-NB15"])

            for path in [
                outputs.journal_feature_table,
                outputs.acsac_feature_table,
            ]:
                self.assertTrue(path.exists(), msg=f"missing schema appendix output: {path}")

            acsac_feature_tex = outputs.acsac_feature_table.read_text(encoding="utf-8")
            normalized_feature_tex = _normalize_latex(acsac_feature_tex)
            quantitative_rows = [
                row
                for row in FEATURE_APPENDIX_ROWS
                if not row["feature_name"].startswith("proto_")
            ]
            self.assertNotIn(r"\resizebox", acsac_feature_tex)
            self.assertIn(r"Formula", acsac_feature_tex)
            self.assertEqual(acsac_feature_tex.count(r"\begin{table*}"), 1)
            self.assertNotIn(r"\begin{table}", acsac_feature_tex)
            self.assertNotIn(r"\label{tab:shared-feature-schema-protocol}", acsac_feature_tex)
            self.assertNotIn(r"proto\_tcp", normalized_feature_tex)
            for row in quantitative_rows:
                self.assertIn(
                    row["feature_name"].replace("_", r"\_"),
                    normalized_feature_tex,
                )
                self.assertIn(
                    "".join(row["definition"].split(";")[0].replace("_", r"\_").split()),
                    normalized_feature_tex,
                )
            self.assertNotIn("log_duration_slog", acsac_feature_tex)
            self.assertNotIn("log Ip", acsac_feature_tex)


if __name__ == "__main__":
    unittest.main()
