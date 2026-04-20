from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from iot_ids.native_auxiliary import NativeAuxiliaryOutputs, run_native_auxiliary


def _dataset_rows(dataset: str) -> list[dict[str, object]]:
    extras: dict[str, object]
    if dataset == "ton_iot":
        extras = {
            "src_ip": "10.0.0.1",
            "dst_ip": "8.8.8.8",
            "src_port": 12345,
            "dst_port": 80,
        }
    elif dataset == "unsw_nb15":
        extras = {"service": "http", "state": "FIN"}
    else:
        extras = {"proto_name": "tcp"}

    rows: list[dict[str, object]] = []
    for idx, role in enumerate(["train", "train", "val", "val", "test", "test"]):
        rows.append(
            {
                "record_id": f"{dataset}_{idx}",
                "group_id": f"{dataset}_{role}_{idx // 2}",
                "label": idx % 2,
                "family_raw": "attack" if idx % 2 else "benign",
                "proto": "tcp" if idx % 3 else "udp",
                "duration_s": 1.0 + idx,
                "src_bytes": 10.0 + idx,
                "dst_bytes": 5.0 + idx,
                "src_pkts": 2.0 + idx,
                "dst_pkts": 1.0 + idx,
                **extras,
            }
        )
    return rows


class NativeAuxiliaryTest(unittest.TestCase):
    def test_native_auxiliary_writes_parallel_artifacts_and_reuses_split_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_paths = {}
            original_csv = {}
            split_rows = []
            for dataset in ["iot23", "ton_iot", "unsw_nb15"]:
                csv_path = root / f"{dataset}.csv"
                frame = pd.DataFrame(_dataset_rows(dataset))
                before = frame.to_csv(index=False)
                csv_path.write_text(before, encoding="utf-8")
                dataset_paths[dataset] = csv_path
                original_csv[dataset] = before
                for idx, role in enumerate(["train", "train", "val", "val", "test", "test"]):
                    split_rows.append(
                        {
                            "dataset": dataset,
                            "record_id": f"{dataset}_{idx}",
                            "fold_role": role,
                        }
                    )

            split_manifest = pd.DataFrame(split_rows)
            split_manifest_path = root / "split_manifest.csv"
            split_manifest.to_csv(split_manifest_path, index=False)

            config = {
                "project": {"seed": 20260413},
                "ontology": {
                    "mapping_csv": str(root / "missing_mapping.csv"),
                    "default_group": "other",
                },
                "datasets": {
                    dataset: {
                        "raw_csv": str(path),
                        "column_map": {
                            "record_id": "record_id",
                            "group_id": "group_id",
                            "label": "label",
                            "family_raw": "family_raw",
                            "proto": "proto",
                            "duration_s": "duration_s",
                            "src_bytes": "src_bytes",
                            "dst_bytes": "dst_bytes",
                            "src_pkts": "src_pkts",
                            "dst_pkts": "dst_pkts",
                        },
                    }
                    for dataset, path in dataset_paths.items()
                },
                "native_auxiliary": {
                    "split_manifest_csv": str(split_manifest_path),
                    "target_fpr": 0.01,
                    "max_categorical_cardinality": 8,
                    "models": ["hist_gradient_boosting"],
                },
            }
            outputs = NativeAuxiliaryOutputs(
                summary_csv=root / "aux" / "summary.csv",
                feature_manifest_csv=root / "aux" / "manifest.csv",
                split_audit_csv=root / "aux" / "split_audit.csv",
                journal_table=root / "aux" / "journal.tex",
                acsac_table=root / "aux" / "acsac.tex",
                staged_root=root / "aux" / "staged",
            )

            summary = run_native_auxiliary(config, outputs)

            self.assertEqual(summary["summary_rows"], 6)
            self.assertTrue(outputs.summary_csv.exists())
            self.assertTrue(outputs.feature_manifest_csv.exists())
            self.assertTrue(outputs.split_audit_csv.exists())
            self.assertTrue(outputs.journal_table.exists())
            self.assertTrue(outputs.acsac_table.exists())

            journal_tex = outputs.journal_table.read_text(encoding="utf-8")
            acsac_tex = outputs.acsac_table.read_text(encoding="utf-8")
            self.assertIn("representative strong shared-schema model", journal_tex)
            self.assertIn("representative strong shared-schema model", acsac_tex)

            manifest = pd.read_csv(outputs.feature_manifest_csv)
            self.assertEqual(
                sorted(manifest["dataset"].unique().tolist()),
                ["iot23", "ton_iot", "unsw_nb15"],
            )
            counts = manifest.groupby("dataset").size().to_dict()
            self.assertTrue(all(count > 0 for count in counts.values()))

            split_audit = pd.read_csv(outputs.split_audit_csv)
            self.assertEqual(set(split_audit["split_role"]), {"train", "val", "test"})

            for dataset, path in dataset_paths.items():
                after = path.read_text(encoding="utf-8")
                self.assertEqual(after, original_csv[dataset])


if __name__ == "__main__":
    unittest.main()
