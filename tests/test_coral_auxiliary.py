from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from iot_ids.config import load_config
from iot_ids.coral_auxiliary import (
    CORAL_AUXILIARY_COLUMNS,
    CoralAdapter,
    CoralAuxiliaryOutputs,
    run_coral_auxiliary,
)


def _synthetic_dataset_rows(dataset: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    proto_cycle = ["tcp", "udp", "icmp", "other"]
    for group_idx in range(10):
        for row_idx, label in enumerate([0, 1]):
            rows.append(
                {
                    "record_id": f"{dataset}_{group_idx}_{row_idx}",
                    "group_id": f"{dataset}_g{group_idx}",
                    "label": label,
                    "family_raw": "benign" if label == 0 else "attack",
                    "proto": proto_cycle[(group_idx + row_idx) % len(proto_cycle)],
                    "duration_s": 1.0 + group_idx + row_idx / 10.0,
                    "src_bytes": 10.0 + (2 * group_idx) + (5 * label),
                    "dst_bytes": 6.0 + group_idx + (3 * label),
                    "src_pkts": 2.0 + (group_idx % 4) + label,
                    "dst_pkts": 1.0 + (group_idx % 3) + (0.5 * label),
                }
            )
    return rows


class CoralAuxiliaryTest(unittest.TestCase):
    def test_coral_adapter_preserves_shape_and_improves_covariance_alignment(self) -> None:
        rng = np.random.default_rng(13)
        source = pd.DataFrame(
            rng.normal(loc=0.0, scale=[1.0, 0.5, 2.0], size=(128, 3)),
            columns=["a", "b", "c"],
        )
        target = pd.DataFrame(
            rng.normal(loc=[1.5, -1.0, 0.5], scale=[2.5, 1.5, 0.7], size=(128, 3)),
            columns=["a", "b", "c"],
        )
        adapter = CoralAdapter()
        adapter.fit(source, target)
        transformed = adapter.transform(source)

        self.assertEqual(transformed.shape, source.shape)
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())

        source_cov = np.cov(source.to_numpy(), rowvar=False)
        target_cov = np.cov(target.to_numpy(), rowvar=False)
        transformed_cov = np.cov(transformed.to_numpy(), rowvar=False)
        before = np.linalg.norm(source_cov - target_cov, ord="fro")
        after = np.linalg.norm(transformed_cov - target_cov, ord="fro")
        self.assertLessEqual(after, before)

    def test_run_coral_auxiliary_writes_deterministic_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_paths: dict[str, Path] = {}
            for dataset in ["iot23", "ton_iot", "unsw_nb15"]:
                csv_path = root / f"{dataset}.csv"
                pd.DataFrame(_synthetic_dataset_rows(dataset)).to_csv(csv_path, index=False)
                dataset_paths[dataset] = csv_path

            config_path = root / "render.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "project:",
                        "  seed: 20260413",
                        "paths:",
                        f"  processed_root: {root / 'processed'}",
                        f"  manifests_root: {root / 'manifests'}",
                        "ontology:",
                        f"  mapping_csv: {root / 'missing_mapping.csv'}",
                        "  default_group: other",
                        "split:",
                        "  train_frac: 0.6",
                        "  val_frac: 0.2",
                        "datasets:",
                        *[
                            "\n".join(
                                [
                                    f"  {dataset}:",
                                    f"    raw_csv: {path}",
                                    "    loader: staged_csv",
                                    "    column_map:",
                                    "      record_id: record_id",
                                    "      group_id: group_id",
                                    "      label: label",
                                    "      family_raw: family_raw",
                                    "      proto: proto",
                                    "      duration_s: duration_s",
                                    "      src_bytes: src_bytes",
                                    "      dst_bytes: dst_bytes",
                                    "      src_pkts: src_pkts",
                                    "      dst_pkts: dst_pkts",
                                ]
                            )
                            for dataset, path in dataset_paths.items()
                        ],
                        "coral_auxiliary:",
                        "  target_fpr: 0.01",
                        "  models:",
                        "    - logistic_regression",
                        "    - extra_trees",
                        "    - hist_gradient_boosting",
                        "    - mlp",
                        "  outputs:",
                        f"    summary_csv: {root / 'artifacts' / 'coral_auxiliary.csv'}",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)
            outputs = CoralAuxiliaryOutputs(
                summary_csv=root / "artifacts" / "coral_auxiliary.csv"
            )
            summary = run_coral_auxiliary(config, outputs)

            self.assertEqual(summary["summary_rows"], 48)
            self.assertTrue(outputs.summary_csv.exists())

            frame = pd.read_csv(outputs.summary_csv)
            self.assertEqual(list(frame.columns), CORAL_AUXILIARY_COLUMNS)
            self.assertEqual(len(frame), 48)
            self.assertEqual(
                sorted(frame["auxiliary_method"].unique().tolist()),
                ["coral", "source_only_matched"],
            )
            self.assertEqual(sorted(frame["model"].unique().tolist()), [
                "extra_trees",
                "hist_gradient_boosting",
                "logistic_regression",
                "mlp",
            ])
            self.assertTrue((frame["target_adaptation_group_count"] > 0).all())
            self.assertTrue((frame["target_evaluation_group_count"] > 0).all())
            self.assertTrue(np.isfinite(frame.select_dtypes(include="number").to_numpy()).all())
            matched = frame.loc[frame["auxiliary_method"] == "source_only_matched"].copy()
            coral = frame.loc[frame["auxiliary_method"] == "coral"].copy()
            join_columns = ["source_dataset", "target_dataset", "model"]
            merged = matched.merge(
                coral[
                    join_columns
                    + [
                        "target_adaptation_group_count",
                        "target_evaluation_group_count",
                        "target_split_seed",
                    ]
                ],
                on=join_columns,
                how="inner",
                suffixes=("_src", "_coral"),
                validate="one_to_one",
            )
            self.assertEqual(len(merged), 24)
            self.assertTrue(
                (
                    merged["target_adaptation_group_count_src"]
                    == merged["target_adaptation_group_count_coral"]
                ).all()
            )
            self.assertTrue(
                (
                    merged["target_evaluation_group_count_src"]
                    == merged["target_evaluation_group_count_coral"]
                ).all()
            )
            self.assertTrue((merged["target_split_seed_src"] == merged["target_split_seed_coral"]).all())

            rerun = run_coral_auxiliary(config, outputs)
            self.assertEqual(rerun["summary_rows"], 48)
            rerun_frame = pd.read_csv(outputs.summary_csv)
            pd.testing.assert_frame_equal(frame, rerun_frame)


if __name__ == "__main__":
    unittest.main()
