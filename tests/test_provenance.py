from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from iot_ids.validate import validate_claims_and_results


class ProvenanceTest(unittest.TestCase):
    def test_validate_claims_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "artifact.csv"
            artifact.write_text("x\n1\n", encoding="utf-8")
            claims_path = root / "claims.csv"
            results_path = root / "results.csv"
            pd.DataFrame(
                [
                    {
                        "claim_id": "c1",
                        "artifact_id": "a1",
                        "source_path": str(artifact),
                    }
                ]
            ).to_csv(claims_path, index=False)
            pd.DataFrame(
                [
                    {
                        "artifact_id": "a1",
                        "path": str(artifact),
                    }
                ]
            ).to_csv(results_path, index=False)
            validate_claims_and_results(claims_path, results_path)

    def test_validate_claims_and_results_rejects_source_path_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "artifact.csv"
            wrong_artifact = root / "wrong.csv"
            artifact.write_text("x\n1\n", encoding="utf-8")
            wrong_artifact.write_text("x\n2\n", encoding="utf-8")
            claims_path = root / "claims.csv"
            results_path = root / "results.csv"
            pd.DataFrame(
                [
                    {
                        "claim_id": "c1",
                        "artifact_id": "a1",
                        "source_path": str(wrong_artifact),
                    }
                ]
            ).to_csv(claims_path, index=False)
            pd.DataFrame(
                [
                    {
                        "artifact_id": "a1",
                        "path": str(artifact),
                    }
                ]
            ).to_csv(results_path, index=False)
            with self.assertRaisesRegex(ValueError, "does not match results manifest path"):
                validate_claims_and_results(claims_path, results_path)


if __name__ == "__main__":
    unittest.main()
