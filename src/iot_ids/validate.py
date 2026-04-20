from __future__ import annotations

from pathlib import Path

import pandas as pd


def require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def validate_no_group_overlap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    group_column: str = "group_id",
) -> None:
    left_groups = set(left[group_column].astype(str))
    right_groups = set(right[group_column].astype(str))
    overlap = sorted(left_groups.intersection(right_groups))
    if overlap:
        raise ValueError(f"Group overlap detected for {group_column}: {overlap[:10]}")


def validate_claims_and_results(
    claims_path: str | Path,
    results_manifest_path: str | Path,
) -> None:
    claims = pd.read_csv(claims_path)
    results = pd.read_csv(results_manifest_path)
    require_columns(claims, ["claim_id", "artifact_id", "source_path"], "claims")
    require_columns(results, ["artifact_id", "path"], "results manifest")
    if claims["claim_id"].astype(str).duplicated().any():
        duplicates = (
            claims.loc[claims["claim_id"].astype(str).duplicated(), "claim_id"].astype(str).tolist()
        )
        raise ValueError(f"Claims contain duplicate claim_id values: {duplicates}")
    if results["artifact_id"].astype(str).duplicated().any():
        duplicates = (
            results.loc[
                results["artifact_id"].astype(str).duplicated(),
                "artifact_id",
            ]
            .astype(str)
            .tolist()
        )
        raise ValueError(f"Results manifest contains duplicate artifact_id values: {duplicates}")
    artifacts = set(results["artifact_id"].astype(str))
    missing = claims.loc[~claims["artifact_id"].astype(str).isin(artifacts), "artifact_id"].tolist()
    if missing:
        raise ValueError(f"Claims reference missing artifact ids: {missing}")
    manifest_paths = {
        str(row.artifact_id): str(row.path)
        for row in results[["artifact_id", "path"]].itertuples(index=False)
    }
    for path in results["path"].astype(str):
        if not Path(path).exists():
            raise FileNotFoundError(f"Artifact path in manifest does not exist: {path}")
    for row in claims.itertuples(index=False):
        source_path = str(row.source_path)
        if not Path(source_path).exists():
            raise FileNotFoundError(f"Claim source_path does not exist: {source_path}")
        manifest_path = manifest_paths[str(row.artifact_id)]
        if source_path != manifest_path:
            raise ValueError(
                "Claim source_path does not match results manifest path for "
                f"artifact_id={row.artifact_id}: {source_path} != {manifest_path}"
            )
