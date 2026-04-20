#!/usr/bin/env python3
from __future__ import annotations

import argparse

from iot_ids.config import load_config
from iot_ids.validate import validate_claims_and_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate artifact provenance.")
    parser.add_argument("--config", required=True, help="Path to render YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_claims_and_results(
        config["render"]["claims_csv"],
        config["render"]["results_manifest"],
    )
    print("Provenance validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
