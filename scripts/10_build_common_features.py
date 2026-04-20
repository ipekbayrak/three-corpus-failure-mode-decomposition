#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from iot_ids.config import load_config
from iot_ids.experiment import prepare_feature_frames


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build common feature-schema files from staged datasets."
    )
    parser.add_argument("--config", required=True, help="Path to base YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    frames = prepare_feature_frames(config)
    processed_root = Path(config["paths"]["processed_root"])
    print(f"Wrote {len(frames)} processed feature files to {processed_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
