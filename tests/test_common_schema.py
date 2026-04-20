from __future__ import annotations

import unittest

import pandas as pd

from iot_ids.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, build_feature_schema_v1


class CommonSchemaTest(unittest.TestCase):
    def test_build_feature_schema_v1(self) -> None:
        frame = pd.DataFrame(
            {
                "duration_s": [1.0, 2.0],
                "src_bytes": [10.0, 20.0],
                "dst_bytes": [5.0, 10.0],
                "src_pkts": [2.0, 4.0],
                "dst_pkts": [1.0, 2.0],
                "proto": ["tcp", "udp"],
            }
        )
        featured = build_feature_schema_v1(frame)
        self.assertEqual(featured["feature_schema_version"].iloc[0], FEATURE_SCHEMA_VERSION)
        for column in FEATURE_COLUMNS:
            self.assertIn(column, featured.columns)
        self.assertAlmostEqual(featured["proto_tcp"].iloc[0], 1.0)
        self.assertAlmostEqual(featured["proto_udp"].iloc[1], 1.0)


if __name__ == "__main__":
    unittest.main()
