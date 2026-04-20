from __future__ import annotations

import unittest

import pandas as pd

from iot_ids.experiment import split_by_group


class SplitTest(unittest.TestCase):
    def test_split_by_group_is_deterministic_and_disjoint(self) -> None:
        frame = pd.DataFrame(
            {
                "group_id": [f"g{i}" for i in range(10) for _ in range(2)],
                "label": [0, 1] * 10,
            }
        )
        first = split_by_group(frame, train_frac=0.6, val_frac=0.2, seed=7)
        second = split_by_group(frame, train_frac=0.6, val_frac=0.2, seed=7)
        self.assertEqual(set(first.train["group_id"]), set(second.train["group_id"]))
        self.assertTrue(set(first.train["group_id"]).isdisjoint(set(first.val["group_id"])))
        self.assertTrue(set(first.train["group_id"]).isdisjoint(set(first.test["group_id"])))


if __name__ == "__main__":
    unittest.main()
