from __future__ import annotations

import unittest

import pandas as pd

from iot_ids.experiment import (
    SplitFrames,
    _make_controlled_source_splits,
    _make_controlled_targets,
)


class ControlledShiftTest(unittest.TestCase):
    def test_controlled_shift_excludes_novel_groups_from_train(self) -> None:
        train = pd.DataFrame(
            {
                "record_id": ["b0", "s0", "n0", "n1"],
                "label": [0, 1, 1, 1],
                "family_group": [
                    "other",
                    "service_disruption",
                    "reconnaissance",
                    "command_control",
                ],
            }
        )
        val = pd.DataFrame({"record_id": ["v0"], "label": [0], "family_group": ["other"]})
        test = pd.DataFrame(
            {
                "record_id": [f"t{i}" for i in range(8)],
                "label": [0, 0, 1, 1, 1, 1, 1, 1],
                "family_group": [
                    "other",
                    "other",
                    "service_disruption",
                    "service_disruption",
                    "reconnaissance",
                    "reconnaissance",
                    "command_control",
                    "command_control",
                ],
            }
        )
        splits = SplitFrames(train=train, val=val, test=test)
        controlled = _make_controlled_source_splits(
            splits,
            ["reconnaissance", "command_control"],
        )

        attack_groups = set(controlled.train.loc[controlled.train["label"] == 1, "family_group"])
        self.assertEqual(attack_groups, {"service_disruption"})

        targets = _make_controlled_targets(
            "iot23",
            controlled,
            [0.0, 0.25, 0.5, 0.75, 1.0],
            ["service_disruption"],
            ["reconnaissance", "command_control"],
        )
        self.assertEqual(len(targets), 5)
        levels = [level for level, _, _ in targets]
        self.assertEqual(levels, [0.0, 0.25, 0.5, 0.75, 1.0])


if __name__ == "__main__":
    unittest.main()
