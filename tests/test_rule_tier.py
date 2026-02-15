from __future__ import annotations

import unittest

from bidagent.models import Block, Location
from bidagent.review import extract_requirements


class RuleTierTests(unittest.TestCase):
    def test_extracts_hard_fail_tier(self) -> None:
        blocks = [
            Block(
                doc_id="tender",
                text="商务要求：投标人必须提供营业执照，否则将被否决。",
                location=Location(block_index=1),
            )
        ]
        requirements = extract_requirements(blocks, focus="business")
        self.assertEqual(len(requirements), 1)
        self.assertEqual(requirements[0].rule_tier, "hard_fail")

    def test_extracts_scored_tier(self) -> None:
        blocks = [
            Block(
                doc_id="tender",
                text="评分项：提供类似业绩可加分，最高得分为5分。",
                location=Location(block_index=1),
            )
        ]
        requirements = extract_requirements(blocks, focus="business")
        self.assertEqual(len(requirements), 1)
        self.assertEqual(requirements[0].rule_tier, "scored")


if __name__ == "__main__":
    unittest.main()

