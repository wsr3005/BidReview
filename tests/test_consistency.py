from __future__ import annotations

import unittest

from bidagent.consistency import find_inconsistencies
from bidagent.models import Block, Location


class ConsistencyTests(unittest.TestCase):
    def test_detects_conflicting_bidder_name(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="投标人：北京为是科技有限公司",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="投标人名称：北京为是科技股份有限公司",
                location=Location(block_index=2, page=2, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="投标人：北京为是科技有限公司",
                location=Location(block_index=3, page=3, section="Normal"),
            ),
        ]
        findings = find_inconsistencies(blocks)
        types = {item.type for item in findings}
        self.assertIn("bidder_name", types)

    def test_does_not_flag_single_value(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="项目名称：芜湖某项目",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="项目名称：芜湖某项目",
                location=Location(block_index=2, page=2, section="Normal"),
            ),
        ]
        findings = find_inconsistencies(blocks)
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
