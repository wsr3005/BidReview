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

    def test_total_price_ignores_unrelated_amounts(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="投标报价：100万元；服务费：200元",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="投标报价：100万元；服务费：300元",
                location=Location(block_index=2, page=2, section="Normal"),
            ),
        ]
        findings = find_inconsistencies(blocks)
        self.assertNotIn("bid_total_price_fen", {item.type for item in findings})

    def test_key_date_binds_to_date_key_value(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="合同签订时间：2024-01-01；投标日期：2024-03-01",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="合同签订时间：2024-02-01；投标日期：2024-03-01",
                location=Location(block_index=2, page=2, section="Normal"),
            ),
        ]
        findings = find_inconsistencies(blocks)
        self.assertNotIn("key_date", {item.type for item in findings})

    def test_uscc_requires_credit_code_key(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="设备序列号：91350211MA12345678",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="设备编码：91350211MA87654321",
                location=Location(block_index=2, page=2, section="Normal"),
            ),
        ]
        findings = find_inconsistencies(blocks)
        self.assertNotIn("uscc", {item.type for item in findings})

    def test_uscc_detects_conflict_when_labeled(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="统一社会信用代码：91350211MA12345678",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="统一社会信用代码：91350211MA87654321",
                location=Location(block_index=2, page=2, section="Normal"),
            ),
        ]
        findings = find_inconsistencies(blocks)
        self.assertIn("uscc", {item.type for item in findings})


if __name__ == "__main__":
    unittest.main()
