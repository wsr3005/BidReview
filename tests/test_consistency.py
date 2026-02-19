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

    def test_detects_tender_no_conflict_inside_bid(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="根据贵公司招标编号：25AT187076602200 招标，我方决定参加。",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="招标编号：25AT187076602232",
                location=Location(block_index=2, page=2, section="Normal"),
            ),
        ]
        findings = find_inconsistencies(blocks)
        by_type = {item.type: item for item in findings}
        self.assertIn("tender_no", by_type)
        tender_finding = by_type["tender_no"]
        self.assertEqual(tender_finding.status, "fail")
        self.assertEqual((tender_finding.comparison or {}).get("conclusion"), "不一致")

    def test_detects_authorized_representative_name_conflict(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="授权委托书 委托代理人：王五",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="授权委托书 委托代理人：赵六",
                location=Location(block_index=2, page=2, section="Normal"),
            ),
        ]
        findings = find_inconsistencies(blocks)
        by_type = {item.type: item for item in findings}
        self.assertIn("authorized_representative_name", by_type)
        self.assertEqual(by_type["authorized_representative_name"].status, "fail")

    def test_detects_placeholder_legal_representative_name(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="法定代表人身份证明",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="姓名：张三 性别：女",
                location=Location(block_index=2, page=1, section="Normal"),
            ),
        ]
        findings = find_inconsistencies(blocks)
        by_type = {item.type: item for item in findings}
        self.assertIn("legal_representative_name_placeholder", by_type)
        self.assertEqual(by_type["legal_representative_name_placeholder"].status, "risk")

    def test_detects_cross_doc_tender_no_mismatch(self) -> None:
        bid_blocks = [
            Block(
                doc_id="bid",
                text="根据贵公司招标编号：25AT187076602200 招标，我方决定参加。",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
        ]
        tender_blocks = [
            Block(
                doc_id="tender",
                text="项目编号：25AT187076602232",
                location=Location(block_index=10, page=3, section="Normal"),
            )
        ]
        findings = find_inconsistencies(bid_blocks, tender_blocks=tender_blocks)
        types = {item.type for item in findings}
        self.assertIn("tender_no_cross_doc", types)

    def test_detects_bidder_name_authorization_company_mismatch(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="投标人：北京为是科技有限公司",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="我方授权王某代表我方北京好玩科技有限公司(投标单位的名称)全权处理本项目投标事宜。",
                location=Location(block_index=2, page=1, section="Normal"),
            ),
        ]
        findings = find_inconsistencies(blocks)
        self.assertIn("bidder_name_authorization_mismatch", {item.type for item in findings})

    def test_detects_bank_receipt_unreadable_when_branch_missing(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="开 户 行 ：中国民生银行股份有限公司北京真我支行",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="账号：610820402",
                location=Location(block_index=2, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="CHINA MINSHENG BANK 账号610820402 电子回单",
                location=Location(block_index=100, page=None, section="OCR_MEDIA"),
            ),
        ]
        findings = find_inconsistencies(blocks)
        by_type = {item.type: item for item in findings}
        self.assertIn("account_bank_receipt_unreadable", by_type)
        self.assertIn(by_type["account_bank_receipt_unreadable"].status, {"needs_ocr", "risk"})

    def test_detects_account_number_conflict_inside_bid(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="开户账号：610820402",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="账号：610820499",
                location=Location(block_index=2, page=2, section="Normal"),
            ),
        ]
        findings = find_inconsistencies(blocks)
        by_type = {item.type: item for item in findings}
        self.assertIn("account_number", by_type)
        self.assertEqual(by_type["account_number"].status, "fail")
        self.assertEqual((by_type["account_number"].comparison or {}).get("conclusion"), "不一致")

    def test_ocr_bank_noise_does_not_trigger_internal_account_bank_conflict(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="开 户 行 ：中国民生银行股份有限公司北京真我支行",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="账号：610820402",
                location=Location(block_index=2, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="回单 开户行 中国民生银行股份有限公司北京真我支行方原流水 账号610820402",
                location=Location(block_index=100, page=None, section="OCR_MEDIA"),
            ),
        ]
        findings = find_inconsistencies(blocks)
        by_type = {item.type: item for item in findings}
        self.assertNotIn("account_bank", by_type)
        self.assertNotIn("account_bank_receipt_mismatch", by_type)

    def test_contract_bank_metadata_in_ocr_does_not_count_as_receipt(self) -> None:
        blocks = [
            Block(
                doc_id="bid",
                text="开 户 行 ：中国民生银行股份有限公司北京真我支行",
                location=Location(block_index=1, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="账号：610820402",
                location=Location(block_index=2, page=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text=(
                    "基于工业互联网平台开发服务合同：乙方开户行名称：招商银行股份有限公司北京首体科技金融支行，"
                    "账号：610820402，甲方按合同条款付款。"
                ),
                location=Location(block_index=120, page=None, section="OCR_MEDIA"),
            ),
        ]
        findings = find_inconsistencies(blocks)
        by_type = {item.type: item for item in findings}
        self.assertNotIn("account_bank_receipt_mismatch", by_type)
        self.assertNotIn("account_bank_receipt_unreadable", by_type)


if __name__ == "__main__":
    unittest.main()
