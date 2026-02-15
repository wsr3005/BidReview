from __future__ import annotations

import unittest

from bidagent.models import Block, Location
from bidagent.review import extract_requirements, review_requirements


class ReviewTests(unittest.TestCase):
    def test_extract_business_requirements_filters_technical(self) -> None:
        blocks = [
            Block(
                doc_id="tender",
                text="投标人必须提供有效营业执照和相关资质证明。",
                location=Location(block_index=1),
            ),
            Block(
                doc_id="tender",
                text="技术参数必须满足性能指标并提供算法说明。",
                location=Location(block_index=2),
            ),
        ]
        requirements = extract_requirements(blocks, focus="business")
        self.assertEqual(len(requirements), 1)
        self.assertIn("营业执照", requirements[0].text)

    def test_review_returns_fail_without_evidence(self) -> None:
        requirements = extract_requirements(
            [
                Block(
                    doc_id="tender",
                    text="投标人必须提供投标保证金缴纳凭证。",
                    location=Location(block_index=1),
                )
            ],
            focus="business",
        )
        bid_blocks = [
            Block(
                doc_id="bid",
                text="本次方案详细描述技术实现路径。",
                location=Location(block_index=1),
            )
        ]
        findings = review_requirements(requirements, bid_blocks)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].status, "fail")

    def test_extract_requirements_merges_duplicate_requirements(self) -> None:
        blocks = [
            Block(
                doc_id="tender",
                text="商务资质要求：投标人必须提供有效营业执照。",
                location=Location(block_index=1),
            ),
            Block(
                doc_id="tender",
                text="商务资质要求：投标人应提供有效营业执照。",
                location=Location(block_index=2),
            ),
        ]
        requirements = extract_requirements(blocks, focus="business")
        self.assertEqual(len(requirements), 1)
        self.assertEqual(requirements[0].source["merged_count"], 2)

    def test_review_marks_high_risk_for_weak_mandatory_evidence(self) -> None:
        requirements = extract_requirements(
            [
                Block(
                    doc_id="tender",
                    text="商务要求：投标人必须提供营业执照、开户许可证、纳税证明和社保证明。",
                    location=Location(block_index=1),
                )
            ],
            focus="business",
        )
        bid_blocks = [
            Block(
                doc_id="bid",
                text="本项目已附开户许可证复印件。",
                location=Location(block_index=1),
            )
        ]
        findings = review_requirements(requirements, bid_blocks)
        self.assertEqual(findings[0].status, "risk")
        self.assertEqual(findings[0].severity, "high")

    def test_extract_requirements_skips_catalog_and_template_noise(self) -> None:
        blocks = [
            Block(
                doc_id="tender",
                text="目录............12",
                location=Location(block_index=1),
            ),
            Block(
                doc_id="tender",
                text="第六章 投标文件格式",
                location=Location(block_index=2),
            ),
            Block(
                doc_id="tender",
                text="商务要求：投标文件格式要求详见附表模板并按样式填写。",
                location=Location(block_index=3),
            ),
            Block(
                doc_id="tender",
                text="商务要求：投标人必须提供有效营业执照。",
                location=Location(block_index=4),
            ),
        ]
        requirements = extract_requirements(blocks, focus="business")
        self.assertEqual(len(requirements), 1)
        self.assertIn("营业执照", requirements[0].text)

    def test_extract_requirements_splits_mixed_sentences(self) -> None:
        blocks = [
            Block(
                doc_id="tender",
                text="商务要求：投标人必须提供有效营业执照。技术参数必须满足性能指标。",
                location=Location(block_index=1),
            )
        ]
        requirements = extract_requirements(blocks, focus="business")
        self.assertEqual(len(requirements), 1)
        self.assertIn("营业执照", requirements[0].text)


if __name__ == "__main__":
    unittest.main()
