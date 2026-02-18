from __future__ import annotations

import unittest
from typing import Any

from bidagent.models import Block, Location, Requirement
from bidagent.review import (
    enforce_evidence_quality_gate,
    extract_requirements,
    extract_requirements_with_llm,
    review_requirements,
)


class ReviewTests(unittest.TestCase):
    def test_extract_requirements_with_llm_validates_schema(self) -> None:
        class _FakeExtractor:
            provider = "mock"
            model = "mock-model"
            prompt_version = "mock-v1"

            def extract_requirements(self, *, block_text: str, focus: str) -> list[dict[str, Any]]:
                return [
                    {
                        "text": "投标人必须提供有效营业执照复印件。",
                        "category": "资质与证照",
                        "mandatory": True,
                        "rule_tier": "hard_fail",
                        "keywords": ["营业执照", "复印件"],
                        "confidence": 0.91,
                    },
                    {
                        "text": "目录",
                        "category": "商务其他",
                        "mandatory": False,
                        "rule_tier": "general",
                        "keywords": ["目录"],
                        "confidence": 0.95,
                    },
                ]

        blocks = [
            Block(
                doc_id="tender",
                text="商务要求：投标人必须提供有效营业执照复印件。",
                location=Location(block_index=1),
            )
        ]
        requirements, stats = extract_requirements_with_llm(
            blocks,
            focus="business",
            extractor=_FakeExtractor(),
            min_confidence=0.6,
        )
        self.assertEqual(len(requirements), 1)
        self.assertEqual(requirements[0].rule_tier, "hard_fail")
        self.assertEqual((requirements[0].source.get("extraction") or {}).get("engine"), "llm_schema_validated")
        self.assertEqual(stats.get("items_raw"), 2)
        self.assertEqual(stats.get("items_accepted"), 1)
        self.assertEqual(stats.get("items_rejected"), 1)

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

    def test_review_returns_insufficient_without_evidence(self) -> None:
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
        self.assertEqual(findings[0].status, "insufficient_evidence")

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

    def test_review_marks_medium_risk_for_weak_mandatory_evidence(self) -> None:
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
        self.assertEqual(findings[0].severity, "medium")

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

    def test_extract_requirements_skips_non_bidder_process_clauses(self) -> None:
        blocks = [
            Block(
                doc_id="tender",
                text="评标委员会对满足招标文件要求的投标文件进行评分并推荐中标候选人。",
                location=Location(block_index=1),
            ),
            Block(
                doc_id="tender",
                text="按本章第2.2.4（1）目规定的评审因素和分值对商务部分计算出得分A。",
                location=Location(block_index=2),
            ),
            Block(
                doc_id="tender",
                text="中标人不能按要求提交履约担保的，视为放弃中标并承担赔偿责任。",
                location=Location(block_index=3),
            ),
            Block(
                doc_id="tender",
                text="商务要求：投标人必须提供有效营业执照复印件。",
                location=Location(block_index=4),
            ),
        ]
        requirements = extract_requirements(blocks, focus="business")
        self.assertEqual(len(requirements), 1)
        self.assertIn("营业执照", requirements[0].text)

    def test_review_ignores_catalog_heading_as_evidence(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0001",
                text="投标人必须提供授权委托书原件。",
                category="资质与证照",
                mandatory=True,
                keywords=["授权委托书", "原件"],
            )
        ]
        bid_blocks = [
            Block(
                doc_id="bid",
                text="5.3 授权委托书 ........ 45",
                location=Location(block_index=1, section="Heading1"),
            )
        ]
        findings = review_requirements(requirements, bid_blocks)
        self.assertEqual(findings[0].status, "insufficient_evidence")

    def test_review_accepts_real_content_not_heading(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0001",
                text="投标人必须提供授权委托书原件。",
                category="资质与证照",
                mandatory=True,
                keywords=["授权委托书", "原件"],
            )
        ]
        bid_blocks = [
            Block(
                doc_id="bid",
                text="我方已提交授权委托书原件并加盖公章。",
                location=Location(block_index=1, section="Normal"),
            )
        ]
        findings = review_requirements(requirements, bid_blocks)
        self.assertEqual(findings[0].status, "pass")

    def test_review_ignores_project_title_like_block(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0001",
                text="投标人必须提供保密承诺函。",
                category="有效期与响应",
                mandatory=True,
                keywords=["保密", "承诺函"],
            )
        ]
        bid_blocks = [
            Block(
                doc_id="bid",
                text="某某有限公司某项目投标文件",
                location=Location(block_index=1, section="Normal"),
            )
        ]
        findings = review_requirements(requirements, bid_blocks)
        self.assertEqual(findings[0].status, "insufficient_evidence")

    def test_review_marks_needs_ocr_for_reference_only_scan_evidence(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0001",
                text="投标人必须提供营业执照。",
                category="资质与证照",
                mandatory=True,
                keywords=["营业执照", "提供"],
            )
        ]
        bid_blocks = [
            Block(
                doc_id="bid",
                text="营业执照扫描件见附件1。",
                location=Location(block_index=1, section="Normal"),
            )
        ]
        findings = review_requirements(requirements, bid_blocks)
        self.assertEqual(findings[0].status, "needs_ocr")

    def test_review_does_not_force_needs_ocr_when_substantive_text_exists(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0001",
                text="投标人必须提供营业执照。",
                category="资质与证照",
                mandatory=True,
                keywords=["营业执照", "提供"],
            )
        ]
        bid_blocks = [
            Block(
                doc_id="bid",
                text="营业执照扫描件见附件1。",
                location=Location(block_index=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="我司已提供营业执照复印件并提交至商务文件。",
                location=Location(block_index=2, section="Normal"),
            ),
        ]
        findings = review_requirements(requirements, bid_blocks)
        self.assertEqual(findings[0].status, "pass")

    def test_review_marks_needs_ocr_when_scan_requirement_only_hits_list_items(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0001",
                text="投标人综合情况简介附营业执照扫描件等材料。",
                category="资质与证照",
                mandatory=True,
                keywords=["营业执照扫描件", "单位名单", "证明材料"],
            )
        ]
        bid_blocks = [
            Block(
                doc_id="bid",
                text="1、营业执照扫描件",
                location=Location(block_index=1, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="3、单位名单",
                location=Location(block_index=2, section="Normal"),
            ),
            Block(
                doc_id="bid",
                text="4、证明材料",
                location=Location(block_index=3, section="Normal"),
            ),
        ]
        findings = review_requirements(requirements, bid_blocks)
        self.assertEqual(findings[0].status, "needs_ocr")

    def test_review_accepts_ocr_media_short_evidence_blocks(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0001",
                text="投标人必须提供投标保证金缴纳凭证或银行回单。",
                category="保证金与担保",
                mandatory=True,
                keywords=["银行回单", "保证金"],
            )
        ]
        bid_blocks = [
            Block(
                doc_id="bid",
                text="银行回单",
                location=Location(block_index=1, section="OCR_MEDIA"),
            )
        ]
        findings = review_requirements(requirements, bid_blocks)
        self.assertTrue(findings[0].evidence)
        self.assertNotEqual(findings[0].status, "fail")

    def test_evidence_gate_downgrades_pass_when_only_reference_evidence(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0001",
                text="提供营业执照扫描件（见附件）",
                category="资质与证照",
                mandatory=False,
                keywords=["营业执照", "扫描件"],
                source={},
            )
        ]
        bid_blocks = [
            Block(
                doc_id="bid",
                text="营业执照扫描件见附件",
                location=Location(block_index=1, section="Normal"),
            )
        ]

        findings = review_requirements(requirements, bid_blocks)
        self.assertEqual(findings[0].status, "pass")

        gated = enforce_evidence_quality_gate(requirements=requirements, findings=findings, min_excerpt_len=10)
        self.assertEqual(gated[0].status, "needs_ocr")
        trace = gated[0].decision_trace or {}
        self.assertEqual((trace.get("evidence_gate") or {}).get("downgraded_to"), "needs_ocr")

    def test_evidence_gate_keeps_pass_when_non_reference_evidence_exists(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0001",
                text="提供营业执照复印件",
                category="资质与证照",
                mandatory=False,
                keywords=["营业执照", "复印件"],
                source={},
            )
        ]
        bid_blocks = [
            Block(
                doc_id="bid",
                text="本公司营业执照复印件如下，统一社会信用代码：9134XXXXXXXXXXXX。",
                location=Location(block_index=1, section="Normal"),
            )
        ]

        findings = review_requirements(requirements, bid_blocks)
        self.assertEqual(findings[0].status, "pass")
        gated = enforce_evidence_quality_gate(requirements=requirements, findings=findings, min_excerpt_len=10)
        self.assertEqual(gated[0].status, "pass")

    def test_review_outputs_traceable_clause_and_evidence(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0001",
                text="投标人必须提供有效营业执照。",
                category="资质与证照",
                mandatory=True,
                keywords=["营业执照", "提供"],
                source={"doc_id": "tender", "location": {"block_index": 7, "page": 2, "section": "Normal"}},
            )
        ]
        bid_blocks = [
            Block(
                doc_id="bid",
                text="我司已提供有效营业执照复印件。",
                location=Location(block_index=11, page=9, section="Normal"),
            )
        ]
        findings = review_requirements(requirements, bid_blocks)
        finding = findings[0]
        self.assertEqual(finding.clause_id, finding.requirement_id)
        self.assertIsNotNone(finding.decision_trace)
        self.assertEqual(finding.decision_trace["clause_id"], finding.requirement_id)
        self.assertEqual(finding.decision_trace["clause_source"]["location"]["block_index"], 7)
        self.assertTrue(finding.evidence[0]["evidence_id"].startswith("E-bid-"))
        self.assertIn("excerpt_hash", finding.evidence[0])
        self.assertEqual(
            finding.decision_trace["evidence_refs"][0]["evidence_id"],
            finding.evidence[0]["evidence_id"],
        )
        self.assertEqual(
            finding.decision_trace["evidence_refs"][0]["excerpt_hash"],
            finding.evidence[0]["excerpt_hash"],
        )
        self.assertEqual(finding.decision_trace["rule"]["version"], "r1-trace-v1")

    def test_review_keeps_trace_when_no_evidence(self) -> None:
        requirements = extract_requirements(
            [
                Block(
                    doc_id="tender",
                    text="商务要求：投标人必须提供投标保证金缴纳凭证。",
                    location=Location(block_index=3, page=1),
                )
            ],
            focus="business",
        )
        findings = review_requirements(requirements, bid_blocks=[])
        finding = findings[0]
        self.assertEqual(finding.status, "insufficient_evidence")
        self.assertIsNotNone(finding.decision_trace)
        self.assertEqual(finding.decision_trace["decision"]["top_score"], 0)
        self.assertEqual(finding.decision_trace["evidence_refs"], [])


if __name__ == "__main__":
    unittest.main()
