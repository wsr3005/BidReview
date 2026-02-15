from __future__ import annotations

import unittest

from bidagent.llm import apply_llm_review
from bidagent.models import Finding, Requirement


class _FakeReviewer:
    provider = "deepseek"
    model = "deepseek-chat"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def review(self, requirement: Requirement, finding: Finding) -> dict:
        if self.fail:
            raise RuntimeError("network error")
        return {
            "status": "pass",
            "severity": "none",
            "reason": "LLM确认合规",
            "confidence": 0.93,
        }


class LlmReviewTests(unittest.TestCase):
    def test_apply_llm_review_overrides_rule_result(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0001",
                text="必须提供营业执照",
                category="资质与证照",
                mandatory=True,
                keywords=["营业执照"],
            )
        ]
        findings = [
            Finding(
                requirement_id="R0001",
                status="risk",
                score=1,
                severity="high",
                reason="规则判定弱证据",
                evidence=[{"excerpt": "已附营业执照"}],
            )
        ]

        result = apply_llm_review(requirements, findings, _FakeReviewer())
        self.assertEqual(result[0].status, "pass")
        self.assertEqual(result[0].severity, "none")
        self.assertIsNotNone(result[0].llm)
        self.assertEqual(result[0].llm["provider"], "deepseek")

    def test_apply_llm_review_fallback_on_error(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0001",
                text="必须提供营业执照",
                category="资质与证照",
                mandatory=True,
                keywords=["营业执照"],
            )
        ]
        findings = [
            Finding(
                requirement_id="R0001",
                status="risk",
                score=1,
                severity="high",
                reason="规则判定弱证据",
            )
        ]

        result = apply_llm_review(requirements, findings, _FakeReviewer(fail=True))
        self.assertEqual(result[0].status, "risk")
        self.assertEqual(result[0].severity, "high")
        self.assertIn("error", result[0].llm)

    def test_apply_llm_review_preserves_input_order_in_parallel(self) -> None:
        class _OrderReviewer:
            provider = "deepseek"
            model = "deepseek-chat"

            def review(self, requirement: Requirement, finding: Finding) -> dict:
                if requirement.requirement_id == "R0002":
                    return {
                        "status": "fail",
                        "severity": "high",
                        "reason": "不符合",
                        "confidence": 0.8,
                    }
                return {
                    "status": "pass",
                    "severity": "none",
                    "reason": "符合",
                    "confidence": 0.9,
                }

        requirements = [
            Requirement(
                requirement_id="R0001",
                text="必须提供营业执照",
                category="资质与证照",
                mandatory=True,
                keywords=["营业执照"],
            ),
            Requirement(
                requirement_id="R0002",
                text="必须提供保证金",
                category="保证金与担保",
                mandatory=True,
                keywords=["保证金"],
            ),
        ]
        findings = [
            Finding(requirement_id="R0001", status="risk", score=1, severity="medium", reason="待确认"),
            Finding(requirement_id="R0002", status="risk", score=1, severity="medium", reason="待确认"),
        ]

        result = apply_llm_review(requirements, findings, _OrderReviewer(), max_workers=8)
        self.assertEqual([item.requirement_id for item in result], ["R0001", "R0002"])
        self.assertEqual(result[0].status, "pass")
        self.assertEqual(result[1].status, "fail")

    def test_apply_llm_review_skips_needs_ocr_findings(self) -> None:
        class _ShouldNotCallReviewer:
            provider = "deepseek"
            model = "deepseek-chat"

            def review(self, requirement: Requirement, finding: Finding) -> dict:  # pragma: no cover - defensive
                raise AssertionError("needs_ocr finding should not call reviewer")

        requirements = [
            Requirement(
                requirement_id="R0001",
                text="必须提供营业执照",
                category="资质与证照",
                mandatory=True,
                keywords=["营业执照"],
            )
        ]
        findings = [
            Finding(
                requirement_id="R0001",
                status="needs_ocr",
                score=1,
                severity="medium",
                reason="仅命中扫描件引用",
                evidence=[{"excerpt": "营业执照扫描件见附件"}],
            )
        ]

        result = apply_llm_review(requirements, findings, _ShouldNotCallReviewer())
        self.assertEqual(result[0].status, "needs_ocr")
        self.assertEqual(result[0].severity, "medium")
        self.assertEqual((result[0].llm or {}).get("skipped"), "needs_ocr")


if __name__ == "__main__":
    unittest.main()
