from __future__ import annotations

import unittest

from bidagent.verdict_policy import apply_cross_audit, counter_conflict_second_pass, requirement_needs_cross_verification


class VerdictPolicyTests(unittest.TestCase):
    def test_requirement_needs_cross_verification_is_not_triggered_by_general_clause(self) -> None:
        requirement = {"rule_tier": "general", "text": "投标人应按要求提交报价文件", "keywords": ["报价"]}
        self.assertFalse(requirement_needs_cross_verification(requirement))

    def test_counter_conflict_second_pass_requires_strong_dominant_counter(self) -> None:
        result = counter_conflict_second_pass(
            status="pass",
            reason="ok",
            confidence=0.92,
            task_packs=[
                {
                    "evidence_pack": [{"score": 10}],
                    "counter_evidence_pack": [{"score": 14, "excerpt": "该条款未提供对应证明", "matched_terms": []}],
                }
            ],
        )
        self.assertEqual(result["status"], "risk")
        self.assertTrue(result["audit"]["downgraded"])

    def test_counter_conflict_second_pass_keeps_pass_for_non_dominant_counter(self) -> None:
        result = counter_conflict_second_pass(
            status="pass",
            reason="ok",
            confidence=0.92,
            task_packs=[
                {
                    "evidence_pack": [{"score": 10}],
                    "counter_evidence_pack": [{"score": 12, "excerpt": "该条款未提供对应证明", "matched_terms": []}],
                }
            ],
        )
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["audit"]["downgraded"])

    def test_cross_audit_downgrades_only_for_high_risk_and_weak_single_channel(self) -> None:
        requirement = {"rule_tier": "hard_fail", "text": "资格审查不通过", "keywords": []}
        weak_support = [{"evidence_id": "E-1", "source_type": "text", "score": 3}]
        result = apply_cross_audit(
            requirement_id="R1",
            requirement=requirement,
            support_refs=weak_support,
            counter_refs=[],
            status="pass",
            reason="ok",
            confidence=0.9,
        )
        self.assertEqual(result["status"], "risk")
        self.assertEqual(result["cross_audit"]["action"], "downgrade_pass_to_risk_cross_verification")

        strong_support = [
            {"evidence_id": "E-1", "source_type": "text", "score": 9},
            {"evidence_id": "E-2", "source_type": "text", "score": 8},
        ]
        result_strong = apply_cross_audit(
            requirement_id="R1",
            requirement=requirement,
            support_refs=strong_support,
            counter_refs=[],
            status="pass",
            reason="ok",
            confidence=0.9,
        )
        self.assertEqual(result_strong["status"], "pass")
        self.assertIsNone(result_strong["cross_audit"]["action"])


if __name__ == "__main__":
    unittest.main()

