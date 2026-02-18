from __future__ import annotations

import unittest

from bidagent.atomic import build_atomic_requirements
from bidagent.models import Requirement


class AtomicTests(unittest.TestCase):
    def test_build_atomic_requirements_splits_compound_hard_clause(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0001",
                text="投标人必须提供营业执照且提交授权委托书。",
                category="资质与证照",
                mandatory=True,
                keywords=["营业执照", "授权委托书"],
                constraints=[],
                rule_tier="hard_fail",
            )
        ]
        atomic_rows = build_atomic_requirements(requirements)
        self.assertGreaterEqual(len(atomic_rows), 2)
        self.assertTrue(all(row["classification"] == "hard" for row in atomic_rows))
        self.assertTrue(all(bool(row["engine_enabled"]) for row in atomic_rows))

    def test_build_atomic_requirements_marks_fluff_as_disabled(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0002",
                text="详见附件模板格式说明。",
                category="商务其他",
                mandatory=False,
                keywords=["模板", "格式"],
                constraints=[],
                rule_tier="general",
            )
        ]
        atomic_rows = build_atomic_requirements(requirements)
        self.assertEqual(len(atomic_rows), 1)
        self.assertEqual(atomic_rows[0]["classification"], "fluff")
        self.assertFalse(bool(atomic_rows[0]["engine_enabled"]))

    def test_build_atomic_requirements_prefers_constraint_and_numeric_evidence(self) -> None:
        requirements = [
            Requirement(
                requirement_id="R0003",
                text="投标保证金不少于50万元。",
                category="保证金与担保",
                mandatory=True,
                keywords=["保证金", "50万元"],
                constraints=[{"type": "amount", "field": "保证金", "op": ">=", "value": 500000, "unit": "元"}],
                rule_tier="hard_fail",
            )
        ]
        atomic_rows = build_atomic_requirements(requirements)
        self.assertEqual(len(atomic_rows), 1)
        self.assertIn(">=", atomic_rows[0]["constraint"])
        self.assertEqual(atomic_rows[0]["evidence_expectation"], "numeric_or_term_evidence")


if __name__ == "__main__":
    unittest.main()
