from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bidagent.io_utils import write_jsonl
from bidagent.models import Requirement
from bidagent.task_planner import decompose_requirement, ensure_review_tasks


class TaskPlannerTests(unittest.TestCase):
    def test_decompose_requirement_builds_base_and_constraint_tasks(self) -> None:
        requirement = Requirement(
            requirement_id="R0007",
            text="投标保证金不低于50万元，交付周期不超过30日。",
            category="business_other",
            mandatory=True,
            keywords=["投标保证金", "交付周期"],
            constraints=[
                {"type": "amount", "op": ">=", "value_fen": 50000000, "unit": "fen", "raw": "不低于50万元"},
                {"type": "term", "op": "<=", "value": 30, "unit": "日", "raw": "不超过30日"},
            ],
            rule_tier="hard_fail",
        )

        tasks = decompose_requirement(requirement)
        self.assertEqual(len(tasks), 4)
        self.assertEqual(tasks[0]["task_id"], "T-R0007-01")
        self.assertEqual(tasks[1]["task_id"], "T-R0007-02")
        self.assertEqual(tasks[2]["task_id"], "T-R0007-03")
        self.assertEqual(tasks[3]["task_id"], "T-R0007-04")
        self.assertEqual(tasks[0]["task_type"], "evidence_check")
        self.assertEqual(tasks[1]["task_type"], "keyword_check")
        self.assertEqual(tasks[2]["task_type"], "amount_check")
        self.assertEqual(tasks[3]["task_type"], "term_check")
        self.assertTrue(all(row["priority"] == "hard_fail" for row in tasks))
        self.assertTrue(all(row["requirement_id"] == "R0007" for row in tasks))

    def test_decompose_requirement_caps_task_count_at_six(self) -> None:
        constraints = [{"type": "quantity", "op": ">=", "value": index, "unit": "份"} for index in range(1, 10)]
        requirement = Requirement(
            requirement_id="R0008",
            text="至少提供若干资料。",
            category="business_other",
            mandatory=False,
            keywords=["提供", "资料"],
            constraints=constraints,
            rule_tier="general",
        )

        tasks = decompose_requirement(requirement)
        self.assertEqual(len(tasks), 6)
        self.assertEqual(tasks[0]["task_type"], "evidence_check")
        self.assertEqual(tasks[1]["task_type"], "keyword_check")
        self.assertEqual(tasks[-1]["task_id"], "T-R0008-06")

    def test_ensure_review_tasks_generates_and_reuses_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            requirements_path = base / "requirements.jsonl"
            review_tasks_path = base / "review-tasks.jsonl"

            write_jsonl(
                requirements_path,
                [
                    {
                        "requirement_id": "R0001",
                        "text": "必须提供营业执照。",
                        "category": "资质",
                        "mandatory": True,
                        "keywords": ["营业执照"],
                        "constraints": [],
                        "rule_tier": "hard_fail",
                    },
                    {
                        "requirement_id": "R0002",
                        "text": "项目交付周期不超过30日。",
                        "category": "履约",
                        "mandatory": True,
                        "keywords": ["交付周期"],
                        "constraints": [{"type": "term", "op": "<=", "value": 30, "unit": "日", "raw": "不超过30日"}],
                        "rule_tier": "general",
                    },
                ],
            )

            generated = ensure_review_tasks(requirements_path, review_tasks_path, resume=False)
            self.assertTrue(review_tasks_path.exists())
            self.assertGreaterEqual(len(generated), 4)

            write_jsonl(
                review_tasks_path,
                [
                    {
                        "task_id": "T-R9999-01",
                        "requirement_id": "R9999",
                        "task_type": "manual",
                        "query": "manual row",
                        "expected_logic": {"mode": "manual"},
                        "priority": "general",
                    }
                ],
            )

            reused = ensure_review_tasks(requirements_path, review_tasks_path, resume=True)
            self.assertEqual(len(reused), 1)
            self.assertEqual(reused[0]["task_id"], "T-R9999-01")

    def test_ensure_review_tasks_skips_fluff_requirements_with_atomic_map(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            requirements_path = base / "requirements.jsonl"
            requirements_atomic_path = base / "requirements.atomic.jsonl"
            review_tasks_path = base / "review-tasks.jsonl"

            write_jsonl(
                requirements_path,
                [
                    {
                        "requirement_id": "R0001",
                        "text": "详见附件模板格式说明。",
                        "category": "商务其他",
                        "mandatory": False,
                        "keywords": ["模板"],
                        "constraints": [],
                        "rule_tier": "general",
                    },
                    {
                        "requirement_id": "R0002",
                        "text": "必须提供营业执照。",
                        "category": "资质",
                        "mandatory": True,
                        "keywords": ["营业执照"],
                        "constraints": [],
                        "rule_tier": "hard_fail",
                    },
                ],
            )
            write_jsonl(
                requirements_atomic_path,
                [
                    {
                        "atomic_id": "R0001-A01",
                        "requirement_id": "R0001",
                        "classification": "fluff",
                        "engine_enabled": False,
                    },
                    {
                        "atomic_id": "R0002-A01",
                        "requirement_id": "R0002",
                        "classification": "hard",
                        "engine_enabled": True,
                    },
                ],
            )

            generated = ensure_review_tasks(
                requirements_path=requirements_path,
                requirements_atomic_path=requirements_atomic_path,
                review_tasks_path=review_tasks_path,
                resume=False,
            )
            self.assertTrue(generated)
            self.assertTrue(all(row["requirement_id"] == "R0002" for row in generated))


if __name__ == "__main__":
    unittest.main()

