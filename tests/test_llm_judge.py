from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from bidagent.io_utils import read_jsonl
from bidagent.llm_judge import judge_tasks_with_llm, write_verdicts_jsonl


class _PassReviewer:
    provider = "deepseek"
    model = "deepseek-chat"

    def review_task(self, task: dict) -> dict:
        return {"status": "pass", "reason": "Evidence is sufficient.", "confidence": 0.94}


class _LowConfidenceReviewer:
    provider = "deepseek"
    model = "deepseek-chat"

    def review_task(self, task: dict) -> dict:
        return {"status": "pass", "reason": "Looks good.", "confidence": 0.22}


class _ErrorReviewer:
    provider = "deepseek"
    model = "deepseek-chat"

    def review_task(self, task: dict) -> dict:
        raise RuntimeError("network error")


class _InvalidStatusReviewer:
    provider = "deepseek"
    model = "deepseek-chat"

    def review_task(self, task: dict) -> dict:
        return {"status": "maybe", "reason": "Unknown", "confidence": 0.8}


class _CountCallsReviewer:
    provider = "deepseek"
    model = "deepseek-chat"

    def __init__(self) -> None:
        self.calls = 0

    def review_task(self, task: dict) -> dict:
        self.calls += 1
        return {"status": "pass", "reason": "ok", "confidence": 0.9}


class _SlowOrderedReviewer:
    provider = "deepseek"
    model = "deepseek-chat"

    def review_task(self, task: dict) -> dict:
        task_id = str(task.get("task_id") or "T0000")
        try:
            order = int(task_id.replace("T", ""))
        except ValueError:
            order = 0
        # Reverse latency to force out-of-order completion under parallel execution.
        time.sleep(max(0, 6 - order) * 0.01)
        return {"status": "pass", "reason": f"ok-{task_id}", "confidence": 0.91}


class LlmJudgeTests(unittest.TestCase):
    def test_judge_tasks_returns_llm_verdict_with_trace(self) -> None:
        tasks = [
            {
                "task_id": "T0001",
                "clause_id": "R0001",
                "rule_status": "risk",
                "rule_reason": "rule check",
                "evidence_refs": [{"evidence_id": "E-1", "location": {"page": 3, "block_index": 9}, "score": 2}],
            }
        ]

        verdicts = judge_tasks_with_llm(tasks, _PassReviewer())
        self.assertEqual(len(verdicts), 1)
        self.assertEqual(verdicts[0]["status"], "pass")
        self.assertIn("confidence", verdicts[0])
        self.assertIsInstance(verdicts[0].get("evidence_refs"), list)
        self.assertEqual((verdicts[0]["decision_trace"] or {}).get("decision", {}).get("source"), "llm")
        self.assertEqual((verdicts[0]["decision_trace"] or {}).get("llm_review", {}).get("provider"), "deepseek")
        self.assertIsInstance((verdicts[0]["decision_trace"] or {}).get("evidence_refs"), list)

    def test_judge_tasks_low_confidence_pass_downgrades_to_risk(self) -> None:
        tasks = [{"task_id": "T0001", "clause_id": "R0001", "rule_status": "risk", "rule_reason": "rule check"}]

        verdicts = judge_tasks_with_llm(tasks, _LowConfidenceReviewer(), min_confidence=0.65)
        self.assertEqual(verdicts[0]["status"], "risk")
        trace = verdicts[0]["decision_trace"] or {}
        self.assertEqual((trace.get("decision") or {}).get("source"), "llm_low_confidence_fallback")
        self.assertEqual((trace.get("low_confidence_fallback") or {}).get("action"), "downgrade_pass_to_risk")

    def test_judge_tasks_error_fallback_keeps_rule_status_and_trace(self) -> None:
        tasks = [{"task_id": "T0001", "clause_id": "R0001", "rule_status": "fail", "rule_reason": "missing evidence"}]

        verdicts = judge_tasks_with_llm(tasks, _ErrorReviewer())
        self.assertEqual(verdicts[0]["status"], "fail")
        trace = verdicts[0]["decision_trace"] or {}
        self.assertEqual((trace.get("decision") or {}).get("source"), "rule_fallback")
        self.assertEqual((trace.get("fallback") or {}).get("type"), "llm_error")
        self.assertIn("error", trace.get("llm_review", {}))

    def test_judge_tasks_legacy_insufficient_status_normalizes_to_missing(self) -> None:
        tasks = [
            {
                "task_id": "T0001",
                "clause_id": "R0001",
                "rule_status": "insufficient_evidence",
                "rule_reason": "缺少证明材料",
            }
        ]

        verdicts = judge_tasks_with_llm(tasks, _ErrorReviewer())
        self.assertEqual(verdicts[0]["status"], "missing")
        trace = verdicts[0]["decision_trace"] or {}
        self.assertEqual((trace.get("decision") or {}).get("source"), "rule_fallback")

    def test_judge_tasks_invalid_llm_status_fallback_has_trace(self) -> None:
        tasks = [{"task_id": "T0001", "clause_id": "R0001", "rule_status": "risk", "rule_reason": "rule check"}]

        verdicts = judge_tasks_with_llm(tasks, _InvalidStatusReviewer())
        self.assertEqual(verdicts[0]["status"], "risk")
        trace = verdicts[0]["decision_trace"] or {}
        self.assertEqual((trace.get("fallback") or {}).get("type"), "invalid_llm_status")
        self.assertEqual((trace.get("decision") or {}).get("source"), "rule_fallback")

    def test_judge_tasks_skips_llm_when_rule_marks_needs_ocr(self) -> None:
        reviewer = _CountCallsReviewer()
        tasks = [
            {
                "task_id": "T0001",
                "clause_id": "R0001",
                "rule_status": "needs_ocr",
                "rule_reason": "OCR required",
            }
        ]

        verdicts = judge_tasks_with_llm(tasks, reviewer)
        self.assertEqual(reviewer.calls, 0)
        self.assertEqual(verdicts[0]["status"], "needs_ocr")
        trace = verdicts[0]["decision_trace"] or {}
        self.assertEqual((trace.get("llm_review") or {}).get("skipped"), "needs_ocr")

    def test_judge_tasks_parallel_keeps_input_order(self) -> None:
        tasks = [
            {"task_id": "T0001", "clause_id": "R0001", "rule_status": "risk", "rule_reason": "rule check"},
            {"task_id": "T0002", "clause_id": "R0001", "rule_status": "risk", "rule_reason": "rule check"},
            {"task_id": "T0003", "clause_id": "R0001", "rule_status": "risk", "rule_reason": "rule check"},
            {"task_id": "T0004", "clause_id": "R0001", "rule_status": "risk", "rule_reason": "rule check"},
        ]

        verdicts = judge_tasks_with_llm(tasks, _SlowOrderedReviewer(), max_workers=4)
        self.assertEqual([row["task_id"] for row in verdicts], [row["task_id"] for row in tasks])
        self.assertTrue(all(row["status"] == "pass" for row in verdicts))

    def test_write_verdicts_jsonl_accepts_allowed_statuses_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "verdicts.jsonl"
            rows = [
                {
                    "task_id": "T0001",
                    "status": "pass",
                    "reason": "ok",
                    "confidence": 0.9,
                    "evidence_refs": [],
                    "decision_trace": {},
                },
                {
                    "task_id": "T0002",
                    "status": "needs_ocr",
                    "reason": "ocr",
                    "confidence": None,
                    "evidence_refs": [],
                    "decision_trace": {},
                },
                {
                    "task_id": "T0003",
                    "status": "missing",
                    "reason": "missing",
                    "confidence": 0.4,
                    "evidence_refs": [],
                    "decision_trace": {},
                },
            ]
            count = write_verdicts_jsonl(path, rows)
            loaded = list(read_jsonl(path))

        self.assertEqual(count, 3)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0]["status"], "pass")
        self.assertEqual(loaded[1]["status"], "needs_ocr")
        self.assertEqual(loaded[2]["status"], "missing")

    def test_write_verdicts_jsonl_rejects_unknown_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "verdicts.jsonl"
            rows = [
                {
                    "task_id": "T0001",
                    "status": "unknown",
                    "reason": "bad",
                    "confidence": 0.1,
                    "evidence_refs": [],
                    "decision_trace": {},
                }
            ]

            with self.assertRaises(ValueError):
                write_verdicts_jsonl(path, rows)

    def test_write_verdicts_jsonl_rejects_missing_protocol_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "verdicts.jsonl"
            rows = [{"task_id": "T0001", "status": "pass", "reason": "ok"}]

            with self.assertRaises(ValueError):
                write_verdicts_jsonl(path, rows)


if __name__ == "__main__":
    unittest.main()
