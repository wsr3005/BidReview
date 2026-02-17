from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bidagent.eval import evaluate_run, validate_gold_rows
from bidagent.io_utils import ensure_dir, write_jsonl


class EvalTests(unittest.TestCase):
    def test_evaluate_run_computes_hard_fail_recall(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            ensure_dir(run_dir / "eval")

            write_jsonl(
                run_dir / "eval" / "gold.jsonl",
                [
                    {"requirement_id": "R0001", "tier": "hard_fail", "expected_status": "fail"},
                    {"requirement_id": "R0002", "tier": "hard_fail", "expected_status": "fail"},
                    {"requirement_id": "R0003", "tier": "general", "expected_status": "pass"},
                ],
            )
            write_jsonl(
                run_dir / "findings.jsonl",
                [
                    {"requirement_id": "R0001", "status": "fail"},
                    {"requirement_id": "R0002", "status": "risk"},
                    {"requirement_id": "R0003", "status": "fail"},
                ],
            )

            out_path = run_dir / "eval" / "metrics.json"
            result = evaluate_run(run_dir, out_path=out_path)
            self.assertTrue(out_path.exists())
            payload = json.loads(out_path.read_text(encoding="utf-8"))

            metrics = payload["metrics"]
            self.assertEqual(metrics["hard_fail_total"], 2)
            self.assertEqual(metrics["hard_fail_fail"], 1)
            self.assertEqual(metrics["hard_fail_missed"], 1)
            self.assertAlmostEqual(metrics["hard_fail_recall"], 0.5)
            self.assertEqual(metrics["false_positive_fail"], 1)

    def test_validate_gold_rows_rejects_unknown_labels(self) -> None:
        rows = [
            {"requirement_id": "R0001", "tier": "hard_fail", "expected_status": "pass"},
            {"requirement_id": "R0002", "tier": "critical", "expected_status": "maybe"},
        ]
        with self.assertRaisesRegex(ValueError, "gold set validation failed"):
            validate_gold_rows(rows)

    def test_evaluate_run_rejects_invalid_gold_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            ensure_dir(run_dir / "eval")

            write_jsonl(
                run_dir / "eval" / "gold.jsonl",
                [
                    {"requirement_id": "R0001", "tier": "hard_fail", "expected_status": "fail"},
                    {"requirement_id": "R0002", "tier": "unknown_tier", "expected_status": "pass"},
                ],
            )
            write_jsonl(
                run_dir / "findings.jsonl",
                [
                    {"requirement_id": "R0001", "status": "fail"},
                    {"requirement_id": "R0002", "status": "pass"},
                ],
            )

            with self.assertRaisesRegex(ValueError, "gold set validation failed"):
                evaluate_run(run_dir)


if __name__ == "__main__":
    unittest.main()

