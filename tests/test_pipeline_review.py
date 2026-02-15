from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bidagent.io_utils import read_jsonl, write_jsonl
from bidagent.models import Finding
from bidagent.pipeline import review


class _DummyReviewer:
    provider = "deepseek"
    model = "deepseek-chat"

    def __init__(self, *args, **kwargs) -> None:
        pass


class PipelineReviewTests(unittest.TestCase):
    def test_resume_ai_partial_llm_triggers_refill(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            ingest_dir = out_dir / "ingest"
            ingest_dir.mkdir(parents=True, exist_ok=True)

            write_jsonl(
                out_dir / "requirements.jsonl",
                [
                    {"requirement_id": "R0001", "text": "必须提供营业执照", "mandatory": True},
                    {"requirement_id": "R0002", "text": "必须提供保证金", "mandatory": True},
                ],
            )
            write_jsonl(
                ingest_dir / "bid_blocks.jsonl",
                [{"doc_id": "bid", "text": "样例", "location": {"block_index": 1}}],
            )
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "risk",
                        "score": 1,
                        "severity": "medium",
                        "reason": "partial",
                        "llm": {"provider": "deepseek"},
                    },
                    {
                        "requirement_id": "R0002",
                        "status": "risk",
                        "score": 1,
                        "severity": "medium",
                        "reason": "partial",
                    },
                ],
            )

            fake_findings = [
                Finding("R0001", "risk", 1, "medium", "规则判定"),
                Finding("R0002", "risk", 1, "medium", "规则判定"),
            ]

            def _fake_apply(requirements, findings, reviewer, max_workers):
                for item in findings:
                    item.llm = {"provider": "deepseek", "model": "deepseek-chat"}
                return findings

            with (
                patch("bidagent.pipeline._load_api_key", return_value="sk-test"),
                patch("bidagent.pipeline.DeepSeekReviewer", _DummyReviewer),
                patch("bidagent.pipeline.review_requirements", return_value=fake_findings) as mocked_review,
                patch("bidagent.pipeline.apply_llm_review", side_effect=_fake_apply) as mocked_apply,
            ):
                result = review(out_dir=out_dir, resume=True, ai_provider="deepseek")

            self.assertEqual(result["findings"], 2)
            self.assertTrue(mocked_review.called)
            self.assertTrue(mocked_apply.called)
            rows = list(read_jsonl(out_dir / "findings.jsonl"))
            self.assertTrue(all((row.get("llm") or {}).get("provider") == "deepseek" for row in rows))

    def test_resume_ai_full_llm_skips_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "pass",
                        "score": 3,
                        "severity": "none",
                        "reason": "ok",
                        "llm": {"provider": "deepseek"},
                    },
                    {
                        "requirement_id": "R0002",
                        "status": "risk",
                        "score": 1,
                        "severity": "medium",
                        "reason": "check",
                        "llm": {"provider": "deepseek"},
                    },
                ],
            )

            with patch("bidagent.pipeline.review_requirements") as mocked_review:
                result = review(out_dir=out_dir, resume=True, ai_provider="deepseek")

            self.assertEqual(result["findings"], 2)
            self.assertFalse(mocked_review.called)


if __name__ == "__main__":
    unittest.main()
