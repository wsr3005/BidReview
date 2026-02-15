from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import json

from bidagent.io_utils import read_jsonl, write_jsonl
from bidagent.models import Block, Finding, Location
from bidagent.pipeline import ingest, review


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

            def _fake_apply(requirements, findings, reviewer, max_workers, **_kwargs):
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

    def test_ingest_appends_ocr_blocks_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            tender = base / "tender.txt"
            bid = base / "bid.txt"
            out_dir = base / "out"
            tender.write_text("商务要求：投标人必须提供营业执照。", encoding="utf-8")
            bid.write_text("我司已提交相关商务文件。", encoding="utf-8")

            fake_ocr_blocks = [
                Block(
                    doc_id="bid",
                    text="营业执照统一社会信用代码：9134XXXXXXXXXX",
                    location=Location(block_index=2, section="OCR_MEDIA"),
                )
            ]
            with (
                patch("bidagent.pipeline.iter_document_ocr_blocks", return_value=iter(fake_ocr_blocks)),
                patch(
                    "bidagent.pipeline.ocr_selfcheck",
                    return_value={"mode": "auto", "engine": "tesseract", "engine_available": True},
                ),
            ):
                result = ingest(
                    tender_path=tender,
                    bid_path=bid,
                    out_dir=out_dir,
                    resume=False,
                    ocr_mode="auto",
                )

            self.assertEqual(result["bid_ocr_blocks"], 1)
            self.assertIn("ocr", result)
            self.assertTrue((result.get("ocr") or {}).get("engine_available"))

            manifest = json.loads((out_dir / "ingest" / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("ocr", manifest)
            self.assertEqual((manifest.get("ocr") or {}).get("mode"), "auto")
            self.assertEqual(result["bid_blocks"], 2)
            rows = list(read_jsonl(out_dir / "ingest" / "bid_blocks.jsonl"))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["location"]["section"], "OCR_MEDIA")


if __name__ == "__main__":
    unittest.main()
