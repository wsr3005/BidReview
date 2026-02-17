from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bidagent.pipeline import run_pipeline


@unittest.skipUnless(importlib.util.find_spec("pypdf") is not None, "pypdf is required")
@unittest.skipUnless(importlib.util.find_spec("PIL") is not None, "Pillow is required")
class PipelinePdfOcrSmokeTests(unittest.TestCase):
    def test_run_pipeline_with_pdf_image_bid_and_mocked_ocr(self) -> None:
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            out_dir = base / "out"
            tender = base / "tender.txt"
            bid_pdf = base / "bid.pdf"
            image_path = base / "bid-image.png"

            tender.write_text(
                "\n".join(
                    [
                        "商务要求：投标人必须提供：营业执照。",
                        "商务要求：投标人应提供：类似项目业绩。",
                    ]
                ),
                encoding="utf-8",
            )
            image = Image.new("RGB", (220, 80), "white")
            draw = ImageDraw.Draw(image)
            draw.text((12, 25), "BUSINESS EVIDENCE", fill="black")
            image.save(image_path)
            image.save(bid_pdf, "PDF")

            with (
                patch.dict(os.environ, {"BIDAGENT_ALLOW_NO_AI": "1"}, clear=False),
                patch(
                    "bidagent.pipeline.ocr_selfcheck",
                    return_value={"mode": "auto", "engine": "mock", "engine_available": True},
                ),
                patch(
                    "bidagent.ocr.load_ocr_engine",
                    return_value=lambda _data: "我司已提供有效营业执照复印件。",
                ),
            ):
                result = run_pipeline(
                    tender_path=tender,
                    bid_path=bid_pdf,
                    out_dir=out_dir,
                    focus="business",
                    resume=False,
                    page_range=None,
                    ocr_mode="auto",
                    ai_provider=None,
                    ai_model="deepseek-chat",
                    ai_api_key_file=None,
                    ai_base_url="https://api.deepseek.com/v1",
                    ai_workers=1,
                    ai_min_confidence=0.65,
                )

            self.assertGreaterEqual(int(result["extract_req"]["requirements"]), 1)
            self.assertGreaterEqual(int(result["ingest"]["bid_ocr_blocks"]), 1)
            self.assertTrue((out_dir / "findings.jsonl").exists())
            self.assertTrue((out_dir / "review-report.md").exists())

            findings = [json.loads(line) for line in (out_dir / "findings.jsonl").read_text(encoding="utf-8").splitlines() if line]
            self.assertGreaterEqual(len(findings), 1)


if __name__ == "__main__":
    unittest.main()
