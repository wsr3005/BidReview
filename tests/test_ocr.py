from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bidagent.ocr import _resolve_tesseract_lang, iter_pdf_ocr_blocks, load_ocr_engine


class OcrBackendSelectionTests(unittest.TestCase):
    def test_load_engine_prefers_paddle_in_auto_mode(self) -> None:
        with (
            patch.dict(os.environ, {"BIDAGENT_OCR_BACKEND": "auto"}, clear=False),
            patch("bidagent.ocr._load_paddle_engine", return_value=lambda _data: "paddle"),
            patch("bidagent.ocr._load_tesseract_engine", return_value=lambda _data: "tesseract"),
        ):
            engine = load_ocr_engine("auto")
        self.assertIsNotNone(engine)
        self.assertEqual(engine(b"demo"), "paddle")

    def test_load_engine_fallbacks_to_tesseract_when_paddle_missing(self) -> None:
        with (
            patch.dict(os.environ, {"BIDAGENT_OCR_BACKEND": "auto"}, clear=False),
            patch("bidagent.ocr._load_paddle_engine", return_value=None),
            patch("bidagent.ocr._load_tesseract_engine", return_value=lambda _data: "tesseract"),
        ):
            engine = load_ocr_engine("auto")
        self.assertIsNotNone(engine)
        self.assertEqual(engine(b"demo"), "tesseract")

    def test_load_engine_honors_tesseract_mode(self) -> None:
        with (
            patch.dict(os.environ, {"BIDAGENT_OCR_BACKEND": "paddle"}, clear=False),
            patch("bidagent.ocr._load_paddle_engine", return_value=lambda _data: "paddle"),
            patch("bidagent.ocr._load_tesseract_engine", return_value=lambda _data: "tesseract"),
        ):
            engine = load_ocr_engine("tesseract")
        self.assertIsNotNone(engine)
        self.assertEqual(engine(b"demo"), "tesseract")

    def test_resolve_tesseract_lang_falls_back_when_chinese_pack_missing(self) -> None:
        with patch("bidagent.ocr._tesseract_available_langs", return_value={"eng", "osd"}):
            effective, meta = _resolve_tesseract_lang("tesseract", "chi_sim+eng")
        self.assertEqual(effective, "eng")
        self.assertTrue(bool(meta.get("lang_degraded")))


@unittest.skipUnless(importlib.util.find_spec("pypdf") is not None, "pypdf is required")
@unittest.skipUnless(importlib.util.find_spec("PIL") is not None, "Pillow is required")
class OcrPdfTests(unittest.TestCase):
    def test_iter_pdf_ocr_blocks_processes_embedded_images(self) -> None:
        from PIL import Image, ImageDraw

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            image_path = base / "evidence.png"
            bid_pdf = base / "bid.pdf"

            image = Image.new("RGB", (220, 80), "white")
            draw = ImageDraw.Draw(image)
            draw.text((12, 25), "BUSINESS EVIDENCE", fill="black")
            image.save(image_path)
            image.save(bid_pdf, "PDF")

            stats: dict[str, int] = {}
            with patch("bidagent.ocr._load_ocr_engines", return_value=[("mock", lambda _data: "营业执照扫描件")]):
                rows = list(
                    iter_pdf_ocr_blocks(
                        path=bid_pdf,
                        doc_id="bid",
                        start_index=0,
                        ocr_mode="auto",
                        stats=stats,
                        max_workers=1,
                    )
                )

            self.assertGreaterEqual(len(rows), 1)
            self.assertEqual(rows[0].location.section, "OCR_MEDIA")
            self.assertEqual(rows[0].location.page, 1)
            self.assertGreaterEqual(int(stats.get("images_total", 0)), 1)
            self.assertGreaterEqual(int(stats.get("images_succeeded", 0)), 1)
            self.assertEqual(int(stats.get("images_failed", 0)), 0)

    def test_iter_pdf_ocr_blocks_falls_back_to_secondary_engine(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            image_path = base / "evidence.png"
            bid_pdf = base / "bid.pdf"

            image = Image.new("RGB", (220, 80), "white")
            image.save(image_path)
            image.save(bid_pdf, "PDF")

            def _primary(_data: bytes) -> str:
                raise RuntimeError("primary failed")

            def _secondary(_data: bytes) -> str:
                return "营业执照扫描件"

            stats: dict[str, int] = {}
            with patch("bidagent.ocr._load_ocr_engines", return_value=[("paddle", _primary), ("tesseract", _secondary)]):
                rows = list(
                    iter_pdf_ocr_blocks(
                        path=bid_pdf,
                        doc_id="bid",
                        start_index=0,
                        ocr_mode="auto",
                        stats=stats,
                        max_workers=1,
                    )
                )

            self.assertGreaterEqual(len(rows), 1)
            self.assertGreaterEqual(int(stats.get("images_succeeded", 0)), 1)
            self.assertEqual(int(stats.get("images_failed", 0)), 0)
            self.assertIn("tesseract", (stats.get("backend_used_counts") or {}))
            self.assertIn("paddle:RuntimeError", (stats.get("backend_fallback_errors") or {}))


if __name__ == "__main__":
    unittest.main()
