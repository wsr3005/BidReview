from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

from bidagent.io_utils import write_jsonl
from bidagent.pipeline import annotate, ingest


def _create_minimal_docx(path: Path, paragraphs: list[str]) -> None:
    document_paragraphs = []
    for text in paragraphs:
        document_paragraphs.append(
            "<w:p><w:r><w:t>"
            + text
            + "</w:t></w:r></w:p>"
        )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(document_paragraphs)
        + "</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                "</Types>"
            ),
        )
        archive.writestr(
            "_rels/.rels",
            (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="word/document.xml"/>'
                "</Relationships>"
            ),
        )
        archive.writestr("word/document.xml", document_xml)


class AnnotateOutputTests(unittest.TestCase):
    def test_annotate_generates_docx_copy_with_notes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            out_dir = base / "out"
            out_dir.mkdir(parents=True, exist_ok=True)

            bid_docx = base / "bid.docx"
            tender_txt = base / "tender.txt"
            _create_minimal_docx(bid_docx, ["投标人必须提供营业执照", "其他内容"])
            tender_txt.write_text("商务要求：投标人必须提供营业执照。", encoding="utf-8")

            # create ingest manifest to allow annotate command to resolve source path
            ingest(
                tender_path=tender_txt,
                bid_path=bid_docx,
                out_dir=out_dir,
                resume=False,
            )

            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "risk",
                        "severity": "high",
                        "reason": "证据不足",
                        "evidence": [{"doc_id": "bid", "location": {"block_index": 1}}],
                    }
                ],
            )

            result = annotate(out_dir=out_dir, resume=False)
            self.assertEqual(result["annotations"], 1)
            self.assertTrue(result.get("annotated_copy"))

            annotated_path = Path(result["annotated_copy"])
            self.assertTrue(annotated_path.exists())
            with zipfile.ZipFile(annotated_path, "r") as archive:
                document_xml = archive.read("word/document.xml").decode("utf-8")
                comments_xml = archive.read("word/comments.xml").decode("utf-8")
                rels_xml = archive.read("word/_rels/document.xml.rels").decode("utf-8")
                self.assertIn("commentRangeStart", document_xml)
                self.assertIn("commentReference", document_xml)
                self.assertIn("R0001", comments_xml)
                self.assertIn("relationships/comments", rels_xml)

    def test_annotate_returns_warning_for_missing_bid_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            write_jsonl(
                out_dir / "findings.jsonl",
                [{"requirement_id": "R0001", "status": "fail", "severity": "high", "reason": "缺失"}],
            )
            result = annotate(out_dir=out_dir, resume=False)
            self.assertEqual(result["annotations"], 1)
            self.assertIsNone(result.get("annotated_copy"))
            self.assertTrue(result.get("annotation_warning"))

    @unittest.skipUnless(importlib.util.find_spec("pypdf") is not None, "pypdf is required")
    def test_annotate_generates_pdf_copy_with_annotations(self) -> None:
        from pypdf import PdfReader, PdfWriter

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            out_dir = base / "out"
            out_dir.mkdir(parents=True, exist_ok=True)

            bid_pdf = base / "bid.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=595, height=842)
            with bid_pdf.open("wb") as handle:
                writer.write(handle)

            manifest_path = out_dir / "ingest"
            manifest_path.mkdir(parents=True, exist_ok=True)
            (manifest_path / "manifest.json").write_text(
                json.dumps({"bid_path": str(bid_pdf)}, ensure_ascii=False),
                encoding="utf-8",
            )

            write_jsonl(
                out_dir / "findings.jsonl",
                [
                    {
                        "requirement_id": "R0001",
                        "status": "risk",
                        "severity": "high",
                        "reason": "证据不足",
                        "evidence": [{"doc_id": "bid", "location": {"page": 1, "block_index": 1}}],
                    }
                ],
            )

            result = annotate(out_dir=out_dir, resume=False)
            annotated_path = Path(result["annotated_copy"])
            self.assertTrue(annotated_path.exists())

            reader = PdfReader(str(annotated_path))
            annots = reader.pages[0].get("/Annots")
            self.assertIsNotNone(annots)


if __name__ == "__main__":
    unittest.main()
