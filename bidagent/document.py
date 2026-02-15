from __future__ import annotations

import re
import xml.etree.ElementTree as etree
import zipfile
from pathlib import Path
from typing import Iterator

from bidagent.models import Block, Location

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def parse_page_range(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)-(\d+)", value.strip())
    if not match:
        raise ValueError("page_range must be in format START-END, e.g. 1-200")
    start = int(match.group(1))
    end = int(match.group(2))
    if start <= 0 or end < start:
        raise ValueError("invalid page range boundaries")
    return (start, end)


def split_text_blocks(text: str, max_chars: int = 1000) -> Iterator[str]:
    if not text:
        return
    chunks = re.split(r"(?:\r?\n){2,}", text)
    for chunk in chunks:
        normalized = re.sub(r"\s+", " ", chunk).strip()
        if not normalized:
            continue
        if len(normalized) <= max_chars:
            yield normalized
            continue
        for start in range(0, len(normalized), max_chars):
            part = normalized[start : start + max_chars].strip()
            if part:
                yield part


def iter_txt_blocks(path: Path, doc_id: str) -> Iterator[Block]:
    index = 0
    text = path.read_text(encoding="utf-8", errors="ignore")
    for chunk in split_text_blocks(text):
        index += 1
        yield Block(doc_id=doc_id, text=chunk, location=Location(block_index=index))


def iter_docx_blocks(path: Path, doc_id: str) -> Iterator[Block]:
    index = 0
    with zipfile.ZipFile(path) as archive:
        if "word/document.xml" not in archive.namelist():
            raise ValueError(f"invalid DOCX file: {path}")
        with archive.open("word/document.xml") as xml_file:
            for _, elem in etree.iterparse(xml_file, events=("end",)):
                if elem.tag != f"{{{W_NS}}}p":
                    continue
                text_parts = []
                for text_node in elem.iterfind(".//w:t", NS):
                    if text_node.text:
                        text_parts.append(text_node.text)
                merged = "".join(text_parts).strip()
                section = None
                style = elem.find(".//w:pStyle", NS)
                if style is not None:
                    section = style.attrib.get(f"{{{W_NS}}}val")
                elem.clear()
                if not merged:
                    continue
                for chunk in split_text_blocks(merged):
                    index += 1
                    yield Block(
                        doc_id=doc_id,
                        text=chunk,
                        location=Location(block_index=index, section=section),
                    )


def iter_pdf_blocks(path: Path, doc_id: str, page_range: tuple[int, int] | None) -> Iterator[Block]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PDF parsing requires optional dependency `pypdf`. "
            "Install with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    total = len(reader.pages)
    start_page = 1
    end_page = total
    if page_range:
        start_page, end_page = page_range
        end_page = min(end_page, total)
    index = 0
    for page_no in range(start_page, end_page + 1):
        page = reader.pages[page_no - 1]
        text = page.extract_text() or ""
        for chunk in split_text_blocks(text):
            index += 1
            yield Block(
                doc_id=doc_id,
                text=chunk,
                location=Location(block_index=index, page=page_no),
            )


def iter_document_blocks(
    path: Path,
    doc_id: str,
    page_range: tuple[int, int] | None = None,
) -> Iterator[Block]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        yield from iter_txt_blocks(path, doc_id)
        return
    if suffix == ".docx":
        yield from iter_docx_blocks(path, doc_id)
        return
    if suffix == ".pdf":
        yield from iter_pdf_blocks(path, doc_id, page_range)
        return
    raise ValueError(f"unsupported file type: {path.suffix}")

