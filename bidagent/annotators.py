from __future__ import annotations

from datetime import datetime, timezone
import xml.etree.ElementTree as etree
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from bidagent.document import split_text_blocks

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
XML_NS = "http://www.w3.org/XML/1998/namespace"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
COMMENTS_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
COMMENTS_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"


def _qname(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def build_issue_note(issue: dict) -> str:
    return (
        f"{issue.get('requirement_id', 'N/A')} "
        f"[{issue.get('severity', 'medium')}]: "
        f"{issue.get('reason', '')}"
    ).strip()


def _ensure_docx_rels(zin: zipfile.ZipFile) -> etree.Element:
    rels_path = "word/_rels/document.xml.rels"
    if rels_path in zin.namelist():
        return etree.fromstring(zin.read(rels_path))
    return etree.Element(_qname(PKG_REL_NS, "Relationships"))


def _next_rid(rels_root: etree.Element) -> str:
    max_id = 0
    for rel in rels_root.findall(_qname(PKG_REL_NS, "Relationship")):
        rel_id = rel.attrib.get("Id", "")
        if rel_id.startswith("rId"):
            suffix = rel_id[3:]
            if suffix.isdigit():
                max_id = max(max_id, int(suffix))
    return f"rId{max_id + 1}"


def _ensure_comments_rel(rels_root: etree.Element) -> None:
    for rel in rels_root.findall(_qname(PKG_REL_NS, "Relationship")):
        if rel.attrib.get("Type") == COMMENTS_REL_TYPE:
            return
    rel = etree.SubElement(rels_root, _qname(PKG_REL_NS, "Relationship"))
    rel.attrib["Id"] = _next_rid(rels_root)
    rel.attrib["Type"] = COMMENTS_REL_TYPE
    rel.attrib["Target"] = "comments.xml"


def _ensure_comments_part(zin: zipfile.ZipFile) -> etree.Element:
    comments_path = "word/comments.xml"
    if comments_path in zin.namelist():
        return etree.fromstring(zin.read(comments_path))
    return etree.Element(_qname(W_NS, "comments"))


def _ensure_content_types_override(zin: zipfile.ZipFile) -> etree.Element:
    if "[Content_Types].xml" not in zin.namelist():
        raise ValueError("Invalid DOCX file: missing [Content_Types].xml")
    root = etree.fromstring(zin.read("[Content_Types].xml"))
    for override in root.findall(_qname(CT_NS, "Override")):
        if override.attrib.get("PartName") == "/word/comments.xml":
            return root
    node = etree.SubElement(root, _qname(CT_NS, "Override"))
    node.attrib["PartName"] = "/word/comments.xml"
    node.attrib["ContentType"] = COMMENTS_CONTENT_TYPE
    return root


def _next_comment_id(comments_root: etree.Element) -> int:
    max_id = -1
    for item in comments_root.findall(_qname(W_NS, "comment")):
        value = item.attrib.get(_qname(W_NS, "id")) or item.attrib.get("id")
        if value is None:
            continue
        try:
            max_id = max(max_id, int(value))
        except ValueError:
            continue
    return max_id + 1


def _append_comment_entry(comments_root: etree.Element, comment_id: int, note: str) -> None:
    comment = etree.SubElement(comments_root, _qname(W_NS, "comment"))
    comment.attrib[_qname(W_NS, "id")] = str(comment_id)
    comment.attrib[_qname(W_NS, "author")] = "BidAgent"
    comment.attrib[_qname(W_NS, "initials")] = "BA"
    comment.attrib[_qname(W_NS, "date")] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    paragraph = etree.SubElement(comment, _qname(W_NS, "p"))
    run = etree.SubElement(paragraph, _qname(W_NS, "r"))
    text_node = etree.SubElement(run, _qname(W_NS, "t"))
    text_node.attrib[_qname(XML_NS, "space")] = "preserve"
    text_node.text = note


def _attach_comment_to_paragraph(paragraph: etree.Element, comment_id: int) -> None:
    run_nodes = paragraph.findall(_qname(W_NS, "r"))
    if not run_nodes:
        run = etree.SubElement(paragraph, _qname(W_NS, "r"))
        etree.SubElement(run, _qname(W_NS, "t")).text = " "
        run_nodes = [run]

    first_run = run_nodes[0]
    children = list(paragraph)
    first_run_index = children.index(first_run)

    start = etree.Element(_qname(W_NS, "commentRangeStart"))
    start.attrib[_qname(W_NS, "id")] = str(comment_id)
    paragraph.insert(first_run_index, start)

    end = etree.Element(_qname(W_NS, "commentRangeEnd"))
    end.attrib[_qname(W_NS, "id")] = str(comment_id)
    paragraph.append(end)

    ref_run = etree.Element(_qname(W_NS, "r"))
    rpr = etree.SubElement(ref_run, _qname(W_NS, "rPr"))
    style = etree.SubElement(rpr, _qname(W_NS, "rStyle"))
    style.attrib[_qname(W_NS, "val")] = "CommentReference"
    ref = etree.SubElement(ref_run, _qname(W_NS, "commentReference"))
    ref.attrib[_qname(W_NS, "id")] = str(comment_id)
    paragraph.append(ref_run)


def annotate_docx_copy(
    source_path: Path,
    output_path: Path,
    issues: Iterable[dict],
) -> dict:
    issues_by_block: dict[int, list[str]] = defaultdict(list)
    for issue in issues:
        location = (issue.get("target") or {}).get("location") or {}
        block_index = location.get("block_index")
        if isinstance(block_index, int) and block_index > 0:
            issues_by_block[block_index].append(build_issue_note(issue))

    if not issues_by_block:
        return {"annotated_notes": 0, "annotated_paragraphs": 0}

    with zipfile.ZipFile(source_path, "r") as zin:
        if "word/document.xml" not in zin.namelist():
            raise ValueError(f"Invalid DOCX file: {source_path}")

        document_root = etree.fromstring(zin.read("word/document.xml"))
        rels_root = _ensure_docx_rels(zin)
        comments_root = _ensure_comments_part(zin)
        content_types_root = _ensure_content_types_override(zin)
        _ensure_comments_rel(rels_root)

        block_to_paragraph: dict[int, etree.Element] = {}
        block_index = 0
        for paragraph in document_root.findall(".//w:p", NS):
            text_parts = []
            for text_node in paragraph.iterfind(".//w:t", NS):
                if text_node.text:
                    text_parts.append(text_node.text)
            merged = "".join(text_parts).strip()
            if not merged:
                continue
            for _ in split_text_blocks(merged):
                block_index += 1
                block_to_paragraph[block_index] = paragraph

        annotated_notes = 0
        annotated_paragraphs = 0
        touched: set[int] = set()
        next_comment_id = _next_comment_id(comments_root)

        for idx, notes in issues_by_block.items():
            paragraph = block_to_paragraph.get(idx)
            if paragraph is None:
                continue
            if idx not in touched:
                annotated_paragraphs += 1
                touched.add(idx)
            for note in notes:
                comment_id = next_comment_id
                next_comment_id += 1
                _attach_comment_to_paragraph(paragraph, comment_id)
                _append_comment_entry(comments_root, comment_id, note)
                annotated_notes += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            existing_files = set(zin.namelist())
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data = etree.tostring(document_root, encoding="utf-8", xml_declaration=True)
                elif item.filename == "word/comments.xml":
                    data = etree.tostring(comments_root, encoding="utf-8", xml_declaration=True)
                elif item.filename == "word/_rels/document.xml.rels":
                    data = etree.tostring(rels_root, encoding="utf-8", xml_declaration=True)
                elif item.filename == "[Content_Types].xml":
                    data = etree.tostring(content_types_root, encoding="utf-8", xml_declaration=True)
                zout.writestr(item, data)

            if "word/comments.xml" not in existing_files:
                zout.writestr(
                    "word/comments.xml",
                    etree.tostring(comments_root, encoding="utf-8", xml_declaration=True),
                )
            if "word/_rels/document.xml.rels" not in existing_files:
                zout.writestr(
                    "word/_rels/document.xml.rels",
                    etree.tostring(rels_root, encoding="utf-8", xml_declaration=True),
                )

    return {"annotated_notes": annotated_notes, "annotated_paragraphs": annotated_paragraphs}


def annotate_pdf_copy(
    source_path: Path,
    output_path: Path,
    issues: Iterable[dict],
) -> dict:
    try:
        from pypdf import PdfReader, PdfWriter
        from pypdf.annotations import Text
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PDF annotation requires optional dependency `pypdf`. Install with: pip install pypdf"
        ) from exc

    issues_by_page: dict[int, list[str]] = defaultdict(list)
    for issue in issues:
        location = (issue.get("target") or {}).get("location") or {}
        page = location.get("page")
        if isinstance(page, int) and page > 0:
            issues_by_page[page].append(build_issue_note(issue))

    if not issues_by_page:
        return {"annotated_notes": 0, "annotated_pages": 0}

    reader = PdfReader(str(source_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    annotated_notes = 0
    annotated_pages = 0
    for page_no, notes in issues_by_page.items():
        if page_no < 1 or page_no > len(writer.pages):
            continue
        page = writer.pages[page_no - 1]
        width = float(page.mediabox.right) - float(page.mediabox.left)
        height = float(page.mediabox.top) - float(page.mediabox.bottom)
        y_top = height - 36
        page_touched = False
        for idx, note in enumerate(notes):
            top = y_top - (idx * 28)
            if top < 48:
                break
            rect = (36, top - 20, min(width - 36, 340), top)
            annotation = Text(
                rect=rect,
                text=note,
                open=False,
            )
            writer.add_annotation(page_number=page_no - 1, annotation=annotation)
            annotated_notes += 1
            page_touched = True
        if page_touched:
            annotated_pages += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        writer.write(handle)

    return {"annotated_notes": annotated_notes, "annotated_pages": annotated_pages}
