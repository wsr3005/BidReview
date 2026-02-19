from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable

_HEADING_STYLE_PATTERN = re.compile(r"(heading|title|标题)", flags=re.IGNORECASE)
_CHAPTER_HEADING_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百零0-9]+[章节编部分篇卷]\s*.*|[0-9]{1,2}(?:\.[0-9]{1,3}){1,4}\s*.+)$"
)
_CHAPTER_IN_TEXT_PATTERN = re.compile(r"第[一二三四五六七八九十百零0-9]+[章节编部分篇卷]")
_LONG_ANCHOR_HINT_PATTERN = re.compile(r"(投标人须知|评标办法|合同条款|技术规范|商务条款|资格审查)")
_TOC_LINE_PATTERN = re.compile(r"^(?P<title>.+?)(?:[\.·•…\-—]{2,}|\s{2,})(?P<page>\d{1,4})$")
_TABLE_DELIMITER_PATTERN = re.compile(r"[|｜│]")
_TABLE_COLUMN_HINT_PATTERN = re.compile(r"\S+\s{2,}\S+")
_TABLE_KEYWORD_PATTERN = re.compile(r"(序号|条款|项目|内容|参数|规格|数量|单位|偏离|响应)")
_SPACE_PATTERN = re.compile(r"\s+")


@dataclass(slots=True)
class _Anchor:
    title: str
    tag: str
    score: int
    block_index: int
    page: int | None
    section: str | None


def _utc_now_z() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_space(value: Any) -> str:
    return _SPACE_PATTERN.sub(" ", str(value or "")).strip()


def _normalize_title_key(value: Any) -> str:
    text = _normalize_space(value).lower()
    text = re.sub(r"^[第\s]*[一二三四五六七八九十百零0-9]+[章节编部分篇卷]\s*", "", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", text)
    return text


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _extract_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        location = row.get("location") if isinstance(row.get("location"), dict) else {}
        text = _normalize_space(row.get("text"))
        block_type = str(row.get("block_type") or "").strip().lower()
        if block_type not in {"text", "ocr", "table"}:
            raw_text = str(row.get("text") or "")
            delimiter_hits = len(_TABLE_DELIMITER_PATTERN.findall(raw_text))
            has_tab = "\t" in raw_text
            spaced_columns = bool(_TABLE_COLUMN_HINT_PATTERN.search(raw_text))
            table_keyword = bool(_TABLE_KEYWORD_PATTERN.search(raw_text))
            if delimiter_hits >= 2 or has_tab or (spaced_columns and table_keyword):
                block_type = "table"
            else:
                block_type = "text"
        normalized_rows.append(
            {
                "doc_id": str(row.get("doc_id") or ""),
                "text": text,
                "block_type": block_type,
                "location": {
                    "block_index": _safe_int(location.get("block_index")),
                    "page": _safe_int(location.get("page")),
                    "section": _normalize_space(location.get("section")) or None,
                },
            }
        )
    return normalized_rows


def _classify_semantic_tag(title: str) -> str:
    normalized = _normalize_space(title)
    compact = re.sub(r"\s+", "", normalized)
    if any(token in compact for token in ("评标", "评审", "评分", "否决", "废标", "资格审查")):
        return "evaluation_risk"
    if any(token in compact for token in ("合同", "付款", "结算", "违约", "质保", "交货", "履约", "商务")):
        return "business_contract"
    if any(token in compact for token in ("技术", "参数", "规格", "性能", "接口", "配置", "方案")):
        return "technical_spec"
    if any(token in compact for token in ("投标人须知", "须知前附表", "投标须知")):
        return "bidder_instruction"
    if any(token in compact for token in ("格式", "附件", "附表", "模板", "封面")):
        return "format_appendix"
    return "other"


def _build_anchor_candidates(rows: list[dict[str, Any]]) -> list[_Anchor]:
    anchors: list[_Anchor] = []
    for row in rows:
        text = str(row.get("text") or "")
        if not text:
            continue
        location = row.get("location") if isinstance(row.get("location"), dict) else {}
        block_type = str(row.get("block_type") or "text")
        section = _normalize_space(location.get("section")) or None
        compact = re.sub(r"\s+", "", text)
        prefix = compact[:240]
        chapter_in_text = bool(_CHAPTER_IN_TEXT_PATTERN.search(compact))
        long_heading_hint = bool(chapter_in_text or _LONG_ANCHOR_HINT_PATTERN.search(prefix))
        if len(compact) > 220 and not long_heading_hint:
            continue
        if _TOC_LINE_PATTERN.match(compact):
            # TOC lines are hints, not physical heading anchors.
            continue

        score = 0
        if section and _HEADING_STYLE_PATTERN.search(section):
            score += 4
        if _CHAPTER_HEADING_PATTERN.match(compact):
            score += 4
        if long_heading_hint:
            score += 3
        if block_type == "table" and long_heading_hint:
            score += 2
        if compact.startswith("第") and any(token in compact[:8] for token in ("章", "节", "篇", "编")):
            score += 2
        if compact.startswith("目录"):
            score += 1
        if len(compact) > 120 and score < 3:
            continue
        if score <= 0:
            continue

        anchors.append(
            _Anchor(
                title=text,
                tag=_classify_semantic_tag(text),
                score=score,
                block_index=int(location.get("block_index") or 0),
                page=_safe_int(location.get("page")),
                section=section,
            )
        )
    anchors.sort(key=lambda item: (item.block_index, item.page or 0))
    return anchors


def _extract_toc_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    toc_entries: list[dict[str, Any]] = []
    first_page_limit = 10
    for row in rows:
        text = str(row.get("text") or "")
        if not text:
            continue
        location = row.get("location") if isinstance(row.get("location"), dict) else {}
        page = _safe_int(location.get("page"))
        block_index = int(location.get("block_index") or 0)
        if page is not None and page > first_page_limit:
            continue
        if page is None and block_index > 160:
            continue
        match = _TOC_LINE_PATTERN.match(re.sub(r"\s+", "", text))
        if not match:
            continue
        title = _normalize_space(match.group("title"))
        logical_page = _safe_int(match.group("page"))
        if not title or logical_page is None:
            continue
        toc_entries.append({"title": title, "logical_page": logical_page})
    return toc_entries


def _estimate_page_offset(anchors: list[_Anchor], toc_entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not anchors or not toc_entries:
        return {"value": None, "votes": 0, "samples": [], "toc_entries": len(toc_entries), "matched_entries": 0}

    anchor_keys: list[tuple[_Anchor, str]] = [(_anchor, _normalize_title_key(_anchor.title)) for _anchor in anchors]
    offsets: list[tuple[int, str, str, int, int]] = []
    matched = 0
    for toc in toc_entries:
        key = _normalize_title_key(toc.get("title"))
        logical_page = _safe_int(toc.get("logical_page"))
        if not key or logical_page is None:
            continue
        best_anchor: _Anchor | None = None
        for anchor, anchor_key in anchor_keys:
            if anchor.page is None:
                continue
            if anchor_key == key:
                best_anchor = anchor
                break
            if len(key) >= 4 and (key in anchor_key or anchor_key in key):
                best_anchor = anchor
                break
        if best_anchor is None or best_anchor.page is None:
            continue
        offset = int(best_anchor.page) - logical_page
        matched += 1
        offsets.append((offset, toc.get("title", ""), best_anchor.title, logical_page, int(best_anchor.page)))

    if not offsets:
        return {"value": None, "votes": 0, "samples": [], "toc_entries": len(toc_entries), "matched_entries": matched}

    counts = Counter(offset for offset, *_ in offsets)
    value, votes = counts.most_common(1)[0]
    samples = [
        {
            "offset": offset,
            "toc_title": toc_title,
            "anchor_title": anchor_title,
            "logical_page": logical_page,
            "physical_page": physical_page,
        }
        for offset, toc_title, anchor_title, logical_page, physical_page in offsets[:5]
    ]
    return {
        "value": int(value),
        "votes": int(votes),
        "samples": samples,
        "toc_entries": len(toc_entries),
        "matched_entries": int(matched),
    }


def _build_sections(anchors: list[_Anchor], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    max_block = max(int((row.get("location") or {}).get("block_index") or 0) for row in rows)
    max_page_candidates = [_safe_int((row.get("location") or {}).get("page")) for row in rows]
    max_page = max((item for item in max_page_candidates if item is not None), default=None)

    if not anchors:
        return [
            {
                "section_id": "S-001",
                "title": "全文",
                "semantic_tag": "other",
                "anchor_score": 0,
                "range": {
                    "start_block": 1,
                    "end_block": max_block,
                    "start_page": 1 if max_page is not None else None,
                    "end_page": max_page,
                },
            }
        ]

    sections: list[dict[str, Any]] = []
    for index, anchor in enumerate(anchors, start=1):
        next_anchor = anchors[index] if index < len(anchors) else None
        end_block = max_block
        if next_anchor is not None:
            end_block = max(anchor.block_index, next_anchor.block_index - 1)
        end_page = max_page
        if next_anchor is not None and next_anchor.page is not None and anchor.page is not None:
            end_page = max(anchor.page, next_anchor.page - 1)
        sections.append(
            {
                "section_id": f"S-{index:03d}",
                "title": anchor.title,
                "semantic_tag": anchor.tag,
                "anchor_score": anchor.score,
                "range": {
                    "start_block": anchor.block_index,
                    "end_block": end_block,
                    "start_page": anchor.page,
                    "end_page": end_page,
                },
            }
        )
    return sections


def _build_doc_entry(*, doc_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    anchors = _build_anchor_candidates(rows)
    toc_entries = _extract_toc_entries(rows)
    page_offset = _estimate_page_offset(anchors, toc_entries)
    sections = _build_sections(anchors, rows)

    pages = sorted(
        {
            page
            for page in (_safe_int((row.get("location") or {}).get("page")) for row in rows)
            if page is not None
        }
    )
    block_type_counts = dict(Counter(str(row.get("block_type") or "text") for row in rows))
    return {
        "doc_id": doc_id,
        "total_blocks": len(rows),
        "total_pages": len(pages),
        "block_type_counts": block_type_counts,
        "anchors": [
            {
                "anchor_id": f"A-{index:03d}",
                "title": anchor.title,
                "semantic_tag": anchor.tag,
                "score": anchor.score,
                "location": {
                    "block_index": anchor.block_index,
                    "page": anchor.page,
                    "section": anchor.section,
                },
            }
            for index, anchor in enumerate(anchors, start=1)
        ],
        "sections": sections,
        "page_offset": page_offset,
    }


def build_ingest_doc_map(*, tender_rows: Iterable[dict[str, Any]], bid_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    tender = _extract_rows(tender_rows)
    bid = _extract_rows(bid_rows)
    return {
        "schema_version": "doc-map-v1",
        "generated_at": _utc_now_z(),
        "docs": [
            _build_doc_entry(doc_id="tender", rows=tender),
            _build_doc_entry(doc_id="bid", rows=bid),
        ],
    }
