from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

SOURCE_TEXT = "text"
SOURCE_TABLE = "table"
SOURCE_OCR = "ocr"
SUPPORTED_SOURCE_TYPES = {SOURCE_TEXT, SOURCE_TABLE, SOURCE_OCR}

_SOURCE_TYPE_ALIASES = {
    "text": SOURCE_TEXT,
    "plain_text": SOURCE_TEXT,
    "paragraph": SOURCE_TEXT,
    "table": SOURCE_TABLE,
    "tabular": SOURCE_TABLE,
    "spreadsheet": SOURCE_TABLE,
    "ocr": SOURCE_OCR,
    "ocr_image": SOURCE_OCR,
    "ocr-media": SOURCE_OCR,
    "ocr_media": SOURCE_OCR,
    "image_ocr": SOURCE_OCR,
}

_TABLE_QUERY_HINTS = {
    "金额",
    "单价",
    "总价",
    "合计",
    "税率",
    "报价",
    "清单",
}

_OCR_QUERY_HINTS = {
    "扫描件",
    "复印件",
    "影印件",
    "附件",
    "ocr",
}

_QUERY_STOP_TERMS = {
    "核验",
    "核对",
    "是否",
    "存在",
    "可定位",
    "证据",
    "支持",
    "要求",
    "该要求",
    "检查",
    "请",
    "task",
    "requirement",
    "evidence",
    "check",
    "whether",
}

_SECTION_TAG_ALIASES = {
    "evaluationrisk": "evaluation_risk",
    "businesscontract": "business_contract",
    "technicalspec": "technical_spec",
    "bidderinstruction": "bidder_instruction",
    "formatappendix": "format_appendix",
    "other": "other",
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_compact(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", (text or "").lower())


def _normalize_source_type(value: Any) -> str | None:
    token = str(value or "").strip().lower().replace(" ", "_")
    if not token:
        return None
    return _SOURCE_TYPE_ALIASES.get(token)


def _normalize_section_tag(value: Any) -> str | None:
    token = _normalize_compact(str(value or ""))
    if not token:
        return None
    if token in _SECTION_TAG_ALIASES:
        return _SECTION_TAG_ALIASES[token]
    if any(key in token for key in ("评标", "评审", "评分", "否决", "废标", "资格审查")):
        return "evaluation_risk"
    if any(key in token for key in ("合同", "付款", "结算", "违约", "质保", "交货", "履约", "商务")):
        return "business_contract"
    if any(key in token for key in ("技术", "参数", "规格", "性能", "接口", "配置", "方案")):
        return "technical_spec"
    if any(key in token for key in ("投标人须知", "须知前附表", "投标须知", "须知")):
        return "bidder_instruction"
    if any(key in token for key in ("格式", "附件", "附表", "模板", "封面")):
        return "format_appendix"
    return "other"


def _looks_like_table_text(text: str) -> bool:
    snippet = str(text or "")
    if not snippet:
        return False
    # Common table-like fragments from markdown/TSV/CSV style extraction.
    if snippet.count("|") >= 2:
        return True
    if "\t" in snippet:
        return True
    if snippet.count(",") >= 2 and re.search(r"[0-9]", snippet):
        return True
    if re.search(r"(?:\S+\s*[:：]\s*\S+\s*){3,}", snippet):
        return True
    return False


def infer_source_type(row: dict[str, Any]) -> str:
    explicit = _normalize_source_type(row.get("source_type")) or _normalize_source_type(row.get("block_type"))
    if explicit:
        return explicit

    location = row.get("location")
    section = ""
    if isinstance(location, dict):
        section = str(location.get("section") or "")
    if not section:
        section = str(row.get("section") or "")
    lowered_section = section.lower()
    if "ocr" in lowered_section:
        return SOURCE_OCR
    if "table" in lowered_section:
        return SOURCE_TABLE

    if _looks_like_table_text(str(row.get("text") or "")):
        return SOURCE_TABLE
    return SOURCE_TEXT


def _safe_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _build_excerpt_hash(text: str) -> str:
    compact = _normalize_compact(text)[:2000]
    digest = hashlib.sha1(compact.encode("utf-8", errors="ignore")).hexdigest()
    return digest[:16]


def _sanitize_doc_id(value: Any) -> str:
    doc_id = str(value or "bid")
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]", "", doc_id)
    return cleaned or "bid"


def _build_base_evidence_id(
    *,
    doc_id: str,
    page: int | None,
    block_index: int | None,
    source_type: str,
    block_id: str | None = None,
) -> str:
    if block_id:
        return f"E-{block_id}-{source_type}"
    page_token = page if page is not None else 0
    block_token = block_index if block_index is not None else 0
    return f"E-{doc_id}-p{page_token}-b{block_token}-{source_type}"


def build_unified_evidence_index(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    index_rows: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue

        text = _normalize_whitespace(str(row.get("text") or ""))
        if not text:
            continue

        location = row.get("location")
        location_dict = dict(location) if isinstance(location, dict) else {}
        block_index = _safe_positive_int(location_dict.get("block_index"))
        page = _safe_positive_int(location_dict.get("page"))
        section = str(location_dict.get("section") or row.get("section") or "").strip() or None
        section_tag = (
            _normalize_section_tag(row.get("section_tag"))
            or _normalize_section_tag(location_dict.get("section_tag"))
            or _normalize_section_tag(section)
        )

        doc_id = _sanitize_doc_id(row.get("doc_id"))
        block_id = str(row.get("block_id") or "").strip() or None
        source_type = infer_source_type(row)
        base_id = _build_base_evidence_id(
            doc_id=doc_id,
            page=page,
            block_index=block_index,
            source_type=source_type,
            block_id=block_id,
        )
        seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
        evidence_id = base_id if seen_ids[base_id] == 1 else f"{base_id}-n{seen_ids[base_id]}"

        location_out: dict[str, Any] = {}
        if block_index is not None:
            location_out["block_index"] = block_index
        if page is not None:
            location_out["page"] = page
        if section:
            location_out["section"] = section

        index_rows.append(
            {
                "evidence_id": evidence_id,
                "block_id": block_id,
                "doc_id": doc_id,
                "source_type": source_type,
                "section_tag": section_tag,
                "location": location_out,
                "excerpt": text[:240],
                "excerpt_hash": _build_excerpt_hash(text),
                "text": text,
                "char_count": len(text),
            }
        )

    return index_rows


def _extract_query_terms(text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[0-9A-Za-z\u4e00-\u9fff]{2,}", text or ""):
        normalized = _normalize_compact(token)
        if not normalized:
            continue
        if normalized in _QUERY_STOP_TERMS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        terms.append(token)
    return terms


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in (_normalize_compact(item) for item in re.findall(r"[0-9A-Za-z\u4e00-\u9fff]{2,}", text or ""))
        if token and token not in _QUERY_STOP_TERMS
    }


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    compact = _normalize_compact(text)
    if len(compact) <= n:
        return {compact} if compact else set()
    return {compact[index : index + n] for index in range(0, len(compact) - n + 1)}


def _semantic_similarity(query: str, text: str, terms: list[str]) -> float:
    query_tokens = _token_set(query)
    text_tokens = _token_set(text)
    token_overlap = 0.0
    if query_tokens and text_tokens:
        token_overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens))

    key_phrase = "".join(_normalize_compact(term) for term in terms[:4])
    if not key_phrase:
        key_phrase = _normalize_compact(query)
    phrase_overlap = 0.0
    if key_phrase:
        query_grams = _char_ngrams(key_phrase, n=2)
        text_grams = _char_ngrams(text, n=2)
        if query_grams and text_grams:
            phrase_overlap = len(query_grams & text_grams) / max(1, len(query_grams))

    return max(0.0, min(1.0, token_overlap * 0.6 + phrase_overlap * 0.4))


def _score_candidate(
    candidate: dict[str, Any],
    *,
    query: str,
    terms: list[str],
    preferred_section_tags: set[str] | None,
) -> float:
    text = str(candidate.get("text") or "")
    compact_text = _normalize_compact(text)
    compact_query = _normalize_compact(query)
    source_type = str(candidate.get("source_type") or SOURCE_TEXT)

    overlap = 0
    for term in terms:
        if _normalize_compact(term) and _normalize_compact(term) in compact_text:
            overlap += 1

    phrase_bonus = 0.0
    if compact_query and compact_query in compact_text:
        phrase_bonus = 1.0

    source_bonus = 0.0
    if source_type == SOURCE_TABLE and any(token in query for token in _TABLE_QUERY_HINTS):
        source_bonus = 0.3
    if source_type == SOURCE_OCR and any(token in query.lower() for token in _OCR_QUERY_HINTS):
        source_bonus = 0.3
    section_bonus = 0.0
    if preferred_section_tags:
        section_tag = _normalize_section_tag(candidate.get("section_tag"))
        if section_tag in preferred_section_tags:
            section_bonus = 0.45
        elif section_tag and section_tag != "other":
            section_bonus = 0.05

    length_bonus = min(len(text), 300) / 1000.0
    semantic_bonus = _semantic_similarity(query, text, terms) * 2.5
    return float(overlap) + phrase_bonus + source_bonus + section_bonus + length_bonus + semantic_bonus


def retrieve_evidence_candidates(
    index_rows: Iterable[dict[str, Any]],
    query: str,
    *,
    top_k: int = 5,
    source_types: Iterable[str] | None = None,
    preferred_section_tags: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    normalized_query = _normalize_whitespace(query)
    if not normalized_query:
        return []

    allowed_types: set[str] | None = None
    if source_types is not None:
        normalized_types = {
            token
            for token in (_normalize_source_type(value) for value in source_types)
            if token in SUPPORTED_SOURCE_TYPES
        }
        allowed_types = normalized_types

    section_tags: set[str] | None = None
    if preferred_section_tags is not None:
        normalized_tags = {
            tag
            for tag in (_normalize_section_tag(value) for value in preferred_section_tags)
            if tag is not None
        }
        section_tags = normalized_tags or None

    terms = _extract_query_terms(normalized_query)
    scored: list[dict[str, Any]] = []
    for row in index_rows:
        if not isinstance(row, dict):
            continue
        source_type = str(row.get("source_type") or SOURCE_TEXT)
        if allowed_types is not None and source_type not in allowed_types:
            continue
        score = _score_candidate(
            row,
            query=normalized_query,
            terms=terms,
            preferred_section_tags=section_tags,
        )
        if score <= 0:
            continue
        candidate = dict(row)
        candidate["score"] = round(score, 6)
        scored.append(candidate)

    scored.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            len(str(item.get("excerpt") or "")),
            str(item.get("evidence_id") or ""),
        ),
        reverse=True,
    )
    return scored[:top_k]


# Backward-friendly alias used by downstream lanes.
build_evidence_index = build_unified_evidence_index

