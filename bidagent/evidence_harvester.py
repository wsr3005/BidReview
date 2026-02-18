from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from bidagent.io_utils import write_jsonl

DEFAULT_TOP_K = 5
DEFAULT_COUNTER_K = 2

_TERM_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]{2,}")
_SPACE_PATTERN = re.compile(r"\s+")

_STOP_TERMS = {
    "check",
    "whether",
    "with",
    "this",
    "that",
    "the",
    "and",
    "for",
    "from",
    "task",
    "requirement",
    "evidence",
}

_SUPPORT_HINTS = {
    "provide",
    "provided",
    "submit",
    "submitted",
    "attach",
    "attached",
    "complies",
    "meet",
    "meets",
    "已提供",
    "提供",
    "已提交",
    "提交",
    "随附",
    "已附",
    "附上",
    "符合",
    "满足",
    "具备",
}

_COUNTER_HINTS = {
    "not",
    "no",
    "without",
    "missing",
    "lack",
    "failed",
    "fail",
    "reject",
    "invalid",
    "未提供",
    "未提交",
    "未附",
    "缺失",
    "缺少",
    "不满足",
    "不符合",
    "不具备",
    "无效",
    "作废",
    "驳回",
}

_STRONG_COUNTER_HINTS = {
    "not",
    "without",
    "missing",
    "invalid",
    "未提供",
    "未提交",
    "缺失",
    "缺少",
    "不满足",
    "不符合",
    "不具备",
    "无效",
    "作废",
    "驳回",
}

_REFERENCE_HINTS = {
    "appendix",
    "annex",
    "refer",
    "see attachment",
    "scan copy",
    "附件",
    "见附件",
    "详见附件",
    "附后",
    "扫描件",
    "复印件",
    "影印件",
}

_GENERIC_QUERY_TERMS = {
    "核验",
    "核对",
    "核查",
    "是否",
    "存在",
    "可定位",
    "证据",
    "支持",
    "该要求",
    "要求",
    "提供",
    "提交",
    "满足",
    "check",
    "whether",
    "requirement",
}

_FOCUS_SUFFIXES = (
    "营业执照",
    "执照",
    "资质",
    "证书",
    "授权书",
    "委托书",
    "法定代表人",
    "授权人",
    "保证金",
    "报价",
    "金额",
    "税率",
    "账号",
    "开户行",
    "编号",
    "名称",
    "日期",
    "期限",
    "合同",
    "发票",
    "业绩",
)


def _normalize_text(text: Any) -> str:
    return _SPACE_PATTERN.sub("", str(text or "")).lower()


def _extract_terms(value: Any, *, limit: int = 24) -> list[str]:
    def _token_variants(token: str) -> list[str]:
        variants = [token]
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            for size in (2, 3, 4):
                if len(token) <= size:
                    continue
                for start in range(0, len(token) - size + 1):
                    variants.append(token[start : start + size])
        return variants

    if isinstance(value, list):
        source = " ".join(str(item) for item in value if item is not None)
    else:
        source = str(value or "")

    terms: list[str] = []
    seen: set[str] = set()
    for match in _TERM_PATTERN.findall(source.lower()):
        for token in _token_variants(match):
            if token in _STOP_TERMS:
                continue
            if token in seen:
                continue
            seen.add(token)
            terms.append(token)
            if len(terms) >= limit:
                break
        if len(terms) >= limit:
            break
    return terms


def _extract_focus_phrases(value: Any, *, limit: int = 8) -> list[str]:
    source = str(value or "")
    phrases = re.findall(r"[\u4e00-\u9fff]{2,16}", source)
    merged: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        normalized = _normalize_text(phrase)
        if not normalized:
            continue
        if normalized in seen:
            continue
        if not any(suffix in phrase for suffix in _FOCUS_SUFFIXES):
            continue
        seen.add(normalized)
        merged.append(phrase)
        if len(merged) >= limit:
            break
    return merged


def _prune_positive_terms(terms: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = _normalize_text(term)
        if not normalized:
            continue
        if normalized in _GENERIC_QUERY_TERMS:
            continue
        if normalized in _STOP_TERMS:
            continue
        if len(normalized) < 2:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        merged.append(term)
    return merged


def _char_ngrams(text: str, *, n: int = 2) -> set[str]:
    if len(text) <= n:
        return {text} if text else set()
    return {text[index : index + n] for index in range(0, len(text) - n + 1)}


def _task_relevance(entry: dict[str, Any], positive_terms: list[str]) -> tuple[float, list[str]]:
    normalized_text = str(entry.get("normalized_text") or "")
    if not normalized_text:
        return 0.0, []
    anchors = [term for term in positive_terms if _normalize_text(term) not in _GENERIC_QUERY_TERMS]
    matched = [term for term in anchors if _normalize_text(term) and _normalize_text(term) in normalized_text]
    token_score = 0.0
    if anchors:
        token_score = len(matched) / max(1, min(len(anchors), 4))

    anchor_phrase = "".join(_normalize_text(term) for term in anchors[:4])
    phrase_score = 0.0
    if anchor_phrase:
        anchor_ngrams = _char_ngrams(anchor_phrase, n=2)
        text_ngrams = _char_ngrams(normalized_text, n=2)
        if anchor_ngrams and text_ngrams:
            phrase_score = len(anchor_ngrams & text_ngrams) / max(1, len(anchor_ngrams))

    relevance = max(0.0, min(1.0, token_score * 0.7 + phrase_score * 0.3))
    return relevance, matched


def _classify_block_type(section: Any) -> str:
    token = str(section or "").strip().lower()
    if "ocr" in token:
        return "ocr"
    if "table" in token:
        return "table"
    return "text"


def _normalize_location(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            "block_index": value.get("block_index"),
            "page": value.get("page"),
            "section": value.get("section"),
            "section_tag": value.get("section_tag"),
        }
    return {"block_index": None, "page": None, "section": None, "section_tag": None}


def _build_evidence_id(*, doc_id: str, location: dict[str, Any], index: int, block_id: str | None = None) -> str:
    if block_id:
        return f"E-{block_id}"
    page = location.get("page") if isinstance(location.get("page"), int) else 0
    block = location.get("block_index") if isinstance(location.get("block_index"), int) else index
    return f"E-{doc_id}-p{page}-b{block}"


def _build_excerpt_hash(text: str) -> str:
    digest = hashlib.sha1(_normalize_text(text).encode("utf-8", errors="ignore")).hexdigest()
    return digest[:16]


def _is_reference_only(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    has_reference = any(hint in normalized for hint in _REFERENCE_HINTS)
    if not has_reference:
        return False
    has_support_action = any(hint in normalized for hint in _SUPPORT_HINTS)
    return not has_support_action


def _iter_block_rows(blocks: Iterable[Any]) -> Iterable[dict[str, Any]]:
    for item in blocks:
        if isinstance(item, dict):
            location = _normalize_location(item.get("location"))
            yield {
                "doc_id": str(item.get("doc_id") or "bid"),
                "block_id": item.get("block_id"),
                "source_type": item.get("source_type"),
                "text": str(item.get("text") or ""),
                "location": location,
            }
            continue
        doc_id = getattr(item, "doc_id", "bid")
        block_id = getattr(item, "block_id", None)
        source_type = getattr(item, "block_type", None)
        text = getattr(item, "text", "")
        location_obj = getattr(item, "location", None)
        location = {
            "block_index": getattr(location_obj, "block_index", None),
            "page": getattr(location_obj, "page", None),
            "section": getattr(location_obj, "section", None),
            "section_tag": getattr(location_obj, "section_tag", None),
        }
        yield {
            "doc_id": str(doc_id or "bid"),
            "block_id": block_id,
            "source_type": source_type,
            "text": str(text or ""),
            "location": location,
        }


def build_evidence_index(blocks: Iterable[Any]) -> list[dict[str, Any]]:
    index_rows: list[dict[str, Any]] = []
    for offset, row in enumerate(_iter_block_rows(blocks), start=1):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        location = _normalize_location(row.get("location"))
        section = location.get("section")
        index_rows.append(
            {
                "evidence_id": _build_evidence_id(
                    doc_id=str(row.get("doc_id") or "bid"),
                    location=location,
                    index=offset,
                    block_id=str(row.get("block_id") or "").strip() or None,
                ),
                "excerpt_hash": _build_excerpt_hash(text),
                "doc_id": str(row.get("doc_id") or "bid"),
                "block_id": row.get("block_id"),
                "location": location,
                "block_type": _classify_block_type(section),
                "source_type": str(row.get("source_type") or _classify_block_type(section)),
                "section_tag": location.get("section_tag"),
                "text": text,
                "normalized_text": _normalize_text(text),
                "terms": _extract_terms(text, limit=64),
                "reference_only": _is_reference_only(text),
            }
        )
    return index_rows


def _collect_task_terms(task: dict[str, Any]) -> tuple[list[str], list[str]]:
    expected_logic = task.get("expected_logic")
    logic = expected_logic if isinstance(expected_logic, dict) else {}

    positive_terms: list[str] = []
    positive_terms.extend(_extract_terms(task.get("query"), limit=12))
    positive_terms.extend(_extract_focus_phrases(task.get("query"), limit=6))
    positive_terms.extend(_extract_terms(task.get("keywords"), limit=8))
    positive_terms.extend(_extract_terms(logic.get("keywords"), limit=8))
    positive_terms.extend(_extract_focus_phrases(logic.get("requirement_text"), limit=6))
    positive_terms.extend(_extract_terms(logic.get("requirement_text"), limit=8))

    positive_terms = _prune_positive_terms(positive_terms)
    merged_positive: list[str] = []
    seen_positive: set[str] = set()
    for term in positive_terms:
        if term in seen_positive:
            continue
        seen_positive.add(term)
        merged_positive.append(term)
    if not merged_positive:
        fallback_terms = _extract_focus_phrases(task.get("query"), limit=6)
        fallback_terms.extend(_extract_terms(task.get("query"), limit=8))
        merged_positive = _prune_positive_terms(fallback_terms)

    counter_terms: list[str] = []
    counter_terms.extend(_extract_terms(logic.get("counter_keywords"), limit=12))
    counter_terms.extend(sorted(_COUNTER_HINTS))

    merged_counter: list[str] = []
    seen_counter: set[str] = set()
    for term in counter_terms:
        if term in seen_counter:
            continue
        seen_counter.add(term)
        merged_counter.append(term)
    return merged_positive, merged_counter


def _support_score(
    entry: dict[str, Any],
    positive_terms: list[str],
    *,
    relevance_score: float,
    relevance_terms: list[str],
) -> tuple[int, list[str]]:
    normalized = str(entry.get("normalized_text") or "")
    matched = [term for term in positive_terms if _normalize_text(term) and _normalize_text(term) in normalized]
    if not matched:
        return 0, []
    if relevance_score < 0.2 and not relevance_terms:
        return 0, []
    score = len(matched) * 3
    has_counter_phrase = any(hint in normalized for hint in _COUNTER_HINTS)
    if (not has_counter_phrase) and any(hint in normalized for hint in _SUPPORT_HINTS):
        score += 2
    if entry.get("block_type") in {"ocr", "table"}:
        score += 1
    score += int(round(relevance_score * 3))
    if bool(entry.get("reference_only")):
        score = 0
    return score, matched


def _counter_score(
    entry: dict[str, Any],
    positive_terms: list[str],
    counter_terms: list[str],
    *,
    relevance_score: float,
    relevance_terms: list[str],
) -> tuple[int, list[str], list[str]]:
    normalized = str(entry.get("normalized_text") or "")
    if not relevance_terms:
        relevance_terms = [term for term in positive_terms if _normalize_text(term) and _normalize_text(term) in normalized]
    if not relevance_terms:
        return 0, [], []
    if relevance_score < 0.25:
        return 0, [], []

    matched_counter = [term for term in counter_terms if _normalize_text(term) and _normalize_text(term) in normalized]
    if not matched_counter:
        return 0, [], []
    score = len(matched_counter) * 3
    strong_hits = [term for term in matched_counter if term in _STRONG_COUNTER_HINTS]
    if strong_hits:
        score += 12
    score += int(round(relevance_score * 6))
    score += len(relevance_terms)
    deduped_counter = list(dict.fromkeys(matched_counter))
    return score, deduped_counter, relevance_terms


def _to_pack_item(
    entry: dict[str, Any],
    *,
    polarity: str,
    score: int,
    matched_terms: list[str],
    relevance_score: float,
) -> dict[str, Any]:
    return {
        "evidence_id": entry.get("evidence_id"),
        "block_id": entry.get("block_id"),
        "excerpt_hash": entry.get("excerpt_hash"),
        "doc_id": entry.get("doc_id"),
        "location": entry.get("location"),
        "score": score,
        "block_type": entry.get("block_type"),
        "source_type": entry.get("source_type"),
        "section_tag": entry.get("section_tag"),
        "reference_only": bool(entry.get("reference_only")),
        "polarity": polarity,
        "matched_terms": matched_terms,
        "relevance_score": round(float(relevance_score), 4),
        "excerpt": str(entry.get("text") or "")[:240],
    }


def _sort_pack(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda row: (-int(row.get("score") or 0), str(row.get("evidence_id") or "")))


def harvest_task_evidence(
    task: dict[str, Any],
    evidence_index: Iterable[dict[str, Any]],
    *,
    top_k: int = DEFAULT_TOP_K,
    counter_k: int = DEFAULT_COUNTER_K,
    min_support_score: int = 3,
    min_counter_score: int = 3,
) -> dict[str, Any]:
    support_limit = max(1, int(top_k))
    counter_limit = max(1, int(counter_k))
    positive_terms, counter_terms = _collect_task_terms(task)

    support_pack: list[dict[str, Any]] = []
    counter_pack: list[dict[str, Any]] = []
    candidate_blocks = 0
    reference_only_hits = 0
    relevant_blocks = 0
    relevance_threshold = 0.18

    for entry in evidence_index:
        if not isinstance(entry, dict):
            continue
        if not str(entry.get("text") or "").strip():
            continue
        candidate_blocks += 1
        if bool(entry.get("reference_only")):
            reference_only_hits += 1

        relevance_score, relevance_terms = _task_relevance(entry, positive_terms)
        if relevance_score >= relevance_threshold:
            relevant_blocks += 1

        support_score, support_terms = _support_score(
            entry,
            positive_terms,
            relevance_score=relevance_score,
            relevance_terms=relevance_terms,
        )
        counter_score, counter_terms_hit, counter_relevance = _counter_score(
            entry,
            positive_terms,
            counter_terms,
            relevance_score=relevance_score,
            relevance_terms=relevance_terms,
        )

        if counter_score >= min_counter_score and counter_score >= support_score:
            counter_pack.append(
                _to_pack_item(
                    entry,
                    polarity="counter",
                    score=counter_score,
                    matched_terms=list(dict.fromkeys(counter_terms_hit + counter_relevance)),
                    relevance_score=relevance_score,
                )
            )
            continue

        if support_score >= min_support_score:
            support_pack.append(
                _to_pack_item(
                    entry,
                    polarity="support",
                    score=support_score,
                    matched_terms=support_terms,
                    relevance_score=relevance_score,
                )
            )
            continue

        if counter_score >= min_counter_score:
            counter_pack.append(
                _to_pack_item(
                    entry,
                    polarity="counter",
                    score=counter_score,
                    matched_terms=list(dict.fromkeys(counter_terms_hit + counter_relevance)),
                    relevance_score=relevance_score,
                )
            )

    support_pack = _sort_pack(support_pack)[:support_limit]
    counter_pack = _sort_pack(counter_pack)[:counter_limit]
    evidence_refs = [
        {
            "evidence_id": item.get("evidence_id"),
            "block_id": item.get("block_id"),
            "excerpt_hash": item.get("excerpt_hash"),
            "doc_id": item.get("doc_id"),
            "location": item.get("location"),
            "source_type": item.get("source_type"),
            "section_tag": item.get("section_tag"),
            "score": item.get("score"),
        }
        for item in support_pack
    ]
    counter_refs = [
        {
            "evidence_id": item.get("evidence_id"),
            "block_id": item.get("block_id"),
            "excerpt_hash": item.get("excerpt_hash"),
            "doc_id": item.get("doc_id"),
            "location": item.get("location"),
            "source_type": item.get("source_type"),
            "section_tag": item.get("section_tag"),
            "score": item.get("score"),
        }
        for item in counter_pack
    ]
    return {
        "task_id": task.get("task_id"),
        "requirement_id": task.get("requirement_id"),
        "task_type": task.get("task_type"),
        "query": task.get("query"),
        "evidence_pack": support_pack,
        "counter_evidence_pack": counter_pack,
        "evidence_refs": evidence_refs,
        "counter_evidence_refs": counter_refs,
        "retrieval_trace": {
            "candidate_blocks": candidate_blocks,
            "positive_terms": positive_terms,
            "counter_terms": counter_terms,
            "reference_only_hits": reference_only_hits,
            "relevant_blocks": relevant_blocks,
            "relevance_threshold": relevance_threshold,
            "top_k": support_limit,
            "counter_k": counter_limit,
            "min_support_score": min_support_score,
            "min_counter_score": min_counter_score,
        },
    }


def harvest_task_evidence_packs(
    tasks: Iterable[dict[str, Any]],
    blocks: Iterable[Any],
    *,
    top_k: int = DEFAULT_TOP_K,
    counter_k: int = DEFAULT_COUNTER_K,
    min_support_score: int = 3,
    min_counter_score: int = 3,
) -> list[dict[str, Any]]:
    evidence_index = build_evidence_index(blocks)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        rows.append(
            harvest_task_evidence(
                task,
                evidence_index,
                top_k=top_k,
                counter_k=counter_k,
                min_support_score=min_support_score,
                min_counter_score=min_counter_score,
            )
        )
    return rows


def write_evidence_packs_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    for index, row in enumerate(rows, start=1):
        if not str(row.get("task_id") or "").strip():
            raise ValueError(f"Missing task_id at row {index}")
        if not isinstance(row.get("evidence_pack"), list):
            raise ValueError(f"Missing or invalid evidence_pack at row {index}")
        if not isinstance(row.get("counter_evidence_pack"), list):
            raise ValueError(f"Missing or invalid counter_evidence_pack at row {index}")
        if not isinstance(row.get("retrieval_trace"), dict):
            raise ValueError(f"Missing or invalid retrieval_trace at row {index}")
    return write_jsonl(path, rows)
