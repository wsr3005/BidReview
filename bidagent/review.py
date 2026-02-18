from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import asdict
from typing import Any, Iterable, Protocol

from bidagent.constraints import extract_constraints
from bidagent.models import Block, Finding, Requirement

REVIEW_RULE_ENGINE = "keyword_match"
REVIEW_RULE_VERSION = "r1-trace-v1"

BUSINESS_KEYWORDS = [
    "商务",
    "报价",
    "资质",
    "业绩",
    "保证金",
    "有效期",
    "付款",
    "合同",
    "偏离",
    "授权",
    "承诺",
    "交付",
    "交货",
    "发票",
    "税",
    "售后",
]

TECHNICAL_HINTS = [
    "技术参数",
    "技术方案",
    "性能",
    "架构",
    "算法",
    "接口协议",
    "源码",
]

MANDATORY_HINTS = ["必须", "应", "须", "不得", "严禁", "要求"]
MANDATORY_STRONG_HINTS = ["必须", "应", "须", "不得", "严禁"]
SCORE_HINTS = ["评分", "得分", "加分", "扣分", "分值", "评分标准"]

TIER_ORDER = {"general": 0, "scored": 1, "hard_fail": 2}

HARD_FAIL_HINTS = [
    "否决",
    "废标",
    "无效投标",
    "不响应",
    "将被否决",
    "将被废标",
    "视为无效",
    "否则",
]


def infer_rule_tier(text: str, *, mandatory: bool) -> str:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return "general"
    if any(token in compact for token in SCORE_HINTS):
        return "scored"
    if any(token in compact for token in HARD_FAIL_HINTS):
        return "hard_fail"
    if mandatory and re.search(r"(不满足|未提供|未提交|未响应).{0,12}(将被|视为|否则)", compact):
        return "hard_fail"
    return "general"

STOP_WORDS = {
    "投标",
    "投标人",
    "招标",
    "文件",
    "要求",
    "内容",
    "进行",
    "相关",
    "必须",
    "应当",
    "条款",
    "商务",
    "提供",
    "提交",
    "满足",
    "符合",
    "承诺",
    "声明",
    "保证",
    "单位",
    "项目",
    "工程",
    "标段",
    "本项目",
    "本次",
    "合同",
    "有效期",
    "招标人",
    "投标方",
}

WEAK_MATCH_KEYWORDS = {
    "提供",
    "提交",
    "满足",
    "符合",
    "承诺",
    "声明",
    "保证",
    "要求",
    "文件",
    "资料",
    "项目",
    "工程",
    "标段",
    "投标",
    "投标人",
    "招标人",
    "响应",
}

NON_CHECKABLE_HINTS = [
    "目录",
    "模板",
    "格式",
    "样式",
    "示例",
    "填写说明",
    "盖章处",
    "签字处",
    "详见",
    "参见",
    "本表",
    "附表",
]

NON_BIDDER_PROCESS_HINTS = [
    "评标委员会",
    "推荐中标候选人",
    "招标人",
    "发包人",
    "监理人",
    "承包人",
]

EVIDENCE_ACTION_HINTS = [
    "提供",
    "提交",
    "附",
    "响应",
    "满足",
    "承诺",
    "声明",
    "保证",
    "签署",
    "盖章",
    "同意",
    "执行",
]

OCR_REFERENCE_HINTS = [
    "扫描件",
    "复印件",
    "影印件",
    "附件",
    "附后",
    "见附",
    "图片",
    "照片",
    "原件照片",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def normalize_compact(text: str) -> str:
    normalized = normalize_text(text)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", normalized)


def normalize_requirement_text(text: str) -> str:
    normalized = normalize_compact(text)
    for token in ("必须", "应当", "应", "须", "需"):
        normalized = normalized.replace(token, "")
    return normalized


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    def _noise_keyword(term: str) -> bool:
        compact = normalize_compact(term)
        if not compact:
            return True
        if term in STOP_WORDS:
            return True
        if normalize_text(term) in WEAK_MATCH_KEYWORDS:
            return True
        if re.fullmatch(r"\d+", compact):
            return True
        if re.fullmatch(r"[0-9a-z]+", compact) and not re.search(r"[a-z]{4,}", compact):
            return True
        return False

    terms = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text)
    ranked = Counter(term for term in terms if not _noise_keyword(term))
    selected = [item for item, _ in ranked.most_common(limit)]
    if selected:
        return selected
    # Fallback: keep at least some non-numeric terms to avoid empty keyword sets.
    fallback = [term for term in terms if not re.fullmatch(r"\d+", normalize_compact(term))]
    return fallback[:limit]


def classify_category(text: str) -> str:
    pairs = {
        "资质与证照": ["资质", "证照", "营业执照", "许可"],
        "业绩与案例": ["业绩", "案例", "合同复印件"],
        "报价与税费": ["报价", "税", "含税", "总价", "单价"],
        "付款与结算": ["付款", "结算", "账期", "发票"],
        "保证金与担保": ["保证金", "保函", "担保"],
        "有效期与响应": ["有效期", "偏离", "响应", "承诺"],
    }
    for category, keywords in pairs.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "商务其他"


def is_business_requirement(text: str, focus: str) -> bool:
    if focus != "business":
        return True
    has_business = any(token in text for token in BUSINESS_KEYWORDS)
    has_technical = any(token in text for token in TECHNICAL_HINTS)
    if has_business:
        return True
    if has_technical:
        return False
    return False


def split_candidate_sentences(text: str) -> list[str]:
    parts = re.split(r"[。；;！？!\r\n]+", text)
    return [item.strip() for item in parts if item and item.strip()]


def is_catalog_or_heading(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.strip())
    if not compact:
        return True
    if compact in {"目录", "投标文件目录", "招标文件目录"}:
        return True
    if "目录" in compact[:16] and re.search(r"[\.·•…]{4,}", compact):
        return True
    if re.search(r"[\.·•…]{8,}", compact) and re.search(r"\d{1,4}", compact):
        return True
    if re.search(r"[\.·•…]{2,}\d+$", compact):
        return True
    if re.match(r"^第[一二三四五六七八九十百零0-9]+[章节条]\S{0,18}$", compact):
        return True
    if re.match(r"^\d+(\.\d+){1,}\S{0,20}$", compact):
        return True
    return False


def is_checkable_statement(text: str) -> bool:
    if is_catalog_or_heading(text):
        return False
    compact = re.sub(r"\s+", "", text)
    if any(token in compact for token in NON_CHECKABLE_HINTS):
        return False
    return True


def is_bidder_obligation(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.strip())
    if not compact:
        return False
    if "评标委员会" in compact:
        return False
    if re.search(r"按本章第[0-9\.（）()]+.*(得分|评分|评审因素|分值)", compact):
        return False
    has_bidder_subject = bool(re.search(r"(投标人|我方|我公司|中标人)", compact))
    if any(token in compact for token in NON_BIDDER_PROCESS_HINTS) and not has_bidder_subject:
        return False
    # Contract administration clauses are usually not direct bid file evidence obligations.
    if "中标人" in compact and re.search(r"(签订合同|履约担保|放弃中标|赔偿)", compact):
        return False
    return True


def is_substantive_bid_block(text: str, section: str | None) -> bool:
    compact = re.sub(r"\s+", "", text.strip())
    if not compact:
        return False
    # OCR blocks are image-derived evidence; keep them unless they are obviously noise.
    if section == "OCR_MEDIA":
        if is_catalog_or_heading(compact):
            return False
        if len(compact) < 4:
            return False
        return True
    if section and re.search(r"(toc|目录)", section, flags=re.IGNORECASE):
        return False
    if "目录" in compact[:16] and re.search(r"[\.·•…]{4,}", compact):
        return False
    if is_catalog_or_heading(compact):
        return False
    if re.search(r"[\.·•…]{2,}\d{1,4}$", compact):
        return False
    has_action = any(token in compact for token in EVIDENCE_ACTION_HINTS)
    has_ocr_ref = any(token in compact for token in OCR_REFERENCE_HINTS)
    if len(compact) <= 30 and not has_action:
        return has_ocr_ref
    # Word heading styles often mark TOC/section titles rather than evidential content.
    if section and re.search(r"(heading|标题)", section, flags=re.IGNORECASE):
        if len(compact) <= 60 and not has_action and not has_ocr_ref:
            return False
    if len(compact) <= 100 and re.search(r"(项目|工程|标段|有限公司|公司)", compact) and not has_action and not has_ocr_ref:
        return False
    # Very short title-like fragments are weak evidence and usually directory entries.
    if len(compact) <= 20 and re.fullmatch(r"[0-9一二三四五六七八九十百零第章节条款\.（）()\-A-Za-z]+", compact):
        return False
    return True


def has_evidence_action(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.strip())
    if not compact:
        return False
    action_tokens = [token for token in EVIDENCE_ACTION_HINTS if token != "附"]
    return any(token in compact for token in action_tokens) or any(
        phrase in compact for phrase in ("已附", "随附", "附有")
    )


def is_reference_only_evidence(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.strip())
    if not compact:
        return False
    has_ocr_ref = any(token in compact for token in OCR_REFERENCE_HINTS)
    if not has_ocr_ref:
        return False
    if has_evidence_action(compact):
        return False
    if len(compact) <= 32:
        return True
    return bool(re.search(r"(见附件|详见附件|附后|附件[0-9一二三四五六七八九十])", compact))


def is_requirement_sentence(text: str) -> bool:
    if len(text) < 10:
        return False
    if any(token in text for token in MANDATORY_STRONG_HINTS):
        return True
    if any(token in text for token in SCORE_HINTS):
        return True
    return bool(re.search(r"(符合|满足|达到).{0,12}要求", text))


def _new_source(block: Block, text: str) -> dict:
    return {
        "doc_id": block.doc_id,
        "location": asdict(block.location),
        "excerpt": text[:120],
    }


def _build_evidence_id(block: Block) -> str:
    page = block.location.page if isinstance(block.location.page, int) else 0
    section = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", block.location.section or "NA")[:16]
    section = section or "NA"
    return f"E-{block.doc_id}-p{page}-b{block.location.block_index}-s{section}"


def _build_excerpt_hash(text: str) -> str:
    # A stable-ish fingerprint for traceability across block_index drift.
    # Intended as "find this snippet again", not a permanent primary key.
    normalized = normalize_compact(text)[:2000]
    digest = hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()
    return digest[:16]


def _merge_keywords(base: list[str], extra: list[str], limit: int = 10) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in base + extra:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def _same_requirement(existing: Requirement, candidate: Requirement) -> bool:
    if normalize_requirement_text(existing.text) == normalize_requirement_text(candidate.text):
        return True
    if existing.category != candidate.category:
        return False
    existing_keys = set(existing.keywords)
    candidate_keys = set(candidate.keywords)
    if not existing_keys or not candidate_keys:
        return False
    overlap = len(existing_keys & candidate_keys) / min(len(existing_keys), len(candidate_keys))
    if overlap >= 0.75:
        return True
    existing_norm = normalize_requirement_text(existing.text)
    candidate_norm = normalize_requirement_text(candidate.text)
    return existing_norm in candidate_norm or candidate_norm in existing_norm


def _merge_requirement(existing: Requirement, candidate: Requirement) -> None:
    if len(candidate.text) > len(existing.text):
        existing.text = candidate.text
    existing.mandatory = existing.mandatory or candidate.mandatory
    existing.keywords = _merge_keywords(existing.keywords, candidate.keywords)
    # Keep union of constraints to preserve checkable fields across duplicates.
    if isinstance(getattr(candidate, "constraints", None), list):
        existing.constraints = list(existing.constraints) + [
            item for item in candidate.constraints if item not in existing.constraints
        ]
    # Elevate tier if the merged candidate is stricter (hard_fail > scored > general).
    existing_tier = getattr(existing, "rule_tier", "general")
    candidate_tier = getattr(candidate, "rule_tier", "general")
    if TIER_ORDER.get(candidate_tier, 0) > TIER_ORDER.get(existing_tier, 0):
        existing.rule_tier = candidate_tier

    merged_sources = existing.source.setdefault("merged_sources", [])
    incoming_sources = candidate.source.get("merged_sources", [])
    if incoming_sources:
        merged_sources.extend(incoming_sources)
    else:
        merged_sources.append(candidate.source)
    existing.source["merged_count"] = len(merged_sources)


def _append_or_merge_requirement(merged_requirements: list[Requirement], candidate: Requirement) -> None:
    for existing in merged_requirements:
        if _same_requirement(existing, candidate):
            _merge_requirement(existing, candidate)
            return
    merged_requirements.append(candidate)


class RequirementExtractor(Protocol):
    provider: str
    model: str
    prompt_version: str

    def extract_requirements(self, *, block_text: str, focus: str) -> list[dict[str, Any]]:
        ...


def _build_validated_requirement_from_llm(
    *,
    block: Block,
    raw_item: dict[str, Any],
    focus: str,
    extractor: RequirementExtractor,
    min_confidence: float,
) -> Requirement | None:
    text = str(raw_item.get("text") or "").strip()
    if not text:
        return None
    if len(text) < 10:
        return None
    if focus == "business":
        has_business = any(token in text for token in BUSINESS_KEYWORDS)
        has_technical = any(token in text for token in TECHNICAL_HINTS)
        # LLM may return concise clauses (e.g. "必须提供营业执照") without explicit "商务" prefix.
        # Guardrail here should only exclude pure technical clauses.
        if has_technical and not has_business:
            return None
    if not is_checkable_statement(text):
        return None
    if not is_bidder_obligation(text):
        return None
    if not is_requirement_sentence(text):
        return None

    confidence = raw_item.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else 1.0
    except (TypeError, ValueError):
        confidence_value = 0.0
    if confidence_value < float(min_confidence):
        return None

    mandatory_raw = raw_item.get("mandatory")
    if isinstance(mandatory_raw, bool):
        mandatory = mandatory_raw
    else:
        mandatory = any(token in text for token in MANDATORY_STRONG_HINTS)

    tier_raw = str(raw_item.get("rule_tier") or "").strip()
    rule_tier = tier_raw if tier_raw in TIER_ORDER else infer_rule_tier(text, mandatory=mandatory)
    category = str(raw_item.get("category") or "").strip() or classify_category(text)

    llm_keywords = [str(item or "").strip() for item in (raw_item.get("keywords") or []) if str(item or "").strip()]
    keywords = _merge_keywords(extract_keywords(text), llm_keywords, limit=10)

    source = _new_source(block, text)
    return Requirement(
        requirement_id="",
        text=text,
        category=category,
        mandatory=mandatory,
        # Guardrail: constraints are deterministically re-extracted from text.
        constraints=extract_constraints(text),
        keywords=keywords,
        rule_tier=rule_tier,
        source={
            "doc_id": source["doc_id"],
            "location": source["location"],
            "merged_count": 1,
            "merged_sources": [source],
            "extraction": {
                "engine": "llm_schema_validated",
                "provider": extractor.provider,
                "model": extractor.model,
                "prompt_version": extractor.prompt_version,
                "confidence": round(confidence_value, 4),
            },
        },
    )


def extract_requirements(
    tender_blocks: Iterable[Block],
    focus: str,
) -> list[Requirement]:
    merged_requirements: list[Requirement] = []
    for block in tender_blocks:
        for sentence in split_candidate_sentences(block.text):
            text = sentence.strip()
            if not text:
                continue
            if not is_business_requirement(text, focus):
                continue
            if not is_checkable_statement(text):
                continue
            if not is_bidder_obligation(text):
                continue
            if not is_requirement_sentence(text):
                continue
            source = _new_source(block, text)
            mandatory = any(token in text for token in MANDATORY_STRONG_HINTS)
            candidate = Requirement(
                requirement_id="",
                text=text,
                category=classify_category(text),
                # "要求" is too generic; use strong obligation hints to reduce false mandatory flags.
                mandatory=mandatory,
                keywords=extract_keywords(text),
                constraints=extract_constraints(text),
                rule_tier=infer_rule_tier(text, mandatory=mandatory),
                source={
                    "doc_id": source["doc_id"],
                    "location": source["location"],
                    "merged_count": 1,
                    "merged_sources": [source],
                },
            )
            _append_or_merge_requirement(merged_requirements, candidate)

    for index, requirement in enumerate(merged_requirements, start=1):
        requirement.requirement_id = f"R{index:04d}"
    return merged_requirements


def _iter_extraction_batches(
    tender_blocks: Iterable[Block],
    *,
    batch_size: int,
    batch_max_chars: int,
) -> Iterable[list[Block]]:
    current_batch: list[Block] = []
    current_chars = 0
    for block in tender_blocks:
        text = str(block.text or "").strip()
        if not text:
            continue
        block_chars = len(text)
        should_flush = bool(current_batch) and (
            len(current_batch) >= batch_size or current_chars + block_chars > batch_max_chars
        )
        if should_flush:
            yield current_batch
            current_batch = []
            current_chars = 0
        current_batch.append(block)
        current_chars += block_chars
    if current_batch:
        yield current_batch


def _merge_batch_text(batch: list[Block]) -> str:
    rows: list[str] = []
    for block in batch:
        text = str(block.text or "").strip()
        if text:
            rows.append(text)
    return "\n\n".join(rows)


def extract_requirements_with_llm(
    tender_blocks: Iterable[Block],
    focus: str,
    extractor: RequirementExtractor,
    *,
    min_confidence: float = 0.6,
    batch_size: int = 8,
    batch_max_chars: int = 8000,
) -> tuple[list[Requirement], dict[str, Any]]:
    merged_requirements: list[Requirement] = []
    resolved_batch_size = max(1, int(batch_size))
    resolved_batch_max_chars = max(2000, int(batch_max_chars))
    stats: dict[str, Any] = {
        "provider": extractor.provider,
        "model": extractor.model,
        "prompt_version": extractor.prompt_version,
        "blocks_total": 0,
        "blocks_with_items": 0,
        "batches_total": 0,
        "batches_with_items": 0,
        "batch_size": resolved_batch_size,
        "batch_max_chars": resolved_batch_max_chars,
        "fallback_batches": 0,
        "timeout_fallback_batches": 0,
        "fallback_items": 0,
        "items_raw": 0,
        "items_accepted": 0,
        "items_rejected": 0,
        "errors": 0,
    }
    for batch in _iter_extraction_batches(
        tender_blocks,
        batch_size=resolved_batch_size,
        batch_max_chars=resolved_batch_max_chars,
    ):
        merged_text = _merge_batch_text(batch)
        if not merged_text:
            continue
        stats["batches_total"] = int(stats["batches_total"]) + 1
        stats["blocks_total"] = int(stats["blocks_total"]) + len(batch)
        try:
            raw_items = extractor.extract_requirements(block_text=merged_text, focus=focus)
        except Exception as exc:  # noqa: BLE001
            stats["errors"] = int(stats["errors"]) + 1
            stats["fallback_batches"] = int(stats["fallback_batches"]) + 1
            message = str(exc).lower()
            if "timeout" in message or "timed out" in message:
                stats["timeout_fallback_batches"] = int(stats["timeout_fallback_batches"]) + 1
            fallback_requirements = extract_requirements(batch, focus=focus)
            stats["fallback_items"] = int(stats["fallback_items"]) + len(fallback_requirements)
            for requirement in fallback_requirements:
                _append_or_merge_requirement(merged_requirements, requirement)
            continue
        normalized_items = [item for item in raw_items if isinstance(item, dict)]
        if normalized_items:
            stats["batches_with_items"] = int(stats["batches_with_items"]) + 1
            stats["blocks_with_items"] = int(stats["blocks_with_items"]) + len(batch)
        anchor_block = batch[0]
        for raw_item in normalized_items:
            stats["items_raw"] = int(stats["items_raw"]) + 1
            candidate = _build_validated_requirement_from_llm(
                block=anchor_block,
                raw_item=raw_item,
                focus=focus,
                extractor=extractor,
                min_confidence=min_confidence,
            )
            if candidate is None:
                stats["items_rejected"] = int(stats["items_rejected"]) + 1
                continue
            stats["items_accepted"] = int(stats["items_accepted"]) + 1
            _append_or_merge_requirement(merged_requirements, candidate)

    for index, requirement in enumerate(merged_requirements, start=1):
        requirement.requirement_id = f"R{index:04d}"
    return merged_requirements, stats


def _push_top_match(
    top_matches: list[dict],
    score: int,
    block: Block,
) -> None:
    candidate = {
        "evidence_id": _build_evidence_id(block),
        "excerpt_hash": _build_excerpt_hash(block.text),
        "score": score,
        "doc_id": block.doc_id,
        "location": asdict(block.location),
        "excerpt": block.text[:240],
        "reference_only": is_reference_only_evidence(block.text),
        "has_action": has_evidence_action(block.text),
    }
    if len(top_matches) < 3:
        top_matches.append(candidate)
        top_matches.sort(key=lambda item: item["score"], reverse=True)
        return
    if score > top_matches[-1]["score"]:
        top_matches[-1] = candidate
        top_matches.sort(key=lambda item: item["score"], reverse=True)


def _build_decision_trace(
    requirement: Requirement,
    *,
    status: str,
    reason: str,
    top_score: int,
    threshold: int,
    evidence: list[dict],
) -> dict:
    source_doc_id = requirement.source.get("doc_id") if isinstance(requirement.source, dict) else None
    source_location = requirement.source.get("location") if isinstance(requirement.source, dict) else None
    evidence_refs = [
        {
            "evidence_id": item.get("evidence_id"),
            "excerpt_hash": item.get("excerpt_hash"),
            "doc_id": item.get("doc_id"),
            "location": item.get("location"),
            "score": item.get("score"),
        }
        for item in evidence
    ]
    return {
        "clause_id": requirement.requirement_id,
        "clause_source": {
            "doc_id": source_doc_id,
            "location": source_location,
        },
        "rule": {
            "engine": REVIEW_RULE_ENGINE,
            "version": REVIEW_RULE_VERSION,
        },
        "decision": {
            "status": status,
            "reason": reason,
            "top_score": top_score,
            "threshold": threshold,
        },
        "evidence_refs": evidence_refs,
    }


def review_requirements(requirements: Iterable[Requirement], bid_blocks: Iterable[Block]) -> list[Finding]:
    requirement_list = list(requirements)
    if not requirement_list:
        return []

    keyword_to_req_ids: dict[str, set[int]] = {}
    indexed_requirements: set[int] = set()
    effective_keyword_count: dict[int, int] = {}
    for index, requirement in enumerate(requirement_list):
        for keyword in requirement.keywords:
            normalized = normalize_text(keyword)
            if not normalized:
                continue
            if normalized in WEAK_MATCH_KEYWORDS:
                continue
            keyword_to_req_ids.setdefault(normalized, set()).add(index)
            indexed_requirements.add(index)
            effective_keyword_count[index] = effective_keyword_count.get(index, 0) + 1

    # If a requirement only has weak keywords, fall back to its original keyword set.
    for index, requirement in enumerate(requirement_list):
        if index in indexed_requirements:
            continue
        for keyword in requirement.keywords:
            normalized = normalize_text(keyword)
            if not normalized:
                continue
            keyword_to_req_ids.setdefault(normalized, set()).add(index)
            effective_keyword_count[index] = effective_keyword_count.get(index, 0) + 1

    req_scores: dict[int, list[dict]] = {index: [] for index in range(len(requirement_list))}
    for block in bid_blocks:
        if not is_substantive_bid_block(block.text, block.location.section):
            continue
        normalized_block = normalize_text(block.text)
        if not normalized_block:
            continue
        hit_counter: Counter[int] = Counter()
        for keyword, req_ids in keyword_to_req_ids.items():
            if keyword not in normalized_block:
                continue
            for req_index in req_ids:
                hit_counter[req_index] += 1
        for req_index, score in hit_counter.items():
            _push_top_match(req_scores[req_index], score, block)

    findings: list[Finding] = []
    for index, requirement in enumerate(requirement_list):
        top_matches = req_scores[index]
        keyword_count = effective_keyword_count.get(index, len(requirement.keywords))
        if keyword_count <= 1:
            threshold = 1
        else:
            threshold = max(2, min(keyword_count, 4))

        if not top_matches:
            # No evidence is not proof of non-compliance.
            # Keep this as a "needs manual supplementation" state.
            status = "insufficient_evidence"
            reason = "未定位到可核验证据，建议补充材料页码后复核"
            severity = "medium" if requirement.mandatory else "low"
            trace = _build_decision_trace(
                requirement,
                status=status,
                reason=reason,
                top_score=0,
                threshold=threshold,
                evidence=[],
            )
            findings.append(
                Finding(
                    requirement_id=requirement.requirement_id,
                    clause_id=requirement.requirement_id,
                    status=status,
                    score=0,
                    severity=severity,
                    reason=reason,
                    decision_trace=trace,
                    evidence=[],
                )
            )
            continue

        top_score = top_matches[0]["score"]
        has_reference_only = all(item.get("reference_only") for item in top_matches)
        requirement_has_ocr_ref = any(token in requirement.text for token in OCR_REFERENCE_HINTS)
        top_reference_only = bool(top_matches[0].get("reference_only"))
        should_mark_needs_ocr = has_reference_only or (requirement_has_ocr_ref and top_reference_only)
        if requirement.mandatory and should_mark_needs_ocr:
            status = "needs_ocr"
            reason = "仅命中扫描件/附件引用，需OCR复核图片证据"
            trace = _build_decision_trace(
                requirement,
                status=status,
                reason=reason,
                top_score=top_score,
                threshold=threshold,
                evidence=top_matches,
            )
            findings.append(
                Finding(
                    requirement_id=requirement.requirement_id,
                    clause_id=requirement.requirement_id,
                    status=status,
                    score=top_score,
                    severity="medium",
                    reason=reason,
                    decision_trace=trace,
                    evidence=top_matches,
                )
            )
            continue

        if top_score >= threshold:
            status = "pass"
            severity = "none"
            reason = "匹配到充分证据"
        elif top_score >= 1:
            status = "risk"
            gap = threshold - top_score
            if requirement.mandatory and gap >= 2:
                severity = "medium"
                reason = "证据不足以确认满足，需人工核对并补充定位信息"
            else:
                severity = "medium" if requirement.mandatory else "low"
                reason = "已检索到部分证据，建议补充关键证据后复核"
        else:
            status = "insufficient_evidence"
            severity = "medium"
            reason = "证据不足以形成结论，建议人工补证复核"

        trace = _build_decision_trace(
            requirement,
            status=status,
            reason=reason,
            top_score=top_score,
            threshold=threshold,
            evidence=top_matches,
        )
        findings.append(
            Finding(
                requirement_id=requirement.requirement_id,
                clause_id=requirement.requirement_id,
                status=status,
                score=top_score,
                severity=severity,
                reason=reason,
                decision_trace=trace,
                evidence=top_matches,
            )
        )
    return findings


def _is_mappable_location(location: object) -> bool:
    if not isinstance(location, dict):
        return False
    block_index = location.get("block_index")
    page = location.get("page")
    return (isinstance(block_index, int) and block_index > 0) or (isinstance(page, int) and page > 0)


def enforce_evidence_quality_gate(
    requirements: Iterable[Requirement],
    findings: Iterable[Finding],
    *,
    min_excerpt_len: int = 20,
) -> list[Finding]:
    """Apply a minimal "evidence must be usable" gate.

    Goals (R5):
    - Avoid producing pass/risk conclusions that only cite directory-like references (reference_only),
      or otherwise lack usable evidence excerpts/locations.
    - Make downgrades traceable in decision_trace.
    """

    req_map = {req.requirement_id: req for req in requirements}
    updated: list[Finding] = []

    for finding in findings:
        requirement = req_map.get(finding.requirement_id)
        mandatory = bool(getattr(requirement, "mandatory", False))

        evidence = finding.evidence if isinstance(finding.evidence, list) else []

        def _excerpt_len(item: dict) -> int:
            return len(str(item.get("excerpt") or "").strip())

        has_mappable = any(_is_mappable_location(item.get("location")) for item in evidence if isinstance(item, dict))
        has_reference_mappable = any(
            bool(item.get("reference_only")) and _is_mappable_location(item.get("location"))
            for item in evidence
            if isinstance(item, dict)
        )
        has_non_reference_mappable = any(
            (not bool(item.get("reference_only")))
            and _is_mappable_location(item.get("location"))
            and _excerpt_len(item) >= min_excerpt_len
            for item in evidence
            if isinstance(item, dict)
        )

        downgraded_to: str | None = None
        downgraded_reason: str | None = None

        if finding.status in {"pass", "risk"}:
            # If the system claims pass/risk, we must have at least one usable, mappable, non-reference snippet.
            if not evidence:
                downgraded_to = "fail" if mandatory else "insufficient_evidence"
                downgraded_reason = "缺少可定位证据（evidence为空）"
            elif not has_mappable:
                downgraded_to = "insufficient_evidence"
                downgraded_reason = "缺少可定位证据（无page/block_index）"
            elif not has_non_reference_mappable:
                # Only reference-like hits (e.g. '见附件/扫描件') should not be treated as pass/risk.
                if has_reference_mappable:
                    downgraded_to = "needs_ocr"
                    downgraded_reason = "仅命中扫描件/附件引用，需OCR复核图像证据"
                else:
                    downgraded_to = "insufficient_evidence"
                    downgraded_reason = "证据摘录过短或不可用，需人工复核"

        if downgraded_to:
            finding.status = downgraded_to
            if downgraded_to == "needs_ocr":
                finding.severity = "medium"
                finding.reason = downgraded_reason or finding.reason
            elif downgraded_to == "insufficient_evidence":
                finding.severity = "high" if mandatory else "medium"
                finding.reason = downgraded_reason or finding.reason
            elif downgraded_to == "fail":
                finding.severity = "high"
                finding.reason = downgraded_reason or finding.reason

        trace = finding.decision_trace if isinstance(finding.decision_trace, dict) else {}
        trace["evidence_gate"] = {
            "min_excerpt_len": min_excerpt_len,
            "has_mappable": has_mappable,
            "has_reference_mappable": has_reference_mappable,
            "has_non_reference_mappable": has_non_reference_mappable,
            "downgraded_to": downgraded_to,
            "downgraded_reason": downgraded_reason,
        }
        # Keep trace decision in sync when we downgrade after rule/llm steps.
        trace.setdefault("decision", {})
        if isinstance(trace.get("decision"), dict):
            trace["decision"]["status"] = finding.status
            trace["decision"]["reason"] = finding.reason
        finding.decision_trace = trace

        updated.append(finding)

    return updated
