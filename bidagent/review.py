from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict
from typing import Iterable

from bidagent.models import Block, Finding, Requirement

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

MANDATORY_HINTS = ["必须", "应", "须", "不得", "严禁", "需", "要求"]

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
}


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
    terms = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text)
    ranked = Counter(term for term in terms if term not in STOP_WORDS)
    return [item for item, _ in ranked.most_common(limit)]


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


def is_requirement_sentence(text: str) -> bool:
    if len(text) < 10:
        return False
    return any(token in text for token in MANDATORY_HINTS)


def _new_source(block: Block, text: str) -> dict:
    return {
        "doc_id": block.doc_id,
        "location": asdict(block.location),
        "excerpt": text[:120],
    }


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

    merged_sources = existing.source.setdefault("merged_sources", [])
    incoming_sources = candidate.source.get("merged_sources", [])
    if incoming_sources:
        merged_sources.extend(incoming_sources)
    else:
        merged_sources.append(candidate.source)
    existing.source["merged_count"] = len(merged_sources)


def extract_requirements(
    tender_blocks: Iterable[Block],
    focus: str,
) -> list[Requirement]:
    merged_requirements: list[Requirement] = []
    for block in tender_blocks:
        text = block.text.strip()
        if not text:
            continue
        if not is_business_requirement(text, focus):
            continue
        if not is_requirement_sentence(text):
            continue
        source = _new_source(block, text)
        candidate = Requirement(
            requirement_id="",
            text=text,
            category=classify_category(text),
            mandatory=any(token in text for token in MANDATORY_HINTS),
            keywords=extract_keywords(text),
            source={
                "doc_id": source["doc_id"],
                "location": source["location"],
                "merged_count": 1,
                "merged_sources": [source],
            },
        )
        matched = False
        for existing in merged_requirements:
            if _same_requirement(existing, candidate):
                _merge_requirement(existing, candidate)
                matched = True
                break
        if not matched:
            merged_requirements.append(candidate)

    for index, requirement in enumerate(merged_requirements, start=1):
        requirement.requirement_id = f"R{index:04d}"
    return merged_requirements


def _push_top_match(
    top_matches: list[dict],
    score: int,
    block: Block,
) -> None:
    candidate = {
        "score": score,
        "doc_id": block.doc_id,
        "location": asdict(block.location),
        "excerpt": block.text[:240],
    }
    if len(top_matches) < 3:
        top_matches.append(candidate)
        top_matches.sort(key=lambda item: item["score"], reverse=True)
        return
    if score > top_matches[-1]["score"]:
        top_matches[-1] = candidate
        top_matches.sort(key=lambda item: item["score"], reverse=True)


def review_requirements(requirements: Iterable[Requirement], bid_blocks: Iterable[Block]) -> list[Finding]:
    requirement_list = list(requirements)
    if not requirement_list:
        return []

    keyword_to_req_ids: dict[str, set[int]] = {}
    for index, requirement in enumerate(requirement_list):
        for keyword in requirement.keywords:
            normalized = normalize_text(keyword)
            if not normalized:
                continue
            keyword_to_req_ids.setdefault(normalized, set()).add(index)

    req_scores: dict[int, list[dict]] = {index: [] for index in range(len(requirement_list))}
    for block in bid_blocks:
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

        if not top_matches:
            status = "fail" if requirement.mandatory else "insufficient_evidence"
            reason = "未检索到相关证据"
            severity = "high" if requirement.mandatory else "medium"
            findings.append(
                Finding(
                    requirement_id=requirement.requirement_id,
                    status=status,
                    score=0,
                    severity=severity,
                    reason=reason,
                    evidence=[],
                )
            )
            continue

        top_score = top_matches[0]["score"]
        threshold = max(2, min(len(requirement.keywords), 4))
        if top_score >= threshold:
            status = "pass"
            severity = "none"
            reason = "匹配到充分证据"
        elif top_score >= 1:
            status = "risk"
            gap = threshold - top_score
            if requirement.mandatory and gap >= 2:
                severity = "high"
                reason = "仅匹配到弱证据，需人工重点复核"
            else:
                severity = "medium" if requirement.mandatory else "low"
                reason = "检索到部分证据，建议人工复核"
        else:
            status = "insufficient_evidence"
            severity = "medium"
            reason = "证据强度不足"

        findings.append(
            Finding(
                requirement_id=requirement.requirement_id,
                status=status,
                score=top_score,
                severity=severity,
                reason=reason,
                evidence=top_matches,
            )
        )
    return findings
