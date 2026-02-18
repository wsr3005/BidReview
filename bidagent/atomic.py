from __future__ import annotations

import re
from typing import Any, Iterable

from bidagent.models import Requirement

_SUBJECT_HINTS = ("投标人", "供应商", "承包人", "申请人", "乙方")
_ACTION_HINTS = ("提供", "提交", "具备", "满足", "承诺", "响应", "签署", "盖章", "缴纳", "达到", "符合")
_FLUFF_HINTS = ("目录", "模板", "格式", "示例", "填写说明", "参见", "详见", "附表")
_ATTACHMENT_HINTS = ("扫描件", "复印件", "影印件", "附件", "附表", "附后", "见附")
_SOFT_HINTS = ("建议", "可", "宜", "优先", "酌情")


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _constraint_to_text(constraint: dict[str, Any]) -> str:
    raw = str(constraint.get("raw") or "").strip()
    if raw:
        return _normalize_space(raw)
    field = str(constraint.get("field") or constraint.get("name") or "").strip()
    op = str(constraint.get("op") or "").strip()
    value = constraint.get("value")
    if value is None:
        value = constraint.get("value_fen")
    unit = str(constraint.get("unit") or "").strip()
    body = " ".join(part for part in (field, op, str(value) if value is not None else "", unit) if part)
    return _normalize_space(body)


def _split_compound(text: str) -> list[str]:
    normalized = _normalize_space(text)
    if not normalized:
        return []
    sentences = [item.strip() for item in re.split(r"[；;。]+", normalized) if item.strip()]
    segments: list[str] = []
    for sentence in sentences:
        parts = [item.strip(" ，,") for item in re.split(r"(?:且|并且|以及|同时|并|及|、|，|,)", sentence) if item.strip()]
        filtered = [part for part in parts if len(part) >= 4]
        if len(filtered) <= 1:
            segments.append(sentence)
            continue
        # Keep only segments that look like checkable actions.
        actionable = [part for part in filtered if any(token in part for token in _ACTION_HINTS)]
        segments.extend(actionable or filtered)
    return segments or [normalized]


def _infer_subject(text: str) -> str:
    for token in _SUBJECT_HINTS:
        if token in text:
            return token
    return "投标人"


def _infer_action(text: str) -> str:
    for token in _ACTION_HINTS:
        if token in text:
            return token
    return "满足"


def _infer_logic(*, text: str, mandatory: bool) -> str:
    compact = re.sub(r"\s+", "", text or "")
    if mandatory or any(token in compact for token in ("必须", "应", "须", "不得", "严禁")):
        return "must"
    if any(token in compact for token in _SOFT_HINTS):
        return "should"
    return "should"


def _infer_evidence_expectation(text: str, constraints: list[dict[str, Any]]) -> str:
    compact = re.sub(r"\s+", "", text or "")
    if any(token in compact for token in _ATTACHMENT_HINTS):
        return "attachment_or_ocr_evidence"
    if constraints:
        return "numeric_or_term_evidence"
    return "direct_text_evidence"


def _classify_level(*, text: str, mandatory: bool, tier: str, constraints: list[dict[str, Any]]) -> str:
    compact = re.sub(r"\s+", "", text or "")
    if any(token in compact for token in _FLUFF_HINTS) and not any(token in compact for token in _ACTION_HINTS):
        return "fluff"
    if tier == "hard_fail" or mandatory:
        return "hard"
    if not constraints and not any(token in compact for token in _ACTION_HINTS):
        return "fluff"
    return "soft"


def build_atomic_requirements(requirements: Iterable[Requirement]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for requirement in requirements:
        text = _normalize_space(requirement.text)
        if not text:
            continue
        segments = _split_compound(text)
        raw_constraints = [item for item in requirement.constraints if isinstance(item, dict)]
        constraint_texts = [value for value in (_constraint_to_text(item) for item in raw_constraints) if value]
        if len(constraint_texts) > len(segments):
            segments = [text] + [item for item in constraint_texts if item != text]

        for index, segment in enumerate(segments, start=1):
            constraint_value = (
                constraint_texts[index - 1]
                if index - 1 < len(constraint_texts)
                else (constraint_texts[0] if constraint_texts else "")
            )
            level = _classify_level(
                text=segment,
                mandatory=bool(requirement.mandatory),
                tier=str(requirement.rule_tier or "general"),
                constraints=raw_constraints,
            )
            rows.append(
                {
                    "atomic_id": f"{requirement.requirement_id}-A{index:02d}",
                    "requirement_id": requirement.requirement_id,
                    "source_text": text,
                    "atomic_text": segment,
                    "subject": _infer_subject(segment),
                    "action": _infer_action(segment),
                    "constraint": constraint_value,
                    "logic": _infer_logic(text=segment, mandatory=bool(requirement.mandatory)),
                    "evidence_expectation": _infer_evidence_expectation(segment, raw_constraints),
                    "tier": str(requirement.rule_tier or "general"),
                    "classification": level,
                    "engine_enabled": level != "fluff",
                    "mandatory": bool(requirement.mandatory),
                    "category": requirement.category,
                    "source": requirement.source,
                }
            )
    return rows
