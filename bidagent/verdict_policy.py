from __future__ import annotations

import re
from typing import Any

CROSS_AUDIT_SCHEMA_VERSION = "cross-audit-v1"

STRONG_CONFLICT_HINTS = (
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
)

DUAL_EVIDENCE_HINT_PATTERNS = (
    r"(至少|需|应|必须).{0,6}(两|二|2).{0,8}(证据|证明|材料|来源|渠道)",
    r"(双证|双重证据|交叉核验|交叉验证|相互印证)",
    r"(同时|分别).{0,12}(提供|提交).{0,18}(与|和|及).{0,12}(提供|提交|证明)",
    r"(承诺函|偏离表|附件|证明).{0,10}(与|和|及).{0,10}(承诺函|偏离表|附件|证明)",
)

_HIGH_RISK_REQUIREMENT_PATTERN = re.compile(r"(将被否决|废标|无效投标|资格审查不通过|不得投标|不予通过)")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _max_pack_score(items: list[dict[str, Any]]) -> int:
    top = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        if isinstance(score, (int, float)):
            top = max(top, int(score))
            continue
        try:
            top = max(top, int(float(score)))
        except (TypeError, ValueError):
            continue
    return top


def _normalize_source_channel(value: Any) -> str | None:
    token = str(value or "").strip().lower().replace(" ", "_")
    if not token:
        return None
    if token in {"ocr", "ocr_image", "ocr_media", "image_ocr"}:
        return "attachment_ocr"
    if token in {"table", "tabular", "spreadsheet"}:
        return "deviation_table"
    if token in {"text", "plain_text", "paragraph"}:
        return "text_clause"
    return token


def _evidence_channel_from_ref(ref: dict[str, Any]) -> str:
    source_type = _normalize_source_channel(ref.get("source_type"))
    if source_type:
        return source_type

    evidence_id = str(ref.get("evidence_id") or "").lower()
    if evidence_id.endswith("-ocr") or "-ocr-" in evidence_id:
        return "attachment_ocr"
    if evidence_id.endswith("-table") or "-table-" in evidence_id:
        return "deviation_table"

    location = ref.get("location") if isinstance(ref.get("location"), dict) else {}
    section = str(location.get("section") or "")
    excerpt = str(ref.get("excerpt") or "")
    compact = f"{section} {excerpt}"
    if any(token in compact for token in ("承诺", "声明", "授权", "委托书", "承诺函")):
        return "commitment_letter"
    if any(token in compact for token in ("附件", "扫描件", "复印件", "影印件")):
        return "attachment_text"
    return "text_clause"


def requirement_needs_cross_verification(requirement: dict[str, Any]) -> bool:
    if not isinstance(requirement, dict):
        return False
    tier = str(requirement.get("rule_tier") or "general")
    text = str(requirement.get("text") or "")
    keywords = " ".join(str(item or "") for item in (requirement.get("keywords") or []))
    probe = f"{text} {keywords}"
    hard_clause = tier == "hard_fail" or bool(_HIGH_RISK_REQUIREMENT_PATTERN.search(probe))
    explicit_dual_evidence = any(
        re.search(pattern, probe, flags=re.IGNORECASE) for pattern in DUAL_EVIDENCE_HINT_PATTERNS
    )
    return bool(hard_clause or explicit_dual_evidence)


def is_high_risk_requirement(requirement: dict[str, Any]) -> bool:
    if not isinstance(requirement, dict):
        return False
    tier = str(requirement.get("rule_tier") or "general")
    if tier == "hard_fail":
        return True
    text = str(requirement.get("text") or "")
    return bool(_HIGH_RISK_REQUIREMENT_PATTERN.search(text))


def support_evidence_is_weak(support_refs: list[dict[str, Any]]) -> bool:
    if not support_refs:
        return True
    scores: list[float] = []
    reference_only_hits = 0
    for item in support_refs:
        if not isinstance(item, dict):
            continue
        score = _safe_float(item.get("score"))
        if score is not None:
            scores.append(score)
        if bool(item.get("reference_only")):
            reference_only_hits += 1
    max_score = max(scores) if scores else 0.0
    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    if max_score >= 8 and avg_score >= 6.5 and len(support_refs) >= 2 and reference_only_hits == 0:
        return False
    if max_score >= 9 and reference_only_hits == 0:
        return False
    return True


def counter_conflict_second_pass(
    *,
    status: str,
    reason: str,
    confidence: float,
    task_packs: list[dict[str, Any]],
) -> dict[str, Any]:
    support_top = 0
    counter_top = 0
    strong_hits = 0
    for pack in task_packs:
        if not isinstance(pack, dict):
            continue
        support_pack = [item for item in (pack.get("evidence_pack") or []) if isinstance(item, dict)]
        counter_pack = [item for item in (pack.get("counter_evidence_pack") or []) if isinstance(item, dict)]
        support_top = max(support_top, _max_pack_score(support_pack))
        counter_top = max(counter_top, _max_pack_score(counter_pack))
        for item in counter_pack:
            excerpt = str(item.get("excerpt") or "")
            terms = [str(term or "") for term in (item.get("matched_terms") or [])]
            text = excerpt + " " + " ".join(terms)
            if any(token in text for token in STRONG_CONFLICT_HINTS):
                strong_hits += 1

    conflict_level = "none"
    downgraded = False
    next_status = status
    next_reason = reason
    next_confidence = confidence

    if status == "pass" and counter_top > 0:
        # Conservative but not over-sensitive: downgrade only when strong counter
        # evidence clearly dominates support evidence.
        if strong_hits > 0 and counter_top >= max(8, support_top + 3):
            conflict_level = "strong"
            downgraded = True
            next_status = "risk"
            next_reason = "命中强反证且分值显著高于支持证据，pass结论不稳定，已降级为risk"
            next_confidence = min(confidence, 0.55)
        else:
            conflict_level = "weak"

    return {
        "status": next_status,
        "reason": next_reason,
        "confidence": next_confidence,
        "audit": {
            "support_top_score": support_top,
            "counter_top_score": counter_top,
            "strong_counter_hits": strong_hits,
            "conflict_level": conflict_level,
            "downgraded": downgraded,
            "action": "downgrade_pass_to_risk_conflict_second_pass" if downgraded else None,
        },
    }


def apply_cross_audit(
    *,
    requirement_id: str,
    requirement: dict[str, Any],
    support_refs: list[dict[str, Any]],
    counter_refs: list[dict[str, Any]],
    status: str,
    reason: str,
    confidence: float,
) -> dict[str, Any]:
    support_channels = sorted({_evidence_channel_from_ref(item) for item in support_refs if isinstance(item, dict)})
    counter_channels = sorted({_evidence_channel_from_ref(item) for item in counter_refs if isinstance(item, dict)})
    required = requirement_needs_cross_verification(requirement)
    applicable = required and bool(support_refs)
    verified = len(support_channels) >= 2 if applicable else False
    high_risk = is_high_risk_requirement(requirement)
    weak_support = support_evidence_is_weak(support_refs)

    next_status = status
    next_reason = reason
    next_confidence = confidence
    action = None
    if applicable and not verified and status == "pass" and high_risk and weak_support:
        next_status = "risk"
        next_reason = "高风险条款跨渠道核验不足且支持证据偏弱，已降级为risk"
        next_confidence = min(confidence, 0.58)
        action = "downgrade_pass_to_risk_cross_verification"

    row = {
        "schema_version": CROSS_AUDIT_SCHEMA_VERSION,
        "requirement_id": requirement_id,
        "required": required,
        "applicable": applicable,
        "cross_verified": verified,
        "support_channels": support_channels,
        "counter_channels": counter_channels,
        "high_risk": high_risk,
        "weak_support": weak_support,
        "support_refs": len(support_refs),
        "counter_refs": len(counter_refs),
        "status_before": status,
        "status_after": next_status,
        "action": action,
    }
    return {
        "status": next_status,
        "reason": next_reason,
        "confidence": next_confidence,
        "cross_audit": row,
    }

