from __future__ import annotations

import re
from typing import Any


_OP_WORDS_GTE = ("不少于", "不低于", "不小于", "至少", "不得低于", "应不少于", "须不少于", "需不少于")
_OP_WORDS_LTE = ("不超过", "不得超过", "不高于", "至多", "最高", "不得高于", "应不超过", "须不超过", "需不超过")
_QTY_INTENT_WORDS = (
    "提供",
    "提交",
    "递交",
    "配备",
    "投入",
    "安排",
    "出具",
    "附上",
    "纸质",
    "原件",
    "复印件",
)
_AMOUNT_FIELD_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bid_bond", ("投标保证金", "保证金", "投标担保", "履约保证金")),
    ("ceiling_price", ("最高限价", "控制价", "拦标价", "上限价")),
    ("budget_price", ("预算金额", "项目预算", "预算价", "采购预算")),
    ("bid_price", ("投标总价", "投标报价", "含税总价", "总价", "报价")),
)


def _detect_op(text: str) -> str | None:
    # Prefer explicit symbols first.
    if "≥" in text or ">=" in text:
        return ">="
    if "≤" in text or "<=" in text:
        return "<="
    for word in _OP_WORDS_GTE:
        if word in text:
            return ">="
    for word in _OP_WORDS_LTE:
        if word in text:
            return "<="
    return None


def _detect_op_near(text: str, start: int, end: int, *, window: int = 16) -> str | None:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return _detect_op(text[left:right])


def _has_qty_intent_near(text: str, start: int, end: int, *, window: int = 16) -> bool:
    left = max(0, start - window)
    right = min(len(text), end + window)
    around = text[left:right]
    return any(word in around for word in _QTY_INTENT_WORDS)


def _parse_amount_fen(raw: str) -> int | None:
    value = raw.strip().replace(",", "")
    unit = 1
    if "万元" in value:
        unit = 10000
        value = value.replace("万元", "")
    value = value.replace("元", "").replace("￥", "").replace("¥", "").strip()
    match = re.search(r"(\d+(?:\.\d{1,2})?)", value)
    if not match:
        return None
    try:
        amount = float(match.group(1))
    except ValueError:
        return None
    return int(round(amount * unit * 100))


def _detect_amount_field_near(text: str, start: int, end: int, *, window: int = 24) -> str | None:
    best_field: str | None = None
    best_distance: int | None = None
    best_is_left: bool | None = None
    for field, keywords in _AMOUNT_FIELD_KEYWORDS:
        for keyword in keywords:
            for match in re.finditer(re.escape(keyword), text):
                k_start, k_end = match.span()
                if k_end <= start:
                    distance = start - k_end
                    is_left = True
                elif k_start >= end:
                    distance = k_start - end
                    is_left = False
                else:
                    distance = 0
                    is_left = True
                if distance > window:
                    continue
                should_update = False
                if best_distance is None or distance < best_distance:
                    should_update = True
                elif distance == best_distance and best_is_left is not None and is_left and not best_is_left:
                    # Tie-breaker: prefer keyword on the left side of amount.
                    should_update = True
                if should_update:
                    best_distance = distance
                    best_field = field
                    best_is_left = is_left
                if distance == 0:
                    return field
    return best_field


def extract_constraints(text: str) -> list[dict[str, Any]]:
    """Extract minimal, checkable constraints from a requirement sentence.

    Focused on three types: amount / term / quantity.
    """
    if not text or not text.strip():
        return []

    constraints: list[dict[str, Any]] = []

    # Split by strong punctuation to avoid one clause's op leaking into another.
    clauses = [item.strip() for item in re.split(r"[；;。]\s*", text) if item and item.strip()]
    for clause in clauses:
        clause_op = _detect_op(clause)

        # Amount: require explicit currency unit to avoid matching bare numbers (e.g., 30天 / 2份).
        if re.search(r"(元|万元|￥|¥|金额|总价|报价|保证金)", clause):
            for match in re.finditer(r"([￥¥]?\s*\d[\d,]*(?:\.\d{1,2})?\s*(?:万元|元))", clause):
                raw = match.group(1).strip()
                fen = _parse_amount_fen(raw)
                if fen is None:
                    continue
                local_op = _detect_op_near(clause, match.start(1), match.end(1)) or clause_op
                amount_field = _detect_amount_field_near(clause, match.start(1), match.end(1))
                constraints.append(
                    {
                        "type": "amount",
                        "field": amount_field,
                        "op": local_op,
                        "value_fen": fen,
                        "unit": "fen",
                        "raw": raw,
                    }
                )

        # Term: "工期/服务期/供货期/质保期 ... 30 天/日/月/年/工作日"
        term_match = re.search(
            r"(工期|服务期|供货期|交付期|合同期限|有效期|质保期|保修期|期限).{0,20}?(\d{1,4})\s*(工作日|日|天|月|年)",
            clause,
        )
        if term_match:
            value = int(term_match.group(2))
            unit = term_match.group(3)
            local_op = _detect_op_near(clause, term_match.start(0), term_match.end(0)) or clause_op
            constraints.append(
                {
                    "type": "term",
                    "op": local_op,
                    "value": value,
                    "unit": unit,
                    "raw": term_match.group(0),
                }
            )

        # Quantity: require explicit op words (e.g. 至少/不少于) OR nearby quantity intent.
        qty_pattern = re.compile(
            r"(不少于|不低于|不小于|至少|不得低于|不超过|不得超过|不高于|至多)?\s*(\d{1,4})\s*(份|套|台|项|个|人|家)"
        )
        for qty_match in qty_pattern.finditer(clause):
            has_explicit_op = bool(qty_match.group(1))
            if not has_explicit_op and not _has_qty_intent_near(clause, qty_match.start(0), qty_match.end(0)):
                continue
            local_op = _detect_op(qty_match.group(0)) or clause_op
            value = int(qty_match.group(2))
            unit = qty_match.group(3)
            constraints.append(
                {
                    "type": "quantity",
                    "op": local_op,
                    "value": value,
                    "unit": unit,
                    "raw": qty_match.group(0),
                }
            )

    # De-dup by a small signature.
    seen: set[tuple] = set()
    deduped: list[dict[str, Any]] = []
    for item in constraints:
        sig = (item.get("type"), item.get("op"), item.get("unit"), item.get("value"), item.get("value_fen"), item.get("raw"))
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(item)
    return deduped
