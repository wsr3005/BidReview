from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Iterable

from bidagent.models import Block


@dataclass(slots=True)
class Occurrence:
    location: dict[str, Any]
    excerpt: str


@dataclass(slots=True)
class ConsistencyFinding:
    type: str
    status: str
    severity: str
    reason: str
    values: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _norm_company(value: str) -> str:
    v = _compact(value)
    v = re.sub(r"[()（）【】\[\]{}<>《》“”\"'`]", "", v)
    # Normalize common suffix variant.
    v = v.replace("有限责任公司", "有限公司")
    return v


def _parse_date(value: str) -> str | None:
    value = value.strip()
    match = re.search(r"(\d{4})[年\-/\.](\d{1,2})[月\-/\.](\d{1,2})", value)
    if not match:
        return None
    y, m, d = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def _parse_amount_fen(value: str) -> int | None:
    # Support "12345.67元" / "12,345.67 元" / "123.45万元"
    raw = value.strip().replace(",", "")
    unit = 1
    if "万元" in raw:
        unit = 10000
        raw = raw.replace("万元", "")
    raw = raw.replace("元", "").replace("￥", "").replace("¥", "").replace("RMB", "").strip()
    match = re.search(r"(\d+(?:\.\d{1,2})?)", raw)
    if not match:
        return None
    try:
        amount = float(match.group(1))
    except ValueError:
        return None
    return int(round(amount * unit * 100))


def _extract_kv_text(text: str, keys: list[str]) -> list[tuple[str, str]]:
    # Returns list of (key, value_raw)
    results: list[tuple[str, str]] = []
    for key in keys:
        # Key can appear as "KEY: VALUE" or "KEY：VALUE"
        pattern = rf"{re.escape(key)}\s*[:：]\s*([^\r\n]{{2,120}})"
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group(1).strip()
        # Stop at obvious trailing punctuation.
        value = re.split(r"[。；;]\s*", value)[0].strip()
        results.append((key, value))
    return results


def _collect_occurrence(block: Block) -> Occurrence:
    return Occurrence(location=block.to_dict()["location"], excerpt=(block.text or "")[:160])


def _severity_for(fact_type: str) -> str:
    if fact_type in {"bidder_name", "bid_total_price_fen", "uscc"}:
        return "high"
    return "medium"


def find_inconsistencies(bid_blocks: Iterable[Block], *, max_examples_per_value: int = 3) -> list[ConsistencyFinding]:
    # type -> norm_value -> {"raw": set[str], "count": int, "examples": list[Occurrence]}
    bucket: dict[str, dict[str, dict[str, Any]]] = {}

    def _add(fact_type: str, value_norm: str, value_raw: str, occ: Occurrence) -> None:
        if not value_norm:
            return
        type_map = bucket.setdefault(fact_type, {})
        entry = type_map.setdefault(value_norm, {"raw": set(), "count": 0, "examples": []})
        entry["raw"].add(value_raw)
        entry["count"] += 1
        if len(entry["examples"]) < max_examples_per_value:
            entry["examples"].append(asdict(occ))

    for block in bid_blocks:
        text = block.text or ""
        if not text.strip():
            continue

        occ = _collect_occurrence(block)

        # Bidder/company name.
        for _, value in _extract_kv_text(text, ["投标人", "投标人名称", "公司名称", "供应商名称"]):
            _add("bidder_name", _norm_company(value), value, occ)

        # Project name (less strict; still useful).
        for _, value in _extract_kv_text(text, ["项目名称", "采购项目名称"]):
            _add("project_name", _compact(value), value, occ)

        # USCC / credit code.
        if "统一社会信用代码" in text or re.search(r"\b[0-9A-Z]{18}\b", text):
            for match in re.finditer(r"\b[0-9A-Z]{18}\b", text):
                code = match.group(0)
                _add("uscc", code, code, occ)

        # Total bid price (keyed).
        if re.search(r"(投标(总价|报价)|报价(总价)?|含税总价|总报价|投标价)", text):
            for match in re.finditer(r"([￥¥]?\s*\d[\d,]*(?:\.\d{1,2})?\s*(?:元|万元)?)", text):
                raw = match.group(1)
                fen = _parse_amount_fen(raw)
                if fen is None:
                    continue
                _add("bid_total_price_fen", str(fen), raw, occ)

        # Date (keyed to common headings).
        if re.search(r"(日期|签署日期|投标日期|开标日期)", text):
            parsed = _parse_date(text)
            if parsed:
                _add("key_date", parsed, parsed, occ)

    findings: list[ConsistencyFinding] = []
    for fact_type, values_map in bucket.items():
        if len(values_map) <= 1:
            continue
        # Multiple distinct values is suspicious; report all.
        values = []
        for norm_value, data in sorted(values_map.items(), key=lambda kv: kv[1]["count"], reverse=True):
            raw_values = sorted(list(data["raw"]))
            values.append(
                {
                    "value_norm": norm_value,
                    "value_raw_examples": raw_values[:3],
                    "count": int(data["count"]),
                    "examples": list(data["examples"]),
                }
            )
        reason = f"字段 `{fact_type}` 出现多个不同取值，可能存在跨章节不一致"
        findings.append(
            ConsistencyFinding(
                type=fact_type,
                status="risk",
                severity=_severity_for(fact_type),
                reason=reason,
                values=values,
            )
        )

    return findings

