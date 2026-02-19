from __future__ import annotations

import re
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Iterable

from bidagent.models import Block


@dataclass(slots=True)
class Occurrence:
    doc_id: str
    location: dict[str, Any]
    excerpt: str


@dataclass(slots=True)
class ConsistencyFinding:
    type: str
    status: str
    severity: str
    reason: str
    values: list[dict[str, Any]]
    pairs: list[dict[str, Any]] = field(default_factory=list)
    comparison: dict[str, Any] | None = None
    scope: str = "bid_internal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _norm_company(value: str) -> str:
    v = _compact(value)
    v = re.sub(r"[()（）【】\[\]{}<>《》“”\"'`]", "", v)
    # Normalize common suffix variant.
    v = v.replace("有限责任公司", "有限公司")
    for token in ("盖章", "签章", "投标人", "单位名称"):
        v = v.replace(token, "")
    v = _compact(v)
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
    def _fuzzy_key_pattern(key: str) -> str:
        # Allow DOCX-extracted text like "开 户 行" to match key "开户行".
        return r"\s*".join(re.escape(ch) for ch in key)

    results: list[tuple[str, str]] = []
    for key in keys:
        # Key can appear as "KEY: VALUE" or "KEY：VALUE"
        pattern = rf"{_fuzzy_key_pattern(key)}\s*[:：]\s*([^\r\n]{{1,160}})"
        for match in re.finditer(pattern, text):
            value = match.group(1).strip()
            # Stop at obvious trailing punctuation.
            value = re.split(r"[。；;，,]\s*", value)[0].strip()
            results.append((key, value))
    return results


def _collect_occurrence(block: Block) -> Occurrence:
    return Occurrence(
        doc_id=block.doc_id,
        location=block.to_dict()["location"],
        excerpt=(block.text or "")[:200],
    )


def _severity_for(fact_type: str) -> str:
    if fact_type in {
        "bidder_name",
        "legal_representative_company",
        "authorization_company",
        "authorized_representative_name",
        "bidder_name_authorization_mismatch",
        "bidder_name_legal_representative_mismatch",
        "tender_no",
        "tender_no_cross_doc",
        "account_bank",
        "account_number",
        "account_bank_receipt_mismatch",
        "account_number_receipt_mismatch",
        "account_bank_receipt_unreadable",
        "bid_total_price_fen",
        "uscc",
        "legal_representative_name",
        "legal_representative_name_placeholder",
    }:
        return "high"
    return "medium"


def _status_for(fact_type: str) -> str:
    if fact_type == "account_bank_receipt_unreadable":
        return "needs_ocr"
    if fact_type == "legal_representative_name_placeholder":
        return "risk"
    if fact_type in {
        "bidder_name",
        "tender_no",
        "legal_representative_name",
        "authorized_representative_name",
        "account_bank",
        "account_number",
        "tender_no_cross_doc",
        "bidder_name_authorization_mismatch",
        "bidder_name_legal_representative_mismatch",
        "account_bank_receipt_mismatch",
        "account_number_receipt_mismatch",
    }:
        return "fail"
    return "risk"


def _extract_uscc_codes(text: str) -> list[str]:
    # Only accept 18-char code when it is explicitly tied to credit-code keys.
    keys = ["统一社会信用代码", "社会信用代码", "信用代码"]
    codes: list[str] = []
    for _, value in _extract_kv_text(text, keys):
        match = re.search(r"\b[0-9A-Z]{18}\b", value.upper())
        if match:
            codes.append(match.group(0))
    for match in re.finditer(r"(统一社会信用代码|社会信用代码|信用代码)\s*(?:[:：]|为)?\s*([0-9A-Z]{18})", text):
        codes.append(match.group(2).upper())
    return codes


def _extract_bid_total_prices(text: str) -> list[tuple[str, int]]:
    keys = ["投标总价", "投标报价", "总报价", "含税总价", "投标价", "报价金额", "报价"]
    results: list[tuple[str, int]] = []

    # Prefer key-value extraction first.
    for _, value in _extract_kv_text(text, keys):
        match = re.search(r"([￥¥]?\s*\d[\d,]*(?:\.\d{1,2})?\s*(?:元|万元))", value)
        if not match:
            continue
        raw = match.group(1)
        fen = _parse_amount_fen(raw)
        if fen is not None:
            results.append((raw, fen))
    if results:
        return results

    # Fallback: free-form phrase with key adjacent to amount.
    for match in re.finditer(
        r"(投标总价|投标报价|总报价|含税总价|投标价|报价金额|报价)\s*(?:金额)?\s*(?:为|是|:|：)?\s*([￥¥]?\s*\d[\d,]*(?:\.\d{1,2})?\s*(?:元|万元))",
        text,
    ):
        raw = match.group(2)
        fen = _parse_amount_fen(raw)
        if fen is not None:
            results.append((raw, fen))
    return results


def _extract_key_dates(text: str) -> list[str]:
    keys = ["签署日期", "投标日期", "开标日期", "日期"]
    values: list[str] = []

    # Prefer explicit key-value extraction.
    for _, value in _extract_kv_text(text, keys):
        parsed = _parse_date(value)
        if parsed:
            values.append(parsed)
    if values:
        return values

    # Fallback: key + nearby date.
    for match in re.finditer(
        r"(签署日期|投标日期|开标日期|日期)\s*(?:为|是|:|：)?\s*(\d{4}[年\-/\.]\d{1,2}[月\-/\.]\d{1,2}日?)",
        text,
    ):
        parsed = _parse_date(match.group(2))
        if parsed:
            values.append(parsed)
    return values


_PLACEHOLDER_PERSON_NAMES = {"张三", "李四", "王五", "赵六", "测试", "test", "xxx", "某某"}
_NOISE_TOKENS = {"", "投标单位名称", "投标人名称", "单位名称", "法定代表人", "项目名称"}
_FIELD_LABELS = {
    "bidder_name": "投标主体名称",
    "legal_representative_company": "法定代表人所属单位",
    "authorized_representative_name": "授权人姓名",
    "authorization_company": "授权函投标单位名称",
    "bidder_name_authorization_mismatch": "投标函主体与投标主体",
    "bidder_name_legal_representative_mismatch": "法定代表人所属单位与投标主体",
    "tender_no": "招标编号",
    "tender_no_cross_doc": "招标编号（招投标对照）",
    "account_bank": "开户行",
    "account_number": "账号",
    "account_name": "开户名",
    "account_bank_receipt_mismatch": "开户行与回单支行",
    "account_number_receipt_mismatch": "账号与回单账号",
    "account_bank_receipt_unreadable": "银行回单识别质量",
    "legal_representative_name": "法定代表人姓名",
    "legal_representative_name_placeholder": "法定代表人姓名",
    "uscc": "统一社会信用代码",
    "bid_total_price_fen": "投标总价",
    "key_date": "关键日期",
    "project_name": "项目名称",
}


def _field_label(fact_type: str) -> str:
    return _FIELD_LABELS.get(fact_type, fact_type)


def _extract_tender_numbers(text: str) -> list[str]:
    compact = _compact(text)
    values: list[str] = []
    keys = ["招标编号", "项目编号", "采购编号", "项目编码"]
    for key in keys:
        start = 0
        while True:
            idx = compact.find(key, start)
            if idx < 0:
                break
            snippet = compact[idx : idx + 80]
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{7,}", snippet):
                values.append(token.upper())
            start = idx + len(key)
    # Keep stable order while deduping.
    deduped: list[str] = []
    seen: set[str] = set()
    for token in values:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _clean_company_candidate(value: str) -> str:
    cleaned = _norm_company(value)
    if not cleaned:
        return ""
    for noise in _NOISE_TOKENS:
        if cleaned == noise:
            return ""
    if "投标单位的名称" in cleaned:
        return ""
    return cleaned


def _extract_company_mentions(text: str) -> dict[str, list[str]]:
    bucket: dict[str, list[str]] = {
        "bidder_name": [],
        "legal_representative_company": [],
        "authorization_company": [],
        "account_name": [],
    }
    for _, value in _extract_kv_text(
        text,
        ["投标人", "投标人名称", "公司名称", "供应商名称", "单位名称"],
    ):
        normalized = _clean_company_candidate(value)
        if normalized:
            bucket["bidder_name"].append(normalized)

    for _, value in _extract_kv_text(text, ["开户名", "开户名称"]):
        normalized = _clean_company_candidate(value)
        if normalized:
            bucket["account_name"].append(normalized)

    compact = _compact(text)
    legal_match = re.search(r"系(.{2,50}?)的法定代表人", compact)
    if legal_match:
        normalized = _clean_company_candidate(legal_match.group(1))
        if normalized:
            bucket["legal_representative_company"].append(normalized)

    auth_match = re.search(r"代表我方(.{2,60}?)[（(]投标单位的名称[)）]", compact)
    if auth_match:
        normalized = _clean_company_candidate(auth_match.group(1))
        if normalized:
            bucket["authorization_company"].append(normalized)

    return bucket


def _extract_account_banks(text: str) -> list[str]:
    values: list[str] = []
    for _, value in _extract_kv_text(text, ["开户行", "开户银行"]):
        compact_value = _compact(value)
        if compact_value:
            values.append(compact_value)
    return values


def _normalize_person_name(value: str) -> str:
    cleaned = re.sub(r"[()（）【】\[\]{}<>《》“”\"'`，,。；;:：\s]", "", value or "")
    return cleaned


def _extract_legal_representative_names(
    blocks: list[Block],
) -> list[tuple[int, str, Block]]:
    results: list[tuple[int, str, Block]] = []
    legal_context_window = 0
    for idx, block in enumerate(blocks):
        text = block.text or ""
        compact_text = _compact(text)
        if any(token in compact_text for token in ("法定代表人身份证明", "法定代表人身份证明书", "法定代表人身份证明文件")):
            legal_context_window = 8

        should_extract = legal_context_window > 0 or ("法定代表人" in compact_text and "姓名" in compact_text)
        if should_extract:
            match = re.search(r"姓名[:：]?([\u4e00-\u9fff]{2,4})(?=性别|年龄|职务|$)", compact_text)
            if not match:
                match = re.search(r"姓名[:：]?([A-Za-z]{2,24})", compact_text)
            if match:
                name = _normalize_person_name(match.group(1))
                if name and len(name) <= 12:
                    results.append((idx, name, block))
                    if legal_context_window > 0:
                        legal_context_window -= 1
                    continue
            for _, value in _extract_kv_text(text, ["姓名"]):
                value = re.split(r"(?:性别|年龄|职务)[:：]?", value)[0]
                name = _normalize_person_name(value)
                if not name:
                    continue
                # Keep strict to avoid grabbing long descriptive text.
                if len(name) > 12:
                    continue
                results.append((idx, name, block))
        if legal_context_window > 0:
            legal_context_window -= 1
    return results


def _extract_authorized_representative_names(
    blocks: list[Block],
) -> list[tuple[int, str, Block]]:
    results: list[tuple[int, str, Block]] = []
    for idx, block in enumerate(blocks):
        text = block.text or ""
        compact_text = _compact(text)
        if not any(token in compact_text for token in ("授权委托书", "授权书", "委托代理人", "授权代表", "被授权人", "授权人")):
            continue

        for _, value in _extract_kv_text(text, ["委托代理人", "授权代表", "被授权人", "授权人", "受托人"]):
            value = re.split(r"(?:性别|年龄|职务|身份证号|身份证号码)[:：]?", value)[0]
            name = _normalize_person_name(value)
            if not name or len(name) > 12:
                continue
            results.append((idx, name, block))

        for match in re.finditer(r"(?:委托代理人|授权代表|被授权人|授权人|受托人)[:：]?([\u4e00-\u9fff]{2,4})", compact_text):
            name = _normalize_person_name(match.group(1))
            if not name or len(name) > 12:
                continue
            results.append((idx, name, block))
    return results


def _loc_text(value: dict[str, Any]) -> str:
    location = value.get("location") if isinstance(value, dict) else {}
    if not isinstance(location, dict):
        return "block=N/A page=N/A"
    return f"block={location.get('block_index')} page={location.get('page')}"


def _format_values(values_map: dict[str, dict[str, Any]], max_examples_per_value: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for norm_value, data in sorted(values_map.items(), key=lambda kv: kv[1]["count"], reverse=True):
        raw_values = sorted(list(data["raw"]))
        rows.append(
            {
                "value_norm": norm_value,
                "value_raw_examples": raw_values[:3],
                "count": int(data["count"]),
                "examples": list(data["examples"][:max_examples_per_value]),
            }
        )
    return rows


def _build_pairs(values: list[dict[str, Any]], max_pairs: int) -> list[dict[str, Any]]:
    if len(values) <= 1:
        return []
    anchor = values[0]
    anchor_example = (anchor.get("examples") or [{}])[0]
    pairs: list[dict[str, Any]] = []
    for row in values[1 : 1 + max_pairs]:
        other_example = (row.get("examples") or [{}])[0]
        pairs.append(
            {
                "left": {
                    "value": (anchor.get("value_raw_examples") or [anchor.get("value_norm") or ""])[0],
                    "count": anchor.get("count"),
                    "doc_id": anchor_example.get("doc_id"),
                    "location": anchor_example.get("location"),
                    "excerpt": anchor_example.get("excerpt"),
                },
                "right": {
                    "value": (row.get("value_raw_examples") or [row.get("value_norm") or ""])[0],
                    "count": row.get("count"),
                    "doc_id": other_example.get("doc_id"),
                    "location": other_example.get("location"),
                    "excerpt": other_example.get("excerpt"),
                },
            }
        )
    return pairs


def _conflict_reason(field_type: str, pairs: list[dict[str, Any]]) -> str:
    if not pairs:
        return f"{_field_label(field_type)}存在多个取值，结论：不一致"
    first = pairs[0]
    left = first.get("left") or {}
    right = first.get("right") or {}
    return (
        f"{_field_label(field_type)}核验："
        f"证据A=“{left.get('value', 'N/A')}”({_loc_text(left)})，"
        f"证据B=“{right.get('value', 'N/A')}”({_loc_text(right)})，"
        "结论：不一致"
    )


def _comparison_from_pair(pair: dict[str, Any] | None, conclusion: str) -> dict[str, Any]:
    left = (pair or {}).get("left") if isinstance(pair, dict) else {}
    right = (pair or {}).get("right") if isinstance(pair, dict) else {}
    left = left if isinstance(left, dict) else {}
    right = right if isinstance(right, dict) else {}
    return {
        "evidence_a": {
            "value": left.get("value"),
            "doc_id": left.get("doc_id"),
            "location": left.get("location"),
            "excerpt": left.get("excerpt"),
        },
        "evidence_b": {
            "value": right.get("value"),
            "doc_id": right.get("doc_id"),
            "location": right.get("location"),
            "excerpt": right.get("excerpt"),
        },
        "conclusion": conclusion,
    }


def _init_bucket() -> dict[str, dict[str, dict[str, Any]]]:
    return {}


def _add_value(bucket: dict[str, dict[str, dict[str, Any]]], *, fact_type: str, value_norm: str, value_raw: str, occ: Occurrence) -> None:
    if not value_norm:
        return
    type_map = bucket.setdefault(fact_type, {})
    entry = type_map.setdefault(value_norm, {"raw": set(), "count": 0, "examples": []})
    entry["raw"].add(value_raw)
    entry["count"] += 1
    if len(entry["examples"]) < 5:
        entry["examples"].append(asdict(occ))


def _top_value(values_map: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
    if not values_map:
        return None
    items = sorted(values_map.items(), key=lambda kv: (int(kv[1].get("count") or 0), kv[0]), reverse=True)
    return items[0]


def _make_pair(left_value: str, left_data: dict[str, Any], right_value: str, right_data: dict[str, Any]) -> dict[str, Any]:
    left_example = (left_data.get("examples") or [{}])[0]
    right_example = (right_data.get("examples") or [{}])[0]
    return {
        "left": {
            "value": left_value,
            "count": left_data.get("count"),
            "doc_id": left_example.get("doc_id"),
            "location": left_example.get("location"),
            "excerpt": left_example.get("excerpt"),
        },
        "right": {
            "value": right_value,
            "count": right_data.get("count"),
            "doc_id": right_example.get("doc_id"),
            "location": right_example.get("location"),
            "excerpt": right_example.get("excerpt"),
        },
    }


def _extract_account_numbers(text: str) -> list[str]:
    values: list[str] = []
    for _, value in _extract_kv_text(text, ["账号", "账户", "开户账号"]):
        for token in re.findall(r"\d{6,30}", value):
            values.append(token)
    # fallback: "账号610820402"
    for match in re.finditer(r"(?:账号|账户|开户账号)\s*[:：]?\s*(\d{6,30})", text):
        values.append(match.group(1))
    return values


def _looks_like_bank_receipt_ocr(text: str, account_numbers: set[str]) -> bool:
    compact = _compact(text).upper()
    if not compact:
        return False
    if any(token in compact for token in ("BANK", "MINSHENG", "回单", "流水", "付款人", "收款人", "账号", "账户", "民生")):
        return True
    if any(number in compact for number in account_numbers):
        return True
    return False


def _extract_branch_candidates(text: str) -> list[str]:
    compact = _compact(text)
    values: list[str] = []
    # Standard Chinese pattern.
    for match in re.finditer(r"([\u4e00-\u9fff]{2,20}支行)", compact):
        values.append(match.group(1))
    # OCR might split around "北京紫竹支行"/"北京真我支行"; keep "北京...支行".
    for match in re.finditer(r"(北京[\u4e00-\u9fff]{1,18}支行)", compact):
        values.append(match.group(1))
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _normalize_branch_value(value: str) -> str:
    compact = _compact(value)
    compact = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", compact)
    compact = compact.replace("中国民生银行股份有限公司", "民生银行")
    compact = compact.replace("中国民生银行", "民生银行")
    match = re.search(r"(.*?支行)", compact)
    if match:
        compact = match.group(1)
    return compact


def _branch_match(left: str, right: str) -> bool:
    left_norm = _normalize_branch_value(left)
    right_norm = _normalize_branch_value(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if left_norm.endswith(right_norm) or right_norm.endswith(left_norm):
        return True
    if left_norm in right_norm or right_norm in left_norm:
        return True
    similarity = SequenceMatcher(None, left_norm, right_norm).ratio()
    return similarity >= 0.72


def _branch_candidate_confident(value: str) -> bool:
    compact = _normalize_branch_value(value)
    if "支行" not in compact:
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff]", compact))
    # Require enough CJK signal to avoid OCR fragments turning into hard-fail mismatches.
    return cjk >= 6 and len(compact) >= 8


def find_inconsistencies(
    bid_blocks: Iterable[Block],
    *,
    tender_blocks: Iterable[Block] | None = None,
    max_examples_per_value: int = 3,
    max_pairs_per_finding: int = 3,
) -> list[ConsistencyFinding]:
    # type -> norm_value -> {"raw": set[str], "count": int, "examples": list[Occurrence]}
    bid_list = list(bid_blocks)
    tender_list = list(tender_blocks or [])
    bucket = _init_bucket()
    account_numbers: set[str] = set()
    ocr_blocks: list[Block] = []

    for block in bid_list:
        text = block.text or ""
        if not text.strip():
            continue

        occ = _collect_occurrence(block)
        if str(block.location.section or "").upper() == "OCR_MEDIA":
            ocr_blocks.append(block)
            # OCR blocks are noisy for core consistency fields. We only use them in
            # dedicated cross-evidence checks (e.g., bank receipt parsing) below.
            continue

        company_mentions = _extract_company_mentions(text)
        for fact_type, values in company_mentions.items():
            for value in values:
                _add_value(bucket, fact_type=fact_type, value_norm=value, value_raw=value, occ=occ)

        # Project name.
        for _, value in _extract_kv_text(text, ["项目名称", "采购项目名称"]):
            norm = _compact(value)
            if norm:
                _add_value(bucket, fact_type="project_name", value_norm=norm, value_raw=value, occ=occ)

        # Tender/project number.
        for code in _extract_tender_numbers(text):
            _add_value(bucket, fact_type="tender_no", value_norm=code, value_raw=code, occ=occ)

        # USCC / credit code.
        for code in _extract_uscc_codes(text):
            _add_value(bucket, fact_type="uscc", value_norm=code, value_raw=code, occ=occ)

        # Total bid price (keyed + nearby fallback).
        for raw, fen in _extract_bid_total_prices(text):
            _add_value(bucket, fact_type="bid_total_price_fen", value_norm=str(fen), value_raw=raw, occ=occ)

        # Date (keyed to common headings).
        for parsed in _extract_key_dates(text):
            _add_value(bucket, fact_type="key_date", value_norm=parsed, value_raw=parsed, occ=occ)

        for number in _extract_account_numbers(text):
            account_numbers.add(number)
            _add_value(bucket, fact_type="account_number", value_norm=number, value_raw=number, occ=occ)

        for bank in _extract_account_banks(text):
            _add_value(bucket, fact_type="account_bank", value_norm=bank, value_raw=bank, occ=occ)

    # Targeted legal representative name extraction with local context to avoid broad "姓名" noise.
    for _, name, block in _extract_legal_representative_names(bid_list):
        occ = _collect_occurrence(block)
        _add_value(
            bucket,
            fact_type="legal_representative_name",
            value_norm=name,
            value_raw=name,
            occ=occ,
        )
    for _, name, block in _extract_authorized_representative_names(bid_list):
        occ = _collect_occurrence(block)
        _add_value(
            bucket,
            fact_type="authorized_representative_name",
            value_norm=name,
            value_raw=name,
            occ=occ,
        )

    findings: list[ConsistencyFinding] = []
    for fact_type, values_map in bucket.items():
        if len(values_map) <= 1:
            continue
        values = _format_values(values_map, max_examples_per_value)
        pairs = _build_pairs(values, max_pairs=max_pairs_per_finding)
        reason = _conflict_reason(fact_type, pairs)
        findings.append(
            ConsistencyFinding(
                type=fact_type,
                status=_status_for(fact_type),
                severity=_severity_for(fact_type),
                reason=reason,
                values=values,
                pairs=pairs,
                comparison=_comparison_from_pair(pairs[0] if pairs else None, "不一致"),
                scope="bid_internal",
            )
        )

    # Cross-field: bidder name should align with authorization/legal-representative company identities.
    bidder_top = _top_value(bucket.get("bidder_name") or {})
    if bidder_top is not None:
        bidder_value, bidder_data = bidder_top
        for field_name, finding_type in (
            ("authorization_company", "bidder_name_authorization_mismatch"),
            ("legal_representative_company", "bidder_name_legal_representative_mismatch"),
        ):
            field_map = bucket.get(field_name) or {}
            if not field_map:
                continue
            mismatches = [(value, data) for value, data in field_map.items() if value != bidder_value]
            if not mismatches:
                continue
            # Keep one high-signal mismatch per field to reduce noise.
            left_value, left_data = sorted(
                mismatches,
                key=lambda item: (int(item[1].get("count") or 0), item[0]),
                reverse=True,
            )[0]
            pairs = [_make_pair(left_value, left_data, bidder_value, bidder_data)]
            values = _format_values(
                {
                    left_value: left_data,
                    bidder_value: bidder_data,
                },
                max_examples_per_value=max_examples_per_value,
            )
            reason = (
                f"{_field_label(finding_type)}核验："
                f"证据A=“{left_value}”({_loc_text(pairs[0]['left'])})，"
                f"证据B=“{bidder_value}”({_loc_text(pairs[0]['right'])})，"
                "结论：不一致"
            )
            findings.append(
                ConsistencyFinding(
                    type=finding_type,
                    status=_status_for(finding_type),
                    severity=_severity_for(finding_type),
                    reason=reason,
                    values=values,
                    pairs=pairs,
                    comparison=_comparison_from_pair(pairs[0], "不一致"),
                    scope="bid_internal",
                )
            )

    receipt_blocks = [
        block
        for block in ocr_blocks
        if _looks_like_bank_receipt_ocr(block.text or "", account_numbers)
    ]

    # Cross-evidence: account bank in text vs bank receipt OCR.
    account_bank_map = bucket.get("account_bank") or {}
    if account_bank_map and receipt_blocks:
        receipt_branch_map: dict[str, dict[str, Any]] = {}
        for block in receipt_blocks:
            occ = _collect_occurrence(block)
            for branch in _extract_branch_candidates(block.text or ""):
                _add_value(
                    {"receipt_branch": receipt_branch_map},
                    fact_type="receipt_branch",
                    value_norm=branch,
                    value_raw=branch,
                    occ=occ,
                )

        if receipt_branch_map:
            account_top = _top_value(account_bank_map)
            receipt_top = _top_value(receipt_branch_map)
            if account_top and receipt_top:
                account_value, account_data = account_top
                receipt_value, receipt_data = receipt_top

                matched = _branch_match(account_value, receipt_value)
                if not matched:
                    if not _branch_candidate_confident(receipt_value):
                        pair = _make_pair(
                            account_value,
                            account_data,
                            "回单图片（OCR支行识别不稳定）",
                            {"count": 1, "examples": receipt_data.get("examples") or []},
                        )
                        account_values = _format_values(
                            {
                                account_value: account_data,
                            },
                            max_examples_per_value=max_examples_per_value,
                        )
                        account_values.append(
                            {
                                "value_norm": "receipt_branch_low_confidence",
                                "value_raw_examples": [receipt_value],
                                "count": int(receipt_data.get("count") or 1),
                                "examples": list((receipt_data.get("examples") or [])[:max_examples_per_value]),
                            }
                        )
                        reason = (
                            f"{_field_label('account_bank_receipt_unreadable')}核验："
                            f"证据A=“{account_value}”({_loc_text(pair['left'])})，"
                            f"证据B=“回单图片（OCR支行识别不稳定）”({_loc_text(pair['right'])})，"
                            "结论：需人工核验回单原图"
                        )
                        findings.append(
                            ConsistencyFinding(
                                type="account_bank_receipt_unreadable",
                                status=_status_for("account_bank_receipt_unreadable"),
                                severity=_severity_for("account_bank_receipt_unreadable"),
                                reason=reason,
                                values=account_values,
                                pairs=[pair],
                                comparison=_comparison_from_pair(pair, "需人工核验"),
                                scope="cross_evidence",
                            )
                        )
                    else:
                        pairs = [_make_pair(account_value, account_data, receipt_value, receipt_data)]
                        values = _format_values(
                            {
                                account_value: account_data,
                                receipt_value: receipt_data,
                            },
                            max_examples_per_value=max_examples_per_value,
                        )
                        reason = (
                            f"{_field_label('account_bank_receipt_mismatch')}核验："
                            f"证据A=“{account_value}”({_loc_text(pairs[0]['left'])})，"
                            f"证据B=“{receipt_value}”({_loc_text(pairs[0]['right'])})，"
                            "结论：不一致"
                        )
                        findings.append(
                            ConsistencyFinding(
                                type="account_bank_receipt_mismatch",
                                status=_status_for("account_bank_receipt_mismatch"),
                                severity=_severity_for("account_bank_receipt_mismatch"),
                                reason=reason,
                                values=values,
                                pairs=pairs,
                                comparison=_comparison_from_pair(pairs[0], "不一致"),
                                scope="cross_evidence",
                            )
                        )
        else:
            account_top = _top_value(account_bank_map)
            receipt_block = receipt_blocks[0]
            occ = _collect_occurrence(receipt_block)
            account_values = _format_values(account_bank_map, max_examples_per_value=max_examples_per_value)
            if account_top:
                account_value, account_data = account_top
                pair = _make_pair(
                    account_value,
                    account_data,
                    "回单图片（OCR未识别出支行）",
                    {"count": 1, "examples": [asdict(occ)]},
                )
                reason = (
                    f"{_field_label('account_bank_receipt_unreadable')}核验："
                    f"证据A=“{account_value}”({_loc_text(pair['left'])})，"
                    f"证据B=“回单图片（OCR未识别出支行）”({_loc_text(pair['right'])})，"
                    "结论：需人工核验回单原图"
                )
            else:
                pair = {}
                reason = (
                    f"{_field_label('account_bank_receipt_unreadable')}核验："
                    "证据A=开户行文本，证据B=回单图片（OCR未识别出支行），结论：需人工核验回单原图"
                )
            account_values.append(
                {
                    "value_norm": "receipt_present_but_unreadable",
                    "value_raw_examples": ["回单图片存在，但支行字段不可识别"],
                    "count": len(receipt_blocks),
                    "examples": [asdict(occ)],
                }
            )
            findings.append(
                ConsistencyFinding(
                    type="account_bank_receipt_unreadable",
                    status=_status_for("account_bank_receipt_unreadable"),
                    severity=_severity_for("account_bank_receipt_unreadable"),
                    reason=reason,
                    values=account_values,
                    pairs=[pair] if pair else [],
                    comparison=_comparison_from_pair(pair if pair else None, "需人工核验"),
                    scope="cross_evidence",
                )
            )

    account_number_map = bucket.get("account_number") or {}
    if account_number_map and receipt_blocks:
        receipt_account_map: dict[str, dict[str, Any]] = {}
        for block in receipt_blocks:
            occ = _collect_occurrence(block)
            for number in _extract_account_numbers(block.text or ""):
                _add_value(
                    {"receipt_account_number": receipt_account_map},
                    fact_type="receipt_account_number",
                    value_norm=number,
                    value_raw=number,
                    occ=occ,
                )
        if receipt_account_map:
            account_top = _top_value(account_number_map)
            receipt_top = _top_value(receipt_account_map)
            if account_top and receipt_top:
                account_value, account_data = account_top
                receipt_value, receipt_data = receipt_top
                if account_value != receipt_value:
                    pairs = [_make_pair(account_value, account_data, receipt_value, receipt_data)]
                    values = _format_values(
                        {
                            account_value: account_data,
                            receipt_value: receipt_data,
                        },
                        max_examples_per_value=max_examples_per_value,
                    )
                    reason = (
                        f"{_field_label('account_number_receipt_mismatch')}核验："
                        f"证据A=“{account_value}”({_loc_text(pairs[0]['left'])})，"
                        f"证据B=“{receipt_value}”({_loc_text(pairs[0]['right'])})，"
                        "结论：不一致"
                    )
                    findings.append(
                        ConsistencyFinding(
                            type="account_number_receipt_mismatch",
                            status=_status_for("account_number_receipt_mismatch"),
                            severity=_severity_for("account_number_receipt_mismatch"),
                            reason=reason,
                            values=values,
                            pairs=pairs,
                            comparison=_comparison_from_pair(pairs[0], "不一致"),
                            scope="cross_evidence",
                        )
                    )

    # High-signal placeholder check for legal representative name in fixed forms.
    legal_name_values = bucket.get("legal_representative_name") or {}
    for norm_value, data in legal_name_values.items():
        if norm_value not in _PLACEHOLDER_PERSON_NAMES:
            continue
        values = _format_values({norm_value: data}, max_examples_per_value=max_examples_per_value)
        example = values[0]["examples"][0] if values and values[0].get("examples") else {}
        reason = (
            f"{_field_label('legal_representative_name_placeholder')}疑似占位符："
            f"“{(values[0].get('value_raw_examples') or [''])[0]}”({_loc_text(example)})，请人工核验真实身份信息"
        )
        findings.append(
            ConsistencyFinding(
                type="legal_representative_name_placeholder",
                status=_status_for("legal_representative_name_placeholder"),
                severity=_severity_for("legal_representative_name_placeholder"),
                reason=reason,
                values=values,
                pairs=[],
                scope="bid_internal",
            )
        )

    # Cross-doc tender-number mismatch (if tender text contains extractable number).
    if tender_list:
        tender_number_bucket = _init_bucket()
        for block in tender_list:
            occ = _collect_occurrence(block)
            for code in _extract_tender_numbers(block.text or ""):
                _add_value(
                    tender_number_bucket,
                    fact_type="tender_no",
                    value_norm=code,
                    value_raw=code,
                    occ=occ,
                )
        tender_values = set(tender_number_bucket.get("tender_no", {}).keys())
        bid_values = set(bucket.get("tender_no", {}).keys())
        if tender_values and bid_values:
            unmatched_bid = sorted(value for value in bid_values if value not in tender_values)
            if unmatched_bid:
                values: list[dict[str, Any]] = []
                for value in unmatched_bid[:3]:
                    bid_entry = (bucket.get("tender_no") or {}).get(value) or {}
                    values.append(
                        {
                            "value_norm": value,
                            "value_raw_examples": sorted(list(bid_entry.get("raw", {value})))[:3],
                            "count": int(bid_entry.get("count") or 0),
                            "examples": list((bid_entry.get("examples") or [])[:max_examples_per_value]),
                            "expected_values": sorted(tender_values)[:3],
                        }
                    )
                pairs: list[dict[str, Any]] = []
                left_example = values[0]["examples"][0] if values and values[0].get("examples") else {}
                tender_first = sorted(tender_values)[0]
                tender_first_entry = (tender_number_bucket.get("tender_no") or {}).get(tender_first) or {}
                tender_example = (tender_first_entry.get("examples") or [{}])[0]
                pairs.append(
                    {
                        "left": {
                            "value": (values[0].get("value_raw_examples") or [values[0].get("value_norm")])[0],
                            "count": values[0].get("count"),
                            "doc_id": left_example.get("doc_id"),
                            "location": left_example.get("location"),
                            "excerpt": left_example.get("excerpt"),
                        },
                        "right": {
                            "value": tender_first,
                            "count": tender_first_entry.get("count"),
                            "doc_id": tender_example.get("doc_id"),
                            "location": tender_example.get("location"),
                            "excerpt": tender_example.get("excerpt"),
                        },
                    }
                )
                reason = (
                    f"{_field_label('tender_no_cross_doc')}核验："
                    f"证据A=“{pairs[0]['left']['value']}”({_loc_text(pairs[0]['left'])})，"
                    f"证据B=“{pairs[0]['right']['value']}”({_loc_text(pairs[0]['right'])})，"
                    "结论：不一致"
                )
                findings.append(
                    ConsistencyFinding(
                        type="tender_no_cross_doc",
                        status=_status_for("tender_no_cross_doc"),
                        severity=_severity_for("tender_no_cross_doc"),
                        reason=reason,
                        values=values,
                        pairs=pairs,
                        comparison=_comparison_from_pair(pairs[0], "不一致"),
                        scope="cross_doc",
                    )
                )

    return findings
