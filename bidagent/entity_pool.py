from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Iterable

_SPACE_PATTERN = re.compile(r"\s+")
_ORG_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9（）()·\-.]{2,64}"
    r"(?:有限责任公司|股份有限公司|集团有限公司|有限公司|集团|公司|研究院|研究所|大学|银行|医院|中心|委员会)"
)
_PERSON_LABEL_PATTERN = re.compile(r"(?:法定代表人|联系人|委托代理人|授权代表)[:：]?\s*([\u4e00-\u9fff·]{2,8})")
_OCR_CONFUSION_MAP = str.maketrans(
    {
        "0": "o",
        "1": "l",
        "2": "z",
        "5": "s",
        "6": "g",
        "8": "b",
        "o": "o",
        "O": "o",
        "I": "l",
        "l": "l",
        "S": "s",
        "B": "b",
    }
)
_ORG_SUFFIX_RE = re.compile(r"(有限责任公司|股份有限公司|集团有限公司|有限公司|集团|公司)$")


def _utc_now_z() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_space(value: Any) -> str:
    return _SPACE_PATTERN.sub("", str(value or "")).strip()


def _normalize_alias(value: Any) -> str:
    compact = _normalize_space(value)
    compact = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", compact)
    return compact


def _normalize_ocr(value: Any) -> str:
    compact = _normalize_alias(value)
    lowered = compact.lower().translate(_OCR_CONFUSION_MAP)
    return lowered


def _org_key(value: Any) -> str:
    compact = _normalize_ocr(value)
    return _ORG_SUFFIX_RE.sub("", compact) or compact


def _person_key(value: Any) -> str:
    return _normalize_ocr(value)


def _iter_mentions(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in rows:
        if not isinstance(row, dict):
            continue
        doc_id = str(row.get("doc_id") or "unknown")
        text = str(row.get("text") or row.get("content") or "").strip()
        if not text:
            continue
        for match in _ORG_PATTERN.findall(text):
            alias = match.strip()
            if len(alias) < 4:
                continue
            yield {"entity_type": "organization", "alias": alias, "doc_id": doc_id}
        for match in _PERSON_LABEL_PATTERN.findall(text):
            alias = match.strip()
            if len(alias) < 2:
                continue
            yield {"entity_type": "person", "alias": alias, "doc_id": doc_id}


def build_entity_pool(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    alias_docs: dict[tuple[str, str], set[str]] = defaultdict(set)
    alias_counts: dict[tuple[str, str], int] = defaultdict(int)

    for mention in _iter_mentions(rows):
        entity_type = str(mention.get("entity_type") or "")
        alias = str(mention.get("alias") or "").strip()
        doc_id = str(mention.get("doc_id") or "unknown")
        if not entity_type or not alias:
            continue
        if entity_type == "organization":
            key = _org_key(alias)
        else:
            key = _person_key(alias)
        if not key:
            continue
        group_key = (entity_type, key)
        if group_key not in grouped:
            grouped[group_key] = {
                "entity_type": entity_type,
                "mentions": 0,
                "aliases": defaultdict(int),
                "doc_ids": set(),
            }
        grouped[group_key]["mentions"] = int(grouped[group_key]["mentions"]) + 1
        grouped[group_key]["aliases"][alias] += 1
        grouped[group_key]["doc_ids"].add(doc_id)
        alias_counts[(entity_type, alias)] += 1
        alias_docs[(entity_type, alias)].add(doc_id)

    entities: list[dict[str, Any]] = []
    for index, ((entity_type, _), payload) in enumerate(
        sorted(grouped.items(), key=lambda item: (-int(item[1].get("mentions") or 0), item[0][0], item[0][1])),
        start=1,
    ):
        alias_counter = payload.get("aliases")
        if not isinstance(alias_counter, dict) or not alias_counter:
            continue
        sorted_aliases = sorted(
            alias_counter.items(),
            key=lambda item: (-int(item[1]), -len(str(item[0])), str(item[0])),
        )
        canonical_name = str(sorted_aliases[0][0])
        aliases = [
            {
                "alias": alias,
                "normalized": _normalize_alias(alias),
                "ocr_normalized": _normalize_ocr(alias),
                "count": int(count),
                "doc_ids": sorted(alias_docs.get((entity_type, alias), set())),
            }
            for alias, count in sorted_aliases
        ]
        token = "ORG" if entity_type == "organization" else "PER"
        entities.append(
            {
                "entity_id": f"ENT-{token}-{index:04d}",
                "entity_type": entity_type,
                "canonical_name": canonical_name,
                "aliases": aliases,
                "doc_ids": sorted(payload.get("doc_ids", set())),
                "mentions": int(payload.get("mentions") or 0),
            }
        )

    alias_index = []
    for entity in entities:
        entity_id = str(entity.get("entity_id") or "")
        for alias in entity.get("aliases") or []:
            if not isinstance(alias, dict):
                continue
            alias_index.append(
                {
                    "alias": alias.get("alias"),
                    "normalized": alias.get("normalized"),
                    "ocr_normalized": alias.get("ocr_normalized"),
                    "entity_id": entity_id,
                    "entity_type": entity.get("entity_type"),
                }
            )

    return {
        "schema_version": "entity-pool-v1",
        "generated_at": _utc_now_z(),
        "entities": entities,
        "alias_index": alias_index,
    }
