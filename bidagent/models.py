from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Location:
    block_index: int
    page: int | None = None
    section: str | None = None
    section_tag: str | None = None


@dataclass(slots=True)
class Block:
    doc_id: str
    text: str
    location: Location
    block_id: str | None = None
    block_type: str = "text"
    section_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        location_dict = asdict(self.location)
        block_id = self.block_id or f"B-{self.doc_id}-{self.location.block_index}"
        section_hint = self.section_hint if self.section_hint is not None else self.location.section
        return {
            "block_id": block_id,
            "doc_id": self.doc_id,
            "block_type": self.block_type,
            "section_hint": section_hint,
            "content": self.text,
            "text": self.text,
            "location": location_dict,
        }


@dataclass(slots=True)
class Requirement:
    requirement_id: str
    text: str
    category: str
    mandatory: bool
    keywords: list[str] = field(default_factory=list)
    # Minimal structured constraints for deterministic checks / consistency.
    # Each item is a dict like: {"type": "amount|term|quantity", "op": ">="|"<="|"="|None, "value": int|float|str,
    # "unit": "元|万元|天|月|年|份|套|台|项|个|..." , "raw": "..."}
    constraints: list[dict[str, Any]] = field(default_factory=list)
    # Requirement tier for prioritization and gating.
    # hard_fail: mandatory knockout clause; scored: evaluation/scoring item; general: other checkable requirement.
    rule_tier: str = "general"
    source: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Finding:
    requirement_id: str
    status: str
    score: int
    severity: str
    reason: str
    clause_id: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    decision_trace: dict[str, Any] | None = None
    llm: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
