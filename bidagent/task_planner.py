from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from bidagent.io_utils import path_ready, read_jsonl, write_jsonl
from bidagent.models import Requirement

MAX_TASKS_PER_REQUIREMENT = 6
_VALID_PRIORITIES = {"hard_fail", "scored", "general"}


def _row_to_requirement(row: dict[str, Any]) -> Requirement:
    return Requirement(
        requirement_id=row["requirement_id"],
        text=row["text"],
        category=row.get("category", "business_other"),
        mandatory=bool(row.get("mandatory", False)),
        keywords=list(row.get("keywords", [])),
        constraints=list(row.get("constraints", [])),
        rule_tier=str(row.get("rule_tier") or "general"),
        source=row.get("source", {}),
    )


def _normalize_priority(rule_tier: str | None) -> str:
    tier = str(rule_tier or "general")
    return tier if tier in _VALID_PRIORITIES else "general"


def _fallback_keywords(text: str, limit: int = 4) -> list[str]:
    terms = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text or "")
    merged: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        merged.append(term)
        if len(merged) >= limit:
            break
    return merged


def _constraint_task_type(constraint: dict[str, Any]) -> str:
    type_name = str(constraint.get("type") or "")
    if type_name == "amount":
        return "amount_check"
    if type_name == "term":
        return "term_check"
    if type_name == "quantity":
        return "quantity_check"
    return "constraint_check"


def _constraint_logic(constraint: dict[str, Any]) -> dict[str, Any]:
    logic: dict[str, Any] = {"mode": "constraint"}
    for key in ("type", "field", "op", "value", "value_fen", "unit", "raw"):
        if key not in constraint:
            continue
        value = constraint.get(key)
        if value is None:
            continue
        logic[key] = value
    return logic


def _constraint_query(requirement_text: str, constraint: dict[str, Any]) -> str:
    type_name = str(constraint.get("type") or "constraint")
    raw = str(constraint.get("raw") or "").strip()
    if raw:
        return f"核验{type_name}约束是否满足: {raw}"
    return f"核验{type_name}约束是否满足: {requirement_text}"


def decompose_requirement(requirement: Requirement) -> list[dict[str, Any]]:
    priority = _normalize_priority(getattr(requirement, "rule_tier", "general"))
    keywords = [item for item in requirement.keywords if isinstance(item, str) and item.strip()]
    if not keywords:
        keywords = _fallback_keywords(requirement.text)

    tasks: list[dict[str, Any]] = [
        {
            "task_type": "evidence_check",
            "query": f"核对是否存在可定位证据支持该要求: {requirement.text}",
            "expected_logic": {
                "mode": "evidence_presence",
                "mandatory": bool(requirement.mandatory),
                "requirement_text": requirement.text,
            },
            "priority": priority,
        },
        {
            "task_type": "keyword_check",
            "query": "核对关键词是否被有效证据覆盖: " + ", ".join(keywords[:5]),
            "expected_logic": {
                "mode": "keyword_coverage",
                "keywords": keywords[:8],
                "min_hits": 1 if keywords else 0,
            },
            "priority": priority,
        },
    ]

    seen_constraints: set[tuple[Any, ...]] = set()
    for constraint in requirement.constraints:
        if not isinstance(constraint, dict):
            continue
        signature = (
            constraint.get("type"),
            constraint.get("field"),
            constraint.get("op"),
            constraint.get("value"),
            constraint.get("value_fen"),
            constraint.get("unit"),
            constraint.get("raw"),
        )
        if signature in seen_constraints:
            continue
        seen_constraints.add(signature)
        tasks.append(
            {
                "task_type": _constraint_task_type(constraint),
                "query": _constraint_query(requirement.text, constraint),
                "expected_logic": _constraint_logic(constraint),
                "priority": priority,
            }
        )
        if len(tasks) >= MAX_TASKS_PER_REQUIREMENT:
            break

    final_tasks = tasks[:MAX_TASKS_PER_REQUIREMENT]
    for index, task in enumerate(final_tasks, start=1):
        task["task_id"] = f"T-{requirement.requirement_id}-{index:02d}"
        task["requirement_id"] = requirement.requirement_id
    return final_tasks


def plan_review_tasks(requirements: Iterable[Requirement]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for requirement in requirements:
        rows.extend(decompose_requirement(requirement))
    return rows


def load_requirements(requirements_path: Path) -> list[Requirement]:
    return [_row_to_requirement(row) for row in read_jsonl(requirements_path)]


def ensure_review_tasks(
    requirements_path: Path,
    review_tasks_path: Path,
    *,
    requirements_atomic_path: Path | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    def _load_atomic_enabled_ids(path: Path) -> set[str]:
        enabled_ids: set[str] = set()
        for row in read_jsonl(path):
            if not isinstance(row, dict):
                continue
            requirement_id = str(row.get("requirement_id") or "").strip()
            if not requirement_id:
                continue
            classification = str(row.get("classification") or "").strip()
            engine_enabled = bool(row.get("engine_enabled"))
            if classification and not engine_enabled:
                if classification != "fluff":
                    enabled_ids.add(requirement_id)
                continue
            if engine_enabled or classification != "fluff":
                enabled_ids.add(requirement_id)
        return enabled_ids

    if path_ready(review_tasks_path, resume):
        return list(read_jsonl(review_tasks_path))

    requirements = load_requirements(requirements_path)
    if requirements_atomic_path is not None and requirements_atomic_path.exists():
        enabled_requirement_ids = _load_atomic_enabled_ids(requirements_atomic_path)
        requirements = [item for item in requirements if item.requirement_id in enabled_requirement_ids]
    planned = plan_review_tasks(requirements)
    review_tasks_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(review_tasks_path, planned)
    return list(read_jsonl(review_tasks_path))


def plan_review_tasks_file(
    requirements_path: Path,
    review_tasks_path: Path,
    *,
    requirements_atomic_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    used_existing = path_ready(review_tasks_path, resume)
    tasks = ensure_review_tasks(
        requirements_path=requirements_path,
        review_tasks_path=review_tasks_path,
        requirements_atomic_path=requirements_atomic_path,
        resume=resume,
    )
    return {
        "review_tasks": len(tasks),
        "used_existing": used_existing,
    }

