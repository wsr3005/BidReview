from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bidagent.io_utils import ensure_dir, read_jsonl

VALID_GOLD_STATUSES = {"pass", "risk", "fail", "needs_ocr", "insufficient_evidence"}
VALID_GOLD_TIERS = {"hard_fail", "scored", "general"}
HARD_FAIL_BLOCKING_STATUSES = {"fail", "risk", "needs_ocr", "insufficient_evidence"}


@dataclass(slots=True)
class EvalMetrics:
    total: int
    hard_fail_total: int
    hard_fail_blocked: int  # hard_fail rows that are blocked by non-pass result
    hard_fail_passed: int  # hard_fail rows that slipped to pass
    hard_fail_fail: int
    hard_fail_missed: int  # alias of hard_fail_passed for compatibility
    hard_fail_recall: float
    non_fail_total: int
    false_positive_fail: int
    false_positive_fail_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_gold_tier(row: dict[str, Any]) -> str:
    return str(row.get("tier") or row.get("rule_tier") or "general").strip()


def validate_gold_rows(
    rows: list[dict[str, Any]],
    *,
    min_rows: int = 1,
    require_all_tiers: bool = False,
    require_all_statuses: bool = False,
    min_per_tier: int = 0,
    min_per_status: int = 0,
) -> dict[str, Any]:
    errors: list[str] = []
    requirement_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    tier_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            errors.append(f"row {index}: must be an object")
            continue

        requirement_id = str(row.get("requirement_id") or "").strip()
        if not requirement_id:
            errors.append(f"row {index}: requirement_id is required")
        elif requirement_id in requirement_ids:
            duplicate_ids.add(requirement_id)
        else:
            requirement_ids.add(requirement_id)

        tier = _normalize_gold_tier(row)
        expected_status = str(row.get("expected_status") or "").strip()

        if tier not in VALID_GOLD_TIERS:
            errors.append(
                f"row {index}: tier '{tier}' is invalid (allowed: {', '.join(sorted(VALID_GOLD_TIERS))})"
            )
        else:
            tier_counts[tier] += 1

        if expected_status not in VALID_GOLD_STATUSES:
            errors.append(
                "row "
                + f"{index}: expected_status '{expected_status}' is invalid "
                + f"(allowed: {', '.join(sorted(VALID_GOLD_STATUSES))})"
            )
        else:
            status_counts[expected_status] += 1

    if duplicate_ids:
        errors.append(f"duplicate requirement_id values: {', '.join(sorted(duplicate_ids))}")

    if len(rows) < min_rows:
        errors.append(f"gold set rows={len(rows)} is below minimum required rows={min_rows}")

    if require_all_tiers:
        missing_tiers = sorted(VALID_GOLD_TIERS - set(tier_counts.keys()))
        if missing_tiers:
            errors.append(f"missing tiers: {', '.join(missing_tiers)}")

    if require_all_statuses:
        missing_statuses = sorted(VALID_GOLD_STATUSES - set(status_counts.keys()))
        if missing_statuses:
            errors.append(f"missing expected_status labels: {', '.join(missing_statuses)}")

    if min_per_tier > 0:
        low_tiers = sorted(
            tier for tier in VALID_GOLD_TIERS if int(tier_counts.get(tier, 0)) < int(min_per_tier)
        )
        if low_tiers:
            errors.append(
                "tier stratification below minimum: "
                + ", ".join(f"{tier}={int(tier_counts.get(tier, 0))}" for tier in low_tiers)
                + f" (min={min_per_tier})"
            )

    if min_per_status > 0:
        low_statuses = sorted(
            status
            for status in VALID_GOLD_STATUSES
            if int(status_counts.get(status, 0)) < int(min_per_status)
        )
        if low_statuses:
            errors.append(
                "status stratification below minimum: "
                + ", ".join(f"{status}={int(status_counts.get(status, 0))}" for status in low_statuses)
                + f" (min={min_per_status})"
            )

    if errors:
        preview = errors[:20]
        detail = "\n".join(f"- {item}" for item in preview)
        if len(errors) > len(preview):
            detail = f"{detail}\n- ... (+{len(errors) - len(preview)} more)"
        raise ValueError(f"gold set validation failed:\n{detail}")

    return {
        "total": len(rows),
        "tier_counts": {key: int(value) for key, value in sorted(tier_counts.items())},
        "status_counts": {key: int(value) for key, value in sorted(status_counts.items())},
    }


def evaluate_run(run_dir: Path, *, out_path: Path | None = None) -> dict[str, Any]:
    """Compute basic quality metrics for a run using a gold set.

    Expects `run_dir/eval/gold.jsonl` with rows:
      - requirement_id: str
      - expected_status: pass|risk|fail|needs_ocr|insufficient_evidence
      - tier: hard_fail|scored|general (optional)
      - note: optional
    """
    gold_path = run_dir / "eval" / "gold.jsonl"
    findings_path = run_dir / "findings.jsonl"
    if not gold_path.exists():
        raise FileNotFoundError(f"gold set not found: {gold_path}")
    if not findings_path.exists():
        raise FileNotFoundError(f"findings not found: {findings_path}")

    gold_rows = list(read_jsonl(gold_path))
    validate_gold_rows(gold_rows)
    gold_map = {row["requirement_id"]: row for row in gold_rows if isinstance(row, dict) and row.get("requirement_id")}

    pred_rows = list(read_jsonl(findings_path))
    pred_map = {row["requirement_id"]: row for row in pred_rows if isinstance(row, dict) and row.get("requirement_id")}

    total = 0
    hard_fail_total = 0
    hard_fail_blocked = 0
    hard_fail_passed = 0
    hard_fail_fail = 0
    hard_fail_missed = 0
    non_fail_total = 0
    false_positive_fail = 0

    per_item: list[dict[str, Any]] = []
    for req_id, gold in gold_map.items():
        pred = pred_map.get(req_id, {})
        expected = str(gold.get("expected_status") or "")
        tier = str(gold.get("tier") or gold.get("rule_tier") or "general")
        got = str(pred.get("status") or "")
        total += 1

        if tier == "hard_fail":
            hard_fail_total += 1
            if got in HARD_FAIL_BLOCKING_STATUSES:
                hard_fail_blocked += 1
            else:
                hard_fail_passed += 1
            if got == "fail":
                hard_fail_fail += 1
            hard_fail_missed = hard_fail_passed

        if expected != "fail" and got == "fail":
            non_fail_total += 1
            false_positive_fail += 1
        elif expected != "fail":
            non_fail_total += 1

        per_item.append(
            {
                "requirement_id": req_id,
                "tier": tier,
                "expected_status": expected,
                "got_status": got,
                "ok": expected == got,
            }
        )

    # Recall is defined as "hard-fail blocking recall":
    # any non-pass result blocks release and is counted as intercepted.
    recall = (hard_fail_blocked / hard_fail_total) if hard_fail_total else 0.0
    false_positive_fail_rate = (false_positive_fail / non_fail_total) if non_fail_total else 0.0
    metrics = EvalMetrics(
        total=total,
        hard_fail_total=hard_fail_total,
        hard_fail_blocked=hard_fail_blocked,
        hard_fail_passed=hard_fail_passed,
        hard_fail_fail=hard_fail_fail,
        hard_fail_missed=hard_fail_missed,
        hard_fail_recall=recall,
        non_fail_total=non_fail_total,
        false_positive_fail=false_positive_fail,
        false_positive_fail_rate=false_positive_fail_rate,
    ).to_dict()

    result = {"metrics": metrics, "items": per_item}
    if out_path:
        ensure_dir(out_path.parent)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_and_write(run_dir: Path) -> dict[str, Any]:
    out_path = run_dir / "eval" / "metrics.json"
    return evaluate_run(run_dir, out_path=out_path)

