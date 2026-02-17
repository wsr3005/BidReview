from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from bidagent.io_utils import write_jsonl

ALLOWED_STATUS = {"pass", "risk", "fail", "needs_ocr", "insufficient_evidence"}


class TaskReviewer(Protocol):
    provider: str
    model: str

    def review_task(self, task: dict[str, Any]) -> dict[str, Any]:
        ...


def _review_task(reviewer: TaskReviewer, task: dict[str, Any]) -> dict[str, Any]:
    review_task = getattr(reviewer, "review_task", None)
    if callable(review_task):
        return review_task(task)
    review = getattr(reviewer, "review", None)
    if callable(review):
        return review(task)
    raise RuntimeError("Reviewer does not implement review_task(task) or review(task)")


def _base_status(task: dict[str, Any]) -> str:
    for key in ("rule_status", "status"):
        status = str(task.get(key) or "").strip()
        if status in ALLOWED_STATUS:
            return status
    if bool(task.get("needs_ocr")):
        return "needs_ocr"
    return "insufficient_evidence"


def _base_reason(task: dict[str, Any], status: str) -> str:
    for key in ("rule_reason", "reason"):
        reason = str(task.get(key) or "").strip()
        if reason:
            return reason
    if status == "needs_ocr":
        return "Only OCR-reference evidence available, OCR verification required."
    if status == "fail":
        return "Rule fallback indicates failed requirement."
    if status == "risk":
        return "Rule fallback indicates manual review risk."
    return "Insufficient evidence for final verdict."


def _normalize_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, confidence))


def _normalize_evidence_refs(task: dict[str, Any]) -> list[dict[str, Any]]:
    raw = task.get("evidence_refs")
    if not isinstance(raw, list):
        raw = task.get("evidence")
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "evidence_id": item.get("evidence_id"),
                "excerpt_hash": item.get("excerpt_hash"),
                "doc_id": item.get("doc_id"),
                "location": item.get("location"),
                "score": item.get("score"),
            }
        )
    return normalized


def _ensure_trace(task: dict[str, Any], *, index: int) -> dict[str, Any]:
    trace = task.get("decision_trace")
    if isinstance(trace, dict):
        result = dict(trace)
    else:
        result = {}
    result.setdefault("task_id", task.get("task_id") or f"T{index + 1:04d}")
    clause_id = task.get("clause_id") or task.get("requirement_id")
    if clause_id:
        result.setdefault("clause_id", clause_id)
    result.setdefault("evidence_refs", _normalize_evidence_refs(task))
    return result


def _apply_final_decision(
    row: dict[str, Any],
    trace: dict[str, Any],
    *,
    status: str,
    reason: str,
    source: str,
    confidence: float | None,
) -> dict[str, Any]:
    final_status = status if status in ALLOWED_STATUS else "insufficient_evidence"
    if final_status != status:
        trace["fallback"] = {
            "type": "invalid_final_status",
            "invalid_status": status,
            "applied_status": final_status,
        }
    trace["decision"] = {
        "status": final_status,
        "reason": reason,
        "source": source,
    }
    row["status"] = final_status
    row["reason"] = reason
    row["confidence"] = confidence
    row["evidence_refs"] = trace.get("evidence_refs", [])
    row["decision_trace"] = trace
    return row


def judge_tasks_with_llm(
    tasks: list[dict[str, Any]],
    reviewer: TaskReviewer,
    *,
    min_confidence: float = 0.65,
) -> list[dict[str, Any]]:
    threshold = float(min_confidence)
    provider = str(getattr(reviewer, "provider", "unknown"))
    model = str(getattr(reviewer, "model", "unknown"))
    verdicts: list[dict[str, Any]] = []

    for index, task in enumerate(tasks):
        row = dict(task)
        trace = _ensure_trace(row, index=index)
        base_status = _base_status(row)
        base_reason = _base_reason(row, base_status)

        if base_status == "needs_ocr":
            trace["llm_review"] = {"provider": provider, "model": model, "skipped": "needs_ocr"}
            verdicts.append(
                _apply_final_decision(
                    row,
                    trace,
                    status="needs_ocr",
                    reason=base_reason,
                    source="rule",
                    confidence=None,
                )
            )
            continue

        try:
            llm_result = _review_task(reviewer, row)
            if not isinstance(llm_result, dict):
                raise RuntimeError("LLM reviewer returned non-dict result")
        except Exception as exc:  # noqa: BLE001
            trace["llm_review"] = {"provider": provider, "model": model, "error": str(exc)}
            trace["fallback"] = {
                "type": "llm_error",
                "applied_status": base_status,
                "applied_reason": base_reason,
            }
            verdicts.append(
                _apply_final_decision(
                    row,
                    trace,
                    status=base_status,
                    reason=base_reason,
                    source="rule_fallback",
                    confidence=None,
                )
            )
            continue

        raw_status = str(llm_result.get("status") or "").strip()
        llm_reason = str(llm_result.get("reason") or "").strip() or base_reason
        confidence = _normalize_confidence(llm_result.get("confidence"))

        trace["llm_review"] = {
            "provider": provider,
            "model": model,
            "status": raw_status,
            "confidence": confidence,
        }

        if raw_status not in ALLOWED_STATUS:
            trace["fallback"] = {
                "type": "invalid_llm_status",
                "invalid_status": raw_status,
                "applied_status": base_status,
                "applied_reason": base_reason,
            }
            verdicts.append(
                _apply_final_decision(
                    row,
                    trace,
                    status=base_status,
                    reason=base_reason,
                    source="rule_fallback",
                    confidence=confidence,
                )
            )
            continue

        if raw_status == "pass" and (confidence is None or confidence < threshold):
            fallback_reason = f"LLM confidence too low ({confidence if confidence is not None else 'N/A'}), downgraded to risk."
            trace["low_confidence_fallback"] = {
                "min_confidence": threshold,
                "confidence": confidence,
                "action": "downgrade_pass_to_risk",
                "previous": {"status": raw_status, "reason": llm_reason},
            }
            verdicts.append(
                _apply_final_decision(
                    row,
                    trace,
                    status="risk",
                    reason=fallback_reason,
                    source="llm_low_confidence_fallback",
                    confidence=confidence,
                )
            )
            continue

        verdicts.append(
            _apply_final_decision(
                row,
                trace,
                status=raw_status,
                reason=llm_reason,
                source="llm",
                confidence=confidence,
            )
        )

    return verdicts


def write_verdicts_jsonl(path: Path, verdicts: list[dict[str, Any]]) -> int:
    for index, row in enumerate(verdicts, start=1):
        status = str(row.get("status") or "").strip()
        if status not in ALLOWED_STATUS:
            raise ValueError(f"Invalid verdict status at row {index}: {status}")
        if "confidence" not in row:
            raise ValueError(f"Missing confidence at row {index}")
        if not isinstance(row.get("evidence_refs"), list):
            raise ValueError(f"Missing or invalid evidence_refs at row {index}")
        if not isinstance(row.get("decision_trace"), dict):
            raise ValueError(f"Missing or invalid decision_trace at row {index}")
    return write_jsonl(path, verdicts)
