from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bidagent.io_utils import ensure_dir, read_jsonl, write_jsonl


@dataclass(slots=True)
class EvalMetrics:
    total: int
    hard_fail_total: int
    hard_fail_fail: int
    hard_fail_missed: int  # expected hard_fail but got not-fail
    hard_fail_recall: float
    false_positive_fail: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    gold_map = {row["requirement_id"]: row for row in gold_rows if isinstance(row, dict) and row.get("requirement_id")}

    pred_rows = list(read_jsonl(findings_path))
    pred_map = {row["requirement_id"]: row for row in pred_rows if isinstance(row, dict) and row.get("requirement_id")}

    total = 0
    hard_fail_total = 0
    hard_fail_fail = 0
    hard_fail_missed = 0
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
            if got == "fail":
                hard_fail_fail += 1
            else:
                hard_fail_missed += 1

        if expected != "fail" and got == "fail":
            false_positive_fail += 1

        per_item.append(
            {
                "requirement_id": req_id,
                "tier": tier,
                "expected_status": expected,
                "got_status": got,
                "ok": expected == got,
            }
        )

    recall = (hard_fail_fail / hard_fail_total) if hard_fail_total else 0.0
    metrics = EvalMetrics(
        total=total,
        hard_fail_total=hard_fail_total,
        hard_fail_fail=hard_fail_fail,
        hard_fail_missed=hard_fail_missed,
        hard_fail_recall=recall,
        false_positive_fail=false_positive_fail,
    ).to_dict()

    result = {"metrics": metrics, "items": per_item}
    if out_path:
        ensure_dir(out_path.parent)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_and_write(run_dir: Path) -> dict[str, Any]:
    out_path = run_dir / "eval" / "metrics.json"
    return evaluate_run(run_dir, out_path=out_path)

