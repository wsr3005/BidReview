from __future__ import annotations

import json
import os
import re
import hashlib
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bidagent import __version__
from bidagent.atomic import build_atomic_requirements
from bidagent.annotators import annotate_docx_copy, annotate_pdf_copy
from bidagent.consistency import find_inconsistencies
from bidagent.document import iter_document_blocks
from bidagent.doc_map import build_ingest_doc_map
from bidagent.evidence_harvester import (
    build_evidence_index as build_task_evidence_index,
    harvest_task_evidence,
    write_evidence_packs_jsonl,
)
from bidagent.evidence_index import build_unified_evidence_index, retrieve_evidence_candidates
from bidagent.entity_pool import build_entity_pool
from bidagent.eval import evaluate_run
from bidagent.io_utils import append_jsonl, ensure_dir, path_ready, read_jsonl, write_jsonl
from bidagent.llm_judge import judge_tasks_with_llm, write_verdicts_jsonl
from bidagent.llm import (
    DEEPSEEK_PROMPT_VERSION,
    DeepSeekRequirementExtractor,
    DeepSeekReviewer,
    apply_llm_review,
)
from bidagent.models import Block, Finding, Location, Requirement
from bidagent.ocr import iter_document_ocr_blocks, ocr_selfcheck
from bidagent.review import (
    REVIEW_RULE_ENGINE,
    REVIEW_RULE_VERSION,
    enforce_evidence_quality_gate,
    extract_requirements,
    extract_requirements_with_llm,
    review_requirements,
)
from bidagent.task_planner import ensure_review_tasks

VALID_RELEASE_MODES = {"assist_only", "auto_final"}
VALID_GATE_FAIL_FAST_MODES = {"off", "critical", "all"}
CRITICAL_GATE_CHECKS = {"hard_fail_recall", "false_positive_fail_rate", "evidence_traceability"}
DEFAULT_GATE_THRESHOLDS = {
    "auto_review_coverage": 0.95,
    "hard_fail_recall": 0.98,
    "false_positive_fail_rate": 0.01,
    "evidence_traceability": 1.0,
    "llm_coverage": 1.0,
    "missing_rate": 1.0,
}
VERDICT_STRATEGY_VERSION = "verdict-harvest-v1"
CANARY_POLICY_VERSION = "release-canary-v1"
RUN_METADATA_SCHEMA_VERSION = "run-metadata-v1"
RELEASE_TRACE_SCHEMA_VERSION = "release-trace-v1"
AUTO_FINAL_GUARD_POLICY_VERSION = "auto-final-guard-v1"
AUTO_FINAL_HISTORY_FILENAME = "auto-final-history.jsonl"
DEFAULT_CANARY_MIN_STREAK = 3
CROSS_AUDIT_SCHEMA_VERSION = "cross-audit-v1"
BLOCKING_FINDING_STATUSES = {"fail", "needs_ocr", "missing", "risk", "insufficient_evidence"}
STATUS_PRIORITY = ("fail", "needs_ocr", "missing", "risk", "pass")
STATUS_PRIORITY_INDEX = {name: index for index, name in enumerate(STATUS_PRIORITY)}
EVIDENCE_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{2,}")
POSITIVE_EVIDENCE_HINTS = (
    "已提供",
    "已提交",
    "已附",
    "提供",
    "提交",
    "附",
    "符合",
    "满足",
    "具备",
    "通过",
    "有效",
    "齐全",
)
NEGATIVE_EVIDENCE_HINTS = (
    "未提供",
    "未提交",
    "未附",
    "未见",
    "缺失",
    "缺少",
    "未满足",
    "不满足",
    "不符合",
    "不具备",
    "不通过",
    "无效",
    "无",
    "没有",
)
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


def _row_to_block(row: dict[str, Any]) -> Block:
    location = row.get("location") if isinstance(row.get("location"), dict) else {}
    section_hint = row.get("section_hint") or location.get("section")
    section_tag = row.get("section_tag") or location.get("section_tag")
    text = str(row.get("text") or row.get("content") or "")
    return Block(
        doc_id=row["doc_id"],
        text=text,
        location=Location(
            block_index=location.get("block_index", 0),
            page=location.get("page"),
            section=location.get("section"),
            section_tag=section_tag,
        ),
        block_id=row.get("block_id"),
        block_type=str(row.get("block_type") or "text"),
        section_hint=section_hint,
    )


def _iter_blocks_from_jsonl(path: Path):
    for row in read_jsonl(path):
        yield _row_to_block(row)


def _row_to_requirement(row: dict[str, Any]) -> Requirement:
    return Requirement(
        requirement_id=row["requirement_id"],
        text=row["text"],
        category=row.get("category", "商务其他"),
        mandatory=bool(row.get("mandatory", False)),
        keywords=list(row.get("keywords", [])),
        constraints=list(row.get("constraints", [])),
        rule_tier=str(row.get("rule_tier") or "general"),
        source=row.get("source", {}),
    )


def _load_api_key(provider: str | None, api_key_file: Path | None) -> str | None:
    if provider != "deepseek":
        return None
    env_key = os.getenv("DEEPSEEK_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    if api_key_file and api_key_file.exists():
        value = api_key_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    default_file = Path("deepseek api key.txt")
    if default_file.exists():
        value = default_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY or provide --ai-api-key-file.")


def _truthy_env(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _count_doc_media(path: Path, page_range: tuple[int, int] | None) -> dict[str, Any]:
    """Best-effort doc stats for OCR gating and run introspection.

    Notes:
    - DOCX: counts embedded images under word/media/.
    - PDF: counts images exposed by pypdf page.images (may be 0 for some PDFs).
    - TXT: images=0.
    """
    suffix = path.suffix.lower()
    if suffix == ".docx":
        import zipfile

        from bidagent.ocr import OCR_IMAGE_EXTENSIONS

        images = 0
        try:
            with zipfile.ZipFile(path, "r") as zf:
                for name in zf.namelist():
                    if not name.startswith("word/media/"):
                        continue
                    if Path(name).suffix.lower() in OCR_IMAGE_EXTENSIONS:
                        images += 1
        except Exception:  # noqa: BLE001
            images = 0
        return {"type": "docx", "images": images}

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ModuleNotFoundError:
            return {"type": "pdf", "pages": None, "images": None}

        try:
            reader = PdfReader(str(path))
            total_pages = len(reader.pages)
        except Exception:  # noqa: BLE001
            return {"type": "pdf", "pages": None, "images": None}

        start_page, end_page = 1, total_pages
        if page_range:
            start_page, end_page = page_range
            end_page = min(end_page, total_pages)

        images = 0
        for page_no in range(start_page, end_page + 1):
            try:
                page = reader.pages[page_no - 1]
                page_images = list(getattr(page, "images", []) or [])
                images += len(page_images)
            except Exception:  # noqa: BLE001
                continue
        return {"type": "pdf", "pages": total_pages, "images": images, "page_range": [start_page, end_page]}

    if suffix == ".txt":
        return {"type": "txt", "images": 0}

    return {"type": suffix.lstrip("."), "images": None}


def _enforce_required_ai(ai_provider: str | None) -> None:
    if ai_provider is not None:
        return
    if _truthy_env("BIDAGENT_ALLOW_NO_AI"):
        return
    raise ValueError(
        "AI review is required. Use --ai-provider deepseek and provide an API key. "
        "For offline tests only, set BIDAGENT_ALLOW_NO_AI=1."
    )


def _enforce_required_ocr(ocr_mode: str, *, doc_meta: dict[str, Any]) -> None:
    if ocr_mode != "off":
        return
    if _truthy_env("BIDAGENT_ALLOW_NO_OCR"):
        return
    images = doc_meta.get("images")
    if images is None or int(images) > 0:
        raise ValueError(
            "OCR is required (image evidence detected). "
            "Install OCR deps + Tesseract, then rerun. "
            "For offline tests only, set BIDAGENT_ALLOW_NO_OCR=1."
        )


def _enforce_ocr_engine_available(ocr_stats: dict[str, Any], *, doc_meta: dict[str, Any]) -> None:
    images = doc_meta.get("images")
    if images is not None and int(images) <= 0:
        return
    if bool(ocr_stats.get("engine_available")):
        return
    reason = ocr_stats.get("reason") or "OCR engine unavailable"
    raise ValueError(
        "OCR is required but not available. "
        f"Reason: {reason}. "
        "Install: pip install pillow pytesseract, and install Tesseract OCR (set TESSERACT_CMD if needed)."
    )


def _parse_manifest_page_range(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    start, end = value
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    return (start, end)


def _llm_coverage_complete(rows: list[dict[str, Any]], provider: str) -> bool:
    if not rows:
        return False
    return all((row.get("llm") or {}).get("provider") == provider for row in rows)


def _validate_release_mode(release_mode: str) -> str:
    value = str(release_mode or "").strip()
    if value in VALID_RELEASE_MODES:
        return value
    allowed = ", ".join(sorted(VALID_RELEASE_MODES))
    raise ValueError(f"release_mode must be one of: {allowed}")


def _utc_now_z() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:  # noqa: BLE001
        return None
    if result.returncode != 0:
        return None
    value = str(result.stdout or "").strip()
    return value or None


def _resolve_prompt_version(provider: str | None) -> str:
    if provider == "deepseek":
        return DEEPSEEK_PROMPT_VERSION
    return "rule-only-v1"


def _build_run_metadata(
    *,
    out_dir: Path,
    focus: str,
    page_range: tuple[int, int] | None,
    ocr_mode: str,
    ai_provider: str | None,
    ai_model: str,
    ai_base_url: str,
    ai_workers: int,
    ai_min_confidence: float,
    extract_batch_size: int,
    extract_batch_max_chars: int,
    extract_timeout_seconds: int,
    requested_release_mode: str,
    canary_min_streak: int,
) -> dict[str, Any]:
    release_dir = out_dir / "release"
    ensure_dir(release_dir)
    out_path = release_dir / "run-metadata.json"

    provider = ai_provider or "none"
    model_name = ai_model if ai_provider else None
    model_base_url = ai_base_url if ai_provider else None
    run_id = datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")
    commit = _safe_git_value("rev-parse", "--short", "HEAD")
    branch = _safe_git_value("rev-parse", "--abbrev-ref", "HEAD")

    payload = {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": _utc_now_z(),
        "bidagent_version": __version__,
        "git": {
            "branch": branch,
            "commit": commit,
        },
        "model": {
            "provider": provider,
            "name": model_name,
            "base_url": model_base_url,
        },
        "prompt": {
            "provider": provider,
            "version": _resolve_prompt_version(ai_provider),
        },
        "strategy": {
            "review_rule_engine": REVIEW_RULE_ENGINE,
            "review_rule_version": REVIEW_RULE_VERSION,
            "verdict_strategy_version": VERDICT_STRATEGY_VERSION,
            "canary_policy_version": CANARY_POLICY_VERSION,
            "auto_final_guard_policy_version": AUTO_FINAL_GUARD_POLICY_VERSION,
        },
        "run_config": {
            "focus": focus,
            "page_range": list(page_range) if page_range else None,
            "ocr_mode": ocr_mode,
            "release_mode_requested": requested_release_mode,
            "ai_workers": int(ai_workers),
            "ai_min_confidence": float(ai_min_confidence),
            "extract_batch_size": int(extract_batch_size),
            "extract_batch_max_chars": int(extract_batch_max_chars),
            "extract_timeout_seconds": int(extract_timeout_seconds),
            "canary_min_streak": int(canary_min_streak),
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _collect_release_artifacts(out_dir: Path) -> list[dict[str, Any]]:
    files = [
        "requirements.jsonl",
        "requirements.atomic.jsonl",
        "review-tasks.jsonl",
        "evidence-packs.jsonl",
        "cross-audit.jsonl",
        "findings.jsonl",
        "verdicts.jsonl",
        "gate-result.json",
        "annotations.jsonl",
        "manual-review.jsonl",
        "consistency-findings.jsonl",
        "review-report.md",
        "eval/metrics.json",
        "ingest/doc-map.json",
        "ingest/entity-pool.json",
    ]
    rows: list[dict[str, Any]] = []
    for rel in files:
        path = out_dir / rel
        if not path.exists() or not path.is_file():
            continue
        rows.append(
            {
                "path": rel.replace("\\", "/"),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _validate_canary_min_streak(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("canary_min_streak must be a positive integer") from exc
    if parsed < 1:
        raise ValueError("canary_min_streak must be >= 1")
    return parsed


def _auto_final_history_path(out_dir: Path) -> Path:
    return out_dir.parent / AUTO_FINAL_HISTORY_FILENAME


def _load_auto_final_core_streak(history_path: Path) -> int:
    if not history_path.exists():
        return 0
    streak = 0
    for row in read_jsonl(history_path):
        if bool(row.get("core_ok")):
            streak += 1
        else:
            streak = 0
    return streak


def _append_auto_final_history(
    *,
    history_path: Path,
    out_dir: Path,
    requested_release_mode: str,
    gate_eligible: bool,
    core_ok: bool,
    guard_ok: bool,
    streak_before: int,
    streak_after: int,
    canary_status: str,
    final_release_mode: str,
) -> None:
    ensure_dir(history_path.parent)
    append_jsonl(
        history_path,
        [
            {
                "generated_at": _utc_now_z(),
                "policy_version": AUTO_FINAL_GUARD_POLICY_VERSION,
                "run_dir": str(out_dir),
                "requested_release_mode": requested_release_mode,
                "gate_eligible_for_auto_final": bool(gate_eligible),
                "core_ok": bool(core_ok),
                "guard_ok": bool(guard_ok),
                "streak_before": int(streak_before),
                "streak_after": int(streak_after),
                "canary_status": canary_status,
                "final_release_mode": final_release_mode,
            }
        ],
    )


def _run_canary(
    *,
    out_dir: Path,
    requested_release_mode: str,
    gate_result: dict[str, Any],
    min_streak: int = DEFAULT_CANARY_MIN_STREAK,
) -> dict[str, Any]:
    release_dir = out_dir / "release"
    ensure_dir(release_dir)
    out_path = release_dir / "canary-result.json"

    findings_path = out_dir / "findings.jsonl"
    verdicts_path = out_dir / "verdicts.jsonl"
    findings_rows = list(read_jsonl(findings_path)) if findings_path.exists() else []
    verdict_rows = list(read_jsonl(verdicts_path)) if verdicts_path.exists() else []

    canary_required = requested_release_mode == "auto_final"
    checks: list[dict[str, Any]] = []
    release_mode = "assist_only"
    status = "skipped"
    min_streak_value = _validate_canary_min_streak(min_streak)
    history_path = _auto_final_history_path(out_dir)
    streak_before = 0
    streak_after = 0
    core_ok = False
    guard_ok = False

    if canary_required:
        # Canary checks release integrity only.
        # Business thresholds must be enforced by gate() and reflected via gate_eligible_for_auto_final.
        gate_ok = bool(gate_result.get("eligible_for_auto_final"))
        checks.append(
            {
                "name": "gate_eligible_for_auto_final",
                "ok": gate_ok,
                "value": bool(gate_result.get("eligible_for_auto_final")),
                "blocking": True,
            }
        )

        required_files = [
            out_dir / "requirements.jsonl",
            out_dir / "requirements.atomic.jsonl",
            out_dir / "review-tasks.jsonl",
            out_dir / "evidence-packs.jsonl",
            out_dir / "cross-audit.jsonl",
            out_dir / "findings.jsonl",
            out_dir / "verdicts.jsonl",
            out_dir / "gate-result.json",
            out_dir / "review-report.md",
        ]
        files_ok = all(path.exists() for path in required_files)
        checks.append(
            {
                "name": "required_release_files_present",
                "ok": files_ok,
                "missing": [str(path.relative_to(out_dir)) for path in required_files if not path.exists()],
                "blocking": True,
            }
        )

        trace_ok = all(
            isinstance(row.get("decision_trace"), dict) and isinstance(row.get("evidence_refs"), list)
            for row in verdict_rows
        )
        checks.append(
            {
                "name": "verdict_trace_complete",
                "ok": trace_ok,
                "sample_size": len(verdict_rows),
                "blocking": True,
            }
        )

        signature_set = {
            (str((row.get("model") or {}).get("provider") or ""), str((row.get("model") or {}).get("name") or ""))
            for row in verdict_rows
        }
        signature_set = {item for item in signature_set if item != ("", "")}
        model_ok = len(signature_set) <= 1 and (len(verdict_rows) == 0 or len(signature_set) == 1)
        checks.append(
            {
                "name": "single_model_signature",
                "ok": model_ok,
                "signatures": [f"{provider}:{name}" for provider, name in sorted(signature_set)],
                "blocking": True,
            }
        )
        core_ok = all(
            bool(item.get("ok"))
            for item in checks
            if bool(item.get("blocking", True))
        )
        streak_before = _load_auto_final_core_streak(history_path)
        streak_after = streak_before + 1 if core_ok else 0
        guard_ok = core_ok and streak_after >= min_streak_value
        checks.append(
            {
                "name": "auto_final_guard_streak",
                "ok": guard_ok,
                "streak_before": streak_before,
                "streak_after": streak_after,
                "min_required": min_streak_value,
                "history_file": str(history_path),
                "blocking": True,
            }
        )
        canary_pass = all(
            bool(item.get("ok"))
            for item in checks
            if bool(item.get("blocking", True))
        )
        status = "pass" if canary_pass else "fail"
        release_mode = "auto_final" if canary_pass else "assist_only"
        _append_auto_final_history(
            history_path=history_path,
            out_dir=out_dir,
            requested_release_mode=requested_release_mode,
            gate_eligible=gate_ok,
            core_ok=core_ok,
            guard_ok=guard_ok,
            streak_before=streak_before,
            streak_after=streak_after,
            canary_status=status,
            final_release_mode=release_mode,
        )

    result = {
        "policy_version": CANARY_POLICY_VERSION,
        "generated_at": _utc_now_z(),
        "requested_release_mode": requested_release_mode,
        "canary_required": canary_required,
        "status": status,
        "release_mode": release_mode,
        "checks": checks,
        "guard": {
            "policy_version": AUTO_FINAL_GUARD_POLICY_VERSION,
            "enabled": canary_required,
            "min_consecutive_core_pass": min_streak_value,
            "history_file": str(history_path),
            "streak_before": streak_before if canary_required else None,
            "streak_after": streak_after if canary_required else None,
            "core_ok": core_ok if canary_required else None,
            "ok": guard_ok if canary_required else None,
        },
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _write_release_trace(
    *,
    out_dir: Path,
    run_metadata: dict[str, Any],
    gate_result: dict[str, Any],
    canary_result: dict[str, Any],
) -> dict[str, Any]:
    release_dir = out_dir / "release"
    ensure_dir(release_dir)
    out_path = release_dir / "release-trace.json"

    artifacts = _collect_release_artifacts(out_dir)
    payload = {
        "schema_version": RELEASE_TRACE_SCHEMA_VERSION,
        "generated_at": _utc_now_z(),
        "final_release_mode": canary_result.get("release_mode", "assist_only"),
        "requested_release_mode": gate_result.get("requested_release_mode", "assist_only"),
        "gate": {
            "eligible_for_auto_final": bool(gate_result.get("eligible_for_auto_final")),
            "release_mode": gate_result.get("release_mode", "assist_only"),
            "checks": gate_result.get("checks", []),
        },
        "canary": {
            "status": canary_result.get("status", "skipped"),
            "checks": canary_result.get("checks", []),
            "guard": canary_result.get("guard", {}),
        },
        "run_metadata": {
            "run_id": run_metadata.get("run_id"),
            "model": run_metadata.get("model"),
            "prompt": run_metadata.get("prompt"),
            "strategy": run_metadata.get("strategy"),
        },
        "artifacts": artifacts,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def ingest(
    tender_path: Path,
    bid_path: Path,
    out_dir: Path,
    resume: bool = False,
    page_range: tuple[int, int] | None = None,
    ocr_mode: str = "off",
) -> dict[str, Any]:
    ingest_dir = out_dir / "ingest"
    ensure_dir(ingest_dir)
    tender_out = ingest_dir / "tender_blocks.jsonl"
    bid_out = ingest_dir / "bid_blocks.jsonl"
    doc_map_path = ingest_dir / "doc-map.json"
    entity_pool_path = ingest_dir / "entity-pool.json"
    manifest_path = ingest_dir / "manifest.json"

    tender_meta = _count_doc_media(tender_path, page_range)
    bid_meta = _count_doc_media(bid_path, page_range)
    _enforce_required_ocr(ocr_mode, doc_meta=bid_meta)

    summary: dict[str, Any] = {
        "tender_blocks": 0,
        "bid_blocks": 0,
        "bid_ocr_blocks": 0,
        "tender_meta": tender_meta,
        "bid_meta": bid_meta,
    }
    previous_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_manifest = {}
    page_range_matches = _parse_manifest_page_range(previous_manifest.get("page_range")) == page_range
    ocr_mode_matches = str(previous_manifest.get("ocr_mode", "off")) == ocr_mode

    tender_can_resume = path_ready(tender_out, resume) and page_range_matches
    if not tender_can_resume:
        tender_rows = (
            block.to_dict()
            for block in iter_document_blocks(tender_path, doc_id="tender", page_range=page_range)
        )
        summary["tender_blocks"] = write_jsonl(tender_out, tender_rows)
    else:
        summary["tender_blocks"] = sum(1 for _ in read_jsonl(tender_out))

    bid_can_resume = path_ready(bid_out, resume) and page_range_matches and ocr_mode_matches
    if not bid_can_resume:
        bid_rows = (
            block.to_dict()
            for block in iter_document_blocks(bid_path, doc_id="bid", page_range=page_range)
        )
        bid_text_count = write_jsonl(bid_out, bid_rows)
        ocr_stats: dict[str, Any] = {}
        ocr_stats.update(ocr_selfcheck(ocr_mode))
        _enforce_ocr_engine_available(ocr_stats, doc_meta=bid_meta)
        ocr_rows = (
            block.to_dict()
            for block in iter_document_ocr_blocks(
                bid_path,
                doc_id="bid",
                start_index=bid_text_count,
                page_range=page_range,
                ocr_mode=ocr_mode,
                stats=ocr_stats,
            )
        )
        bid_ocr_count = append_jsonl(bid_out, ocr_rows)
        summary["bid_blocks"] = bid_text_count + bid_ocr_count
        summary["bid_ocr_blocks"] = bid_ocr_count
        # Merge OCR stats for visibility and debugging.
        summary["ocr"] = {
            **ocr_stats,
            "blocks_emitted": int(ocr_stats.get("blocks_emitted", bid_ocr_count) or 0),
            "images_total": int(ocr_stats.get("images_total", 0) or 0),
            "images_succeeded": int(ocr_stats.get("images_succeeded", 0) or 0),
            "images_failed": int(ocr_stats.get("images_failed", 0) or 0),
            "chars_total": int(ocr_stats.get("chars_total", 0) or 0),
        }
    else:
        summary["bid_blocks"] = sum(1 for _ in read_jsonl(bid_out))
        summary["bid_ocr_blocks"] = int(previous_manifest.get("bid_ocr_blocks", 0))
        if isinstance(previous_manifest.get("ocr"), dict):
            summary["ocr"] = previous_manifest.get("ocr")

    doc_map_can_resume = path_ready(doc_map_path, resume) and page_range_matches and ocr_mode_matches
    if not doc_map_can_resume:
        doc_map = build_ingest_doc_map(
            tender_rows=read_jsonl(tender_out),
            bid_rows=read_jsonl(bid_out),
        )
        doc_map_path.write_text(json.dumps(doc_map, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        try:
            doc_map = json.loads(doc_map_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc_map = build_ingest_doc_map(
                tender_rows=read_jsonl(tender_out),
                bid_rows=read_jsonl(bid_out),
            )
            doc_map_path.write_text(json.dumps(doc_map, ensure_ascii=False, indent=2), encoding="utf-8")

    doc_entries = doc_map.get("docs") if isinstance(doc_map, dict) else []
    summary["doc_map"] = {
        "path": str(doc_map_path),
        "schema_version": str((doc_map or {}).get("schema_version") or "doc-map-v1"),
        "docs": len(doc_entries) if isinstance(doc_entries, list) else 0,
    }

    entity_pool_can_resume = path_ready(entity_pool_path, resume) and page_range_matches and ocr_mode_matches
    if not entity_pool_can_resume:
        entity_pool = build_entity_pool(
            [*read_jsonl(tender_out), *read_jsonl(bid_out)],
        )
        entity_pool_path.write_text(json.dumps(entity_pool, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        try:
            entity_pool = json.loads(entity_pool_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            entity_pool = build_entity_pool(
                [*read_jsonl(tender_out), *read_jsonl(bid_out)],
            )
            entity_pool_path.write_text(json.dumps(entity_pool, ensure_ascii=False, indent=2), encoding="utf-8")

    entities = entity_pool.get("entities") if isinstance(entity_pool, dict) else []
    summary["entity_pool"] = {
        "path": str(entity_pool_path),
        "schema_version": str((entity_pool or {}).get("schema_version") or "entity-pool-v1"),
        "entities": len(entities) if isinstance(entities, list) else 0,
    }

    manifest = {
        "tender_path": str(tender_path.resolve()),
        "bid_path": str(bid_path.resolve()),
        "ingest_cwd": str(Path.cwd().resolve()),
        "page_range": list(page_range) if page_range else None,
        "ocr_mode": ocr_mode,
        "bid_ocr_blocks": summary["bid_ocr_blocks"],
        "tender_meta": tender_meta,
        "bid_meta": bid_meta,
        "ocr": summary.get("ocr", ocr_selfcheck(ocr_mode)),
        "doc_map_path": str(doc_map_path),
        "doc_map_schema_version": str((doc_map or {}).get("schema_version") or "doc-map-v1"),
        "entity_pool_path": str(entity_pool_path),
        "entity_pool_schema_version": str((entity_pool or {}).get("schema_version") or "entity-pool-v1"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def extract_req(
    out_dir: Path,
    focus: str,
    resume: bool = False,
    ai_provider: str | None = None,
    ai_model: str = "deepseek-chat",
    ai_api_key_file: Path | None = None,
    ai_base_url: str = "https://api.deepseek.com/v1",
    ai_min_confidence: float = 0.65,
    extract_batch_size: int = 8,
    extract_batch_max_chars: int = 8000,
    extract_timeout_seconds: int = 45,
) -> dict[str, Any]:
    ingest_dir = out_dir / "ingest"
    tender_path = ingest_dir / "tender_blocks.jsonl"
    req_path = out_dir / "requirements.jsonl"
    atomic_path = out_dir / "requirements.atomic.jsonl"
    if path_ready(req_path, resume):
        requirement_rows = list(read_jsonl(req_path))
        total = len(requirement_rows)
        if atomic_path.exists():
            atomic_rows = list(read_jsonl(atomic_path))
        else:
            atomic_rows = build_atomic_requirements(_row_to_requirement(row) for row in requirement_rows)
            write_jsonl(atomic_path, atomic_rows)
        classification_counts = Counter(str(item.get("classification") or "") for item in atomic_rows)
        return {
            "requirements": total,
            "atomic_requirements": len(atomic_rows),
            "atomic_classification_counts": dict(classification_counts),
        }

    tender_rows = list(read_jsonl(tender_path))
    doc_map_data: dict[str, Any] = {}
    doc_map_path = ingest_dir / "doc-map.json"
    if doc_map_path.exists():
        try:
            doc_map_data = json.loads(doc_map_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc_map_data = {}
    tagged_tender_rows = _attach_semantic_section_tags(tender_rows, doc_map=doc_map_data, doc_id="tender")
    tender_blocks = [_row_to_block(row) for row in tagged_tender_rows]
    section_tag_counts = Counter(
        str((row.get("location") or {}).get("section_tag") or row.get("section_tag") or "other")
        for row in tagged_tender_rows
        if isinstance(row, dict)
    )
    requirements = extract_requirements(tender_blocks, focus=focus)
    summary: dict[str, Any] = {
        "extract_engine": "rule",
        "section_tag_counts": dict(section_tag_counts),
    }
    if ai_provider == "deepseek":
        cache_dir = out_dir / "cache"
        cache_path = cache_dir / "extract-semantic-cache.json"
        semantic_cache: dict[str, list[dict[str, Any]]] = {}
        if cache_path.exists():
            try:
                raw_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw_cache = {}
            if isinstance(raw_cache, dict):
                for key, value in raw_cache.items():
                    if not isinstance(key, str):
                        continue
                    if isinstance(value, list):
                        semantic_cache[key] = [item for item in value if isinstance(item, dict)]
        try:
            api_key = _load_api_key(ai_provider, ai_api_key_file)
            extractor = DeepSeekRequirementExtractor(
                api_key=api_key or "",
                model=ai_model,
                base_url=ai_base_url,
                timeout_seconds=max(10, int(extract_timeout_seconds)),
            )
            llm_requirements, llm_stats = extract_requirements_with_llm(
                tender_blocks=tender_blocks,
                focus=focus,
                extractor=extractor,
                min_confidence=max(0.5, float(ai_min_confidence) - 0.1),
                batch_size=max(1, int(extract_batch_size)),
                batch_max_chars=max(2000, int(extract_batch_max_chars)),
                semantic_cache=semantic_cache,
            )
            summary["extract_llm"] = llm_stats
            ensure_dir(cache_dir)
            cache_path.write_text(json.dumps(semantic_cache, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["extract_semantic_cache"] = {
                "path": str(cache_path),
                "hits": int((llm_stats or {}).get("cache_hits") or 0),
                "misses": int((llm_stats or {}).get("cache_misses") or 0),
                "entries": int((llm_stats or {}).get("cache_entries") or len(semantic_cache)),
            }
            rule_total = len(requirements)
            llm_total = len(llm_requirements)
            acceptance_ratio = llm_total / max(1, rule_total)
            if llm_total > 0 and acceptance_ratio >= 0.3:
                requirements = llm_requirements
                summary["extract_engine"] = "llm_schema_validated"
            else:
                summary["extract_engine"] = "rule_fallback"
                summary["extract_fallback_reason"] = "llm_low_coverage_or_empty"
        except Exception as exc:  # noqa: BLE001
            summary["extract_engine"] = "rule_fallback"
            summary["extract_fallback_reason"] = str(exc)

    total = write_jsonl(req_path, (item.to_dict() for item in requirements))
    atomic_rows = build_atomic_requirements(requirements)
    atomic_total = write_jsonl(atomic_path, atomic_rows)
    summary["atomic_requirements"] = atomic_total
    summary["atomic_classification_counts"] = dict(
        Counter(str(item.get("classification") or "") for item in atomic_rows)
    )
    summary["requirements"] = total
    return summary


def review(
    out_dir: Path,
    resume: bool = False,
    ai_provider: str | None = None,
    ai_model: str = "deepseek-chat",
    ai_api_key_file: Path | None = None,
    ai_base_url: str = "https://api.deepseek.com/v1",
    ai_workers: int = 4,
    ai_min_confidence: float = 0.65,
) -> dict[str, Any]:
    _enforce_required_ai(ai_provider)
    req_path = out_dir / "requirements.jsonl"
    bid_path = out_dir / "ingest" / "bid_blocks.jsonl"
    findings_path = out_dir / "findings.jsonl"
    if path_ready(findings_path, resume):
        existing_rows = list(read_jsonl(findings_path))
        if ai_provider is None:
            counts = Counter(row["status"] for row in existing_rows)
            return {"findings": sum(counts.values()), "status_counts": dict(counts)}
        if _llm_coverage_complete(existing_rows, ai_provider):
            counts = Counter(row["status"] for row in existing_rows)
            return {"findings": sum(counts.values()), "status_counts": dict(counts)}

    requirements = [_row_to_requirement(row) for row in read_jsonl(req_path)]

    bid_blocks = _iter_blocks_from_jsonl(bid_path)
    findings = review_requirements(requirements=requirements, bid_blocks=bid_blocks)
    # Apply a first-pass evidence gate to avoid wasting LLM calls on non-actionable findings.
    findings = enforce_evidence_quality_gate(requirements=requirements, findings=findings)

    if ai_provider == "deepseek":
        api_key = _load_api_key(ai_provider, ai_api_key_file)
        reviewer = DeepSeekReviewer(
            api_key=api_key or "",
            model=ai_model,
            base_url=ai_base_url,
        )
        findings = apply_llm_review(
            requirements=requirements,
            findings=findings,
            reviewer=reviewer,
            max_workers=ai_workers,
            min_confidence=ai_min_confidence,
        )
        # Re-apply after LLM in case the model upgraded a finding without usable evidence.
        findings = enforce_evidence_quality_gate(requirements=requirements, findings=findings)

    write_jsonl(findings_path, (item.to_dict() for item in findings))

    counts = Counter(item.status for item in findings)
    return {"findings": len(findings), "status_counts": dict(counts)}


def _resolve_bid_source(out_dir: Path, bid_source: Path | None) -> Path | None:
    if bid_source:
        resolved = bid_source.resolve()
        return resolved if resolved.exists() else bid_source
    manifest_path = out_dir / "ingest" / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    source = data.get("bid_path")
    if not source:
        return None
    path = Path(source)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        ingest_cwd = data.get("ingest_cwd")
        if ingest_cwd:
            candidates.append((Path(ingest_cwd) / path).resolve())
        candidates.append((manifest_path.parent / path).resolve())
        candidates.append(path.resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _is_blocking_finding_status(status: str) -> bool:
    return str(status or "").strip() in BLOCKING_FINDING_STATUSES


def _review_action_for_status(status: str) -> str:
    if status == "fail":
        return "请优先核查该条款是否确实不满足；如已满足，请补充对应证据页码。"
    if status == "needs_ocr":
        return "请人工核读扫描件/附件图片内容，确认是否满足该条款。"
    if status == "missing":
        return "请补充必需证明材料或页码定位后复核。"
    if status == "risk":
        return "请补充关键证明材料并重新复核。"
    return "请补充可定位证据后复核。"


def _consistency_evidence_candidates(
    row: dict[str, Any],
    *,
    preferred_doc_id: str | None = "bid",
) -> list[dict[str, Any]]:
    fact_type = str(row.get("type") or "consistency").strip() or "consistency"
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, int | None, str]] = set()

    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _append_candidate(
        *,
        doc_id: Any,
        location: Any,
        excerpt: Any,
        score: Any = 0,
        reference_only: bool = False,
    ) -> None:
        if not isinstance(location, dict):
            return
        doc = str(doc_id or "").strip()
        if not doc:
            return
        if preferred_doc_id and doc != str(preferred_doc_id).strip():
            return
        block_index = location.get("block_index")
        page = location.get("page")
        if not (isinstance(block_index, int) and block_index > 0) and not (isinstance(page, int) and page > 0):
            return
        excerpt_text = str(excerpt or "").strip()
        key = (
            doc,
            block_index if isinstance(block_index, int) else None,
            page if isinstance(page, int) else None,
            excerpt_text[:80],
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "evidence_id": f"C-{fact_type}-{len(candidates) + 1}",
                "doc_id": doc,
                "location": location,
                "excerpt": excerpt_text,
                "score": _safe_int(score),
                "reference_only": bool(reference_only),
                "has_action": len(excerpt_text) >= 12,
            }
        )

    comparison = row.get("comparison")
    if isinstance(comparison, dict):
        evidence_a = comparison.get("evidence_a")
        if isinstance(evidence_a, dict):
            _append_candidate(
                doc_id=evidence_a.get("doc_id"),
                location=evidence_a.get("location"),
                excerpt=evidence_a.get("excerpt"),
                score=3,
            )
        evidence_b = comparison.get("evidence_b")
        if isinstance(evidence_b, dict):
            _append_candidate(
                doc_id=evidence_b.get("doc_id"),
                location=evidence_b.get("location"),
                excerpt=evidence_b.get("excerpt"),
                score=2,
            )

    for pair in row.get("pairs") or []:
        if not isinstance(pair, dict):
            continue
        for side in ("left", "right"):
            item = pair.get(side)
            if not isinstance(item, dict):
                continue
            _append_candidate(
                doc_id=item.get("doc_id"),
                location=item.get("location"),
                excerpt=item.get("excerpt"),
                score=item.get("count"),
            )

    for value in row.get("values") or []:
        if not isinstance(value, dict):
            continue
        value_score = value.get("count")
        for example in value.get("examples") or []:
            if not isinstance(example, dict):
                continue
            _append_candidate(
                doc_id=example.get("doc_id"),
                location=example.get("location"),
                excerpt=example.get("excerpt"),
                score=value_score,
            )
    return candidates


def _choose_consistency_primary_evidence(
    row: dict[str, Any],
    evidence: Any,
    *,
    status: str | None = None,
    preferred_doc_id: str | None = None,
) -> dict[str, Any]:
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _location_rank(location: Any) -> tuple[int, int]:
        if not isinstance(location, dict):
            return (1, 1)
        section = str(location.get("section") or "").strip().upper()
        is_ocr_media = 1 if section == "OCR_MEDIA" else 0
        block_index = _safe_int(location.get("block_index"), default=0)
        return (is_ocr_media, -block_index)

    if isinstance(row, dict):
        value_candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, int | None, int | None, str]] = set()
        for value in row.get("values") or []:
            if not isinstance(value, dict):
                continue
            conflict_count = max(0, _safe_int(value.get("count"), default=0))
            for example in value.get("examples") or []:
                if not isinstance(example, dict):
                    continue
                doc_id = str(example.get("doc_id") or "").strip()
                if not doc_id:
                    continue
                if preferred_doc_id and doc_id != str(preferred_doc_id).strip():
                    continue
                location = example.get("location")
                if not _is_mappable_location(location):
                    continue
                excerpt = str(example.get("excerpt") or "").strip()
                block_index = location.get("block_index") if isinstance(location, dict) else None
                page = location.get("page") if isinstance(location, dict) else None
                key = (
                    doc_id,
                    block_index if isinstance(block_index, int) else None,
                    page if isinstance(page, int) else None,
                    excerpt[:80],
                )
                if key in seen:
                    continue
                seen.add(key)
                value_candidates.append(
                    {
                        "doc_id": doc_id,
                        "location": location,
                        "excerpt": excerpt,
                        "score": conflict_count,
                        "reference_only": False,
                        "has_action": len(excerpt) >= 12,
                        "_conflict_count": conflict_count,
                    }
                )
        if value_candidates:
            # For consistency issues, annotate the outlier value first so reviewers jump
            # to the suspicious edit instead of the dominant repeated value.
            value_candidates.sort(
                key=lambda item: (
                    int(item.get("_conflict_count") or 0),
                    _location_rank(item.get("location")),
                    -len(str(item.get("excerpt") or "")),
                )
            )
            chosen = value_candidates[0]
            chosen_doc = str(chosen.get("doc_id") or "").strip()
            chosen_location = chosen.get("location") if isinstance(chosen.get("location"), dict) else {}
            chosen_excerpt = str(chosen.get("excerpt") or "").strip()
            chosen_block = chosen_location.get("block_index") if isinstance(chosen_location, dict) else None
            chosen_page = chosen_location.get("page") if isinstance(chosen_location, dict) else None
            for item in evidence if isinstance(evidence, list) else []:
                if not isinstance(item, dict):
                    continue
                item_doc = str(item.get("doc_id") or "").strip()
                if item_doc != chosen_doc:
                    continue
                item_location = item.get("location") if isinstance(item.get("location"), dict) else {}
                item_block = item_location.get("block_index")
                item_page = item_location.get("page")
                if item_block != chosen_block or item_page != chosen_page:
                    continue
                item_excerpt = str(item.get("excerpt") or "").strip()
                if chosen_excerpt and item_excerpt and item_excerpt[:80] != chosen_excerpt[:80]:
                    continue
                return item
            chosen.pop("_conflict_count", None)
            return chosen

    return _choose_primary_evidence(
        evidence,
        status=status,
        preferred_doc_id=preferred_doc_id,
    )


def annotate(
    out_dir: Path,
    resume: bool = False,
    bid_source: Path | None = None,
    blocking_only: bool = False,
) -> dict[str, Any]:
    annotations_path = out_dir / "annotations.jsonl"
    markdown_path = out_dir / "annotations.md"
    findings_path = out_dir / "findings.jsonl"
    consistency_path = out_dir / "consistency-findings.jsonl"
    annotation_doc_id = "bid"

    source_path = _resolve_bid_source(out_dir=out_dir, bid_source=bid_source)
    annotated_dir = out_dir / "annotated"
    pointer_path = annotated_dir / "annotated-copy.json"

    def _load_last_copy() -> Path | None:
        if pointer_path.exists():
            try:
                data = json.loads(pointer_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            candidate = data.get("annotated_copy")
            if candidate:
                path = Path(candidate)
                if path.exists():
                    return path
        # Backward-compatible fallback: older runs used a fixed name without timestamp,
        # and some resume flows may not have written the pointer yet.
        if source_path is not None:
            legacy = annotated_dir / f"{source_path.stem}.annotated{source_path.suffix}"
            if legacy.exists():
                return legacy
            pattern = f"{source_path.stem}.annotated.*{source_path.suffix}"
            matches = list(annotated_dir.glob(pattern))
            if matches:
                matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return matches[0]
        return None

    if path_ready(annotations_path, resume) and path_ready(markdown_path, resume):
        total = sum(1 for _ in read_jsonl(annotations_path))
        response: dict[str, Any] = {"annotations": total}
        last_copy = _load_last_copy()
        if last_copy is not None:
            response["annotated_copy"] = str(last_copy)
        return response

    findings = list(read_jsonl(findings_path)) if findings_path.exists() else []
    issue_rows = []
    for row in findings:
        if row["status"] == "pass":
            continue
        if blocking_only and not _is_blocking_finding_status(str(row.get("status") or "")):
            continue
        evidence = row.get("evidence", [])
        primary = _choose_primary_evidence(
            evidence,
            status=str(row.get("status") or ""),
            preferred_doc_id=annotation_doc_id,
        )
        if not primary:
            primary = {}
        alternates = []
        for item in _choose_alternate_evidence(
            evidence,
            primary,
            status=str(row.get("status") or ""),
            preferred_doc_id=annotation_doc_id,
            limit=2,
        ):
            alternates.append(
                {
                    "doc_id": item.get("doc_id", "bid"),
                    "location": item.get("location"),
                    "excerpt": item.get("excerpt"),
                    "score": item.get("score"),
                }
            )
        issue_rows.append(
            {
                "requirement_id": row["requirement_id"],
                "status": row["status"],
                "severity": row["severity"],
                "reason": row["reason"],
                "review_action": _review_action_for_status(str(row.get("status") or "")),
                "target": {
                    "doc_id": primary.get("doc_id", "bid"),
                    "location": primary.get("location"),
                    "excerpt": primary.get("excerpt"),
                    "score": primary.get("score"),
                },
                "alternate_targets": alternates,
                "note": (
                    f"[{row['severity']}] {row['requirement_id']} {row['status']}: "
                    f"{row['reason']} | 复核动作：{_review_action_for_status(str(row.get('status') or ''))}"
                ),
            }
        )

    if consistency_path.exists():
        for row in read_jsonl(consistency_path):
            status = str(row.get("status") or "").strip()
            if not status or status == "pass":
                continue
            if blocking_only and not _is_blocking_finding_status(status):
                continue
            evidence = _consistency_evidence_candidates(row, preferred_doc_id=annotation_doc_id)
            primary = _choose_consistency_primary_evidence(
                row,
                evidence,
                status=status,
                preferred_doc_id=annotation_doc_id,
            )
            if not primary:
                primary = {}
            alternates = []
            for item in _choose_alternate_evidence(
                evidence,
                primary,
                status=status,
                preferred_doc_id=annotation_doc_id,
                limit=2,
            ):
                alternates.append(
                    {
                        "doc_id": item.get("doc_id", annotation_doc_id),
                        "location": item.get("location"),
                        "excerpt": item.get("excerpt"),
                        "score": item.get("score"),
                    }
                )
            issue_id = f"C-{str(row.get('type') or 'consistency').strip() or 'consistency'}"
            severity = str(row.get("severity") or "medium")
            reason = str(row.get("reason") or "一致性核验发现异常")
            issue_rows.append(
                {
                    "requirement_id": issue_id,
                    "status": status,
                    "severity": severity,
                    "reason": reason,
                    "review_action": _review_action_for_status(status),
                    "target": {
                        "doc_id": primary.get("doc_id", annotation_doc_id),
                        "location": primary.get("location"),
                        "excerpt": primary.get("excerpt"),
                        "score": primary.get("score"),
                    },
                    "alternate_targets": alternates,
                    "source": "consistency",
                    "note": (
                        f"[{severity}] {issue_id} {status}: "
                        f"{reason} | 复核动作：{_review_action_for_status(status)}"
                    ),
                }
            )

    write_jsonl(annotations_path, issue_rows)

    lines = ["# Review Annotations", ""]
    for item in issue_rows:
        location = item["target"].get("location") or {}
        alt_count = len(item.get("alternate_targets", []))
        lines.append(
            "- "
            + (
                f"{item['note']} | block={location.get('block_index')} page={location.get('page')} "
                f"score={item['target'].get('score')} alt={alt_count}"
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = {"annotations": len(issue_rows)}
    if not issue_rows:
        result["annotated_copy"] = None
        result["annotation_note"] = "未发现阻断问题，未生成批注副本。"
        return result

    if source_path is None or not source_path.exists():
        result["annotated_copy"] = None
        result["annotation_warning"] = "Bid source file not found. Sidecar annotation files were generated."
        return result

    suffix = source_path.suffix.lower()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{source_path.stem}.annotated.{timestamp}{source_path.suffix}"
    output_path = annotated_dir / base_name
    # Avoid collisions if called multiple times within the same second.
    counter = 2
    while output_path.exists():
        output_path = annotated_dir / f"{source_path.stem}.annotated.{timestamp}-{counter}{source_path.suffix}"
        counter += 1
    try:
        annotated_dir.mkdir(parents=True, exist_ok=True)
        if suffix == ".docx":
            stats = annotate_docx_copy(source_path=source_path, output_path=output_path, issues=issue_rows)
        elif suffix == ".pdf":
            stats = annotate_pdf_copy(source_path=source_path, output_path=output_path, issues=issue_rows)
        else:
            result["annotated_copy"] = None
            result["annotation_warning"] = f"Unsupported annotation source type: {source_path.suffix}"
            return result
    except Exception as exc:  # noqa: BLE001
        result["annotated_copy"] = None
        result["annotation_warning"] = str(exc)
        return result

    if not output_path.exists():
        result["annotated_copy"] = None
        result["annotation_warning"] = (
            "No document copy was generated because findings lack mappable locations "
            "(block_index/page). Sidecar files were generated."
        )
        result.update(stats)
        return result

    result["annotated_copy"] = str(output_path)
    pointer_path.write_text(
        json.dumps(
            {
                "annotated_copy": str(output_path),
                "generated_at": timestamp,
                "source": str(source_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    result.update(stats)
    return result


def checklist(out_dir: Path, resume: bool = False) -> dict[str, Any]:
    review_jsonl = out_dir / "manual-review.jsonl"
    review_md = out_dir / "manual-review.md"
    findings_path = out_dir / "findings.jsonl"
    requirements_path = out_dir / "requirements.jsonl"

    if path_ready(review_jsonl, resume) and path_ready(review_md, resume):
        total = sum(1 for _ in read_jsonl(review_jsonl))
        return {"manual_review": total}

    req_map = {item["requirement_id"]: item for item in read_jsonl(requirements_path)}
    review_rows: list[dict[str, Any]] = []
    for finding in read_jsonl(findings_path):
        requirement = req_map.get(finding["requirement_id"], {})
        tier = str(requirement.get("rule_tier") or "general")
        is_manual = finding.get("status") != "pass"
        if not is_manual:
            continue
        evidence = finding.get("evidence", [])
        normalized_status = _normalize_status(finding.get("status") or "missing")
        primary = _choose_primary_evidence(evidence, status=normalized_status)
        if not primary:
            primary = {}
        review_rows.append(
            {
                "requirement_id": finding["requirement_id"],
                "tier": tier,
                "status": normalized_status,
                "severity": finding.get("severity", "medium"),
                "reason": finding.get("reason", ""),
                "category": requirement.get("category", "N/A"),
                "requirement_text": requirement.get("text", ""),
                "target": {
                    "doc_id": primary.get("doc_id", "bid"),
                    "location": primary.get("location"),
                    "excerpt": primary.get("excerpt"),
                },
            }
        )

    tier_rank = {"hard_fail": 0, "scored": 1, "general": 2}
    severity_rank = {"high": 0, "medium": 1, "low": 2, "none": 3}
    review_rows.sort(
        key=lambda row: (
            tier_rank.get(str(row.get("tier") or "general"), 9),
            0 if str(row.get("status")) in {"fail"} else 1 if str(row.get("status")) in {"missing"} else 2,
            severity_rank.get(str(row.get("severity") or "medium"), 9),
            str(row.get("requirement_id") or ""),
        )
    )

    write_jsonl(review_jsonl, review_rows)

    lines = [
        "# Manual Review Checklist",
        "",
        f"- total: {len(review_rows)}",
        "",
    ]
    for row in review_rows:
        location = row["target"].get("location") or {}
        lines.append(
            "- "
            + f"[{row.get('tier','general')}/{row['status']}/{row['severity']}] {row['requirement_id']} "
            + f"({row['category']}) block={location.get('block_index')} page={location.get('page')} "
            + f"| {row['reason']}"
        )
    review_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"manual_review": len(review_rows)}


def consistency(out_dir: Path, resume: bool = False) -> dict[str, Any]:
    out_path = out_dir / "consistency-findings.jsonl"
    if path_ready(out_path, resume):
        total = sum(1 for _ in read_jsonl(out_path))
        return {"consistency_findings": total}

    bid_path = out_dir / "ingest" / "bid_blocks.jsonl"
    tender_path = out_dir / "ingest" / "tender_blocks.jsonl"
    bid_blocks = _iter_blocks_from_jsonl(bid_path)
    tender_blocks = _iter_blocks_from_jsonl(tender_path) if tender_path.exists() else None
    findings = find_inconsistencies(bid_blocks, tender_blocks=tender_blocks)
    total = write_jsonl(out_path, (item.to_dict() for item in findings))
    return {"consistency_findings": total}


def _format_trace_location(location: Any) -> str:
    if not isinstance(location, dict):
        return "block=N/A page=N/A"
    return f"block={location.get('block_index')} page={location.get('page')}"


def _sanitize_md_text(value: Any, *, limit: int = 200) -> str:
    text = str(value or "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # Avoid breaking the surrounding quotes in report formatting.
    text = text.replace('"', "'")
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _is_mappable_location(location: Any) -> bool:
    if not isinstance(location, dict):
        return False
    block_index = location.get("block_index")
    page = location.get("page")
    return (isinstance(block_index, int) and block_index > 0) or (isinstance(page, int) and page > 0)


def _choose_primary_evidence(
    evidence: Any,
    *,
    status: str | None = None,
    preferred_doc_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(evidence, list) or not evidence:
        return {}

    candidates = [item for item in evidence if isinstance(item, dict)]
    if preferred_doc_id:
        preferred = [
            item for item in candidates if str(item.get("doc_id") or "").strip() == str(preferred_doc_id).strip()
        ]
        if preferred:
            candidates = preferred
    if status == "needs_ocr":
        reference_candidates = [item for item in candidates if item.get("reference_only")]
        if reference_candidates:
            candidates = reference_candidates
    else:
        non_reference_candidates = [item for item in candidates if not item.get("reference_only")]
        if non_reference_candidates:
            candidates = non_reference_candidates

    def _ocr_hint_score(excerpt: str) -> int:
        score = 0
        for token in ("营业执照", "扫描件", "复印件", "影印件", "附件"):
            if token in excerpt:
                score += 1
        return score

    def _excerpt_readability_score(excerpt: str) -> int:
        compact = re.sub(r"\s+", "", excerpt or "")
        if not compact:
            return 0
        total = len(compact)
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", compact))
        latin_runs = len(re.findall(r"[A-Za-z]{3,}", compact))
        ratio = cjk_chars / max(1, total)
        if ratio >= 0.45 and cjk_chars >= 10:
            score = 3
        elif ratio >= 0.25 and cjk_chars >= 6:
            score = 2
        elif ratio >= 0.10 and cjk_chars >= 4:
            score = 1
        else:
            score = 0
        # De-prioritize OCR noise strings (long latin garbage mixed with sparse CJK).
        if latin_runs >= 8 and ratio < 0.20:
            score = max(0, score - 1)
        return score

    best: dict[str, Any] = {}
    best_key: tuple[int, int, int, int, int] = (-1, -1, -1, -1, -1)
    for item in candidates:
        if not isinstance(item, dict):
            continue
        location = item.get("location")
        mappable = 1 if _is_mappable_location(location) else 0
        has_action = 1 if item.get("has_action") else 0
        excerpt = str(item.get("excerpt") or "").strip()
        excerpt_len = len(excerpt)
        readable = _excerpt_readability_score(excerpt)
        try:
            score = int(item.get("score") or 0)
        except (TypeError, ValueError):
            score = 0
        if status == "needs_ocr":
            key = (mappable, _ocr_hint_score(excerpt), score, excerpt_len, has_action)
        else:
            key = (mappable, readable, score, has_action, excerpt_len)
        if key > best_key:
            best_key = key
            best = item
    return best


def _choose_alternate_evidence(
    evidence: Any,
    primary: dict[str, Any],
    *,
    status: str | None = None,
    preferred_doc_id: str | None = None,
    limit: int = 2,
) -> list[dict[str, Any]]:
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    if not isinstance(evidence, list) or not evidence:
        return []
    primary_id = primary.get("evidence_id")
    candidates = [item for item in evidence if isinstance(item, dict)]
    if preferred_doc_id:
        preferred = [
            item for item in candidates if str(item.get("doc_id") or "").strip() == str(preferred_doc_id).strip()
        ]
        if preferred:
            candidates = preferred
    alternates: list[dict[str, Any]] = []
    for item in sorted(
        candidates,
        key=lambda item: (
            1 if _is_mappable_location(item.get("location")) else 0,
            0 if status == "needs_ocr" and item.get("reference_only") else 1,
            0 if status != "needs_ocr" and item.get("reference_only") else 1,
            _safe_int(item.get("score") or 0),
            len(str(item.get("excerpt") or "")),
        ),
        reverse=True,
    ):
        if not isinstance(item, dict):
            continue
        if primary_id and item.get("evidence_id") == primary_id:
            continue
        alternates.append(item)
        if len(alternates) >= limit:
            break
    return alternates


def _format_finding_trace(finding: dict[str, Any]) -> str:
    trace = finding.get("decision_trace")
    clause_id = finding.get("clause_id") or finding.get("requirement_id", "N/A")
    if not isinstance(trace, dict):
        evidence = finding.get("evidence", [])
        if evidence:
            primary = evidence[0]
            evidence_ref = f"{primary.get('evidence_id', 'N/A')}@{_format_trace_location(primary.get('location'))}"
        else:
            evidence_ref = "none"
        return _sanitize_md_text(f"clause={clause_id}; evidence={evidence_ref}; rule=unknown", limit=240)

    clause_source = trace.get("clause_source", {})
    clause_location = _format_trace_location(clause_source.get("location"))
    evidence_refs = trace.get("evidence_refs", [])
    if evidence_refs:
        first_ref = evidence_refs[0]
        evidence_ref = f"{first_ref.get('evidence_id', 'N/A')}@{_format_trace_location(first_ref.get('location'))}"
    else:
        evidence_ref = "none"
    rule = trace.get("rule", {})
    rule_ref = f"{rule.get('engine', 'N/A')}:{rule.get('version', 'N/A')}"
    return _sanitize_md_text(
        f"clause={clause_id}@{clause_location}; evidence={evidence_ref}; rule={rule_ref}",
        limit=240,
    )


def _format_primary_evidence(finding: dict[str, Any], limit: int = 80) -> str:
    evidence = finding.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        return "evidence: none"
    primary = _choose_primary_evidence(evidence, status=str(finding.get("status") or ""))
    if not primary:
        return "evidence: none"
    location = primary.get("location")
    excerpt = _sanitize_md_text(primary.get("excerpt") or "", limit=limit)
    return (
        "evidence: "
        + f"{primary.get('evidence_id', 'N/A')} "
        + f"{_format_trace_location(location)} "
        + f"score={primary.get('score', 'N/A')} "
        + f"\"{excerpt}\""
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_gate_fail_fast_mode(fail_fast: str) -> str:
    value = str(fail_fast or "off").strip().lower()
    if value in VALID_GATE_FAIL_FAST_MODES:
        return value
    allowed = ", ".join(sorted(VALID_GATE_FAIL_FAST_MODES))
    raise ValueError(f"gate fail_fast must be one of: {allowed}")


def _resolve_gate_thresholds(threshold_overrides: dict[str, Any] | None) -> dict[str, float]:
    thresholds = dict(DEFAULT_GATE_THRESHOLDS)
    if not threshold_overrides:
        return thresholds

    allowed = ", ".join(sorted(DEFAULT_GATE_THRESHOLDS))
    for name, raw_value in threshold_overrides.items():
        if name not in DEFAULT_GATE_THRESHOLDS:
            raise ValueError(f"unknown gate threshold override '{name}', allowed: {allowed}")
        value = _safe_float(raw_value)
        if value is None:
            raise ValueError(f"gate threshold '{name}' must be numeric")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"gate threshold '{name}' must be between 0 and 1")
        thresholds[name] = value
    return thresholds


def _status_confidence_hint(status: str) -> float:
    normalized = _normalize_status(status)
    if normalized == "pass":
        return 0.70
    if normalized == "fail":
        return 0.80
    if normalized == "risk":
        return 0.50
    if normalized == "missing":
        return 0.40
    if normalized == "needs_ocr":
        return 0.35
    return 0.30


def _normalize_status(value: Any, *, default: str = "missing") -> str:
    status = str(value or "").strip()
    if status == "insufficient_evidence":
        return "missing"
    if status in STATUS_PRIORITY_INDEX:
        return status
    return default


def _status_not_looser_than(current: str, floor: str) -> str:
    current_normalized = _normalize_status(current)
    floor_normalized = _normalize_status(floor)
    if STATUS_PRIORITY_INDEX[floor_normalized] < STATUS_PRIORITY_INDEX[current_normalized]:
        return floor_normalized
    return current_normalized


def _is_real_llm_provider(provider: Any) -> bool:
    return str(provider or "").strip().lower() in {"deepseek"}


def _normalize_search_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", "", text)
    return text


def _extract_query_terms(values: list[Any], *, limit: int = 16) -> list[str]:
    def _token_variants(token: str) -> list[str]:
        variants = [token]
        # For long Chinese strings, add short contiguous slices to improve recall
        # without introducing a full tokenizer dependency.
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            for size in (2, 3, 4):
                if len(token) <= size:
                    continue
                for start in range(0, len(token) - size + 1):
                    variants.append(token[start : start + size])
        return variants

    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        raw = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple, set)) else str(value or "")
        for token in EVIDENCE_TOKEN_PATTERN.findall(raw):
            for variant in _token_variants(_normalize_search_text(token)):
                if len(variant) < 2 or variant in seen:
                    continue
                seen.add(variant)
                terms.append(variant)
                if len(terms) >= limit:
                    return terms
    return terms


def _build_evidence_id_from_row(row: dict[str, Any]) -> str:
    block_id = str(row.get("block_id") or "").strip()
    if block_id:
        return f"E-{block_id}"
    location = row.get("location") if isinstance(row.get("location"), dict) else {}
    page = location.get("page") if isinstance(location, dict) else None
    page_no = page if isinstance(page, int) else 0
    block_index = location.get("block_index") if isinstance(location, dict) else None
    block_no = block_index if isinstance(block_index, int) else 0
    section = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", str((location or {}).get("section") or "NA"))[:16]
    section = section or "NA"
    doc_id = str(row.get("doc_id") or "bid")
    return f"E-{doc_id}-p{page_no}-b{block_no}-s{section}"


def _to_evidence_ref(item: dict[str, Any], *, source: str, harvest_score: int | None = None) -> dict[str, Any]:
    location = item.get("location") if isinstance(item.get("location"), dict) else None
    if isinstance(location, dict):
        location = dict(location)
    evidence_id = str(item.get("evidence_id") or "").strip() or _build_evidence_id_from_row(item)
    source_type = str(item.get("source_type") or "").strip().lower() or None
    if source_type is None and isinstance(location, dict):
        section = str(location.get("section") or "").lower()
        if "ocr" in section:
            source_type = "ocr"
        elif "table" in section:
            source_type = "table"
        else:
            source_type = "text"
    section_tag = item.get("section_tag")
    if section_tag is None and isinstance(location, dict):
        section_tag = location.get("section_tag")
    score = _safe_float(item.get("score"))
    if harvest_score is not None:
        score = float(harvest_score)
    return {
        "evidence_id": evidence_id,
        "block_id": item.get("block_id"),
        "doc_id": item.get("doc_id") or "bid",
        "location": location,
        "source_type": source_type,
        "section_tag": section_tag,
        "score": score,
        "excerpt": item.get("excerpt"),
        "source": source,
    }


def _merge_evidence_refs(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("evidence_id") or "").strip()
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            merged.append(item)
    return merged


def _load_evidence_index(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    postings: dict[str, list[str]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return {
            "records": records,
            "postings": postings,
            "by_id": by_id,
            "stats": {"blocks_indexed": 0, "tokens_indexed": 0},
        }

    for row in read_jsonl(path):
        if not isinstance(row, dict):
            continue
        evidence_id = str(row.get("evidence_id") or "").strip()
        if not evidence_id:
            evidence_id = _build_evidence_id_from_row(row)
        if evidence_id in by_id:
            continue

        text = str(row.get("text") or "")
        terms = _extract_query_terms([text], limit=64)
        record = {
            "evidence_id": evidence_id,
            "block_id": row.get("block_id"),
            "doc_id": row.get("doc_id") or "bid",
            "location": row.get("location"),
            "source_type": row.get("source_type"),
            "section_tag": row.get("section_tag"),
            "excerpt": text[:240],
            "normalized_text": _normalize_search_text(text),
            "terms": terms,
        }
        records.append(record)
        by_id[evidence_id] = record
        for term in terms:
            postings.setdefault(term, []).append(evidence_id)

    return {
        "records": records,
        "postings": postings,
        "by_id": by_id,
        "stats": {"blocks_indexed": len(records), "tokens_indexed": len(postings)},
    }


def _evidence_polarity(text: str) -> int:
    compact = _normalize_search_text(text)
    if not compact:
        return 0
    positive_score = sum(1 for token in POSITIVE_EVIDENCE_HINTS if token in compact)
    negative_score = sum(1 for token in NEGATIVE_EVIDENCE_HINTS if token in compact)
    if positive_score > negative_score and positive_score > 0:
        return 1
    if negative_score > positive_score and negative_score > 0:
        return -1
    return 0


def _collect_query_terms_for_requirement(
    requirement_id: str,
    finding: dict[str, Any],
    task_rows: list[dict[str, Any]],
) -> list[str]:
    inputs: list[Any] = []
    for task in task_rows:
        if not isinstance(task, dict):
            continue
        inputs.append(task.get("query"))
        inputs.append(task.get("expected_logic"))
    inputs.extend([requirement_id, finding.get("reason"), finding.get("clause_id")])
    for item in finding.get("evidence") or []:
        if isinstance(item, dict):
            inputs.append(item.get("excerpt"))
    return _extract_query_terms(inputs, limit=120)


def _normalize_section_tag(value: Any) -> str | None:
    compact = _normalize_search_text(value)
    if not compact:
        return None
    if compact in {"evaluationrisk"}:
        return "evaluation_risk"
    if compact in {"businesscontract"}:
        return "business_contract"
    if compact in {"technicalspec"}:
        return "technical_spec"
    if compact in {"bidderinstruction"}:
        return "bidder_instruction"
    if compact in {"formatappendix"}:
        return "format_appendix"
    if compact in {"other"}:
        return "other"
    if any(token in compact for token in ("评标", "评审", "评分", "否决", "废标", "资格审查")):
        return "evaluation_risk"
    if any(token in compact for token in ("合同", "付款", "结算", "违约", "质保", "交货", "履约", "商务")):
        return "business_contract"
    if any(token in compact for token in ("技术", "参数", "规格", "性能", "接口", "配置", "方案")):
        return "technical_spec"
    if any(token in compact for token in ("投标人须知", "须知前附表", "投标须知", "须知")):
        return "bidder_instruction"
    if any(token in compact for token in ("格式", "附件", "附表", "模板", "封面")):
        return "format_appendix"
    return "other"


def _safe_block_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _doc_section_ranges(doc_map: dict[str, Any], *, doc_id: str) -> list[dict[str, Any]]:
    docs = doc_map.get("docs") if isinstance(doc_map, dict) else []
    if not isinstance(docs, list):
        return []
    for entry in docs:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("doc_id") or "") != doc_id:
            continue
        sections = entry.get("sections")
        if not isinstance(sections, list):
            return []
        rows: list[dict[str, Any]] = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            range_row = section.get("range") if isinstance(section.get("range"), dict) else {}
            start_block = _safe_block_int(range_row.get("start_block"))
            end_block = _safe_block_int(range_row.get("end_block"))
            if start_block is None or end_block is None:
                continue
            if end_block < start_block:
                end_block = start_block
            rows.append(
                {
                    "start_block": start_block,
                    "end_block": end_block,
                    "semantic_tag": _normalize_section_tag(section.get("semantic_tag")) or "other",
                }
            )
        rows.sort(key=lambda item: int(item.get("start_block") or 0))
        return rows
    return []


def _lookup_section_tag(block_index: Any, ranges: list[dict[str, Any]]) -> str | None:
    block = _safe_block_int(block_index)
    if block is None:
        return None
    for row in ranges:
        start_block = _safe_block_int(row.get("start_block"))
        end_block = _safe_block_int(row.get("end_block"))
        if start_block is None or end_block is None:
            continue
        if start_block <= block <= end_block:
            return _normalize_section_tag(row.get("semantic_tag")) or "other"
    return None


def _attach_semantic_section_tags(
    rows: list[dict[str, Any]],
    *,
    doc_map: dict[str, Any] | None,
    doc_id: str,
) -> list[dict[str, Any]]:
    section_ranges = _doc_section_ranges(doc_map or {}, doc_id=doc_id)
    if not section_ranges:
        return [dict(row) for row in rows if isinstance(row, dict)]

    tagged_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        output = dict(row)
        location = output.get("location") if isinstance(output.get("location"), dict) else {}
        block_index = location.get("block_index") if isinstance(location, dict) else None
        section_tag = _lookup_section_tag(block_index, section_ranges)
        if section_tag is not None:
            output["section_tag"] = section_tag
            if isinstance(location, dict):
                location_out = dict(location)
                location_out["section_tag"] = section_tag
                output["location"] = location_out
        tagged_rows.append(output)
    return tagged_rows


def _preferred_section_tags(*values: Any) -> list[str]:
    text = " ".join(str(value or "") for value in values)
    compact = _normalize_search_text(text)
    if not compact:
        return []
    tags: list[str] = []
    seen: set[str] = set()

    def _push(tag: str) -> None:
        if tag in seen:
            return
        seen.add(tag)
        tags.append(tag)

    if any(token in compact for token in ("评标", "评审", "评分", "否决", "废标", "资格审查")):
        _push("evaluation_risk")
    if any(token in compact for token in ("合同", "付款", "结算", "违约", "质保", "交货", "履约", "商务")):
        _push("business_contract")
    if any(token in compact for token in ("技术", "参数", "规格", "性能", "接口", "配置", "方案")):
        _push("technical_spec")
    if any(token in compact for token in ("投标人须知", "须知前附表", "投标须知", "须知")):
        _push("bidder_instruction")
    if any(token in compact for token in ("格式", "附件", "附表", "模板", "封面")):
        _push("format_appendix")
    return tags


def _harvest_evidence_refs(
    evidence_index: dict[str, Any],
    *,
    query_terms: list[str],
    existing_refs: list[dict[str, Any]],
    status: str,
    support_top_k: int = 3,
    counter_top_k: int = 2,
) -> dict[str, Any]:
    by_id = evidence_index.get("by_id", {})
    postings = evidence_index.get("postings", {})

    existing = _merge_evidence_refs(existing_refs)
    existing_ids = {str(item.get("evidence_id") or "") for item in existing}

    candidate_ids: set[str] = set()
    for term in query_terms:
        for evidence_id in postings.get(term, []):
            candidate_ids.add(str(evidence_id))

    if not candidate_ids:
        candidate_ids = {str(item.get("evidence_id") or "") for item in by_id.values()}

    ranked: list[dict[str, Any]] = []
    for evidence_id in candidate_ids:
        record = by_id.get(evidence_id)
        if not isinstance(record, dict):
            continue
        normalized_text = str(record.get("normalized_text") or "")
        hit_count = sum(1 for term in query_terms if term and term in normalized_text)
        if hit_count <= 0 and query_terms:
            continue
        polarity = _evidence_polarity(str(record.get("excerpt") or ""))
        rank_score = hit_count * 4 + (2 if _is_mappable_location(record.get("location")) else 0)
        ranked.append(
            {
                **record,
                "hit_count": hit_count,
                "harvest_score": rank_score,
                "polarity": polarity,
            }
        )

    ranked.sort(
        key=lambda item: (
            int(item.get("harvest_score") or 0),
            int(item.get("hit_count") or 0),
            len(str(item.get("excerpt") or "")),
        ),
        reverse=True,
    )

    expected_sign = 1 if status == "pass" else -1 if status == "fail" else 0
    harvested_support: list[dict[str, Any]] = []
    harvested_counter: list[dict[str, Any]] = []
    for item in ranked:
        evidence_id = str(item.get("evidence_id") or "")
        if not evidence_id or evidence_id in existing_ids:
            continue
        polarity = int(item.get("polarity") or 0)
        ref = _to_evidence_ref(item, source="evidence_harvester", harvest_score=int(item.get("harvest_score") or 0))
        if expected_sign != 0 and polarity == -expected_sign and len(harvested_counter) < counter_top_k:
            harvested_counter.append(ref)
            continue
        if len(harvested_support) < support_top_k:
            harvested_support.append(ref)
        if len(harvested_support) >= support_top_k and len(harvested_counter) >= counter_top_k:
            break

    existing_counter: list[dict[str, Any]] = []
    if expected_sign != 0:
        for item in existing:
            if _evidence_polarity(str(item.get("excerpt") or "")) == -expected_sign:
                existing_counter.append(item)

    support_refs = _merge_evidence_refs(existing, harvested_support)
    counter_refs = _merge_evidence_refs(existing_counter, harvested_counter)
    return {
        "support_refs": support_refs,
        "counter_refs": counter_refs,
        "query_terms": query_terms,
    }


def _requirement_task_id(requirement_id: str, index: int) -> str:
    token = "".join(ch for ch in requirement_id if ch.isalnum())
    if not token:
        token = f"R{index:04d}"
    return f"T-{token}-01"


def plan_tasks(out_dir: Path, resume: bool = False) -> dict[str, Any]:
    requirements_path = out_dir / "requirements.jsonl"
    requirements_atomic_path = out_dir / "requirements.atomic.jsonl"
    tasks_path = out_dir / "review-tasks.jsonl"
    tasks = ensure_review_tasks(
        requirements_path=requirements_path,
        requirements_atomic_path=requirements_atomic_path,
        review_tasks_path=tasks_path,
        resume=resume,
    )
    return {"review_tasks": len(tasks)}


class _PipelineTaskReviewer:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        default_status: str,
        default_reason: str,
        default_confidence: float | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self._default_status = default_status
        self._default_reason = default_reason
        self._default_confidence = default_confidence

    def review_task(self, task: dict[str, Any]) -> dict[str, Any]:
        status = _normalize_status(task.get("rule_status") or task.get("status") or self._default_status or "missing")
        reason = str(task.get("rule_reason") or task.get("reason") or self._default_reason or "Rule fallback")
        confidence = _safe_float(task.get("rule_confidence"))
        if confidence is None:
            confidence = self._default_confidence
        if confidence is None:
            confidence = _status_confidence_hint(status)
        return {
            "status": status,
            "reason": reason,
            "confidence": confidence,
        }


def _task_keywords(task: dict[str, Any], limit: int = 8) -> list[str]:
    values: list[Any] = [task.get("query"), task.get("task_type")]
    expected_logic = task.get("expected_logic")
    if isinstance(expected_logic, dict):
        values.append(expected_logic.get("keywords"))
        values.append(expected_logic.get("requirement_text"))
    return _extract_query_terms(values, limit=limit)


def _task_evidence_for_llm(task: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("evidence_pack", "counter_evidence_pack"):
        for item in (task.get(key) or []):
            if not isinstance(item, dict):
                continue
            excerpt = str(item.get("excerpt") or "")
            polarity = str(item.get("polarity") or ("counter" if key == "counter_evidence_pack" else "support"))
            rows.append(
                {
                    "score": item.get("score"),
                    "location": item.get("location"),
                    "excerpt": f"[{polarity}] {excerpt}".strip(),
                }
            )
    if rows:
        return rows

    for item in (task.get("evidence_refs") or []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "score": item.get("score"),
                "location": item.get("location"),
                "excerpt": str(item.get("excerpt") or ""),
            }
        )
    return rows


def _status_to_severity(status: str) -> str:
    if status == "pass":
        return "none"
    if status == "risk":
        return "medium"
    if status == "fail":
        return "high"
    if status == "needs_ocr":
        return "medium"
    return "medium"


class _PipelineDeepSeekTaskReviewer:
    provider = "deepseek"

    def __init__(self, *, api_key: str, model: str, base_url: str) -> None:
        self._reviewer = DeepSeekReviewer(api_key=api_key, model=model, base_url=base_url)
        self.model = model

    def review_task(self, task: dict[str, Any]) -> dict[str, Any]:
        requirement_id = str(task.get("requirement_id") or task.get("task_id") or "T")
        query = str(task.get("query") or "")
        expected_logic = task.get("expected_logic")
        expected_logic_dict = expected_logic if isinstance(expected_logic, dict) else {}
        requirement = Requirement(
            requirement_id=requirement_id,
            text=query,
            category=str(task.get("task_type") or "task"),
            mandatory=bool(expected_logic_dict.get("mandatory", False)),
            keywords=_task_keywords(task),
            constraints=list(expected_logic_dict.get("constraints") or []),
            rule_tier=str(task.get("priority") or "general"),
            source={"task_id": task.get("task_id")},
        )
        rule_status = _normalize_status(task.get("rule_status") or task.get("status") or "missing")
        rule_reason = str(task.get("rule_reason") or task.get("reason") or "task rule fallback")
        finding = Finding(
            requirement_id=requirement_id,
            status=rule_status,
            score=0,
            severity=_status_to_severity(rule_status),
            reason=rule_reason,
            clause_id=requirement_id,
            evidence=_task_evidence_for_llm(task),
            decision_trace=task.get("decision_trace") if isinstance(task.get("decision_trace"), dict) else {},
            llm=None,
        )
        result = self._reviewer.review(requirement, finding)
        return {
            "status": str(result.get("status") or rule_status),
            "reason": str(result.get("reason") or rule_reason),
            "confidence": _safe_float(result.get("confidence")),
        }


def _aggregate_task_verdicts(task_verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    if not task_verdicts:
        return {
            "status": "missing",
            "reason": "未生成任务判决",
            "confidence": _status_confidence_hint("missing"),
            "status_counts": {},
        }

    status_order = ("fail", "needs_ocr", "missing", "risk", "pass")
    status_counts = Counter(_normalize_status(row.get("status") or "missing") for row in task_verdicts)
    selected_status = "missing"
    for status in status_order:
        if status_counts.get(status):
            selected_status = status
            break

    selected_reason = ""
    for row in task_verdicts:
        if str(row.get("status") or "") != selected_status:
            continue
        selected_reason = str(row.get("reason") or "").strip()
        if selected_reason:
            break
    if not selected_reason:
        selected_reason = f"任务聚合结论: {selected_status}"

    confidence_values = [value for value in (_safe_float(row.get("confidence")) for row in task_verdicts) if value is not None]
    confidence = (
        max(0.0, min(1.0, sum(confidence_values) / len(confidence_values)))
        if confidence_values
        else _status_confidence_hint(selected_status)
    )
    return {
        "status": selected_status,
        "reason": selected_reason,
        "confidence": confidence,
        "status_counts": dict(status_counts),
    }


def _max_pack_score(items: list[dict[str, Any]]) -> int:
    scores: list[int] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            scores.append(int(item.get("score") or 0))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else 0


def _derive_task_rule_decision(
    *,
    task: dict[str, Any],
    pack: dict[str, Any],
    mandatory: bool,
    default_status: str,
    default_reason: str,
    default_confidence: float | None,
) -> dict[str, Any]:
    support_pack = [item for item in (pack.get("evidence_pack") or []) if isinstance(item, dict)]
    counter_pack = [item for item in (pack.get("counter_evidence_pack") or []) if isinstance(item, dict)]
    support_score = _max_pack_score(support_pack)
    counter_score = _max_pack_score(counter_pack)

    support_reference_only = bool(support_pack) and all(bool(item.get("reference_only")) for item in support_pack)
    has_any_support = support_score > 0
    has_any_counter = counter_score > 0

    status = _normalize_status(default_status or "missing")
    reason = default_reason or "任务规则判定"
    if not has_any_support and not has_any_counter:
        status = "missing"
        reason = "任务未检索到有效证据"
    elif support_reference_only and not has_any_counter:
        status = "needs_ocr" if mandatory else "missing"
        reason = "任务仅命中附件/扫描件引用，需OCR复核"
    elif has_any_counter and counter_score >= max(6, support_score + 2):
        if has_any_support:
            status = "risk"
            reason = "任务支持与高置信反证冲突证据并存，需人工复核"
        else:
            status = "fail" if mandatory else "risk"
            reason = "任务命中高置信反证，当前结论不成立"
    elif has_any_support and support_score >= max(5, counter_score + 1):
        status = "pass"
        reason = "任务命中支持证据且强于反证"
    elif has_any_support and has_any_counter:
        status = "risk"
        reason = "任务支持证据与反证并存，需人工复核"
    elif has_any_support:
        status = "risk"
        reason = "任务仅命中弱支持证据，需人工复核"
    elif has_any_counter:
        status = "fail" if mandatory and counter_score >= 4 else "risk"
        reason = "任务命中反证但缺少充分支持证据"

    confidence = default_confidence
    if confidence is None:
        confidence = _status_confidence_hint(status)
    confidence = max(0.0, min(1.0, float(confidence)))
    return {
        "status": status,
        "reason": reason,
        "confidence": confidence,
        "trace": {
            "task_type": str(task.get("task_type") or ""),
            "support_score": support_score,
            "counter_score": counter_score,
            "support_reference_only": support_reference_only,
        },
    }


def _counter_conflict_second_pass(
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


def _requirement_needs_cross_verification(requirement: dict[str, Any]) -> bool:
    if not isinstance(requirement, dict):
        return False
    tier = str(requirement.get("rule_tier") or "general")
    text = str(requirement.get("text") or "")
    keywords = " ".join(str(item or "") for item in (requirement.get("keywords") or []))
    probe = f"{text} {keywords}"
    hard_clause = tier == "hard_fail" or bool(
        re.search(r"(将被否决|废标|无效投标|资格审查不通过|不得投标|不予通过)", probe)
    )
    explicit_dual_evidence = any(
        re.search(pattern, probe, flags=re.IGNORECASE) for pattern in DUAL_EVIDENCE_HINT_PATTERNS
    )
    return bool(hard_clause or explicit_dual_evidence)


def _is_high_risk_requirement(requirement: dict[str, Any]) -> bool:
    if not isinstance(requirement, dict):
        return False
    tier = str(requirement.get("rule_tier") or "general")
    if tier == "hard_fail":
        return True
    text = str(requirement.get("text") or "")
    return bool(re.search(r"(将被否决|废标|无效投标|资格审查不通过|不得投标|不予通过)", text))


def _support_evidence_is_weak(support_refs: list[dict[str, Any]]) -> bool:
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


def _apply_cross_audit(
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
    required = _requirement_needs_cross_verification(requirement)
    applicable = required and bool(support_refs)
    verified = len(support_channels) >= 2 if applicable else False
    high_risk = _is_high_risk_requirement(requirement)
    weak_support = _support_evidence_is_weak(support_refs)

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


def verdict(
    out_dir: Path,
    resume: bool = False,
    ai_provider: str | None = None,
    ai_model: str = "deepseek-chat",
    ai_api_key_file: Path | None = None,
    ai_base_url: str = "https://api.deepseek.com/v1",
    ai_workers: int = 4,
    ai_min_confidence: float = 0.65,
) -> dict[str, Any]:
    requirements_path = out_dir / "requirements.jsonl"
    findings_path = out_dir / "findings.jsonl"
    tasks_path = out_dir / "review-tasks.jsonl"
    bid_blocks_path = out_dir / "ingest" / "bid_blocks.jsonl"
    evidence_packs_path = out_dir / "evidence-packs.jsonl"
    cross_audit_path = out_dir / "cross-audit.jsonl"
    verdicts_path = out_dir / "verdicts.jsonl"
    if path_ready(verdicts_path, resume):
        rows = list(read_jsonl(verdicts_path))
        counts = Counter(_normalize_status(row.get("status")) for row in rows)
        llm_coverage = _safe_ratio(
            sum(
                1
                for row in rows
                if isinstance(row.get("model"), dict)
                and _is_real_llm_provider(row.get("model", {}).get("provider"))
            ),
            len(rows),
        )
        evidence_packs_total = sum(1 for _ in read_jsonl(evidence_packs_path)) if evidence_packs_path.exists() else 0
        cross_audit_rows = list(read_jsonl(cross_audit_path)) if cross_audit_path.exists() else []
        cross_required = sum(1 for row in cross_audit_rows if bool(row.get("required")))
        cross_verified = sum(1 for row in cross_audit_rows if bool(row.get("cross_verified")))
        return {
            "verdicts": len(rows),
            "evidence_packs": evidence_packs_total,
            "cross_audit": len(cross_audit_rows),
            "cross_audit_required": cross_required,
            "cross_audit_verified": cross_verified,
            "status_counts": dict(counts),
            "llm_coverage": llm_coverage,
        }

    tasks_by_requirement: dict[str, list[dict[str, Any]]] = {}
    if tasks_path.exists():
        for row in read_jsonl(tasks_path):
            requirement_id = str(row.get("requirement_id") or "").strip()
            if requirement_id:
                tasks_by_requirement.setdefault(requirement_id, []).append(row)

    bid_block_rows = list(read_jsonl(bid_blocks_path)) if bid_blocks_path.exists() else []
    doc_map_path = out_dir / "ingest" / "doc-map.json"
    doc_map_data: dict[str, Any] = {}
    if doc_map_path.exists():
        try:
            doc_map_data = json.loads(doc_map_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            doc_map_data = {}
    tagged_bid_rows = _attach_semantic_section_tags(bid_block_rows, doc_map=doc_map_data, doc_id="bid")
    unified_evidence_index = build_unified_evidence_index(tagged_bid_rows)
    task_evidence_index = build_task_evidence_index(tagged_bid_rows)
    source_type_counts = Counter(str(item.get("source_type") or "unknown") for item in unified_evidence_index)
    section_tag_counts = Counter(str(item.get("section_tag") or "other") for item in unified_evidence_index)
    evidence_index_stats = {
        "unified_blocks_indexed": len(unified_evidence_index),
        "harvest_blocks_indexed": len(task_evidence_index),
        "source_type_counts": dict(source_type_counts),
        "section_tag_counts": dict(section_tag_counts),
    }
    requirements_rows = list(read_jsonl(requirements_path)) if requirements_path.exists() else []
    requirements_by_id: dict[str, dict[str, Any]] = {}
    requirement_order: list[str] = []
    for index, row in enumerate(requirements_rows, start=1):
        requirement_id = str(row.get("requirement_id") or f"R{index:04d}").strip()
        if requirement_id and requirement_id not in requirement_order:
            requirement_order.append(requirement_id)
        if requirement_id:
            requirements_by_id[requirement_id] = row

    finding_rows = list(read_jsonl(findings_path)) if findings_path.exists() else []
    finding_by_requirement: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(finding_rows, start=1):
        requirement_id = str(row.get("requirement_id") or f"R{index:04d}").strip()
        if not requirement_id:
            continue
        finding_by_requirement[requirement_id] = row
        if requirement_id not in requirement_order:
            requirement_order.append(requirement_id)

    for requirement_id in tasks_by_requirement:
        if requirement_id not in requirement_order:
            requirement_order.append(requirement_id)

    shared_task_llm_reviewer: _PipelineDeepSeekTaskReviewer | None = None
    task_llm_workers = 1
    task_llm_rate_limit: dict[str, Any] = {
        "enabled": False,
        "requested_workers": 0,
        "max_workers": 1,
        "strategy": "single_thread",
    }
    if ai_provider == "deepseek":
        requested_workers = max(1, int(ai_workers))
        safe_cap = 4
        task_llm_workers = min(requested_workers, safe_cap)
        task_llm_rate_limit = {
            "enabled": True,
            "requested_workers": requested_workers,
            "max_workers": task_llm_workers,
            "strategy": "deepseek_concurrency_cap_4",
        }
        api_key = _load_api_key(ai_provider, ai_api_key_file)
        if api_key:
            shared_task_llm_reviewer = _PipelineDeepSeekTaskReviewer(
                api_key=api_key,
                model=ai_model,
                base_url=ai_base_url,
            )
        else:
            task_llm_rate_limit["enabled"] = False
            task_llm_rate_limit["strategy"] = "no_api_key_fallback_rule_only"

    verdict_rows: list[dict[str, Any]] = []
    evidence_pack_rows: list[dict[str, Any]] = []
    cross_audit_rows: list[dict[str, Any]] = []
    for index, requirement_id in enumerate(requirement_order, start=1):
        finding = finding_by_requirement.get(requirement_id, {})
        task_rows = list(tasks_by_requirement.get(requirement_id, []))
        fallback_status = _normalize_status(finding.get("status") or "missing")
        llm = finding.get("llm") if isinstance(finding.get("llm"), dict) else {}
        llm_confidence = _safe_float((llm or {}).get("confidence"))
        mandatory = bool((requirements_by_id.get(requirement_id) or {}).get("mandatory", False))

        existing_refs = [
            _to_evidence_ref(item, source="finding")
            for item in (finding.get("evidence") or [])
            if isinstance(item, dict)
        ]
        if not task_rows:
            task_rows = [
                {
                    "task_id": _requirement_task_id(requirement_id, index),
                    "requirement_id": requirement_id,
                    "task_type": "requirement_check",
                    "query": str(finding.get("reason") or ""),
                }
            ]
        base_reason = str(finding.get("reason") or "规则判定结果")
        requirement_row = requirements_by_id.get(requirement_id) or {}

        task_pack_rows: list[dict[str, Any]] = []
        for task in task_rows:
            pack = harvest_task_evidence(
                task,
                task_evidence_index,
                top_k=3,
                counter_k=2,
            )
            query = str(task.get("query") or "")
            preferred_tags = _preferred_section_tags(
                requirement_row.get("text"),
                requirement_row.get("category"),
                task.get("task_type"),
                query,
                base_reason,
            )
            retrieved_candidates = retrieve_evidence_candidates(
                unified_evidence_index,
                query,
                top_k=3,
                preferred_section_tags=preferred_tags,
            )
            retrieved_refs = [_to_evidence_ref(item, source="evidence_index_retrieval") for item in retrieved_candidates]
            pack["evidence_refs"] = _merge_evidence_refs(list(pack.get("evidence_refs") or []), retrieved_refs)
            trace = pack.get("retrieval_trace")
            if not isinstance(trace, dict):
                trace = {}
            trace["retriever_candidates"] = len(retrieved_candidates)
            trace["soft_routing_tags"] = preferred_tags
            pack["retrieval_trace"] = trace
            task_pack_rows.append(pack)

        evidence_pack_rows.extend(task_pack_rows)

        provider = str((llm or {}).get("provider") or "rule_fallback")
        model_name = str((llm or {}).get("model") or "rule-only")
        task_reviewer: Any
        if shared_task_llm_reviewer is not None:
            task_reviewer = shared_task_llm_reviewer
        else:
            task_reviewer = _PipelineTaskReviewer(
                provider=provider,
                model=model_name,
                default_status=fallback_status,
                default_reason=base_reason,
                default_confidence=llm_confidence,
            )
        early_exit_hard_fail = (
            _normalize_status(fallback_status) == "fail"
            and str((requirement_row or {}).get("rule_tier") or "") == "hard_fail"
        )
        task_verdict_rows: list[dict[str, Any]] = []
        if early_exit_hard_fail:
            aggregated = {
                "status": "fail",
                "reason": "硬否决条款预审已判定fail，触发提前终止策略",
                "confidence": _status_confidence_hint("fail"),
                "status_counts": {"fail": len(task_rows)},
            }
        else:
            task_inputs: list[dict[str, Any]] = []
            for task, task_pack in zip(task_rows, task_pack_rows, strict=False):
                task_rule = _derive_task_rule_decision(
                    task=task,
                    pack=task_pack,
                    mandatory=mandatory,
                    default_status=fallback_status,
                    default_reason=base_reason,
                    default_confidence=llm_confidence,
                )
                task_row = dict(task)
                task_row["rule_status"] = task_rule["status"]
                task_row["rule_reason"] = task_rule["reason"]
                task_row["rule_confidence"] = task_rule["confidence"]
                task_row["evidence_refs"] = _merge_evidence_refs(
                    existing_refs,
                    [item for item in (task_pack.get("evidence_refs") or []) if isinstance(item, dict)],
                )
                trace = task_row.get("decision_trace")
                if not isinstance(trace, dict):
                    trace = {}
                trace.setdefault("source", "pipeline_task_bridge")
                trace.setdefault("requirement_id", requirement_id)
                trace["task_rule"] = task_rule.get("trace")
                task_row["decision_trace"] = trace
                task_inputs.append(task_row)
            task_verdict_rows = judge_tasks_with_llm(
                task_inputs,
                task_reviewer,
                min_confidence=ai_min_confidence,
                max_workers=task_llm_workers,
            )
            aggregated = _aggregate_task_verdicts(task_verdict_rows)
        status = _normalize_status(aggregated.get("status") or "missing")
        confidence = max(0.0, min(1.0, _safe_float(aggregated.get("confidence")) or _status_confidence_hint(status)))

        task_support_refs: list[dict[str, Any]] = []
        task_counter_refs: list[dict[str, Any]] = []
        query_terms: list[str] = []
        for pack in task_pack_rows:
            task_support_refs.extend(
                item for item in (pack.get("evidence_refs") or []) if isinstance(item, dict)
            )
            task_counter_refs.extend(
                item for item in (pack.get("counter_evidence_refs") or []) if isinstance(item, dict)
            )
            trace = pack.get("retrieval_trace")
            if isinstance(trace, dict):
                for term in (trace.get("positive_terms") or []):
                    token = str(term or "").strip()
                    if token:
                        query_terms.append(token)

        if not query_terms:
            query_terms = _collect_query_terms_for_requirement(requirement_id, finding, task_rows)

        support_refs = _merge_evidence_refs(existing_refs, task_support_refs)
        counter_refs = _merge_evidence_refs(task_counter_refs)
        evidence_refs = [str(item.get("evidence_id")) for item in support_refs if item.get("evidence_id")]
        counter_evidence_refs = [str(item.get("evidence_id")) for item in counter_refs if item.get("evidence_id")]

        reason = str(aggregated.get("reason") or base_reason)
        status_before_audit = status
        second_pass = _counter_conflict_second_pass(
            status=status,
            reason=reason,
            confidence=confidence,
            task_packs=task_pack_rows,
        )
        status = str(second_pass.get("status") or status)
        reason = str(second_pass.get("reason") or reason)
        confidence = max(0.0, min(1.0, _safe_float(second_pass.get("confidence")) or confidence))
        second_pass_audit = second_pass.get("audit") if isinstance(second_pass.get("audit"), dict) else {}
        downgrade_action = second_pass_audit.get("action") if second_pass_audit else None
        status_before_floor = status
        floor_enforced = bool(finding) and _normalize_status(fallback_status) in {"fail", "needs_ocr"}
        if floor_enforced:
            status = _status_not_looser_than(status, fallback_status)
            if status != status_before_floor:
                reason = (
                    f"最终结论不得优于预审结论({fallback_status})，"
                    f"已由{status_before_floor}收敛为{status}"
                )
                confidence = min(confidence, _status_confidence_hint(status))
        status_after_floor = status

        cross_audit = _apply_cross_audit(
            requirement_id=requirement_id,
            requirement=requirement_row,
            support_refs=support_refs,
            counter_refs=counter_refs,
            status=status,
            reason=reason,
            confidence=confidence,
        )
        status = _normalize_status(cross_audit.get("status") or status)
        reason = str(cross_audit.get("reason") or reason)
        confidence = max(0.0, min(1.0, _safe_float(cross_audit.get("confidence")) or confidence))
        cross_audit_row = cross_audit.get("cross_audit") if isinstance(cross_audit.get("cross_audit"), dict) else {}
        cross_action = cross_audit_row.get("action") if isinstance(cross_audit_row, dict) else None
        if cross_audit_row:
            cross_audit_rows.append(cross_audit_row)

        model = {}
        provider = str((llm or {}).get("provider") or getattr(task_reviewer, "provider", "") or "")
        model_name = str((llm or {}).get("model") or getattr(task_reviewer, "model", "") or "")
        prompt_version = str((llm or {}).get("prompt_version") or "")
        if not prompt_version and provider == "deepseek":
            prompt_version = DEEPSEEK_PROMPT_VERSION
        if provider or model_name or prompt_version:
            model = {"provider": provider, "name": model_name, "prompt_version": prompt_version or None}

        decision_trace = finding.get("decision_trace")
        if not isinstance(decision_trace, dict):
            decision_trace = {"source": "pipeline_findings_bridge", "fallbacks": []}
        decision_trace["task_verdicts"] = {
            "total": len(task_verdict_rows),
            "status_counts": aggregated.get("status_counts", {}),
            "llm_rate_limit": dict(task_llm_rate_limit),
            "early_exit_hard_fail": early_exit_hard_fail,
            "sample": [
                {
                    "task_id": row.get("task_id"),
                    "status": row.get("status"),
                    "confidence": row.get("confidence"),
                }
                for row in task_verdict_rows[:5]
            ],
        }
        decision_trace["evidence_index"] = evidence_index_stats
        decision_trace["evidence_harvest"] = {
            "query_terms": query_terms,
            "support_refs": support_refs,
            "task_packs": len(task_pack_rows),
        }
        decision_trace["counter_evidence_audit"] = {
            "counter_evidence_refs": counter_refs,
            "conflict_detected": bool(counter_evidence_refs)
            or bool(second_pass_audit.get("conflict_level") not in {None, "none"}),
            "status_before": status_before_audit,
            "status_after": status,
            "action": downgrade_action,
            "second_pass": second_pass_audit,
        }
        decision_trace["status_floor"] = {
            "enabled": floor_enforced,
            "floor_status": _normalize_status(fallback_status),
            "status_before_floor": status_before_floor,
            "status_after_floor": status_after_floor,
            "downgraded": floor_enforced and status_after_floor != status_before_floor,
        }
        decision_trace["cross_audit"] = cross_audit_row
        decision_trace["evidence_refs"] = support_refs
        decision_trace["counter_evidence_refs"] = counter_refs
        decision_trace.setdefault("decision", {})
        if isinstance(decision_trace.get("decision"), dict):
            decision_trace["decision"]["status"] = status
            decision_trace["decision"]["reason"] = reason
            if cross_action:
                decision_trace["decision"]["source"] = "pipeline_cross_audit"
            elif downgrade_action:
                decision_trace["decision"]["source"] = "pipeline_counter_evidence_audit"
            else:
                decision_trace["decision"]["source"] = "pipeline_findings_bridge"

        verdict_rows.append(
            {
                "task_id": str((task_rows[0] or {}).get("task_id") or _requirement_task_id(requirement_id, index)),
                "requirement_id": requirement_id,
                "status": status,
                "confidence": confidence,
                "reason": reason,
                "evidence_refs": evidence_refs,
                "counter_evidence_refs": counter_evidence_refs,
                "model": model,
                "decision_trace": decision_trace,
            }
        )

    write_evidence_packs_jsonl(evidence_packs_path, evidence_pack_rows)
    write_jsonl(cross_audit_path, cross_audit_rows)
    write_verdicts_jsonl(verdicts_path, verdict_rows)
    counts = Counter(_normalize_status(row.get("status")) for row in verdict_rows)
    cross_required = sum(1 for row in cross_audit_rows if bool(row.get("required")))
    cross_verified = sum(1 for row in cross_audit_rows if bool(row.get("cross_verified")))
    llm_coverage = _safe_ratio(
        sum(
            1
            for row in verdict_rows
            if isinstance(row.get("model"), dict)
            and _is_real_llm_provider(row.get("model", {}).get("provider"))
        ),
        len(verdict_rows),
    )
    return {
        "verdicts": len(verdict_rows),
        "evidence_packs": len(evidence_pack_rows),
        "cross_audit": len(cross_audit_rows),
        "cross_audit_required": cross_required,
        "cross_audit_verified": cross_verified,
        "status_counts": dict(counts),
        "llm_coverage": llm_coverage,
    }


def _sync_findings_from_verdicts(out_dir: Path) -> dict[str, Any]:
    """Materialize downstream findings from verdict results.

    Business intent:
    - Use the task-aggregated verdict as the final requirement-level status.
    - Carry forward traceable evidence refs so annotation/report/checklist align with
      the latest review strategy rather than stale pre-verdict findings.
    """
    verdicts_path = out_dir / "verdicts.jsonl"
    findings_path = out_dir / "findings.jsonl"
    if not verdicts_path.exists():
        return {"synced": False, "reason": "verdicts_missing"}

    existing_rows = list(read_jsonl(findings_path)) if findings_path.exists() else []
    existing_by_requirement = {
        str(row.get("requirement_id") or "").strip(): row
        for row in existing_rows
        if str(row.get("requirement_id") or "").strip()
    }

    synced_rows: list[dict[str, Any]] = []
    for row in read_jsonl(verdicts_path):
        requirement_id = str(row.get("requirement_id") or "").strip()
        if not requirement_id:
            continue
        base = existing_by_requirement.get(requirement_id, {})
        status = _normalize_status(row.get("status") or base.get("status") or "missing")
        decision_trace = row.get("decision_trace") if isinstance(row.get("decision_trace"), dict) else {}

        evidence: list[dict[str, Any]] = []
        for item in (decision_trace.get("evidence_refs") if isinstance(decision_trace, dict) else []) or []:
            if not isinstance(item, dict):
                continue
            evidence.append(
                {
                    "evidence_id": item.get("evidence_id"),
                    "block_id": item.get("block_id"),
                    "excerpt_hash": item.get("excerpt_hash"),
                    "doc_id": item.get("doc_id"),
                    "location": item.get("location"),
                    "source_type": item.get("source_type"),
                    "section_tag": item.get("section_tag"),
                    "score": item.get("score"),
                    "excerpt": item.get("excerpt"),
                    "source": item.get("source"),
                    "reference_only": item.get("reference_only"),
                    "has_action": item.get("has_action"),
                }
            )
        if not evidence:
            for item in base.get("evidence") or []:
                if isinstance(item, dict):
                    evidence.append(item)

        score = 0
        for item in evidence:
            try:
                item_score = int(item.get("score") or 0)
            except (TypeError, ValueError):
                item_score = 0
            score = max(score, item_score)

        model = row.get("model") if isinstance(row.get("model"), dict) else {}
        confidence = _safe_float(row.get("confidence"))
        llm_payload = {
            "provider": model.get("provider"),
            "model": model.get("name"),
            "prompt_version": model.get("prompt_version"),
            "confidence": confidence,
        }

        synced_rows.append(
            {
                "requirement_id": requirement_id,
                "clause_id": str(base.get("clause_id") or requirement_id),
                "status": status,
                "score": score,
                "severity": _status_to_severity(status),
                "reason": str(row.get("reason") or base.get("reason") or "任务聚合结论"),
                "decision_trace": decision_trace or base.get("decision_trace"),
                "evidence": evidence,
                "llm": llm_payload,
            }
        )

    write_jsonl(findings_path, synced_rows)
    counts = Counter(_normalize_status(item.get("status")) for item in synced_rows)
    return {"synced": True, "findings": len(synced_rows), "status_counts": dict(counts)}


def _load_eval_metrics(out_dir: Path) -> dict[str, Any] | None:
    metrics_path = out_dir / "eval" / "metrics.json"
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics")
            if isinstance(metrics, dict):
                return metrics
        except json.JSONDecodeError:
            pass

    gold_path = out_dir / "eval" / "gold.jsonl"
    findings_path = out_dir / "findings.jsonl"
    if not gold_path.exists() or not findings_path.exists():
        return None

    try:
        result = evaluate_run(out_dir, out_path=metrics_path)
    except FileNotFoundError:
        return None
    metrics = result.get("metrics")
    return metrics if isinstance(metrics, dict) else None


def _finding_has_traceable_evidence(finding: dict[str, Any]) -> bool:
    for evidence in finding.get("evidence") or []:
        if not isinstance(evidence, dict):
            continue
        if _is_mappable_location(evidence.get("location")):
            return True

    trace = finding.get("decision_trace")
    if not isinstance(trace, dict):
        return False

    for evidence_ref in trace.get("evidence_refs") or []:
        if not isinstance(evidence_ref, dict):
            continue
        if _is_mappable_location(evidence_ref.get("location")):
            return True

    return False


def gate(
    out_dir: Path,
    *,
    requested_release_mode: str = "assist_only",
    threshold_overrides: dict[str, Any] | None = None,
    fail_fast: str = "off",
) -> dict[str, Any]:
    requested_mode = _validate_release_mode(requested_release_mode)
    fail_fast_mode = _validate_gate_fail_fast_mode(fail_fast)
    thresholds = _resolve_gate_thresholds(threshold_overrides)

    requirements_path = out_dir / "requirements.jsonl"
    findings_path = out_dir / "findings.jsonl"
    verdicts_path = out_dir / "verdicts.jsonl"
    gate_out = out_dir / "gate-result.json"

    requirements_total = 0
    if requirements_path.exists():
        requirements_total = sum(1 for _ in read_jsonl(requirements_path))

    finding_rows = list(read_jsonl(findings_path)) if findings_path.exists() else []
    verdict_rows = list(read_jsonl(verdicts_path)) if verdicts_path.exists() else []
    if requirements_total <= 0:
        requirements_total = max(len(finding_rows), len(verdict_rows))

    verdict_requirement_ids = {
        str(row.get("requirement_id") or "").strip()
        for row in verdict_rows
        if str(row.get("requirement_id") or "").strip()
    }
    llm_requirement_ids = {
        str(row.get("requirement_id") or "").strip()
        for row in verdict_rows
        if str(row.get("requirement_id") or "").strip()
        and isinstance(row.get("model"), dict)
        and _is_real_llm_provider(row.get("model", {}).get("provider"))
    }

    auto_review_coverage = _safe_ratio(len(verdict_requirement_ids), requirements_total)
    missing_total = sum(1 for row in verdict_rows if _normalize_status(row.get("status")) == "missing")
    missing_rate = _safe_ratio(missing_total, requirements_total)
    findings_with_traceable_evidence = sum(1 for row in finding_rows if _finding_has_traceable_evidence(row))
    evidence_traceability = _safe_ratio(
        findings_with_traceable_evidence,
        len(finding_rows),
    )
    llm_coverage = _safe_ratio(len(llm_requirement_ids), requirements_total)

    eval_metrics = _load_eval_metrics(out_dir)
    hard_fail_recall = _safe_float((eval_metrics or {}).get("hard_fail_recall"))
    false_positive_fail_rate = _safe_float((eval_metrics or {}).get("false_positive_fail_rate"))
    if false_positive_fail_rate is None and eval_metrics:
        false_positive_fail = _safe_float(eval_metrics.get("false_positive_fail")) or 0.0
        non_fail_total = _safe_float(eval_metrics.get("non_fail_total")) or 0.0
        false_positive_fail_rate = (false_positive_fail / non_fail_total) if non_fail_total > 0 else 0.0

    check_defs = [
        {
            "name": "auto_review_coverage",
            "value": auto_review_coverage,
            "threshold": thresholds["auto_review_coverage"],
            "ok": auto_review_coverage >= thresholds["auto_review_coverage"],
        },
        {
            "name": "hard_fail_recall",
            "value": hard_fail_recall,
            "threshold": thresholds["hard_fail_recall"],
            "ok": hard_fail_recall is not None and hard_fail_recall >= thresholds["hard_fail_recall"],
        },
        {
            "name": "false_positive_fail_rate",
            "value": false_positive_fail_rate,
            "threshold": thresholds["false_positive_fail_rate"],
            "ok": false_positive_fail_rate is not None
            and false_positive_fail_rate <= thresholds["false_positive_fail_rate"],
        },
        {
            "name": "evidence_traceability",
            "value": evidence_traceability,
            "threshold": thresholds["evidence_traceability"],
            "ok": evidence_traceability >= thresholds["evidence_traceability"],
        },
        {
            "name": "llm_coverage",
            "value": llm_coverage,
            "threshold": thresholds["llm_coverage"],
            "ok": llm_coverage >= thresholds["llm_coverage"],
        },
        {
            "name": "missing_rate",
            "value": missing_rate,
            "threshold": thresholds["missing_rate"],
            "ok": missing_rate <= thresholds["missing_rate"],
        },
    ]
    checks: list[dict[str, Any]] = []
    fail_fast_triggered_by: str | None = None
    for check in check_defs:
        checks.append(dict(check))
        if check.get("ok"):
            continue
        should_fail_fast = fail_fast_mode == "all" or (
            fail_fast_mode == "critical" and check["name"] in CRITICAL_GATE_CHECKS
        )
        if should_fail_fast:
            fail_fast_triggered_by = check["name"]
            break

    if fail_fast_triggered_by is not None:
        evaluated = {item["name"] for item in checks}
        for check in check_defs:
            if check["name"] in evaluated:
                continue
            checks.append(
                {
                    "name": check["name"],
                    "value": check["value"],
                    "threshold": check["threshold"],
                    "ok": False,
                    "skipped": True,
                    "reason": f"fail_fast_triggered_by:{fail_fast_triggered_by}",
                }
            )

    all_checks_pass = fail_fast_triggered_by is None and all(bool(item.get("ok")) for item in checks)
    release_mode = "auto_final" if requested_mode == "auto_final" and all_checks_pass else "assist_only"

    result = {
        "requested_release_mode": requested_mode,
        "release_mode": release_mode,
        "eligible_for_auto_final": all_checks_pass,
        "thresholds": thresholds,
        "fail_fast": {
            "mode": fail_fast_mode,
            "triggered": fail_fast_triggered_by is not None,
            "triggered_by": fail_fast_triggered_by,
        },
        "metrics": {
            "requirements_total": requirements_total,
            "verdict_total": len(verdict_rows),
            "verdict_requirement_covered": len(verdict_requirement_ids),
            "llm_requirement_covered": len(llm_requirement_ids),
            "findings_total": len(finding_rows),
            "requirements_with_evidence": findings_with_traceable_evidence,
            "requirements_without_evidence": max(0, len(finding_rows) - findings_with_traceable_evidence),
            "auto_review_coverage": auto_review_coverage,
            "hard_fail_recall": hard_fail_recall,
            "false_positive_fail_rate": false_positive_fail_rate,
            "evidence_traceability": evidence_traceability,
            "llm_coverage": llm_coverage,
            "missing_total": missing_total,
            "missing_rate": missing_rate,
            "eval_metrics": eval_metrics or {},
        },
        "checks": checks,
        "generated_at": _utc_now_z(),
    }
    gate_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def report(out_dir: Path) -> dict[str, Any]:
    requirements = list(read_jsonl(out_dir / "requirements.jsonl"))
    findings = list(read_jsonl(out_dir / "findings.jsonl"))
    consistency_path = out_dir / "consistency-findings.jsonl"
    consistency_findings: list[dict[str, Any]] = []
    if consistency_path.exists():
        consistency_findings = list(read_jsonl(consistency_path))
    manual_review_path = out_dir / "manual-review.jsonl"
    manual_review_total = 0
    if manual_review_path.exists():
        manual_review_total = sum(1 for _ in read_jsonl(manual_review_path))
    counts = Counter(_normalize_status(item.get("status")) for item in findings)
    blocking_findings = [item for item in findings if _is_blocking_finding_status(str(item.get("status") or ""))]
    tier_counts = Counter(str(item.get("rule_tier") or "general") for item in requirements)
    req_tier_map = {item.get("requirement_id"): str(item.get("rule_tier") or "general") for item in requirements}
    hard_fail_status = Counter(
        _normalize_status(item.get("status"))
        for item in findings
        if req_tier_map.get(item.get("requirement_id")) == "hard_fail"
    )
    report_path = out_dir / "review-report.md"

    lines = [
        "# Bid Review Report",
        "",
        "## Summary",
        "",
        f"- Total requirements: {len(requirements)}",
        f"- hard_fail_requirements: {tier_counts.get('hard_fail', 0)}",
        f"- scored_requirements: {tier_counts.get('scored', 0)}",
        f"- general_requirements: {tier_counts.get('general', 0)}",
        f"- pass: {counts.get('pass', 0)}",
        f"- blocking_items(non-pass): {len(blocking_findings)}",
        f"- fail: {counts.get('fail', 0)}",
        f"- risk: {counts.get('risk', 0)}",
        f"- needs_ocr: {counts.get('needs_ocr', 0)}",
        f"- missing: {counts.get('missing', 0)}",
        f"- insufficient_evidence(legacy): {counts.get('insufficient_evidence', 0)}",
        f"- hard_fail_pass: {hard_fail_status.get('pass', 0)}",
        f"- hard_fail_fail: {hard_fail_status.get('fail', 0)}",
        f"- hard_fail_risk: {hard_fail_status.get('risk', 0)}",
        f"- hard_fail_needs_ocr: {hard_fail_status.get('needs_ocr', 0)}",
        f"- hard_fail_missing: {hard_fail_status.get('missing', 0)}",
        f"- hard_fail_insufficient_evidence(legacy): {hard_fail_status.get('insufficient_evidence', 0)}",
        f"- consistency_findings: {len(consistency_findings)}",
        f"- manual_review_items: {manual_review_total}",
        "",
        "## Blocking Findings (Need Action, Non-Pass)",
        "",
    ]

    req_map = {item["requirement_id"]: item for item in requirements}
    if not blocking_findings:
        lines.append("- none")
    for item in blocking_findings:
        requirement = req_map.get(item["requirement_id"], {})
        trace_text = _format_finding_trace(item)
        evidence_text = _format_primary_evidence(item)
        tier = str(requirement.get("rule_tier") or "general")
        lines.append(
            "- "
                + f"[{item['status']}/{item['severity']}] {item['requirement_id']} "
                + f"({tier}/{requirement.get('category', 'N/A')}): {item['reason']} | {evidence_text} | trace: {trace_text}"
        )

    if consistency_findings:
        lines.extend(["", "## Consistency Findings", ""])
        for item in consistency_findings:
            values = item.get("values") or []
            pairs = item.get("pairs") or []
            comparison = item.get("comparison") if isinstance(item.get("comparison"), dict) else {}
            # Keep the report readable; include top 2 values only.
            top_values = []
            for value_row in values[:2]:
                top_values.append(
                    f"{value_row.get('value_raw_examples', [''])[0]}(count={value_row.get('count')})"
                )
            more_count = max(0, len(values) - len(top_values))
            values_text = ", ".join(_sanitize_md_text(v, limit=60) for v in top_values if v)
            if more_count > 0:
                if values_text:
                    values_text = f"{values_text} (+{more_count} more)"
                else:
                    values_text = f"(+{more_count} more)"
            pair_text = ""
            if comparison:
                left = comparison.get("evidence_a") if isinstance(comparison.get("evidence_a"), dict) else {}
                right = comparison.get("evidence_b") if isinstance(comparison.get("evidence_b"), dict) else {}
                left_loc = _format_trace_location(left.get("location")) if left else "block=N/A page=N/A"
                right_loc = _format_trace_location(right.get("location")) if right else "block=N/A page=N/A"
                conclusion = _sanitize_md_text(comparison.get("conclusion"), limit=30)
                pair_text = (
                    " | 证据A: "
                    + f"{_sanitize_md_text(left.get('value'), limit=40)}@{left_loc}"
                    + " | 证据B: "
                    + f"{_sanitize_md_text(right.get('value'), limit=40)}@{right_loc}"
                    + f" | 结论: {conclusion}"
                )
            elif pairs:
                first_pair = pairs[0] if isinstance(pairs[0], dict) else {}
                left = first_pair.get("left") if isinstance(first_pair.get("left"), dict) else {}
                right = first_pair.get("right") if isinstance(first_pair.get("right"), dict) else {}
                left_loc = _format_trace_location(left.get("location"))
                right_loc = _format_trace_location(right.get("location"))
                pair_text = (
                    " | 证据A: "
                    + f"{_sanitize_md_text(left.get('value'), limit=40)}@{left_loc}"
                    + " | 证据B: "
                    + f"{_sanitize_md_text(right.get('value'), limit=40)}@{right_loc}"
                    + " | 结论: 不一致"
                )
            lines.append(
                "- "
                + f"[{item.get('status', 'risk')}/{item.get('severity', 'medium')}] "
                + f"{item.get('type')}: {item.get('reason')} "
                + f"| values: {values_text}"
                + pair_text
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str(report_path), "status_counts": dict(counts)}


def run_pipeline(
    tender_path: Path,
    bid_path: Path,
    out_dir: Path,
    focus: str,
    resume: bool = False,
    page_range: tuple[int, int] | None = None,
    ocr_mode: str = "off",
    ai_provider: str | None = None,
    ai_model: str = "deepseek-chat",
    ai_api_key_file: Path | None = None,
    ai_base_url: str = "https://api.deepseek.com/v1",
    ai_workers: int = 4,
    ai_min_confidence: float = 0.65,
    extract_batch_size: int = 8,
    extract_batch_max_chars: int = 8000,
    extract_timeout_seconds: int = 45,
    release_mode: str = "assist_only",
    gate_threshold_overrides: dict[str, Any] | None = None,
    gate_fail_fast: str = "off",
    canary_min_streak: int = DEFAULT_CANARY_MIN_STREAK,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    summary: dict[str, Any] = {}
    requested_release_mode = _validate_release_mode(release_mode)
    min_streak_value = _validate_canary_min_streak(canary_min_streak)
    summary["ingest"] = ingest(
        tender_path=tender_path,
        bid_path=bid_path,
        out_dir=out_dir,
        resume=resume,
        page_range=page_range,
        ocr_mode=ocr_mode,
    )
    summary["extract_req"] = extract_req(
        out_dir=out_dir,
        focus=focus,
        resume=resume,
        ai_provider=ai_provider,
        ai_model=ai_model,
        ai_api_key_file=ai_api_key_file,
        ai_base_url=ai_base_url,
        ai_min_confidence=ai_min_confidence,
        extract_batch_size=extract_batch_size,
        extract_batch_max_chars=extract_batch_max_chars,
        extract_timeout_seconds=extract_timeout_seconds,
    )
    summary["plan_tasks"] = plan_tasks(out_dir=out_dir, resume=resume)
    summary["review"] = review(
        out_dir=out_dir,
        resume=resume,
        ai_provider=ai_provider,
        ai_model=ai_model,
        ai_api_key_file=ai_api_key_file,
        ai_base_url=ai_base_url,
        ai_workers=ai_workers,
        ai_min_confidence=ai_min_confidence,
    )
    summary["verdict"] = verdict(
        out_dir=out_dir,
        resume=False,
        ai_provider=ai_provider,
        ai_model=ai_model,
        ai_api_key_file=ai_api_key_file,
        ai_base_url=ai_base_url,
        ai_workers=ai_workers,
        ai_min_confidence=ai_min_confidence,
    )
    summary["sync_findings"] = _sync_findings_from_verdicts(out_dir=out_dir)
    # Even when resuming expensive upstream stages, keep downstream deliverables fresh.
    # This also guarantees a new timestamped annotated copy after each `run`.
    downstream_resume = False
    summary["annotate"] = annotate(
        out_dir=out_dir,
        resume=downstream_resume,
        bid_source=bid_path,
        blocking_only=True,
    )
    summary["checklist"] = checklist(out_dir=out_dir, resume=downstream_resume)
    summary["consistency"] = consistency(out_dir=out_dir, resume=downstream_resume)
    summary["report"] = report(out_dir=out_dir)
    eval_metrics = _load_eval_metrics(out_dir)
    summary["eval"] = {
        "metrics_available": eval_metrics is not None,
        "metrics": eval_metrics,
    }
    gate_result = gate(
        out_dir=out_dir,
        requested_release_mode=requested_release_mode,
        threshold_overrides=gate_threshold_overrides,
        fail_fast=gate_fail_fast,
    )
    summary["gate"] = gate_result
    summary["run_metadata"] = _build_run_metadata(
        out_dir=out_dir,
        focus=focus,
        page_range=page_range,
        ocr_mode=ocr_mode,
        ai_provider=ai_provider,
        ai_model=ai_model,
        ai_base_url=ai_base_url,
        ai_workers=ai_workers,
        ai_min_confidence=ai_min_confidence,
        extract_batch_size=extract_batch_size,
        extract_batch_max_chars=extract_batch_max_chars,
        extract_timeout_seconds=extract_timeout_seconds,
        requested_release_mode=requested_release_mode,
        canary_min_streak=min_streak_value,
    )
    summary["canary"] = _run_canary(
        out_dir=out_dir,
        requested_release_mode=requested_release_mode,
        gate_result=gate_result,
        min_streak=min_streak_value,
    )
    summary["release_trace"] = _write_release_trace(
        out_dir=out_dir,
        run_metadata=summary["run_metadata"],
        gate_result=gate_result,
        canary_result=summary["canary"],
    )
    summary["release_mode"] = str((summary.get("canary") or {}).get("release_mode") or "assist_only")
    return summary
