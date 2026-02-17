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
from bidagent.annotators import annotate_docx_copy, annotate_pdf_copy
from bidagent.consistency import find_inconsistencies
from bidagent.document import iter_document_blocks
from bidagent.evidence_harvester import (
    build_evidence_index as build_task_evidence_index,
    harvest_task_evidence,
    write_evidence_packs_jsonl,
)
from bidagent.evidence_index import build_unified_evidence_index, retrieve_evidence_candidates
from bidagent.eval import evaluate_run
from bidagent.io_utils import append_jsonl, ensure_dir, path_ready, read_jsonl, write_jsonl
from bidagent.llm_judge import judge_tasks_with_llm, write_verdicts_jsonl
from bidagent.llm import DEEPSEEK_PROMPT_VERSION, DeepSeekReviewer, apply_llm_review
from bidagent.models import Block, Finding, Location, Requirement
from bidagent.ocr import iter_document_ocr_blocks, ocr_selfcheck
from bidagent.review import (
    REVIEW_RULE_ENGINE,
    REVIEW_RULE_VERSION,
    enforce_evidence_quality_gate,
    extract_requirements,
    review_requirements,
)
from bidagent.task_planner import ensure_review_tasks

VALID_RELEASE_MODES = {"assist_only", "auto_final"}
VALID_GATE_FAIL_FAST_MODES = {"off", "critical", "all"}
CRITICAL_GATE_CHECKS = {"hard_fail_recall", "false_positive_fail_rate"}
DEFAULT_GATE_THRESHOLDS = {
    "auto_review_coverage": 0.95,
    "hard_fail_recall": 0.98,
    "false_positive_fail_rate": 0.01,
    "evidence_traceability": 0.99,
    "llm_coverage": 1.0,
}
VERDICT_STRATEGY_VERSION = "verdict-harvest-v1"
CANARY_POLICY_VERSION = "release-canary-v1"
RUN_METADATA_SCHEMA_VERSION = "run-metadata-v1"
RELEASE_TRACE_SCHEMA_VERSION = "release-trace-v1"
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


def _row_to_block(row: dict[str, Any]) -> Block:
    location = row.get("location", {})
    return Block(
        doc_id=row["doc_id"],
        text=row["text"],
        location=Location(
            block_index=location.get("block_index", 0),
            page=location.get("page"),
            section=location.get("section"),
        ),
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
    requested_release_mode: str,
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
        },
        "run_config": {
            "focus": focus,
            "page_range": list(page_range) if page_range else None,
            "ocr_mode": ocr_mode,
            "release_mode_requested": requested_release_mode,
            "ai_workers": int(ai_workers),
            "ai_min_confidence": float(ai_min_confidence),
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
        "review-tasks.jsonl",
        "evidence-packs.jsonl",
        "findings.jsonl",
        "verdicts.jsonl",
        "gate-result.json",
        "annotations.jsonl",
        "manual-review.jsonl",
        "consistency-findings.jsonl",
        "review-report.md",
        "eval/metrics.json",
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


def _run_canary(
    *,
    out_dir: Path,
    requested_release_mode: str,
    gate_result: dict[str, Any],
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
            out_dir / "review-tasks.jsonl",
            out_dir / "evidence-packs.jsonl",
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
        canary_pass = all(
            bool(item.get("ok"))
            for item in checks
            if bool(item.get("blocking", True))
        )
        status = "pass" if canary_pass else "fail"
        release_mode = "auto_final" if canary_pass else "assist_only"

    result = {
        "policy_version": CANARY_POLICY_VERSION,
        "generated_at": _utc_now_z(),
        "requested_release_mode": requested_release_mode,
        "canary_required": canary_required,
        "status": status,
        "release_mode": release_mode,
        "checks": checks,
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
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def extract_req(out_dir: Path, focus: str, resume: bool = False) -> dict[str, Any]:
    ingest_dir = out_dir / "ingest"
    tender_path = ingest_dir / "tender_blocks.jsonl"
    req_path = out_dir / "requirements.jsonl"
    if path_ready(req_path, resume):
        total = sum(1 for _ in read_jsonl(req_path))
        return {"requirements": total}

    tender_blocks = _iter_blocks_from_jsonl(tender_path)
    requirements = extract_requirements(tender_blocks, focus=focus)
    total = write_jsonl(req_path, (item.to_dict() for item in requirements))
    return {"requirements": total}


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


def annotate(
    out_dir: Path,
    resume: bool = False,
    bid_source: Path | None = None,
) -> dict[str, Any]:
    annotations_path = out_dir / "annotations.jsonl"
    markdown_path = out_dir / "annotations.md"
    findings_path = out_dir / "findings.jsonl"

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

    findings = list(read_jsonl(findings_path))
    issue_rows = []
    for row in findings:
        if row["status"] == "pass":
            continue
        evidence = row.get("evidence", [])
        primary = _choose_primary_evidence(evidence, status=str(row.get("status") or ""))
        if not primary:
            primary = {}
        alternates = []
        for item in _choose_alternate_evidence(
            evidence,
            primary,
            status=str(row.get("status") or ""),
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
                "target": {
                    "doc_id": primary.get("doc_id", "bid"),
                    "location": primary.get("location"),
                    "excerpt": primary.get("excerpt"),
                    "score": primary.get("score"),
                },
                "alternate_targets": alternates,
                "note": (
                    f"[{row['severity']}] {row['requirement_id']} {row['status']}: "
                    f"{row['reason']}"
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
        is_manual = (
            finding["status"] in {"fail", "needs_ocr"}
            or finding.get("severity") == "high"
            or (tier == "hard_fail" and finding.get("status") != "pass")
        )
        if not is_manual:
            continue
        evidence = finding.get("evidence", [])
        primary = _choose_primary_evidence(evidence, status=str(finding.get("status") or ""))
        if not primary:
            primary = {}
        review_rows.append(
            {
                "requirement_id": finding["requirement_id"],
                "tier": tier,
                "status": finding["status"],
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
            0 if str(row.get("status")) in {"fail"} else 1,
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
    bid_blocks = _iter_blocks_from_jsonl(bid_path)
    findings = find_inconsistencies(bid_blocks)
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


def _choose_primary_evidence(evidence: Any, *, status: str | None = None) -> dict[str, Any]:
    if not isinstance(evidence, list) or not evidence:
        return {}

    candidates = [item for item in evidence if isinstance(item, dict)]
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
        try:
            score = int(item.get("score") or 0)
        except (TypeError, ValueError):
            score = 0
        if status == "needs_ocr":
            key = (mappable, _ocr_hint_score(excerpt), excerpt_len, score, has_action)
        else:
            key = (mappable, score, has_action, excerpt_len, 0)
        if key > best_key:
            best_key = key
            best = item
    return best


def _choose_alternate_evidence(
    evidence: Any,
    primary: dict[str, Any],
    *,
    status: str | None = None,
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
    alternates: list[dict[str, Any]] = []
    for item in sorted(
        (item for item in evidence if isinstance(item, dict)),
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
    if status == "pass":
        return 0.70
    if status == "fail":
        return 0.80
    if status == "risk":
        return 0.50
    if status == "needs_ocr":
        return 0.35
    return 0.30


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
    evidence_id = str(item.get("evidence_id") or "").strip() or _build_evidence_id_from_row(item)
    score = _safe_float(item.get("score"))
    if harvest_score is not None:
        score = float(harvest_score)
    return {
        "evidence_id": evidence_id,
        "doc_id": item.get("doc_id") or "bid",
        "location": location,
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
            "doc_id": row.get("doc_id") or "bid",
            "location": row.get("location"),
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
    tasks_path = out_dir / "review-tasks.jsonl"
    tasks = ensure_review_tasks(requirements_path=requirements_path, review_tasks_path=tasks_path, resume=resume)
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
        status = str(task.get("rule_status") or task.get("status") or self._default_status or "insufficient_evidence")
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
        rule_status = str(task.get("rule_status") or task.get("status") or "insufficient_evidence")
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
            "status": "insufficient_evidence",
            "reason": "未生成任务判决",
            "confidence": _status_confidence_hint("insufficient_evidence"),
            "status_counts": {},
        }

    status_order = ("fail", "needs_ocr", "risk", "insufficient_evidence", "pass")
    status_counts = Counter(str(row.get("status") or "insufficient_evidence") for row in task_verdicts)
    selected_status = "insufficient_evidence"
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

    status = default_status or "insufficient_evidence"
    reason = default_reason or "任务规则判定"
    if not has_any_support and not has_any_counter:
        status = "insufficient_evidence"
        reason = "任务未检索到有效证据"
    elif support_reference_only and not has_any_counter:
        status = "needs_ocr" if mandatory else "insufficient_evidence"
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
        if strong_hits > 0 or counter_top >= support_top:
            conflict_level = "strong"
            downgraded = True
            next_status = "risk"
            next_reason = "命中强反证/冲突证据，pass结论不稳定，已降级为risk"
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


def verdict(
    out_dir: Path,
    resume: bool = False,
    ai_provider: str | None = None,
    ai_model: str = "deepseek-chat",
    ai_api_key_file: Path | None = None,
    ai_base_url: str = "https://api.deepseek.com/v1",
    ai_min_confidence: float = 0.65,
) -> dict[str, Any]:
    requirements_path = out_dir / "requirements.jsonl"
    findings_path = out_dir / "findings.jsonl"
    tasks_path = out_dir / "review-tasks.jsonl"
    bid_blocks_path = out_dir / "ingest" / "bid_blocks.jsonl"
    evidence_packs_path = out_dir / "evidence-packs.jsonl"
    verdicts_path = out_dir / "verdicts.jsonl"
    if path_ready(verdicts_path, resume):
        rows = list(read_jsonl(verdicts_path))
        counts = Counter(str(row.get("status") or "") for row in rows)
        llm_coverage = _safe_ratio(
            sum(1 for row in rows if isinstance(row.get("model"), dict) and row.get("model", {}).get("provider")),
            len(rows),
        )
        evidence_packs_total = sum(1 for _ in read_jsonl(evidence_packs_path)) if evidence_packs_path.exists() else 0
        return {
            "verdicts": len(rows),
            "evidence_packs": evidence_packs_total,
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
    unified_evidence_index = build_unified_evidence_index(bid_block_rows)
    task_evidence_index = build_task_evidence_index(bid_block_rows)
    source_type_counts = Counter(str(item.get("source_type") or "unknown") for item in unified_evidence_index)
    evidence_index_stats = {
        "unified_blocks_indexed": len(unified_evidence_index),
        "harvest_blocks_indexed": len(task_evidence_index),
        "source_type_counts": dict(source_type_counts),
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
    if ai_provider == "deepseek":
        api_key = _load_api_key(ai_provider, ai_api_key_file)
        if api_key:
            shared_task_llm_reviewer = _PipelineDeepSeekTaskReviewer(
                api_key=api_key,
                model=ai_model,
                base_url=ai_base_url,
            )

    verdict_rows: list[dict[str, Any]] = []
    evidence_pack_rows: list[dict[str, Any]] = []
    for index, requirement_id in enumerate(requirement_order, start=1):
        finding = finding_by_requirement.get(requirement_id, {})
        task_rows = list(tasks_by_requirement.get(requirement_id, []))
        fallback_status = str(finding.get("status") or "insufficient_evidence")
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

        task_pack_rows: list[dict[str, Any]] = []
        for task in task_rows:
            pack = harvest_task_evidence(
                task,
                task_evidence_index,
                top_k=3,
                counter_k=2,
            )
            query = str(task.get("query") or "")
            retrieved_candidates = retrieve_evidence_candidates(unified_evidence_index, query, top_k=3)
            retrieved_refs = [_to_evidence_ref(item, source="evidence_index_retrieval") for item in retrieved_candidates]
            pack["evidence_refs"] = _merge_evidence_refs(list(pack.get("evidence_refs") or []), retrieved_refs)
            trace = pack.get("retrieval_trace")
            if not isinstance(trace, dict):
                trace = {}
            trace["retriever_candidates"] = len(retrieved_candidates)
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
        task_verdict_rows = judge_tasks_with_llm(task_inputs, task_reviewer, min_confidence=ai_min_confidence)
        aggregated = _aggregate_task_verdicts(task_verdict_rows)
        status = str(aggregated.get("status") or "insufficient_evidence")
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
        decision_trace["evidence_refs"] = support_refs
        decision_trace["counter_evidence_refs"] = counter_refs
        decision_trace.setdefault("decision", {})
        if isinstance(decision_trace.get("decision"), dict):
            decision_trace["decision"]["status"] = status
            decision_trace["decision"]["reason"] = reason
            decision_trace["decision"]["source"] = (
                "pipeline_counter_evidence_audit" if downgrade_action else "pipeline_findings_bridge"
            )

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
    write_verdicts_jsonl(verdicts_path, verdict_rows)
    counts = Counter(str(row.get("status") or "") for row in verdict_rows)
    llm_coverage = _safe_ratio(
        sum(1 for row in verdict_rows if isinstance(row.get("model"), dict) and row.get("model", {}).get("provider")),
        len(verdict_rows),
    )
    return {
        "verdicts": len(verdict_rows),
        "evidence_packs": len(evidence_pack_rows),
        "status_counts": dict(counts),
        "llm_coverage": llm_coverage,
    }


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
        and row.get("model", {}).get("provider")
    }

    auto_review_coverage = _safe_ratio(len(verdict_requirement_ids), requirements_total)
    evidence_traceability = _safe_ratio(
        sum(1 for row in finding_rows if _finding_has_traceable_evidence(row)),
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
            "auto_review_coverage": auto_review_coverage,
            "hard_fail_recall": hard_fail_recall,
            "false_positive_fail_rate": false_positive_fail_rate,
            "evidence_traceability": evidence_traceability,
            "llm_coverage": llm_coverage,
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
    counts = Counter(item["status"] for item in findings)
    tier_counts = Counter(str(item.get("rule_tier") or "general") for item in requirements)
    req_tier_map = {item.get("requirement_id"): str(item.get("rule_tier") or "general") for item in requirements}
    hard_fail_status = Counter(
        item["status"]
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
        f"- fail: {counts.get('fail', 0)}",
        f"- risk: {counts.get('risk', 0)}",
        f"- needs_ocr: {counts.get('needs_ocr', 0)}",
        f"- insufficient_evidence: {counts.get('insufficient_evidence', 0)}",
        f"- hard_fail_pass: {hard_fail_status.get('pass', 0)}",
        f"- hard_fail_fail: {hard_fail_status.get('fail', 0)}",
        f"- hard_fail_risk: {hard_fail_status.get('risk', 0)}",
        f"- hard_fail_needs_ocr: {hard_fail_status.get('needs_ocr', 0)}",
        f"- hard_fail_insufficient_evidence: {hard_fail_status.get('insufficient_evidence', 0)}",
        f"- consistency_findings: {len(consistency_findings)}",
        f"- manual_review_items: {manual_review_total}",
        "",
        "## Findings",
        "",
    ]

    req_map = {item["requirement_id"]: item for item in requirements}
    for item in findings:
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
            lines.append(
                "- "
                + f"[{item.get('status', 'risk')}/{item.get('severity', 'medium')}] "
                + f"{item.get('type')}: {item.get('reason')} "
                + f"| values: {values_text}"
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
    release_mode: str = "assist_only",
    gate_threshold_overrides: dict[str, Any] | None = None,
    gate_fail_fast: str = "off",
) -> dict[str, Any]:
    ensure_dir(out_dir)
    summary: dict[str, Any] = {}
    requested_release_mode = _validate_release_mode(release_mode)
    summary["ingest"] = ingest(
        tender_path=tender_path,
        bid_path=bid_path,
        out_dir=out_dir,
        resume=resume,
        page_range=page_range,
        ocr_mode=ocr_mode,
    )
    summary["extract_req"] = extract_req(out_dir=out_dir, focus=focus, resume=resume)
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
        ai_min_confidence=ai_min_confidence,
    )
    # Even when resuming expensive upstream stages, keep downstream deliverables fresh.
    # This also guarantees a new timestamped annotated copy after each `run`.
    downstream_resume = False
    summary["annotate"] = annotate(
        out_dir=out_dir,
        resume=downstream_resume,
        bid_source=bid_path,
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
        requested_release_mode=requested_release_mode,
    )
    summary["canary"] = _run_canary(
        out_dir=out_dir,
        requested_release_mode=requested_release_mode,
        gate_result=gate_result,
    )
    summary["release_trace"] = _write_release_trace(
        out_dir=out_dir,
        run_metadata=summary["run_metadata"],
        gate_result=gate_result,
        canary_result=summary["canary"],
    )
    summary["release_mode"] = str((summary.get("canary") or {}).get("release_mode") or "assist_only")
    return summary
