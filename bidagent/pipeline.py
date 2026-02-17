from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bidagent.annotators import annotate_docx_copy, annotate_pdf_copy
from bidagent.consistency import find_inconsistencies
from bidagent.document import iter_document_blocks
from bidagent.eval import evaluate_run
from bidagent.io_utils import append_jsonl, ensure_dir, path_ready, read_jsonl, write_jsonl
from bidagent.llm import DeepSeekReviewer, apply_llm_review
from bidagent.models import Block, Location, Requirement
from bidagent.ocr import iter_document_ocr_blocks, ocr_selfcheck
from bidagent.review import enforce_evidence_quality_gate, extract_requirements, review_requirements

VALID_RELEASE_MODES = {"assist_only", "auto_final"}
GATE_THRESHOLDS = {
    "auto_review_coverage": 0.95,
    "hard_fail_recall": 0.98,
    "false_positive_fail_rate": 0.01,
    "evidence_traceability": 0.99,
    "llm_coverage": 1.0,
}
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
    if path_ready(tasks_path, resume):
        total = sum(1 for _ in read_jsonl(tasks_path))
        return {"review_tasks": total}

    requirements = list(read_jsonl(requirements_path))
    task_rows: list[dict[str, Any]] = []
    for index, requirement in enumerate(requirements, start=1):
        requirement_id = str(requirement.get("requirement_id") or f"R{index:04d}")
        tier = str(requirement.get("rule_tier") or "general")
        task_rows.append(
            {
                "task_id": _requirement_task_id(requirement_id, index),
                "requirement_id": requirement_id,
                "task_type": "requirement_check",
                "query": str(requirement.get("text") or ""),
                "expected_logic": {
                    "constraints": list(requirement.get("constraints") or []),
                    "mandatory": bool(requirement.get("mandatory", False)),
                },
                "priority": tier,
            }
        )

    write_jsonl(tasks_path, task_rows)
    return {"review_tasks": len(task_rows)}


def verdict(out_dir: Path, resume: bool = False) -> dict[str, Any]:
    findings_path = out_dir / "findings.jsonl"
    tasks_path = out_dir / "review-tasks.jsonl"
    bid_blocks_path = out_dir / "ingest" / "bid_blocks.jsonl"
    verdicts_path = out_dir / "verdicts.jsonl"
    if path_ready(verdicts_path, resume):
        rows = list(read_jsonl(verdicts_path))
        counts = Counter(str(row.get("status") or "") for row in rows)
        llm_coverage = _safe_ratio(
            sum(1 for row in rows if isinstance(row.get("model"), dict) and row.get("model", {}).get("provider")),
            len(rows),
        )
        return {"verdicts": len(rows), "status_counts": dict(counts), "llm_coverage": llm_coverage}

    task_by_requirement: dict[str, str] = {}
    tasks_by_requirement: dict[str, list[dict[str, Any]]] = {}
    if tasks_path.exists():
        for row in read_jsonl(tasks_path):
            requirement_id = str(row.get("requirement_id") or "").strip()
            task_id = str(row.get("task_id") or "").strip()
            if requirement_id and task_id and requirement_id not in task_by_requirement:
                task_by_requirement[requirement_id] = task_id
            if requirement_id:
                tasks_by_requirement.setdefault(requirement_id, []).append(row)

    evidence_index = _load_evidence_index(bid_blocks_path)
    evidence_index_stats = dict(evidence_index.get("stats") or {})

    finding_rows = list(read_jsonl(findings_path))
    verdict_rows: list[dict[str, Any]] = []
    for index, finding in enumerate(finding_rows, start=1):
        requirement_id = str(finding.get("requirement_id") or f"R{index:04d}")
        task_rows = tasks_by_requirement.get(requirement_id, [])
        status = str(finding.get("status") or "risk")
        llm = finding.get("llm") if isinstance(finding.get("llm"), dict) else {}
        confidence = _safe_float((llm or {}).get("confidence"))
        if confidence is None:
            confidence = _status_confidence_hint(status)
        confidence = max(0.0, min(1.0, confidence))
        existing_refs = [
            _to_evidence_ref(item, source="finding")
            for item in (finding.get("evidence") or [])
            if isinstance(item, dict)
        ]
        query_terms = _collect_query_terms_for_requirement(requirement_id, finding, task_rows)
        harvested = _harvest_evidence_refs(
            evidence_index,
            query_terms=query_terms,
            existing_refs=existing_refs,
            status=status,
        )
        support_refs = list(harvested.get("support_refs") or [])
        counter_refs = list(harvested.get("counter_refs") or [])
        evidence_refs = [str(item.get("evidence_id")) for item in support_refs if item.get("evidence_id")]
        counter_evidence_refs = [str(item.get("evidence_id")) for item in counter_refs if item.get("evidence_id")]

        reason = str(finding.get("reason") or "")
        status_before_audit = status
        downgrade_action: str | None = None
        if status == "pass" and counter_evidence_refs:
            status = "risk"
            reason = "命中反证/冲突证据，pass结论不稳定，已降级为risk"
            confidence = min(confidence, 0.55)
            downgrade_action = "downgrade_pass_to_risk_conflict"

        model = {}
        provider = str((llm or {}).get("provider") or "")
        model_name = str((llm or {}).get("model") or "")
        if provider or model_name:
            model = {"provider": provider, "name": model_name}

        decision_trace = finding.get("decision_trace")
        if not isinstance(decision_trace, dict):
            decision_trace = {"source": "pipeline_findings_bridge", "fallbacks": []}
        decision_trace["evidence_index"] = evidence_index_stats
        decision_trace["evidence_harvest"] = {
            "query_terms": query_terms,
            "support_refs": support_refs,
        }
        decision_trace["counter_evidence_audit"] = {
            "counter_evidence_refs": counter_refs,
            "conflict_detected": bool(counter_evidence_refs),
            "status_before": status_before_audit,
            "status_after": status,
            "action": downgrade_action,
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
                "task_id": task_by_requirement.get(requirement_id, _requirement_task_id(requirement_id, index)),
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

    write_jsonl(verdicts_path, verdict_rows)
    counts = Counter(str(row.get("status") or "") for row in verdict_rows)
    llm_coverage = _safe_ratio(
        sum(1 for row in verdict_rows if isinstance(row.get("model"), dict) and row.get("model", {}).get("provider")),
        len(verdict_rows),
    )
    return {"verdicts": len(verdict_rows), "status_counts": dict(counts), "llm_coverage": llm_coverage}


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


def gate(out_dir: Path, *, requested_release_mode: str = "assist_only") -> dict[str, Any]:
    requested_mode = _validate_release_mode(requested_release_mode)

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

    auto_review_coverage = _safe_ratio(len(verdict_rows), requirements_total)
    evidence_traceability = _safe_ratio(
        sum(1 for row in finding_rows if _finding_has_traceable_evidence(row)),
        len(finding_rows),
    )
    llm_coverage = _safe_ratio(
        sum(1 for row in verdict_rows if isinstance(row.get("model"), dict) and row.get("model", {}).get("provider")),
        len(verdict_rows),
    )

    eval_metrics = _load_eval_metrics(out_dir)
    hard_fail_recall = _safe_float((eval_metrics or {}).get("hard_fail_recall"))
    false_positive_fail_rate = _safe_float((eval_metrics or {}).get("false_positive_fail_rate"))
    if false_positive_fail_rate is None and eval_metrics:
        false_positive_fail = _safe_float(eval_metrics.get("false_positive_fail")) or 0.0
        non_fail_total = _safe_float(eval_metrics.get("non_fail_total")) or 0.0
        false_positive_fail_rate = (false_positive_fail / non_fail_total) if non_fail_total > 0 else 0.0

    checks = [
        {
            "name": "auto_review_coverage",
            "value": auto_review_coverage,
            "threshold": GATE_THRESHOLDS["auto_review_coverage"],
            "ok": auto_review_coverage >= GATE_THRESHOLDS["auto_review_coverage"],
        },
        {
            "name": "hard_fail_recall",
            "value": hard_fail_recall,
            "threshold": GATE_THRESHOLDS["hard_fail_recall"],
            "ok": hard_fail_recall is not None and hard_fail_recall >= GATE_THRESHOLDS["hard_fail_recall"],
        },
        {
            "name": "false_positive_fail_rate",
            "value": false_positive_fail_rate,
            "threshold": GATE_THRESHOLDS["false_positive_fail_rate"],
            "ok": false_positive_fail_rate is not None
            and false_positive_fail_rate <= GATE_THRESHOLDS["false_positive_fail_rate"],
        },
        {
            "name": "evidence_traceability",
            "value": evidence_traceability,
            "threshold": GATE_THRESHOLDS["evidence_traceability"],
            "ok": evidence_traceability >= GATE_THRESHOLDS["evidence_traceability"],
        },
        {
            "name": "llm_coverage",
            "value": llm_coverage,
            "threshold": GATE_THRESHOLDS["llm_coverage"],
            "ok": llm_coverage >= GATE_THRESHOLDS["llm_coverage"],
        },
    ]
    all_checks_pass = all(bool(item.get("ok")) for item in checks)
    release_mode = "auto_final" if requested_mode == "auto_final" and all_checks_pass else "assist_only"

    result = {
        "requested_release_mode": requested_mode,
        "release_mode": release_mode,
        "eligible_for_auto_final": all_checks_pass,
        "thresholds": GATE_THRESHOLDS,
        "metrics": {
            "requirements_total": requirements_total,
            "verdict_total": len(verdict_rows),
            "findings_total": len(finding_rows),
            "auto_review_coverage": auto_review_coverage,
            "hard_fail_recall": hard_fail_recall,
            "false_positive_fail_rate": false_positive_fail_rate,
            "evidence_traceability": evidence_traceability,
            "llm_coverage": llm_coverage,
            "eval_metrics": eval_metrics or {},
        },
        "checks": checks,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
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
    summary["verdict"] = verdict(out_dir=out_dir, resume=False)
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
    summary["gate"] = gate(out_dir=out_dir, requested_release_mode=requested_release_mode)
    return summary
