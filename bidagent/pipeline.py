from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from bidagent.annotators import annotate_docx_copy, annotate_pdf_copy
from bidagent.consistency import find_inconsistencies
from bidagent.document import iter_document_blocks
from bidagent.io_utils import append_jsonl, ensure_dir, path_ready, read_jsonl, write_jsonl
from bidagent.llm import DeepSeekReviewer, apply_llm_review
from bidagent.models import Block, Location, Requirement
from bidagent.ocr import iter_document_ocr_blocks, ocr_selfcheck
from bidagent.review import enforce_evidence_quality_gate, extract_requirements, review_requirements


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

    summary: dict[str, Any] = {"tender_blocks": 0, "bid_blocks": 0, "bid_ocr_blocks": 0}
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
    expected_copy: Path | None = None
    if source_path is not None:
        expected_copy = out_dir / "annotated" / f"{source_path.stem}.annotated{source_path.suffix}"

    if path_ready(annotations_path, resume) and path_ready(markdown_path, resume):
        total = sum(1 for _ in read_jsonl(annotations_path))
        if expected_copy is None or expected_copy.exists():
            response: dict[str, Any] = {"annotations": total}
            if expected_copy is not None:
                response["annotated_copy"] = str(expected_copy)
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

    annotated_dir = out_dir / "annotated"
    suffix = source_path.suffix.lower()
    output_path = annotated_dir / f"{source_path.stem}.annotated{source_path.suffix}"
    try:
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
        is_manual = finding["status"] in {"fail", "needs_ocr"} or finding.get("severity") == "high"
        if not is_manual:
            continue
        requirement = req_map.get(finding["requirement_id"], {})
        evidence = finding.get("evidence", [])
        primary = _choose_primary_evidence(evidence, status=str(finding.get("status") or ""))
        if not primary:
            primary = {}
        review_rows.append(
            {
                "requirement_id": finding["requirement_id"],
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
            + f"[{row['status']}/{row['severity']}] {row['requirement_id']} "
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
    report_path = out_dir / "review-report.md"

    lines = [
        "# Bid Review Report",
        "",
        "## Summary",
        "",
        f"- Total requirements: {len(requirements)}",
        f"- pass: {counts.get('pass', 0)}",
        f"- fail: {counts.get('fail', 0)}",
        f"- risk: {counts.get('risk', 0)}",
        f"- needs_ocr: {counts.get('needs_ocr', 0)}",
        f"- insufficient_evidence: {counts.get('insufficient_evidence', 0)}",
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
        lines.append(
            "- "
            + f"[{item['status']}/{item['severity']}] {item['requirement_id']} "
            + f"({requirement.get('category', 'N/A')}): {item['reason']} | {evidence_text} | trace: {trace_text}"
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
            lines.append(
                "- "
                + f"[{item.get('status', 'risk')}/{item.get('severity', 'medium')}] "
                + f"{item.get('type')}: {item.get('reason')} "
                + f"| values: {', '.join(_sanitize_md_text(v, limit=60) for v in top_values if v)}"
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
) -> dict[str, Any]:
    ensure_dir(out_dir)
    summary: dict[str, Any] = {}
    summary["ingest"] = ingest(
        tender_path=tender_path,
        bid_path=bid_path,
        out_dir=out_dir,
        resume=resume,
        page_range=page_range,
        ocr_mode=ocr_mode,
    )
    summary["extract_req"] = extract_req(out_dir=out_dir, focus=focus, resume=resume)
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
    downstream_resume = resume and ai_provider is None
    summary["annotate"] = annotate(
        out_dir=out_dir,
        resume=downstream_resume,
        bid_source=bid_path,
    )
    summary["checklist"] = checklist(out_dir=out_dir, resume=downstream_resume)
    summary["consistency"] = consistency(out_dir=out_dir, resume=downstream_resume)
    summary["report"] = report(out_dir=out_dir)
    return summary
