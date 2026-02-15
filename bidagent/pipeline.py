from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from bidagent.document import iter_document_blocks
from bidagent.io_utils import ensure_dir, path_ready, read_jsonl, write_jsonl
from bidagent.llm import DeepSeekReviewer, apply_llm_review
from bidagent.models import Block, Location, Requirement
from bidagent.review import extract_requirements, review_requirements


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
) -> dict[str, Any]:
    ingest_dir = out_dir / "ingest"
    ensure_dir(ingest_dir)
    tender_out = ingest_dir / "tender_blocks.jsonl"
    bid_out = ingest_dir / "bid_blocks.jsonl"

    summary: dict[str, Any] = {"tender_blocks": 0, "bid_blocks": 0}

    if not path_ready(tender_out, resume):
        tender_rows = (
            block.to_dict()
            for block in iter_document_blocks(tender_path, doc_id="tender", page_range=page_range)
        )
        summary["tender_blocks"] = write_jsonl(tender_out, tender_rows)
    else:
        summary["tender_blocks"] = sum(1 for _ in read_jsonl(tender_out))

    if not path_ready(bid_out, resume):
        bid_rows = (
            block.to_dict()
            for block in iter_document_blocks(bid_path, doc_id="bid", page_range=page_range)
        )
        summary["bid_blocks"] = write_jsonl(bid_out, bid_rows)
    else:
        summary["bid_blocks"] = sum(1 for _ in read_jsonl(bid_out))

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
        )

    write_jsonl(findings_path, (item.to_dict() for item in findings))

    counts = Counter(item.status for item in findings)
    return {"findings": len(findings), "status_counts": dict(counts)}


def annotate(out_dir: Path, resume: bool = False) -> dict[str, Any]:
    annotations_path = out_dir / "annotations.jsonl"
    markdown_path = out_dir / "annotations.md"
    findings_path = out_dir / "findings.jsonl"
    if path_ready(annotations_path, resume) and path_ready(markdown_path, resume):
        total = sum(1 for _ in read_jsonl(annotations_path))
        return {"annotations": total}

    findings = list(read_jsonl(findings_path))
    issue_rows = []
    for row in findings:
        if row["status"] == "pass":
            continue
        evidence = row.get("evidence", [])
        primary = evidence[0] if evidence else {}
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
                },
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
        lines.append(
            "- "
            + f"{item['note']} | block={location.get('block_index')} page={location.get('page')}"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"annotations": len(issue_rows)}


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
        is_manual = finding["status"] == "fail" or finding.get("severity") == "high"
        if not is_manual:
            continue
        requirement = req_map.get(finding["requirement_id"], {})
        evidence = finding.get("evidence", [])
        primary = evidence[0] if evidence else {}
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


def report(out_dir: Path) -> dict[str, Any]:
    requirements = list(read_jsonl(out_dir / "requirements.jsonl"))
    findings = list(read_jsonl(out_dir / "findings.jsonl"))
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
        f"- insufficient_evidence: {counts.get('insufficient_evidence', 0)}",
        f"- manual_review_items: {manual_review_total}",
        "",
        "## Findings",
        "",
    ]

    req_map = {item["requirement_id"]: item for item in requirements}
    for item in findings:
        requirement = req_map.get(item["requirement_id"], {})
        lines.append(
            "- "
            + f"[{item['status']}/{item['severity']}] {item['requirement_id']} "
            + f"({requirement.get('category', 'N/A')}): {item['reason']}"
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
    ai_provider: str | None = None,
    ai_model: str = "deepseek-chat",
    ai_api_key_file: Path | None = None,
    ai_base_url: str = "https://api.deepseek.com/v1",
    ai_workers: int = 4,
) -> dict[str, Any]:
    ensure_dir(out_dir)
    summary: dict[str, Any] = {}
    summary["ingest"] = ingest(
        tender_path=tender_path,
        bid_path=bid_path,
        out_dir=out_dir,
        resume=resume,
        page_range=page_range,
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
    )
    downstream_resume = resume and ai_provider is None
    summary["annotate"] = annotate(out_dir=out_dir, resume=downstream_resume)
    summary["checklist"] = checklist(out_dir=out_dir, resume=downstream_resume)
    summary["report"] = report(out_dir=out_dir)
    return summary
