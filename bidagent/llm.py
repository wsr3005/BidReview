from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

from bidagent.models import Finding, Requirement

ALLOWED_STATUS = {"pass", "risk", "fail", "needs_ocr", "insufficient_evidence"}
ALLOWED_SEVERITY = {"none", "low", "medium", "high"}


class Reviewer(Protocol):
    provider: str
    model: str

    def review(self, requirement: Requirement, finding: Finding) -> dict:
        ...


def _strip_code_block(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _extract_json_object(text: str) -> dict:
    cleaned = _strip_code_block(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM response is not valid JSON")
        return json.loads(match.group(0))


class DeepSeekReviewer:
    provider = "deepseek"

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        timeout_seconds: int = 90,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _build_messages(self, requirement: Requirement, finding: Finding) -> list[dict]:
        evidence_rows = []
        for item in finding.evidence:
            location = item.get("location") or {}
            evidence_rows.append(
                {
                    "score": item.get("score"),
                    "page": location.get("page"),
                    "block_index": location.get("block_index"),
                    "excerpt": item.get("excerpt"),
                }
            )
        user_payload = {
            "requirement_id": requirement.requirement_id,
            "requirement_text": requirement.text,
            "category": requirement.category,
            "mandatory": requirement.mandatory,
            "rule_status": finding.status,
            "rule_severity": finding.severity,
            "rule_reason": finding.reason,
            "evidence": evidence_rows,
            "task": "仅审查商务合规，不审查技术方案。基于证据作结论，禁止臆测。",
            "output_schema": {
                "status": "pass|risk|fail|needs_ocr|insufficient_evidence",
                "severity": "none|low|medium|high",
                "reason": "string(<=80 chars)",
                "confidence": "number 0.0-1.0",
            },
        }
        return [
            {
                "role": "system",
                "content": (
                    "你是招投标商务审查专家。只根据给定证据判断，不得编造。"
                    "输出必须是 JSON 对象，不要输出其它文本。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ]

    def review(self, requirement: Requirement, finding: Finding) -> dict:
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": self._build_messages(requirement, finding),
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"DeepSeek HTTP {exc.code}: {detail[:200]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DeepSeek request failed: {exc.reason}") from exc

        data_obj = json.loads(body)
        choices = data_obj.get("choices", [])
        if not choices:
            raise RuntimeError("DeepSeek response missing choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not content:
            raise RuntimeError("DeepSeek response missing message content")

        parsed = _extract_json_object(content)
        status = parsed.get("status")
        severity = parsed.get("severity")
        reason = str(parsed.get("reason", "")).strip()
        confidence = parsed.get("confidence")

        if status not in ALLOWED_STATUS:
            raise RuntimeError(f"Invalid LLM status: {status}")
        if severity not in ALLOWED_SEVERITY:
            raise RuntimeError(f"Invalid LLM severity: {severity}")
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Invalid LLM confidence") from exc

        confidence_value = min(1.0, max(0.0, confidence_value))
        return {
            "status": status,
            "severity": severity,
            "reason": reason[:120],
            "confidence": confidence_value,
        }


def apply_llm_review(
    requirements: list[Requirement],
    findings: list[Finding],
    reviewer: Reviewer,
    max_workers: int = 4,
) -> list[Finding]:
    worker_count = max(1, int(max_workers))
    req_map = {item.requirement_id: item for item in requirements}

    def _review_one(index: int, finding: Finding) -> tuple[int, Finding]:
        requirement = req_map.get(finding.requirement_id)
        if requirement is None:
            return index, finding
        if finding.status == "needs_ocr":
            trace = finding.decision_trace if isinstance(finding.decision_trace, dict) else {}
            trace.setdefault("decision", {})
            trace["decision"]["status"] = finding.status
            trace["decision"]["reason"] = finding.reason
            trace["decision"]["source"] = "rule"
            trace["llm_review"] = {"skipped": "needs_ocr"}
            finding.decision_trace = trace
            finding.llm = {
                "provider": reviewer.provider,
                "model": reviewer.model,
                "skipped": "needs_ocr",
            }
            return index, finding
        try:
            llm_result = reviewer.review(requirement, finding)
        except Exception as exc:  # noqa: BLE001
            trace = finding.decision_trace if isinstance(finding.decision_trace, dict) else {}
            trace["llm_review"] = {
                "provider": reviewer.provider,
                "model": reviewer.model,
                "error": str(exc),
            }
            finding.decision_trace = trace
            finding.llm = {
                "provider": reviewer.provider,
                "model": reviewer.model,
                "error": str(exc),
            }
            return index, finding

        finding.status = llm_result["status"]
        finding.severity = llm_result["severity"]
        finding.reason = llm_result["reason"] or finding.reason
        trace = finding.decision_trace if isinstance(finding.decision_trace, dict) else {}
        trace.setdefault("decision", {})
        trace["decision"]["status"] = finding.status
        trace["decision"]["reason"] = finding.reason
        trace["decision"]["source"] = "llm"
        trace["llm_review"] = {
            "provider": reviewer.provider,
            "model": reviewer.model,
            "status": llm_result["status"],
            "severity": llm_result["severity"],
            "confidence": llm_result["confidence"],
        }
        finding.decision_trace = trace
        finding.llm = {
            "provider": reviewer.provider,
            "model": reviewer.model,
            "confidence": llm_result["confidence"],
            "decision": llm_result,
            "requirement": {
                "requirement_id": requirement.requirement_id,
                "text": requirement.text[:240],
                "category": requirement.category,
                "mandatory": requirement.mandatory,
            },
        }
        return index, finding

    results: list[Finding] = [*findings]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_review_one, idx, finding) for idx, finding in enumerate(findings)]
        for future in as_completed(futures):
            idx, updated_finding = future.result()
            results[idx] = updated_finding
    return results
