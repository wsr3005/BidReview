from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from bidagent.models import Finding, Requirement

ALLOWED_STATUS = {"pass", "risk", "fail", "needs_ocr", "missing", "insufficient_evidence"}
ALLOWED_SEVERITY = {"none", "low", "medium", "high"}
DEEPSEEK_PROMPT_VERSION = "deepseek-review-v1"
DEEPSEEK_REQUIREMENT_PROMPT_VERSION = "deepseek-requirement-schema-v1"


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
    prompt_version = DEEPSEEK_PROMPT_VERSION

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        timeout_seconds: int = 90,
        max_retries: int = 4,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))
        # Shared cooldown to reduce thundering herd on 429 across worker threads.
        self._cooldown_lock = threading.Lock()
        self._cooldown_until = 0.0

    def _maybe_sleep_for_cooldown(self) -> None:
        with self._cooldown_lock:
            until = float(self._cooldown_until)
        now = time.time()
        if until > now:
            time.sleep(until - now)

    def _set_cooldown(self, seconds: float) -> None:
        delay = max(0.0, float(seconds))
        now = time.time()
        with self._cooldown_lock:
            self._cooldown_until = max(float(self._cooldown_until), now + delay)

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            seconds = float(text)
        except ValueError:
            seconds = None
        if seconds is not None:
            if seconds < 0:
                return None
            return seconds

        try:
            retry_at = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, retry_at.timestamp() - time.time())

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
            "constraints": getattr(requirement, "constraints", []),
            "rule_tier": getattr(requirement, "rule_tier", "general"),
            "rule_status": finding.status,
            "rule_severity": finding.severity,
            "rule_reason": finding.reason,
            "evidence": evidence_rows,
            "task": "仅审查商务合规，不审查技术方案。基于证据作结论，禁止臆测。",
            "output_schema": {
                "status": "pass|risk|fail|needs_ocr|missing|insufficient_evidence",
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

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._maybe_sleep_for_cooldown()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                last_error = None
                break
            except urllib.error.HTTPError as exc:
                retry_after = self._parse_retry_after(exc.headers.get("Retry-After"))
                code = int(getattr(exc, "code", 0) or 0)
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    detail = ""

                # Retry policy:
                # - 429: respect Retry-After when present; otherwise backoff. Also set a shared cooldown.
                # - 5xx/408: retry with backoff.
                # - others: fail fast.
                if attempt < self.max_retries and (code == 429 or code == 408 or 500 <= code <= 599):
                    if retry_after is not None:
                        delay = retry_after
                    else:
                        delay = min(60.0, (1.0 * (2**attempt)) + random.uniform(0.0, 0.5))
                    if code == 429:
                        self._set_cooldown(delay)
                    time.sleep(delay)
                    last_error = RuntimeError(f"DeepSeek HTTP {code}: {detail[:200]}")
                    continue
                raise RuntimeError(f"DeepSeek HTTP {code}: {detail[:200]}") from exc
            except urllib.error.URLError as exc:
                # Transient network errors: retry with backoff.
                if attempt < self.max_retries:
                    delay = min(60.0, (1.0 * (2**attempt)) + random.uniform(0.0, 0.5))
                    time.sleep(delay)
                    last_error = RuntimeError(f"DeepSeek request failed: {exc.reason}")
                    continue
                raise RuntimeError(f"DeepSeek request failed: {exc.reason}") from exc

        if last_error is not None:
            raise last_error

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

        if status == "insufficient_evidence":
            status = "missing"
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


class DeepSeekRequirementExtractor:
    provider = "deepseek"
    prompt_version = DEEPSEEK_REQUIREMENT_PROMPT_VERSION

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        timeout_seconds: int = 90,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))

    def _build_messages(self, *, block_text: str, focus: str) -> list[dict[str, str]]:
        payload = {
            "focus": focus,
            "task": "从招标文本中提取可核验的商务条款，必须输出JSON。",
            "schema": {
                "items": [
                    {
                        "text": "string",
                        "category": "string",
                        "mandatory": "boolean",
                        "rule_tier": "hard_fail|scored|general",
                        "keywords": ["string"],
                        "confidence": "number 0.0-1.0",
                    }
                ]
            },
            "rules": [
                "只提取投标人义务条款，不提取评标流程描述",
                "不要输出目录、模板说明、章节标题",
                "不能编造原文不存在的金额、期限、资质",
            ],
            "text": block_text[:2400],
        }
        return [
            {
                "role": "system",
                "content": (
                    "你是招投标商务条款提取器。"
                    "输出必须是JSON对象，且顶层字段必须为items。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]

    def extract_requirements(self, *, block_text: str, focus: str) -> list[dict[str, Any]]:
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": self._build_messages(block_text=block_text, focus=focus),
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

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                last_error = None
                break
            except urllib.error.HTTPError as exc:
                code = int(getattr(exc, "code", 0) or 0)
                if attempt < self.max_retries and (code == 429 or code == 408 or 500 <= code <= 599):
                    time.sleep(min(20.0, 1.0 * (2**attempt)))
                    last_error = RuntimeError(f"DeepSeek HTTP {code}")
                    continue
                raise RuntimeError(f"DeepSeek HTTP {code}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(min(20.0, 1.0 * (2**attempt)))
                    last_error = RuntimeError(f"DeepSeek request failed: {exc.reason}")
                    continue
                raise RuntimeError(f"DeepSeek request failed: {exc.reason}") from exc

        if last_error is not None:
            raise last_error

        data_obj = json.loads(body)
        choices = data_obj.get("choices", [])
        if not choices:
            return []
        message = choices[0].get("message", {})
        content = str(message.get("content") or "").strip()
        if not content:
            return []
        parsed = _extract_json_object(content)
        rows = parsed.get("items")
        if not isinstance(rows, list):
            return []
        return [item for item in rows if isinstance(item, dict)]


def apply_llm_review(
    requirements: list[Requirement],
    findings: list[Finding],
    reviewer: Reviewer,
    max_workers: int = 4,
    *,
    min_confidence: float = 0.65,
) -> list[Finding]:
    worker_count = max(1, int(max_workers))
    prompt_version = str(getattr(reviewer, "prompt_version", "") or "").strip() or None
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
                "prompt_version": prompt_version,
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
                "prompt_version": prompt_version,
                "error": str(exc),
            }
            finding.decision_trace = trace
            finding.llm = {
                "provider": reviewer.provider,
                "model": reviewer.model,
                "prompt_version": prompt_version,
                "error": str(exc),
            }
            return index, finding

        finding.status = llm_result["status"]
        finding.severity = llm_result["severity"]
        finding.reason = llm_result["reason"] or finding.reason
        # Low-confidence guardrail: do not allow low-confidence pass to slip through as "ok".
        if finding.status == "pass" and llm_result["confidence"] < float(min_confidence):
            previous = {"status": finding.status, "severity": finding.severity, "reason": finding.reason}
            finding.status = "risk"
            finding.severity = "high"
            finding.reason = f"LLM置信度低({llm_result['confidence']:.2f})，需人工复核"
            trace = finding.decision_trace if isinstance(finding.decision_trace, dict) else {}
            trace["low_confidence_fallback"] = {
                "min_confidence": float(min_confidence),
                "confidence": llm_result["confidence"],
                "previous": previous,
                "action": "downgrade_pass_to_risk_high",
            }
            finding.decision_trace = trace
        trace = finding.decision_trace if isinstance(finding.decision_trace, dict) else {}
        trace.setdefault("decision", {})
        trace["decision"]["status"] = finding.status
        trace["decision"]["reason"] = finding.reason
        trace["decision"]["source"] = "llm"
        trace["llm_review"] = {
            "provider": reviewer.provider,
            "model": reviewer.model,
            "prompt_version": prompt_version,
            "status": llm_result["status"],
            "severity": llm_result["severity"],
            "confidence": llm_result["confidence"],
        }
        finding.decision_trace = trace
        finding.llm = {
            "provider": reviewer.provider,
            "model": reviewer.model,
            "prompt_version": prompt_version,
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

    # Bounded scheduling: avoid creating thousands of Future objects at once on huge runs.
    results: list[Finding] = [*findings]
    if not findings:
        return results

    max_in_flight = max(2, worker_count * 2)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        it = iter(enumerate(findings))
        in_flight = set()

        def _submit_next() -> bool:
            try:
                idx, finding = next(it)
            except StopIteration:
                return False
            in_flight.add(executor.submit(_review_one, idx, finding))
            return True

        while len(in_flight) < max_in_flight and _submit_next():
            pass

        while in_flight:
            done = next(as_completed(in_flight))
            in_flight.remove(done)
            idx, updated_finding = done.result()
            results[idx] = updated_finding
            _submit_next()

    return results
