from __future__ import annotations

import io
import json
import threading
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from bidagent.llm import DeepSeekReviewer
from bidagent.models import Finding, Requirement


class _FakeClock:
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._now = float(start)
        self._lock = threading.Lock()
        self.sleep_calls: list[float] = []

    def time(self) -> float:
        with self._lock:
            return self._now

    def sleep(self, seconds: float) -> None:
        delay = float(seconds)
        if delay < 0:
            raise AssertionError("sleep called with negative delay")
        with self._lock:
            self.sleep_calls.append(delay)
            self._now += delay


class _FakeHTTPResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _success_response() -> _FakeHTTPResponse:
    decision = {
        "status": "pass",
        "severity": "none",
        "reason": "ok",
        "confidence": 0.9,
    }
    body = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(decision, ensure_ascii=False),
                    }
                }
            ]
        },
        ensure_ascii=False,
    )
    return _FakeHTTPResponse(body)


def _http_error(code: int, *, retry_after: str | None = None, body: str = "error") -> urllib.error.HTTPError:
    headers: dict[str, str] = {}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        url="https://api.deepseek.com/v1/chat/completions",
        code=code,
        msg=f"HTTP {code}",
        hdrs=headers,
        fp=io.BytesIO(body.encode("utf-8")),
    )


def _sample_requirement() -> Requirement:
    return Requirement(
        requirement_id="R0001",
        text="必须提供营业执照",
        category="资质与证照",
        mandatory=True,
        keywords=["营业执照"],
    )


def _sample_finding() -> Finding:
    return Finding(
        requirement_id="R0001",
        status="risk",
        score=1,
        severity="medium",
        reason="待确认",
        evidence=[{"excerpt": "已提交营业执照"}],
    )


class DeepSeekReviewerTests(unittest.TestCase):
    def test_parse_retry_after_supports_http_date(self) -> None:
        clock = _FakeClock(start=1_700_000_000.0)
        retry_at = datetime.fromtimestamp(clock.time(), tz=timezone.utc) + timedelta(seconds=125)
        retry_after = retry_at.strftime("%a, %d %b %Y %H:%M:%S GMT")

        with patch("bidagent.llm.time.time", side_effect=clock.time):
            delay = DeepSeekReviewer._parse_retry_after(retry_after)

        self.assertIsNotNone(delay)
        self.assertAlmostEqual(float(delay), 125.0, delta=0.01)

    def test_review_429_respects_long_retry_after_seconds_without_cap(self) -> None:
        clock = _FakeClock()
        reviewer = DeepSeekReviewer(api_key="sk-test", max_retries=1)
        responses: list[Exception | _FakeHTTPResponse] = [
            _http_error(429, retry_after="120"),
            _success_response(),
        ]

        def _fake_urlopen(*_args, **_kwargs):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with (
            patch("bidagent.llm.urllib.request.urlopen", side_effect=_fake_urlopen),
            patch("bidagent.llm.time.time", side_effect=clock.time),
            patch("bidagent.llm.time.sleep", side_effect=clock.sleep),
            patch("bidagent.llm.random.uniform", return_value=0.5) as mocked_jitter,
        ):
            result = reviewer.review(_sample_requirement(), _sample_finding())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(clock.sleep_calls, [120.0])
        mocked_jitter.assert_not_called()

    def test_review_429_parses_http_date_retry_after(self) -> None:
        clock = _FakeClock()
        reviewer = DeepSeekReviewer(api_key="sk-test", max_retries=1)
        retry_at = datetime.fromtimestamp(clock.time(), tz=timezone.utc) + timedelta(seconds=75)
        retry_after = retry_at.strftime("%a, %d %b %Y %H:%M:%S GMT")
        responses: list[Exception | _FakeHTTPResponse] = [
            _http_error(429, retry_after=retry_after),
            _success_response(),
        ]

        def _fake_urlopen(*_args, **_kwargs):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with (
            patch("bidagent.llm.urllib.request.urlopen", side_effect=_fake_urlopen),
            patch("bidagent.llm.time.time", side_effect=clock.time),
            patch("bidagent.llm.time.sleep", side_effect=clock.sleep),
        ):
            result = reviewer.review(_sample_requirement(), _sample_finding())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(clock.sleep_calls, [75.0])

    def test_review_retries_5xx_with_backoff(self) -> None:
        clock = _FakeClock()
        reviewer = DeepSeekReviewer(api_key="sk-test", max_retries=1)
        responses: list[Exception | _FakeHTTPResponse] = [
            _http_error(500),
            _success_response(),
        ]

        def _fake_urlopen(*_args, **_kwargs):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with (
            patch("bidagent.llm.urllib.request.urlopen", side_effect=_fake_urlopen),
            patch("bidagent.llm.time.time", side_effect=clock.time),
            patch("bidagent.llm.time.sleep", side_effect=clock.sleep),
            patch("bidagent.llm.random.uniform", return_value=0.25),
        ):
            result = reviewer.review(_sample_requirement(), _sample_finding())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(clock.sleep_calls, [1.25])

    def test_review_retries_on_urlerror(self) -> None:
        clock = _FakeClock()
        reviewer = DeepSeekReviewer(api_key="sk-test", max_retries=1)
        responses: list[Exception | _FakeHTTPResponse] = [
            urllib.error.URLError("temporary failure"),
            _success_response(),
        ]

        def _fake_urlopen(*_args, **_kwargs):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with (
            patch("bidagent.llm.urllib.request.urlopen", side_effect=_fake_urlopen),
            patch("bidagent.llm.time.time", side_effect=clock.time),
            patch("bidagent.llm.time.sleep", side_effect=clock.sleep),
            patch("bidagent.llm.random.uniform", return_value=0.1),
        ):
            result = reviewer.review(_sample_requirement(), _sample_finding())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(clock.sleep_calls, [1.1])

    def test_review_waits_for_cooldown_set_by_another_thread(self) -> None:
        clock = _FakeClock()
        reviewer = DeepSeekReviewer(api_key="sk-test", max_retries=0)
        with patch("bidagent.llm.time.time", side_effect=clock.time):
            setter = threading.Thread(target=reviewer._set_cooldown, args=(3.0,))
            setter.start()
            setter.join()

            with (
                patch("bidagent.llm.urllib.request.urlopen", return_value=_success_response()),
                patch("bidagent.llm.time.sleep", side_effect=clock.sleep),
            ):
                result = reviewer.review(_sample_requirement(), _sample_finding())

        self.assertEqual(result["status"], "pass")
        self.assertEqual(clock.sleep_calls, [3.0])


if __name__ == "__main__":
    unittest.main()
