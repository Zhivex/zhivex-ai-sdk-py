from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai.errors import ProviderHTTPError
from zhivex_ai.runtime import with_retry


class RuntimeTests(IsolatedAsyncioTestCase):
    async def test_provider_http_error_marks_retryable_statuses(self) -> None:
        self.assertTrue(ProviderHTTPError("rate limit", 429).retryable)
        self.assertFalse(ProviderHTTPError("bad request", 400).retryable)

    async def test_provider_http_error_parses_retry_after_seconds(self) -> None:
        error = ProviderHTTPError("rate limit", 429, response_headers={"Retry-After": "2"})
        self.assertEqual(error.retry_after_ms, 2000)

    async def test_provider_http_error_parses_retry_after_http_date(self) -> None:
        retry_at = datetime.now(timezone.utc) + timedelta(seconds=3)
        error = ProviderHTTPError("rate limit", 429, response_headers={"Retry-After": retry_at.strftime("%a, %d %b %Y %H:%M:%S GMT")})
        self.assertIsNotNone(error.retry_after_ms)
        assert error.retry_after_ms is not None
        self.assertGreaterEqual(error.retry_after_ms, 0)
        self.assertLessEqual(error.retry_after_ms, 3000)

    async def test_with_retry_uses_exponential_backoff(self) -> None:
        attempts = 0
        sleeps: list[int] = []

        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ProviderHTTPError("rate limit", 429)
            return "ok"

        async def fake_sleep(ms: int) -> None:
            sleeps.append(ms)

        with patch("zhivex_ai.runtime.sleep", fake_sleep):
            result = await with_retry(flaky, max_retries=2, retry_backoff_ms=100)

        self.assertEqual(result, "ok")
        self.assertEqual(sleeps, [100, 200])

    async def test_with_retry_prefers_retry_after_header(self) -> None:
        attempts = 0
        sleeps: list[int] = []

        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ProviderHTTPError("rate limit", 429, response_headers={"Retry-After": "3"})
            return "ok"

        async def fake_sleep(ms: int) -> None:
            sleeps.append(ms)

        with patch("zhivex_ai.runtime.sleep", fake_sleep):
            result = await with_retry(flaky, max_retries=1, retry_backoff_ms=100)

        self.assertEqual(result, "ok")
        self.assertEqual(sleeps, [3000])
