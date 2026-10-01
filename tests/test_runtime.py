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

    async def test_total_budget_bounds_operation_and_retry_after(self) -> None:
        import asyncio
        from zhivex_ai.runtime import execution_scope

        attempts = 0

        async def unavailable() -> str:
            nonlocal attempts
            attempts += 1
            raise ProviderHTTPError("busy", 429, response_headers={"Retry-After": "60"})

        with self.assertRaises(TimeoutError):
            async with execution_scope(30):
                await with_retry(unavailable, max_retries=5)
        self.assertEqual(attempts, 1)

        cancelled = False

        async def slow() -> str:
            nonlocal cancelled
            try:
                await asyncio.sleep(1)
                return "late"
            finally:
                cancelled = True

        with self.assertRaises(TimeoutError):
            await with_retry(slow, total_timeout_ms=20)
        self.assertTrue(cancelled)

    async def test_nested_budget_cannot_extend_parent_or_leak(self) -> None:
        import asyncio
        from zhivex_ai.runtime import current_execution_budget, execution_scope

        self.assertIsNone(current_execution_budget())
        with self.assertRaises(TimeoutError):
            async with execution_scope(20):
                async with execution_scope(1000):
                    await asyncio.sleep(1)
        self.assertIsNone(current_execution_budget())

    async def test_jitter_is_opt_in_and_never_reduces_retry_after(self) -> None:
        from zhivex_ai.runtime import retry_delay_ms

        error = ProviderHTTPError("busy", 429)
        with patch("zhivex_ai.runtime.random.uniform", return_value=1.5):
            self.assertEqual(retry_delay_ms(error, attempt=1, retry_backoff_ms=100), 200)
            self.assertEqual(retry_delay_ms(error, attempt=1, retry_backoff_ms=100, retry_jitter=0.5), 300)
            retry_after = ProviderHTTPError("busy", 429, response_headers={"Retry-After": "2"})
            self.assertEqual(retry_delay_ms(retry_after, attempt=1, retry_backoff_ms=100, retry_jitter=0.5), 2000)

    async def test_external_cancellation_is_not_retried(self) -> None:
        import asyncio

        attempts = 0

        async def cancelled() -> str:
            nonlocal attempts
            attempts += 1
            raise asyncio.CancelledError

        with self.assertRaises(asyncio.CancelledError):
            await with_retry(cancelled, max_retries=3, total_timeout_ms=100)
        self.assertEqual(attempts, 1)
