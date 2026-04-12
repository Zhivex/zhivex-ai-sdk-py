from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .errors import ProviderHTTPError

T = TypeVar("T")


async def sleep(ms: int) -> None:
    await asyncio.sleep(ms / 1000)


def is_retryable_error(error: Exception) -> bool:
    if isinstance(error, ProviderHTTPError):
        return error.retryable
    return False


async def with_retry(operation: Callable[[], Awaitable[T]], max_retries: int = 0, retry_backoff_ms: int = 250) -> T:
    last_error: Exception | None = None
    for attempt in range(max(0, max_retries) + 1):
        try:
            return await operation()
        except Exception as error:
            last_error = error
            if attempt >= max_retries or not is_retryable_error(error):
                raise
            await sleep(_retry_delay_ms(error, attempt=attempt, retry_backoff_ms=retry_backoff_ms))
    if last_error is None:
        raise RuntimeError("Retry loop exited without a result.")
    raise last_error


def _retry_delay_ms(error: Exception, *, attempt: int, retry_backoff_ms: int) -> int:
    if isinstance(error, ProviderHTTPError) and error.retry_after_ms is not None:
        return max(error.retry_after_ms, 0)
    base = max(retry_backoff_ms, 0)
    if base == 0:
        return 0
    return base * (2**attempt)
