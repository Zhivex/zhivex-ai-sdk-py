from __future__ import annotations

import asyncio
import math
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from .errors import ProviderHTTPError, ValidationError

T = TypeVar("T")
P = ParamSpec("P")


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    """Internal request deadline, shared by nested operations and retries."""

    deadline: float

    def remaining_seconds(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("The total execution budget was exhausted.")
        return remaining


_budget: ContextVar[ExecutionBudget | None] = ContextVar(
    "zhivex_execution_budget", default=None
)
_jitter: ContextVar[float] = ContextVar("zhivex_retry_jitter", default=0.0)


def current_execution_budget() -> ExecutionBudget | None:
    return _budget.get()


def _validate_retry_jitter(retry_jitter: float | None) -> None:
    if retry_jitter is not None and (
        isinstance(retry_jitter, bool)
        or not isinstance(retry_jitter, (int, float))
        or not 0 <= retry_jitter <= 1
        or not math.isfinite(retry_jitter)
    ):
        raise ValidationError(
            'The "retry_jitter" field must be a finite number between zero and one.'
        )


def validate_execution_options(
    total_timeout_ms: int | None, retry_jitter: float | None = None
) -> float | None:
    """Validate execution settings before starting work and return seconds."""
    seconds = None
    if total_timeout_ms is not None:
        if (
            isinstance(total_timeout_ms, bool)
            or not isinstance(total_timeout_ms, int)
            or total_timeout_ms <= 0
        ):
            raise ValidationError(
                'The "total_timeout_ms" field must be a positive integer.'
            )
        try:
            seconds = total_timeout_ms / 1000
        except OverflowError as error:
            raise ValidationError(
                'The "total_timeout_ms" field is too large.'
            ) from error
        if not math.isfinite(seconds):
            raise ValidationError('The "total_timeout_ms" field must be finite.')
    _validate_retry_jitter(retry_jitter)
    return seconds


@asynccontextmanager
async def execution_scope(
    total_timeout_ms: int | None = None, *, retry_jitter: float | None = None
) -> AsyncIterator[ExecutionBudget | None]:
    """Bound all awaited work; nested scopes cannot extend a parent's deadline."""
    seconds = validate_execution_options(total_timeout_ms, retry_jitter)
    parent = _budget.get()
    deadline = time.monotonic() + seconds if seconds is not None else None
    if parent is not None:
        deadline = (
            min(deadline, parent.deadline) if deadline is not None else parent.deadline
        )
    budget = ExecutionBudget(deadline) if deadline is not None else None
    budget_token = _budget.set(budget)
    jitter_token = _jitter.set(
        retry_jitter if retry_jitter is not None else _jitter.get()
    )
    try:
        async with asyncio.timeout(
            budget.remaining_seconds() if budget is not None else None
        ):
            yield budget
    finally:
        _jitter.reset(jitter_token)
        _budget.reset(budget_token)


def budgeted(operation: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    @wraps(operation)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        options: dict[str, Any] = dict(kwargs)
        async with execution_scope(
            options.get("total_timeout_ms"), retry_jitter=options.get("retry_jitter")
        ):
            return await operation(*args, **kwargs)

    return wrapped


async def sleep(ms: int | float) -> None:
    await asyncio.sleep(ms / 1000)


def is_retryable_error(error: Exception) -> bool:
    return isinstance(error, ProviderHTTPError) and error.retryable


def retry_delay_ms(
    error: Exception,
    *,
    attempt: int,
    retry_backoff_ms: int,
    exponential: bool = True,
    retry_jitter: float | None = None,
) -> int:
    jitter = _jitter.get() if retry_jitter is None else retry_jitter
    _validate_retry_jitter(jitter)
    # Retry-After is a minimum requested by the provider: never jitter it downward.
    if isinstance(error, ProviderHTTPError) and error.retry_after_ms is not None:
        return max(error.retry_after_ms, 0)
    base = max(retry_backoff_ms, 0) * (2**attempt if exponential else attempt + 1)
    return round(base * random.uniform(1 - jitter, 1 + jitter)) if jitter else base


async def retry_sleep(delay_ms: int) -> None:
    budget = _budget.get()
    if budget is None:
        await sleep(delay_ms)
        return
    remaining = budget.remaining_seconds()
    if delay_ms / 1000 >= remaining:
        # Exhaust the remaining budget without issuing another external attempt.
        await sleep(remaining * 1000)
        raise TimeoutError(
            "The total execution budget was exhausted during retry backoff."
        )
    await sleep(delay_ms)


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    max_retries: int = 0,
    retry_backoff_ms: int = 250,
    *,
    total_timeout_ms: int | None = None,
    retry_jitter: float | None = None,
) -> T:
    async with execution_scope(total_timeout_ms, retry_jitter=retry_jitter):
        for attempt in range(max(0, max_retries) + 1):
            budget = _budget.get()
            if budget is not None:
                budget.remaining_seconds()
            try:
                return await operation()
            except Exception as error:
                if attempt >= max_retries or not is_retryable_error(error):
                    raise
                await retry_sleep(
                    retry_delay_ms(
                        error, attempt=attempt, retry_backoff_ms=retry_backoff_ms
                    )
                )
    raise RuntimeError("Retry loop exited without a result.")


# Retained for callers/tests of the former internal helper.
_retry_delay_ms = retry_delay_ms
