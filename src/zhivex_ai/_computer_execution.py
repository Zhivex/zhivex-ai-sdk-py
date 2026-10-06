"""Private, invocation-scoped tracking of possible computer effects.

The mutable state is shared with child asyncio tasks, but never persisted or
retained across invocations. It covers the executor and result finalization, so a
post-execution hook/guardrail failure cannot be mistaken for a pre-effect denial.
"""
from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from .errors import ToolExecutionOutcomeUnknown
from .types import ToolExecutionContext


@dataclass(slots=True)
class _ComputerEffectState:
    correlation: tuple[str, str, str] | None = None
    timeout_ms: int = 0


_effect: ContextVar[_ComputerEffectState | None] = ContextVar("computer_effect", default=None)


def mark_computer_effect_started(context: ToolExecutionContext[Any], timeout_ms: int) -> None:
    state = _effect.get()
    if state is not None:
        state.correlation = (context.tool_name, context.tool_call_id, context.idempotency_key or context.tool_call_id)
        state.timeout_ms = timeout_ms


@contextmanager
def computer_effect_scope(*, enabled: bool) -> Iterator[None]:
    state = _ComputerEffectState()
    token = _effect.set(state if enabled else None)
    try:
        yield
    except BaseException as error:
        if state.correlation is not None and isinstance(error, (Exception, asyncio.CancelledError)) and not isinstance(error, ToolExecutionOutcomeUnknown):
            tool_name, call_id, idempotency_key = state.correlation
            raise ToolExecutionOutcomeUnknown(
                f'Computer call "{call_id}" failed after executor entry; external outcome is unknown. '
                f'Reconcile idempotency key "{idempotency_key}" before retrying.',
                tool_name=tool_name, tool_call_id=call_id,
                timeout_ms=state.timeout_ms,
                idempotency_key=idempotency_key,
            ) from error
        raise
    finally:
        _effect.reset(token)
