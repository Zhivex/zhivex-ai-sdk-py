"""Shared event retention and task ownership for SDK streams."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterable
from typing import Any, Generic, Self, TypeVar

from .errors import ValidationError

EventT = TypeVar("EventT")
ResultT = TypeVar("ResultT")
# Preserve full replay by default on the Stable surface. Servers should set
# an explicit event limit (for example 4096) for request-owned streams.
DEFAULT_STREAM_BUFFER_SIZE: int | None = None


class Broadcast(Generic[EventT]):
    """One optionally bounded replay buffer, with no per-subscriber event queues.

    A consumer starts at sequence zero. If that sequence has been evicted it
    fails explicitly, rather than silently receiving an incomplete history.
    Collection of the final result never depends on event consumers.
    """

    def __init__(self, *, max_events: int | None = DEFAULT_STREAM_BUFFER_SIZE) -> None:
        if max_events is not None and (type(max_events) is not int or max_events < 1):
            raise ValidationError(
                "stream_buffer_size must be a positive integer or None."
            )
        self.history: list[EventT] | deque[EventT] = (
            [] if max_events is None else deque(maxlen=max_events)
        )
        self.done = False
        self._published = 0
        self._changed = asyncio.Event()

    async def publish(self, event: EventT) -> None:
        self.history.append(event)
        self._published += 1
        self._changed.set()

    async def close(self) -> None:
        self.done = True
        self._changed.set()

    def stream(self) -> AsyncIterable[EventT]:
        async def generator() -> AsyncIterable[EventT]:
            cursor = 0
            while True:
                first = self._published - len(self.history)
                if cursor < first:
                    raise ValidationError(
                        "Stream consumer exceeded the retained event history; "
                        "increase stream_buffer_size, consume events sooner, "
                        "or use collect() for the final result."
                    )
                if cursor < self._published:
                    event = self.history[cursor - first]
                    cursor += 1
                    yield event
                elif self.done:
                    return
                else:
                    # No await between checking the cursor and clearing the
                    # notification: a publisher cannot race this transition.
                    self._changed.clear()
                    await self._changed.wait()

        return generator()


class OwnedStream(Generic[ResultT]):
    _runner: asyncio.Task[ResultT]
    _broadcast: Broadcast[Any]

    async def aclose(self) -> None:
        """Cancel and join this stream, including upstream cleanup."""
        if not self._runner.done():
            self._runner.cancel()
        await asyncio.gather(self._runner, return_exceptions=True)
        await self._broadcast.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
