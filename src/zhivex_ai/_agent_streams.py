"""Agent streams implementation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from typing import Any

from ._agent_contracts import AgentEvent as AgentEvent
from ._agent_contracts import AgentLiveEvent as AgentLiveEvent
from ._agent_contracts import AgentOutputT as AgentOutputT
from ._agent_contracts import AgentRunResult as AgentRunResult
from ._agent_contracts import AgentTextDeltaEvent as AgentTextDeltaEvent
from ._streaming import Broadcast, OwnedStream
from .types import AudioFrame, ToolChoiceName, ToolSet


class AgentStreamResult(OwnedStream[AgentRunResult[AgentOutputT]]):
    def __init__(
        self,
        runner: asyncio.Task[AgentRunResult[AgentOutputT]],
        broadcast: Broadcast[AgentEvent],
    ) -> None:
        self._runner = runner
        self._broadcast = broadcast

    def event_stream(self) -> AsyncIterable[AgentEvent]:
        return self._broadcast.stream()

    def text_stream(self) -> AsyncIterable[str]:
        async def generator() -> AsyncIterable[str]:
            async for event in self._broadcast.stream():
                if isinstance(event, AgentTextDeltaEvent):
                    yield event.text_delta

        return generator()

    async def collect(self) -> AgentRunResult[AgentOutputT]:
        return await self._runner


class LiveAgentStreamResult(OwnedStream[AgentRunResult[AgentOutputT]]):
    def __init__(
        self,
        runner: asyncio.Task[AgentRunResult[AgentOutputT]],
        broadcast: Broadcast[AgentLiveEvent],
        live_session: asyncio.Future[Any],
    ) -> None:
        self._runner = runner
        self._broadcast = broadcast
        self._live_session = live_session

    def event_stream(self) -> AsyncIterable[AgentLiveEvent]:
        return self._broadcast.stream()

    async def send_audio(self, frame: AudioFrame) -> None:
        await (await asyncio.shield(self._live_session)).send_audio(frame)

    async def send_text(self, text: str) -> None:
        await (await asyncio.shield(self._live_session)).send_text(text)

    async def update(
        self,
        *,
        instructions: str | None = None,
        voice: str | None = None,
        tools: ToolSet | None = None,
        tool_choice: str | ToolChoiceName | None = None,
        turn_detection: dict[str, Any] | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> None:
        await (await asyncio.shield(self._live_session)).update(
            instructions=instructions,
            voice=voice,
            tools=tools,
            tool_choice=tool_choice,
            turn_detection=turn_detection,
            provider_options=provider_options,
        )

    async def aclose(self) -> None:
        await super().aclose()

    async def collect(self) -> AgentRunResult[AgentOutputT]:
        return await self._runner


# Preserve historical public class paths.
AgentStreamResult.__module__ = "zhivex_ai.agent"
LiveAgentStreamResult.__module__ = "zhivex_ai.agent"
