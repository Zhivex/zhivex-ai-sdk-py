"""In-memory agent memory and checkpoint storage, independent of execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._agent_context import _message_text
from ._agent_contracts import AgentCheckpoint, AgentMemoryState, SummaryConfig

if TYPE_CHECKING:
    from .agent import Agent


class InMemoryAgentMemory:
    def __init__(self, *, summary_config: SummaryConfig | None = None) -> None:
        self.summary_config = summary_config or SummaryConfig()
        self._store: dict[str, AgentMemoryState] = {}

    async def load(self, session_id: str) -> AgentMemoryState:
        state = self._store.get(session_id)
        if state is None:
            return AgentMemoryState()
        return AgentMemoryState(
            messages=list(state.messages),
            summary=state.summary,
            metadata=dict(state.metadata),
        )

    async def save(self, session_id: str, state: AgentMemoryState) -> None:
        self._store[session_id] = AgentMemoryState(
            messages=list(state.messages),
            summary=state.summary,
            metadata=dict(state.metadata),
        )

    async def summarize(
        self,
        *,
        session_id: str,
        state: AgentMemoryState,
        agent: Agent,
    ) -> str | None:
        if not state.messages:
            return state.summary
        older_messages = state.messages[: -self.summary_config.preserve_recent_messages]
        if not older_messages:
            return state.summary
        existing = state.summary.strip() if state.summary else ""
        transcript = _message_text(older_messages)
        parts = [part for part in [existing, transcript] if part]
        if not parts:
            return state.summary
        summary = "\n".join(parts)
        if len(summary) > self.summary_config.max_summary_chars:
            summary = summary[: self.summary_config.max_summary_chars]
        return summary.strip() or None


class InMemoryAgentCheckpointStore:
    def __init__(self) -> None:
        self._items: list[AgentCheckpoint] = []

    async def save(self, checkpoint: AgentCheckpoint) -> None:
        self._items.append(checkpoint)

    async def get_latest(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentCheckpoint | None:
        items = await self.list(session_id=session_id, run_id=run_id)
        if not items:
            return None
        return max(items, key=lambda item: (item.saved_at_ms, item.step_index))

    async def list(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> list[AgentCheckpoint]:
        items = list(self._items)
        if session_id is not None:
            items = [item for item in items if item.session_id == session_id]
        if run_id is not None:
            items = [item for item in items if item.run_id == run_id]
        return items


InMemoryAgentMemory.__module__ = "zhivex_ai.agent"
InMemoryAgentCheckpointStore.__module__ = "zhivex_ai.agent"
