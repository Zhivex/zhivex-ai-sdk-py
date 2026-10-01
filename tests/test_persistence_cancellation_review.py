"""Independent regressions for background writes and failed context entry."""
from __future__ import annotations

import asyncio
import pickle
import sys
import threading
import types
from typing import get_type_hints
from unittest.mock import patch

import pytest

from zhivex_ai import AgentContext, AgentRunState, AgentSession, AgentTrace, PostgresAgentRunStore, SQLiteAgentRunStore
from zhivex_ai.agent import InMemoryAgentMemory
from tests.test_agent_storage_concurrency import FakePool


@pytest.mark.asyncio
async def test_cancelled_sqlite_write_uses_snapshot_without_mutating_caller(tmp_path) -> None:
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    class PausedStore(SQLiteAgentRunStore):
        pause = False

        def _save(self, state):
            if self.pause:
                entered.set()
                assert release.wait(5)
            try:
                return super()._save(state)
            finally:
                if self.pause:
                    completed.set()

    store = await PausedStore.open(str(tmp_path / "runs.sqlite"))
    state = AgentRunState("run", "agent", "provider", "model")
    await store.save(state)
    state.output_text = "at dispatch"
    store.pause = True
    operation = asyncio.create_task(store.save(state))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        state.output_text = "after cancellation"
    finally:
        release.set()
        assert await asyncio.to_thread(completed.wait, 5)
    saved = await store.load("run")
    assert saved.output_text == "at dispatch"
    assert saved.revision == 1
    assert state.revision == 0


@pytest.mark.asyncio
async def test_cancelled_sqlite_claim_keeps_original_idempotency_key(tmp_path) -> None:
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    class PausedStore(SQLiteAgentRunStore):
        def _claim_idempotency_key(self, state):
            entered.set()
            assert release.wait(5)
            try:
                return super()._claim_idempotency_key(state)
            finally:
                completed.set()

    store = await PausedStore.open(str(tmp_path / "claims.sqlite"))
    state = AgentRunState("run", "agent", "provider", "model", idempotency_key="original")
    operation = asyncio.create_task(store.claim_idempotency_key(state))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        state.idempotency_key = "changed"
    finally:
        release.set()
        assert await asyncio.to_thread(completed.wait, 5)
    claimed = await store.find_by_idempotency_key("original")
    assert claimed is not None
    assert claimed.run_id == "run"
    assert await store.find_by_idempotency_key("changed") is None


@pytest.mark.asyncio
async def test_failed_owned_postgres_context_entry_closes_pool() -> None:
    pool = FakePool()
    pool.connection.fail_schema = True

    async def create_pool(**kwargs):
        return pool

    with patch.dict(sys.modules, {"asyncpg": types.SimpleNamespace(create_pool=create_pool)}):
        store = PostgresAgentRunStore("postgres://example")
        with pytest.raises(RuntimeError, match="schema failed"):
            async with store:
                pytest.fail("schema must be initialized before entry")
    assert pool.acquired == pool.released
    assert pool.closed == 1


@pytest.mark.asyncio
async def test_failed_borrowed_postgres_context_entry_preserves_pool_owner() -> None:
    pool = FakePool()
    pool.connection.fail_schema = True
    store = PostgresAgentRunStore(pool=pool)
    with pytest.raises(RuntimeError, match="schema failed"):
        async with store:
            pytest.fail("schema must be initialized before entry")
    assert pool.acquired == pool.released
    assert pool.closed == 0


def test_extracted_contracts_preserve_pickle_paths_and_annotations() -> None:
    session = AgentSession("session")
    context = AgentContext("run", "session", "agent", session=session)
    for value in (context, session, AgentTrace("run", "session", "agent", 0), InMemoryAgentMemory()):
        assert type(value).__module__ == "zhivex_ai.agent"
        restored = pickle.loads(pickle.dumps(value))
        assert type(restored) is type(value)
    hints = get_type_hints(AgentContext)
    assert "session" in hints
    assert AgentContext("run", "session", "agent").agent_name == "agent"
