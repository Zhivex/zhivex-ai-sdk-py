"""Single-host contention and namespace boundaries for the Stable local store."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from zhivex_ai import AgentRunState, ValidationError, create_sqlite_agent_run_store


def test_sqlite_independent_connections_claim_one_idempotency_key(tmp_path: Path) -> None:
    path = str(tmp_path / "runs.sqlite3")
    stores = [create_sqlite_agent_run_store(path) for _ in range(4)]
    barrier = Barrier(len(stores))

    def claim(index: int) -> str:
        state = AgentRunState(
            run_id=f"run-{index}", agent_name="test", provider="mock", model_id="test",
            idempotency_key="same-job",
        )
        barrier.wait(timeout=10)
        return asyncio.run(stores[index].claim_idempotency_key(state)).run_id

    with ThreadPoolExecutor(max_workers=len(stores)) as pool:
        run_ids = list(pool.map(claim, range(len(stores))))
    assert len(set(run_ids)) == 1
    reopened = create_sqlite_agent_run_store(path)
    persisted = asyncio.run(reopened.find_by_idempotency_key("same-job"))
    assert persisted is not None and persisted.run_id == run_ids[0]


@pytest.mark.asyncio
async def test_sqlite_reopened_revision_and_namespace_isolation(tmp_path: Path) -> None:
    path = str(tmp_path / "runs.sqlite3")
    first = create_sqlite_agent_run_store(path, namespace="first")
    second = create_sqlite_agent_run_store(path, namespace="second")
    for store in (first, second):
        await store.claim_idempotency_key(AgentRunState(
            run_id="shared-id", agent_name="test", provider="mock", model_id="test",
            idempotency_key="shared-key",
        ))
    stale = await first.load("shared-id")
    assert stale is not None
    reopened = create_sqlite_agent_run_store(path, namespace="first")
    await reopened.cancel_run("shared-id", reason="stop")
    stale.output_text = "late-write"
    with pytest.raises(ValidationError, match="revision conflict"):
        await first.save(stale)
    other = await second.load("shared-id")
    assert other is not None and other.status == "running" and other.revision == 0
