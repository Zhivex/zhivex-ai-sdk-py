from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
import time
import types
from contextlib import closing
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from zhivex_ai import (
    AgentRunState,
    PendingApproval,
    PostgresAgentRunStore,
    SQLiteAgentRunStore,
    ValidationError,
)

from zhivex_ai._agent_persistence import (
    PostgresAgentCheckpointStore,
    PostgresAgentMemoryStore,
)


def state(run_id: str, *, key: str | None = None) -> AgentRunState:
    return AgentRunState(
        run_id=run_id,
        agent_name="test",
        provider="test",
        model_id="test",
        idempotency_key=key,
    )


class SQLiteRunStoreConcurrencyTests(IsolatedAsyncioTestCase):
    async def test_concurrent_open_serializes_legacy_schema_migration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "runs.sqlite")
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("""CREATE TABLE zhivex_agent_runs (
                    namespace TEXT NOT NULL, run_id TEXT NOT NULL, idempotency_key TEXT,
                    parent_run_id TEXT, state_json TEXT NOT NULL, updated_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (namespace, run_id))""")
                connection.commit()
            stores = await asyncio.gather(
                *(SQLiteAgentRunStore.open(path) for _ in range(8))
            )
            await stores[0].save(state("run"))
            self.assertTrue(
                all(
                    item is not None
                    for item in await asyncio.gather(
                        *(store.load("run") for store in stores)
                    )
                )
            )

    async def test_write_lock_does_not_block_other_coroutines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "runs.sqlite")
            store = await SQLiteAgentRunStore.open(path)
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("BEGIN IMMEDIATE")
                operation = asyncio.create_task(store.save(state("run")))
                # The writer must wait for this lock, while the event loop releases it.
                # wait_for also catches an event-loop stall once SQLite times out.
                try:
                    started = time.monotonic()
                    await asyncio.sleep(0.03)
                    elapsed = time.monotonic() - started
                    self.assertFalse(operation.done())
                    self.assertLess(elapsed, 0.5)
                finally:
                    connection.rollback()
                    await operation
            self.assertIsNotNone(await store.load("run"))

    async def test_idempotency_claims_have_one_winner_across_stores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "runs.sqlite")
            stores = [await SQLiteAgentRunStore.open(path) for _ in range(8)]
            claimed = await asyncio.gather(
                *(
                    store.claim_idempotency_key(state(f"run-{index}", key="same"))
                    for index, store in enumerate(stores)
                )
            )
            self.assertEqual(len({item.run_id for item in claimed}), 1)
            saved = await asyncio.gather(
                *(store.load(f"run-{index}") for index, store in enumerate(stores))
            )
            self.assertEqual(len([item for item in saved if item is not None]), 1)

    async def test_pending_approval_is_claimed_by_only_one_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "runs.sqlite")
            stores = [await SQLiteAgentRunStore.open(path) for _ in range(8)]
            suspended = state("run")
            suspended.status = "suspended"
            suspended.pending_approvals = [
                PendingApproval(id="approval", name="lookup")
            ]
            await stores[0].save(suspended)
            claims = await asyncio.gather(
                *(
                    store.claim_pending_approval(
                        "run",
                        "approval",
                        claim_token=f"claim-{index}",
                        claimed_at_ms=100,
                    )
                    for index, store in enumerate(stores)
                )
            )
            winners = [claim for claim in claims if claim is not None]
            self.assertEqual(len(winners), 1)
            persisted = await stores[0].load("run")
            assert persisted is not None
            self.assertEqual(
                persisted.metadata["resume_claim"], winners[0].metadata["resume_claim"]
            )

    async def test_concurrent_saves_reject_stale_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "runs.sqlite")
            stores = [await SQLiteAgentRunStore.open(path) for _ in range(2)]
            await stores[0].save(state("run"))
            states = await asyncio.gather(*(store.load("run") for store in stores))
            assert all(item is not None for item in states)
            results = await asyncio.gather(
                *(
                    store.save(item)
                    for store, item in zip(stores, states)
                    if item is not None
                ),
                return_exceptions=True,
            )
            self.assertEqual(
                sum(isinstance(item, AgentRunState) for item in results), 1
            )
            self.assertEqual(
                sum(isinstance(item, ValidationError) for item in results), 1
            )


class FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class FakeConnection:
    def __init__(self) -> None:
        self.sql: list[str] = []
        self.fail_schema = False

    def transaction(self):
        return FakeTransaction()

    async def execute(self, sql, *args):
        self.sql.append(sql)
        await asyncio.sleep(0)
        if self.fail_schema and "CREATE TABLE" in sql:
            self.fail_schema = False
            raise RuntimeError("schema failed")
        return "OK"

    async def fetchrow(self, sql, *args):
        return None

    async def fetch(self, sql, *args):
        return []


class FakePool:
    def __init__(self) -> None:
        self.connection = FakeConnection()
        self.acquired = 0
        self.released = 0
        self.closed = 0

    async def acquire(self):
        self.acquired += 1
        return self.connection

    async def release(self, connection):
        self.released += 1

    async def close(self):
        self.closed += 1


class PostgresAgentPoolTests(IsolatedAsyncioTestCase):
    async def test_schema_is_initialized_once_and_borrowed_pool_is_not_closed(
        self,
    ) -> None:
        for cls in (
            PostgresAgentRunStore,
            PostgresAgentMemoryStore,
            PostgresAgentCheckpointStore,
        ):
            with self.subTest(store=cls.__name__):
                pool = FakePool()
                store = cls(pool=pool)
                async with store:
                    await asyncio.gather(*(store.initialize() for _ in range(10)))
                    if isinstance(store, PostgresAgentCheckpointStore):
                        await store.list()
                        await store.get_latest(run_id="missing")
                    else:
                        await store.load("missing")
                        await store.load("missing")
                self.assertEqual(
                    sum("CREATE TABLE" in sql for sql in pool.connection.sql), 1
                )
                self.assertEqual(pool.acquired, pool.released)
                self.assertEqual(pool.closed, 0)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    await store.initialize()

    async def test_owned_pool_is_created_once_and_closed_once(self) -> None:
        pool = FakePool()
        calls = []

        async def create_pool(**kwargs):
            calls.append(kwargs)
            return pool

        with patch.dict(
            sys.modules, {"asyncpg": types.SimpleNamespace(create_pool=create_pool)}
        ):
            store = PostgresAgentRunStore(
                "postgres://example", pool_min_size=0, pool_max_size=3
            )
            await asyncio.gather(*(store.load("missing") for _ in range(10)))
            await store.close()
            await store.close()
        self.assertEqual(
            calls, [{"dsn": "postgres://example", "min_size": 0, "max_size": 3}]
        )
        self.assertEqual(pool.closed, 1)
        self.assertEqual(pool.acquired, pool.released)

    async def test_schema_failure_releases_lease_and_allows_retry(self) -> None:
        pool = FakePool()
        pool.connection.fail_schema = True
        store = PostgresAgentRunStore(pool=pool)
        with self.assertRaisesRegex(RuntimeError, "schema failed"):
            await store.initialize()
        self.assertEqual(pool.acquired, pool.released)
        await store.initialize()
        await store.load("missing")
        self.assertEqual(sum("CREATE TABLE" in sql for sql in pool.connection.sql), 2)
        self.assertEqual(pool.acquired, pool.released)
        await store.close()

    async def test_invalid_pool_bounds_and_missing_dsn_fail_before_io(self) -> None:
        for cls in (
            PostgresAgentRunStore,
            PostgresAgentMemoryStore,
            PostgresAgentCheckpointStore,
        ):
            with self.subTest(store=cls.__name__):
                with self.assertRaises(ValidationError):
                    cls()
                for minimum, maximum in ((-1, 5), (2, 1), (0, 0), (True, 5)):
                    with self.assertRaises(ValidationError):
                        cls(
                            pool=FakePool(),
                            pool_min_size=minimum,
                            pool_max_size=maximum,
                        )
