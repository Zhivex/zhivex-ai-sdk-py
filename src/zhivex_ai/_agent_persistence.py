"""Internal memory/checkpoint persistence; public imports remain in agent.py."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from typing import TYPE_CHECKING, Any

from .errors import ValidationError
from ._serde import (
    deserialize_generate_result,
    deserialize_messages,
    deserialize_model_generate_input,
    serialize_generate_result,
    serialize_messages,
    serialize_model_generate_input,
)

if TYPE_CHECKING:
    from .agent import Agent, AgentMemoryState, AgentCheckpoint, SummaryConfig


def _now_ms() -> int:
    from .agent import _now_ms as now_ms

    return now_ms()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _json_loads(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)


def _coerce_json_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        decoded = _json_loads(value)
        return dict(decoded or {})
    return dict(value or {})


def _serialize_agent_memory_state(state: AgentMemoryState) -> dict[str, Any]:
    return {
        "messages": serialize_messages(state.messages),
        "summary": state.summary,
        "metadata": dict(state.metadata),
    }


def _deserialize_agent_memory_state(payload: dict[str, Any] | None) -> AgentMemoryState:
    from .agent import AgentMemoryState

    if payload is None:
        return AgentMemoryState()
    return AgentMemoryState(
        messages=deserialize_messages(payload.get("messages")),
        summary=payload.get("summary"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _serialize_agent_checkpoint(checkpoint: AgentCheckpoint) -> dict[str, Any]:
    return {
        "run_id": checkpoint.run_id,
        "session_id": checkpoint.session_id,
        "agent_name": checkpoint.agent_name,
        "step_index": checkpoint.step_index,
        "request": serialize_model_generate_input(
            checkpoint.request,
            redact_tool_credentials=True,
            redact_provider_options=True,
        ),
        "response": serialize_generate_result(
            checkpoint.response, redact_raw_response=True
        ),
        "saved_at_ms": checkpoint.saved_at_ms,
        "is_final": checkpoint.is_final,
    }


def _deserialize_agent_checkpoint(payload: dict[str, Any]) -> AgentCheckpoint:
    from .agent import AgentCheckpoint

    return AgentCheckpoint(
        run_id=str(payload.get("run_id", "")),
        session_id=str(payload.get("session_id", "")),
        agent_name=str(payload.get("agent_name", "")),
        step_index=int(payload.get("step_index", 0)),
        request=deserialize_model_generate_input(dict(payload.get("request") or {})),
        response=deserialize_generate_result(dict(payload.get("response") or {})),
        saved_at_ms=int(payload.get("saved_at_ms", 0)),
        is_final=bool(payload.get("is_final", False)),
    )


class SQLiteAgentMemoryStore:
    def __init__(
        self,
        path: str,
        *,
        summary_config: SummaryConfig | None = None,
        namespace: str = "default",
    ) -> None:
        from .agent import SummaryConfig

        self.summary_config = summary_config or SummaryConfig()
        self._path = path
        self._namespace = namespace
        self._ready = False

    async def _execute(
        self, sql: str, params: tuple[Any, ...] = (), *, fetchone: bool = False
    ) -> Any:
        def runner() -> Any:
            connection = sqlite3.connect(self._path)
            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS zhivex_agent_memory (
                        namespace TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        state_json TEXT NOT NULL,
                        updated_at_ms INTEGER NOT NULL,
                        PRIMARY KEY (namespace, session_id)
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS zhivex_agent_memory_updated_idx ON zhivex_agent_memory (updated_at_ms)"
                )
                cursor = connection.execute(sql, params)
                row = cursor.fetchone() if fetchone else None
                connection.commit()
                return row
            finally:
                connection.close()

        return await asyncio.to_thread(runner)

    async def load(self, session_id: str) -> AgentMemoryState:
        row = await self._execute(
            "SELECT state_json FROM zhivex_agent_memory WHERE namespace = ? AND session_id = ?",
            (self._namespace, session_id),
            fetchone=True,
        )
        if row is None:
            return _deserialize_agent_memory_state(None)
        return _deserialize_agent_memory_state(_json_loads(row[0]))

    async def save(self, session_id: str, state: AgentMemoryState) -> None:
        payload = _json_dumps(_serialize_agent_memory_state(state))
        await self._execute(
            """
            INSERT INTO zhivex_agent_memory (namespace, session_id, state_json, updated_at_ms)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(namespace, session_id)
            DO UPDATE SET state_json = excluded.state_json, updated_at_ms = excluded.updated_at_ms
            """,
            (self._namespace, session_id, payload, _now_ms()),
        )

    async def summarize(
        self,
        *,
        session_id: str,
        state: AgentMemoryState,
        agent: "Agent",
    ) -> str | None:
        from .agent import InMemoryAgentMemory

        return await InMemoryAgentMemory(summary_config=self.summary_config).summarize(
            session_id=session_id,
            state=state,
            agent=agent,
        )


class SQLiteAgentCheckpointStore:
    def __init__(self, path: str, *, namespace: str = "default") -> None:
        self._path = path
        self._namespace = namespace

    async def _execute(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
        *,
        fetchone: bool = False,
        fetchall: bool = False,
    ) -> Any:
        def runner() -> Any:
            connection = sqlite3.connect(self._path)
            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS zhivex_agent_checkpoints (
                        namespace TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        agent_name TEXT NOT NULL,
                        step_index INTEGER NOT NULL,
                        saved_at_ms INTEGER NOT NULL,
                        is_final INTEGER NOT NULL,
                        checkpoint_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS zhivex_agent_checkpoints_session_idx
                    ON zhivex_agent_checkpoints (namespace, session_id, saved_at_ms, step_index)
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS zhivex_agent_checkpoints_run_idx
                    ON zhivex_agent_checkpoints (namespace, run_id, saved_at_ms, step_index)
                    """
                )
                cursor = connection.execute(sql, params)
                if fetchone:
                    row = cursor.fetchone()
                elif fetchall:
                    row = cursor.fetchall()
                else:
                    row = None
                connection.commit()
                return row
            finally:
                connection.close()

        return await asyncio.to_thread(runner)

    async def save(self, checkpoint: AgentCheckpoint) -> None:
        payload = _json_dumps(_serialize_agent_checkpoint(checkpoint))
        await self._execute(
            """
            INSERT INTO zhivex_agent_checkpoints (
                namespace, run_id, session_id, agent_name, step_index, saved_at_ms, is_final, checkpoint_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._namespace,
                checkpoint.run_id,
                checkpoint.session_id,
                checkpoint.agent_name,
                checkpoint.step_index,
                checkpoint.saved_at_ms,
                1 if checkpoint.is_final else 0,
                payload,
            ),
        )

    async def get_latest(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentCheckpoint | None:
        if session_id is None and run_id is None:
            raise ValidationError(
                'Pass either "session_id" or "run_id" to get_latest().'
            )
        if session_id is not None:
            row = await self._execute(
                """
                SELECT checkpoint_json
                FROM zhivex_agent_checkpoints
                WHERE namespace = ? AND session_id = ?
                ORDER BY saved_at_ms DESC, step_index DESC
                LIMIT 1
                """,
                (self._namespace, session_id),
                fetchone=True,
            )
        else:
            row = await self._execute(
                """
                SELECT checkpoint_json
                FROM zhivex_agent_checkpoints
                WHERE namespace = ? AND run_id = ?
                ORDER BY saved_at_ms DESC, step_index DESC
                LIMIT 1
                """,
                (self._namespace, run_id),
                fetchone=True,
            )
        if row is None:
            return None
        return _deserialize_agent_checkpoint(_json_loads(row[0]))

    async def list(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> list[AgentCheckpoint]:
        sql = "SELECT checkpoint_json FROM zhivex_agent_checkpoints WHERE namespace = ?"
        params: list[Any] = [self._namespace]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " ORDER BY saved_at_ms ASC, step_index ASC"
        rows = await self._execute(sql, tuple(params), fetchall=True)
        return [
            _deserialize_agent_checkpoint(_json_loads(row[0])) for row in rows or []
        ]


class PostgresAgentMemoryStore:
    def __init__(
        self,
        dsn: str,
        *,
        summary_config: SummaryConfig | None = None,
        table_prefix: str = "zhivex_ai",
    ) -> None:
        from .agent import SummaryConfig

        self.summary_config = summary_config or SummaryConfig()
        self._dsn = dsn
        from .agent import _validate_postgres_table_prefix

        self._table_prefix = _validate_postgres_table_prefix(table_prefix)

    def _table(self) -> str:
        return f"{self._table_prefix}_agent_memory"

    async def _connect(self) -> Any:
        try:
            import asyncpg  # type: ignore[import-not-found,import-untyped]
        except Exception as error:
            raise RuntimeError(
                'Postgres support requires the optional dependency "asyncpg".'
            ) from error
        return await asyncpg.connect(self._dsn)

    async def _ensure_schema(self, connection: Any) -> None:
        table = self._table()
        await connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                session_id TEXT PRIMARY KEY,
                state_json JSONB NOT NULL,
                updated_at_ms BIGINT NOT NULL
            )
            """
        )
        await connection.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_updated_idx ON {table} (updated_at_ms)"
        )

    async def load(self, session_id: str) -> AgentMemoryState:
        connection = await self._connect()
        try:
            await self._ensure_schema(connection)
            row = await connection.fetchrow(
                f"SELECT state_json FROM {self._table()} WHERE session_id = $1",
                session_id,
            )
        finally:
            await connection.close()
        if row is None:
            return _deserialize_agent_memory_state(None)
        return _deserialize_agent_memory_state(_coerce_json_payload(row["state_json"]))

    async def save(self, session_id: str, state: AgentMemoryState) -> None:
        connection = await self._connect()
        try:
            await self._ensure_schema(connection)
            await connection.execute(
                f"""
                INSERT INTO {self._table()} (session_id, state_json, updated_at_ms)
                VALUES ($1, $2::jsonb, $3)
                ON CONFLICT(session_id)
                DO UPDATE SET state_json = EXCLUDED.state_json, updated_at_ms = EXCLUDED.updated_at_ms
                """,
                session_id,
                _json_dumps(_serialize_agent_memory_state(state)),
                _now_ms(),
            )
        finally:
            await connection.close()

    async def summarize(
        self,
        *,
        session_id: str,
        state: AgentMemoryState,
        agent: "Agent",
    ) -> str | None:
        from .agent import InMemoryAgentMemory

        return await InMemoryAgentMemory(summary_config=self.summary_config).summarize(
            session_id=session_id,
            state=state,
            agent=agent,
        )


class PostgresAgentCheckpointStore:
    def __init__(self, dsn: str, *, table_prefix: str = "zhivex_ai") -> None:
        self._dsn = dsn
        from .agent import _validate_postgres_table_prefix

        self._table_prefix = _validate_postgres_table_prefix(table_prefix)

    def _table(self) -> str:
        return f"{self._table_prefix}_agent_checkpoints"

    async def _connect(self) -> Any:
        try:
            import asyncpg  # type: ignore[import-not-found,import-untyped]
        except Exception as error:
            raise RuntimeError(
                'Postgres support requires the optional dependency "asyncpg".'
            ) from error
        return await asyncpg.connect(self._dsn)

    async def _ensure_schema(self, connection: Any) -> None:
        table = self._table()
        await connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                run_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                saved_at_ms BIGINT NOT NULL,
                is_final BOOLEAN NOT NULL,
                checkpoint_json JSONB NOT NULL
            )
            """
        )
        await connection.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_session_idx ON {table} (session_id, saved_at_ms, step_index)"
        )
        await connection.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_run_idx ON {table} (run_id, saved_at_ms, step_index)"
        )

    async def save(self, checkpoint: AgentCheckpoint) -> None:
        connection = await self._connect()
        try:
            await self._ensure_schema(connection)
            await connection.execute(
                f"""
                INSERT INTO {self._table()} (
                    run_id, session_id, agent_name, step_index, saved_at_ms, is_final, checkpoint_json
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                """,
                checkpoint.run_id,
                checkpoint.session_id,
                checkpoint.agent_name,
                checkpoint.step_index,
                checkpoint.saved_at_ms,
                checkpoint.is_final,
                _json_dumps(_serialize_agent_checkpoint(checkpoint)),
            )
        finally:
            await connection.close()

    async def get_latest(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> AgentCheckpoint | None:
        if session_id is None and run_id is None:
            raise ValidationError(
                'Pass either "session_id" or "run_id" to get_latest().'
            )
        connection = await self._connect()
        try:
            await self._ensure_schema(connection)
            if session_id is not None:
                row = await connection.fetchrow(
                    f"""
                    SELECT checkpoint_json
                    FROM {self._table()}
                    WHERE session_id = $1
                    ORDER BY saved_at_ms DESC, step_index DESC
                    LIMIT 1
                    """,
                    session_id,
                )
            else:
                row = await connection.fetchrow(
                    f"""
                    SELECT checkpoint_json
                    FROM {self._table()}
                    WHERE run_id = $1
                    ORDER BY saved_at_ms DESC, step_index DESC
                    LIMIT 1
                    """,
                    run_id,
                )
        finally:
            await connection.close()
        if row is None:
            return None
        return _deserialize_agent_checkpoint(
            _coerce_json_payload(row["checkpoint_json"])
        )

    async def list(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> list[AgentCheckpoint]:
        connection = await self._connect()
        try:
            await self._ensure_schema(connection)
            if session_id is not None and run_id is not None:
                rows = await connection.fetch(
                    f"""
                    SELECT checkpoint_json
                    FROM {self._table()}
                    WHERE session_id = $1 AND run_id = $2
                    ORDER BY saved_at_ms ASC, step_index ASC
                    """,
                    session_id,
                    run_id,
                )
            elif session_id is not None:
                rows = await connection.fetch(
                    f"""
                    SELECT checkpoint_json
                    FROM {self._table()}
                    WHERE session_id = $1
                    ORDER BY saved_at_ms ASC, step_index ASC
                    """,
                    session_id,
                )
            elif run_id is not None:
                rows = await connection.fetch(
                    f"""
                    SELECT checkpoint_json
                    FROM {self._table()}
                    WHERE run_id = $1
                    ORDER BY saved_at_ms ASC, step_index ASC
                    """,
                    run_id,
                )
            else:
                rows = await connection.fetch(
                    f"""
                    SELECT checkpoint_json
                    FROM {self._table()}
                    ORDER BY saved_at_ms ASC, step_index ASC
                    """
                )
        finally:
            await connection.close()
        return [
            _deserialize_agent_checkpoint(_coerce_json_payload(row["checkpoint_json"]))
            for row in rows
        ]


def create_sqlite_agent_memory_store(
    path: str,
    *,
    summary_config: SummaryConfig | None = None,
    namespace: str = "default",
) -> SQLiteAgentMemoryStore:
    return SQLiteAgentMemoryStore(
        path, summary_config=summary_config, namespace=namespace
    )


def create_sqlite_checkpoint_store(
    path: str, *, namespace: str = "default"
) -> SQLiteAgentCheckpointStore:
    return SQLiteAgentCheckpointStore(path, namespace=namespace)


def create_postgres_agent_memory_store(
    dsn: str,
    *,
    summary_config: SummaryConfig | None = None,
    table_prefix: str = "zhivex_ai",
) -> PostgresAgentMemoryStore:
    return PostgresAgentMemoryStore(
        dsn, summary_config=summary_config, table_prefix=table_prefix
    )


def create_postgres_checkpoint_store(
    dsn: str,
    *,
    table_prefix: str = "zhivex_ai",
) -> PostgresAgentCheckpointStore:
    return PostgresAgentCheckpointStore(dsn, table_prefix=table_prefix)
