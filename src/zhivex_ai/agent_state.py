from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

from ._serde import deserialize_messages
from .errors import ValidationError
from .types import FinishReason, JsonValue, ModelMessage, TokenUsage, ToolCall, ToolExecutionResult

AgentRunStatus = Literal["running", "completed", "failed", "cancelled", "suspended"]

_POSTGRES_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_postgres_table_prefix(table_prefix: str) -> str:
    if not _POSTGRES_IDENTIFIER_RE.match(table_prefix):
        raise ValidationError(
            'The "table_prefix" field must match the SQL identifier pattern [A-Za-z_][A-Za-z0-9_]*.'
        )
    return table_prefix


def _to_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _to_plain(item) for key, item in asdict(cast("DataclassInstance", value)).items()}
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_to_plain(value), separators=(",", ":"), sort_keys=True)


def _json_loads(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValidationError("Serialized agent run state must be a JSON object.")
    return payload


def _json_value_or_none(value: Any) -> JsonValue | None:
    plain = _to_plain(value)
    if plain is None or isinstance(plain, (str, int, float)):
        return plain
    if isinstance(plain, list):
        return [_json_value_or_none(item) for item in plain]
    if isinstance(plain, dict):
        return {str(key): _json_value_or_none(item) for key, item in plain.items()}
    return str(plain)


def _json_metadata_from_payload(payload: Any) -> dict[str, JsonValue]:
    if not isinstance(payload, dict):
        return {}
    return {str(key): _json_value_or_none(value) for key, value in payload.items()}


@dataclass(slots=True)
class PendingApproval:
    id: str
    name: str
    arguments: JsonValue | None = None
    provider: str | None = None
    reason: str | None = None
    tool_call_id: str | None = None
    permissions: list[str] = field(default_factory=list)
    source: str = "local"
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    created_at_ms: int | None = None
    handoff_path: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentChildRun:
    run_id: str
    agent_name: str
    parent_run_id: str
    status: AgentRunStatus
    output_text: str = ""
    tool_name: str | None = None
    error: str | None = None
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    usage: TokenUsage | None = None


@dataclass(slots=True)
class AgentRunStep:
    index: int
    status: AgentRunStatus
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolExecutionResult] = field(default_factory=list)
    usage: TokenUsage | None = None
    messages: list[ModelMessage] = field(default_factory=list)
    error: str | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None


@dataclass(slots=True)
class AgentRunState:
    run_id: str
    agent_name: str
    provider: str
    model_id: str
    status: AgentRunStatus = "running"
    session_id: str | None = None
    parent_run_id: str | None = None
    idempotency_key: str | None = None
    started_at_ms: int | None = None
    updated_at_ms: int | None = None
    finished_at_ms: int | None = None
    current_step: int = 0
    steps: list[AgentRunStep] = field(default_factory=list)
    child_runs: list[AgentChildRun] = field(default_factory=list)
    pending_approvals: list[PendingApproval] = field(default_factory=list)
    tool_results: list[ToolExecutionResult] = field(default_factory=list)
    usage: TokenUsage | None = None
    output_text: str = ""
    finish_reason: FinishReason | None = None
    error: str | None = None
    cancellation_reason: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


class AgentRunStore(Protocol):
    async def load(self, run_id: str) -> AgentRunState | None: ...

    async def find_by_idempotency_key(self, idempotency_key: str) -> AgentRunState | None: ...

    async def find_by_parent_run_id(self, parent_run_id: str) -> list[AgentRunState]: ...

    async def save(self, state: AgentRunState) -> None: ...


@dataclass(slots=True)
class AgentRunTreeCancellationResult:
    root: AgentRunState | None
    cancelled: list[AgentRunState] = field(default_factory=list)
    missing_parent_lookup: bool = False


def serialize_agent_run_state(state: AgentRunState) -> dict[str, Any]:
    return _to_plain(state)


def _usage_from_payload(payload: Any) -> TokenUsage | None:
    if not isinstance(payload, dict):
        return None
    return TokenUsage(
        input_tokens=payload.get("input_tokens"),
        output_tokens=payload.get("output_tokens"),
        total_tokens=payload.get("total_tokens"),
    )


def _tool_call_from_payload(payload: Any) -> ToolCall | None:
    if not isinstance(payload, dict):
        return None
    return ToolCall(
        id=str(payload.get("id", "")),
        name=str(payload.get("name", "")),
        input=payload.get("input") or {},
        provider_metadata=payload.get("provider_metadata") or {},
    )


def _tool_result_from_payload(payload: Any) -> ToolExecutionResult | None:
    from .types import ToolExecutionError

    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    return ToolExecutionResult(
        tool_call_id=str(payload.get("tool_call_id", "")),
        tool_name=str(payload.get("tool_name", "")),
        output=payload.get("output"),
        error=ToolExecutionError(message=str(error.get("message", ""))) if isinstance(error, dict) else None,
        is_error=bool(payload.get("is_error", False)),
        provider_metadata=dict(payload.get("provider_metadata") or {}),
    )


def deserialize_agent_run_state(payload: dict[str, Any]) -> AgentRunState:
    steps: list[AgentRunStep] = []
    for raw_step in payload.get("steps") or []:
        if not isinstance(raw_step, dict):
            continue
        tool_calls: list[ToolCall] = []
        for raw_call in raw_step.get("tool_calls") or []:
            tool_call = _tool_call_from_payload(raw_call)
            if tool_call is not None:
                tool_calls.append(tool_call)
        step_tool_results: list[ToolExecutionResult] = []
        for raw_result in raw_step.get("tool_results") or []:
            tool_result = _tool_result_from_payload(raw_result)
            if tool_result is not None:
                step_tool_results.append(tool_result)
        steps.append(
            AgentRunStep(
                index=int(raw_step.get("index", 0)),
                status=raw_step.get("status", "running"),
                tool_calls=tool_calls,
                tool_results=step_tool_results,
                usage=_usage_from_payload(raw_step.get("usage")),
                messages=deserialize_messages(raw_step.get("messages") if isinstance(raw_step.get("messages"), list) else None),
                error=raw_step.get("error"),
                started_at_ms=raw_step.get("started_at_ms"),
                finished_at_ms=raw_step.get("finished_at_ms"),
            )
        )
    child_runs = [
        AgentChildRun(
            run_id=str(item.get("run_id", "")),
            agent_name=str(item.get("agent_name", "")),
            parent_run_id=str(item.get("parent_run_id", "")),
            status=item.get("status", "running"),
            output_text=str(item.get("output_text", "")),
            tool_name=item.get("tool_name"),
            error=item.get("error"),
            steps=int(item.get("steps", 0)),
            tool_calls=int(item.get("tool_calls", 0)),
            tool_errors=int(item.get("tool_errors", 0)),
            usage=_usage_from_payload(item.get("usage")),
        )
        for item in payload.get("child_runs") or []
        if isinstance(item, dict)
    ]
    pending = [
        PendingApproval(
            id=str(item.get("id", "")),
            name=str(item.get("name", "")),
            arguments=item.get("arguments"),
            provider=item.get("provider"),
            reason=item.get("reason"),
            tool_call_id=item.get("tool_call_id"),
            permissions=[str(permission) for permission in item.get("permissions") or []],
            source=str(item.get("source") or "local"),
            metadata=_json_metadata_from_payload(item.get("metadata")),
            created_at_ms=item.get("created_at_ms"),
            handoff_path=[str(path_item) for path_item in item.get("handoff_path") or []],
        )
        for item in payload.get("pending_approvals") or []
        if isinstance(item, dict)
    ]
    state_tool_results: list[ToolExecutionResult] = []
    for raw_result in payload.get("tool_results") or []:
        tool_result = _tool_result_from_payload(raw_result)
        if tool_result is not None:
            state_tool_results.append(tool_result)
    return AgentRunState(
        run_id=str(payload.get("run_id", "")),
        agent_name=str(payload.get("agent_name", "")),
        provider=str(payload.get("provider", "")),
        model_id=str(payload.get("model_id", "")),
        status=payload.get("status", "running"),
        session_id=payload.get("session_id"),
        parent_run_id=payload.get("parent_run_id"),
        idempotency_key=payload.get("idempotency_key"),
        started_at_ms=payload.get("started_at_ms"),
        updated_at_ms=payload.get("updated_at_ms"),
        finished_at_ms=payload.get("finished_at_ms"),
        current_step=int(payload.get("current_step", 0)),
        steps=steps,
        child_runs=child_runs,
        pending_approvals=pending,
        tool_results=state_tool_results,
        usage=_usage_from_payload(payload.get("usage")),
        output_text=str(payload.get("output_text", "")),
        finish_reason=payload.get("finish_reason"),
        error=payload.get("error"),
        cancellation_reason=payload.get("cancellation_reason"),
        metadata=_json_metadata_from_payload(payload.get("metadata")),
    )


def agent_run_state_from_json(value: str) -> AgentRunState:
    return deserialize_agent_run_state(_json_loads(value))


def agent_run_state_to_json(state: AgentRunState) -> str:
    return _json_dumps(state)


class InMemoryAgentRunStore:
    def __init__(self) -> None:
        self._states: dict[str, AgentRunState] = {}

    async def load(self, run_id: str) -> AgentRunState | None:
        state = self._states.get(run_id)
        return agent_run_state_from_json(agent_run_state_to_json(state)) if state is not None else None

    async def find_by_idempotency_key(self, idempotency_key: str) -> AgentRunState | None:
        for state in self._states.values():
            if state.idempotency_key == idempotency_key:
                return agent_run_state_from_json(agent_run_state_to_json(state))
        return None

    async def find_by_parent_run_id(self, parent_run_id: str) -> list[AgentRunState]:
        return [
            agent_run_state_from_json(agent_run_state_to_json(state))
            for state in self._states.values()
            if state.parent_run_id == parent_run_id
        ]

    async def save(self, state: AgentRunState) -> None:
        self._states[state.run_id] = agent_run_state_from_json(agent_run_state_to_json(state))


class SQLiteAgentRunStore:
    def __init__(self, path: str, *, namespace: str = "default") -> None:
        self._path = path
        self._namespace = namespace
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS zhivex_agent_runs (
                    namespace TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    idempotency_key TEXT,
                    parent_run_id TEXT,
                    state_json TEXT NOT NULL,
                    updated_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (namespace, run_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS zhivex_agent_runs_idempotency_idx ON zhivex_agent_runs (namespace, idempotency_key)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS zhivex_agent_runs_parent_idx ON zhivex_agent_runs (namespace, parent_run_id)"
            )
            connection.commit()
        finally:
            connection.close()

    async def load(self, run_id: str) -> AgentRunState | None:
        connection = sqlite3.connect(self._path)
        try:
            row = connection.execute(
                "SELECT state_json FROM zhivex_agent_runs WHERE namespace = ? AND run_id = ?",
                (self._namespace, run_id),
            ).fetchone()
        finally:
            connection.close()
        return agent_run_state_from_json(row[0]) if row else None

    async def find_by_idempotency_key(self, idempotency_key: str) -> AgentRunState | None:
        connection = sqlite3.connect(self._path)
        try:
            row = connection.execute(
                "SELECT state_json FROM zhivex_agent_runs WHERE namespace = ? AND idempotency_key = ?",
                (self._namespace, idempotency_key),
            ).fetchone()
        finally:
            connection.close()
        return agent_run_state_from_json(row[0]) if row else None

    async def find_by_parent_run_id(self, parent_run_id: str) -> list[AgentRunState]:
        connection = sqlite3.connect(self._path)
        try:
            rows = connection.execute(
                "SELECT state_json FROM zhivex_agent_runs WHERE namespace = ? AND parent_run_id = ? ORDER BY updated_at_ms",
                (self._namespace, parent_run_id),
            ).fetchall()
        finally:
            connection.close()
        return [agent_run_state_from_json(row[0]) for row in rows]

    async def save(self, state: AgentRunState) -> None:
        payload = agent_run_state_to_json(state)
        connection = sqlite3.connect(self._path)
        try:
            connection.execute(
                """
                INSERT INTO zhivex_agent_runs
                    (namespace, run_id, idempotency_key, parent_run_id, state_json, updated_at_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, run_id)
                DO UPDATE SET
                    idempotency_key = excluded.idempotency_key,
                    parent_run_id = excluded.parent_run_id,
                    state_json = excluded.state_json,
                    updated_at_ms = excluded.updated_at_ms
                """,
                (self._namespace, state.run_id, state.idempotency_key, state.parent_run_id, payload, state.updated_at_ms or 0),
            )
            connection.commit()
        finally:
            connection.close()


class PostgresAgentRunStore:
    def __init__(self, dsn: str, *, table_prefix: str = "zhivex_agent") -> None:
        self._dsn = dsn
        self._table = f"{_validate_postgres_table_prefix(table_prefix)}_runs"

    async def _connect(self) -> Any:
        try:
            import asyncpg  # type: ignore[import-not-found,import-untyped]
        except Exception as error:
            raise RuntimeError('Postgres support requires the optional dependency "asyncpg".') from error
        connection = await asyncpg.connect(self._dsn)
        await connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                run_id TEXT PRIMARY KEY,
                idempotency_key TEXT,
                parent_run_id TEXT,
                state_json JSONB NOT NULL,
                updated_at_ms BIGINT NOT NULL
            )
            """
        )
        await connection.execute(f"CREATE INDEX IF NOT EXISTS {self._table}_idempotency_idx ON {self._table} (idempotency_key)")
        await connection.execute(f"CREATE INDEX IF NOT EXISTS {self._table}_parent_idx ON {self._table} (parent_run_id)")
        return connection

    async def load(self, run_id: str) -> AgentRunState | None:
        connection = await self._connect()
        try:
            row = await connection.fetchrow(f"SELECT state_json FROM {self._table} WHERE run_id = $1", run_id)
            if row is None:
                return None
            payload = row["state_json"]
            return deserialize_agent_run_state(payload if isinstance(payload, dict) else json.loads(payload))
        finally:
            await connection.close()

    async def find_by_idempotency_key(self, idempotency_key: str) -> AgentRunState | None:
        connection = await self._connect()
        try:
            row = await connection.fetchrow(
                f"SELECT state_json FROM {self._table} WHERE idempotency_key = $1",
                idempotency_key,
            )
            if row is None:
                return None
            payload = row["state_json"]
            return deserialize_agent_run_state(payload if isinstance(payload, dict) else json.loads(payload))
        finally:
            await connection.close()

    async def find_by_parent_run_id(self, parent_run_id: str) -> list[AgentRunState]:
        connection = await self._connect()
        try:
            rows = await connection.fetch(
                f"SELECT state_json FROM {self._table} WHERE parent_run_id = $1 ORDER BY updated_at_ms",
                parent_run_id,
            )
            states: list[AgentRunState] = []
            for row in rows:
                payload = row["state_json"]
                states.append(deserialize_agent_run_state(payload if isinstance(payload, dict) else json.loads(payload)))
            return states
        finally:
            await connection.close()

    async def save(self, state: AgentRunState) -> None:
        connection = await self._connect()
        try:
            await connection.execute(
                f"""
                INSERT INTO {self._table} (run_id, idempotency_key, parent_run_id, state_json, updated_at_ms)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                ON CONFLICT(run_id)
                DO UPDATE SET
                    idempotency_key = excluded.idempotency_key,
                    parent_run_id = excluded.parent_run_id,
                    state_json = excluded.state_json,
                    updated_at_ms = excluded.updated_at_ms
                """,
                state.run_id,
                state.idempotency_key,
                state.parent_run_id,
                agent_run_state_to_json(state),
                state.updated_at_ms or 0,
            )
        finally:
            await connection.close()


def create_in_memory_agent_run_store() -> InMemoryAgentRunStore:
    return InMemoryAgentRunStore()


def create_sqlite_agent_run_store(path: str, *, namespace: str = "default") -> SQLiteAgentRunStore:
    return SQLiteAgentRunStore(path, namespace=namespace)


def create_postgres_agent_run_store(dsn: str, *, table_prefix: str = "zhivex_agent") -> PostgresAgentRunStore:
    return PostgresAgentRunStore(dsn, table_prefix=table_prefix)


async def get_pending_agent_approvals(store: AgentRunStore, run_id: str) -> list[PendingApproval]:
    state = await store.load(run_id)
    return list(state.pending_approvals) if state is not None and state.status == "suspended" else []


async def cancel_agent_run(
    store: AgentRunStore,
    run_id: str,
    *,
    reason: str | None = None,
    now_ms: int | None = None,
) -> AgentRunState | None:
    state = await store.load(run_id)
    if state is None:
        return None
    state.status = "cancelled"
    state.cancellation_reason = reason
    state.updated_at_ms = now_ms
    state.finished_at_ms = now_ms
    await store.save(state)
    return state


async def cancel_agent_run_tree(
    store: AgentRunStore,
    run_id: str,
    *,
    reason: str | None = None,
    now_ms: int | None = None,
) -> AgentRunTreeCancellationResult:
    root = await cancel_agent_run(store, run_id, reason=reason, now_ms=now_ms)
    if root is None:
        return AgentRunTreeCancellationResult(root=None)
    cancelled = [root]

    async def collect(parent_run_id: str) -> None:
        children = await store.find_by_parent_run_id(parent_run_id)
        for child in children:
            cancelled_child = await cancel_agent_run(store, child.run_id, reason=reason, now_ms=now_ms)
            if cancelled_child is not None:
                cancelled.append(cancelled_child)
                await collect(cancelled_child.run_id)

    await collect(run_id)
    return AgentRunTreeCancellationResult(root=root, cancelled=cancelled)


def agent_child_run_from_state(state: AgentRunState, *, tool_name: str | None = None) -> AgentChildRun:
    return AgentChildRun(
        run_id=state.run_id,
        agent_name=state.agent_name,
        parent_run_id=state.parent_run_id or "",
        status=state.status,
        output_text=state.output_text,
        tool_name=tool_name,
        error=state.error,
        steps=state.current_step,
        tool_calls=sum(len(step.tool_calls) for step in state.steps),
        tool_errors=sum(1 for result in state.tool_results if result.is_error),
        usage=state.usage,
    )
