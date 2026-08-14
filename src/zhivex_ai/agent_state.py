from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

from ._serde import deserialize_messages
from .errors import ValidationError
from .types import FinishReason, JsonValue, ModelMessage, TokenUsage, ToolCall, ToolExecutionResult

AgentRunStatus = Literal["running", "completed", "failed", "cancelled", "suspended"]

AGENT_RUN_STATE_SCHEMA_VERSION = 1
_TERMINAL_AGENT_RUN_STATUSES: frozenset[AgentRunStatus] = frozenset({"completed", "failed", "cancelled"})
_POSTGRES_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _now_ms() -> int:
    return int(time.time() * 1000)


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
    tool_fingerprint: str | None = None


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
    schema_version: int = AGENT_RUN_STATE_SCHEMA_VERSION
    revision: int = 0
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

    async def save(self, state: AgentRunState) -> AgentRunState | None: ...


@dataclass(slots=True)
class AgentRunTreeCancellationResult:
    root: AgentRunState | None
    cancelled: list[AgentRunState] = field(default_factory=list)
    missing_parent_lookup: bool = False


def serialize_agent_run_state(state: AgentRunState) -> dict[str, Any]:
    _validate_state_version(state.schema_version, state.revision)
    return _to_plain(state)


def _validate_state_version(schema_version: Any, revision: Any) -> tuple[int, int]:
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version < 1:
        raise ValidationError("Agent run state schema_version must be a positive integer.")
    if schema_version > AGENT_RUN_STATE_SCHEMA_VERSION:
        raise ValidationError(
            "Agent run state uses unsupported future schema_version "
            f"{schema_version}; this SDK supports up to {AGENT_RUN_STATE_SCHEMA_VERSION}."
        )
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValidationError("Agent run state revision must be a non-negative integer.")
    return schema_version, revision


def _clone_agent_run_state(state: AgentRunState) -> AgentRunState:
    return agent_run_state_from_json(agent_run_state_to_json(state))


def _prepare_new_state(state: AgentRunState) -> AgentRunState:
    _validate_state_version(state.schema_version, state.revision)
    if state.revision != 0:
        raise ValidationError(f'New agent run "{state.run_id}" must start at revision 0.')
    return _clone_agent_run_state(state)


def _prepare_state_update(current: AgentRunState, candidate: AgentRunState) -> AgentRunState:
    _validate_state_version(candidate.schema_version, candidate.revision)
    if candidate.schema_version != current.schema_version:
        raise ValidationError(
            f'Agent run "{candidate.run_id}" schema_version cannot change during an update.'
        )
    if candidate.revision != current.revision:
        raise ValidationError(
            f'Agent run "{candidate.run_id}" revision conflict: expected {candidate.revision}, '
            f"stored revision is {current.revision}. Reload the state before retrying."
        )
    if current.status in _TERMINAL_AGENT_RUN_STATUSES:
        raise ValidationError(
            f'Agent run "{candidate.run_id}" is terminal with status "{current.status}" and cannot be overwritten.'
        )
    updated = _clone_agent_run_state(candidate)
    updated.revision = current.revision + 1
    return updated


def _copy_persisted_revision(target: AgentRunState, persisted: AgentRunState) -> AgentRunState:
    target.schema_version = persisted.schema_version
    target.revision = persisted.revision
    return _clone_agent_run_state(persisted)


def _cancel_state(
    current: AgentRunState,
    *,
    reason: str | None,
    cancelled_at_ms: int | None,
) -> AgentRunState:
    if current.status in _TERMINAL_AGENT_RUN_STATUSES:
        return _clone_agent_run_state(current)
    effective_cancelled_at_ms = cancelled_at_ms if cancelled_at_ms is not None else _now_ms()
    cancelled = _clone_agent_run_state(current)
    cancelled.status = "cancelled"
    cancelled.cancellation_reason = reason
    cancelled.updated_at_ms = effective_cancelled_at_ms
    cancelled.finished_at_ms = effective_cancelled_at_ms
    return _prepare_state_update(current, cancelled)


def _fail_resume_claim_state(
    current: AgentRunState,
    *,
    claim_token: str,
    reason: str,
    failed_at_ms: int | None,
) -> AgentRunState | None:
    raw_claim = current.metadata.get("resume_claim")
    if current.status != "running" or not isinstance(raw_claim, dict):
        return None
    if raw_claim.get("claim_token") != claim_token:
        return None
    effective_failed_at_ms = failed_at_ms if failed_at_ms is not None else _now_ms()
    failed = _clone_agent_run_state(current)
    failed.status = "failed"
    failed.error = reason
    failed.updated_at_ms = effective_failed_at_ms
    failed.finished_at_ms = effective_failed_at_ms
    failed.metadata = {
        **failed.metadata,
        "resume_claim_failure": {
            "claim_token": claim_token,
            "failed_at_ms": effective_failed_at_ms,
            "reason": reason,
        },
    }
    return _prepare_state_update(current, failed)


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
    schema_version, revision = _validate_state_version(
        payload.get("schema_version", AGENT_RUN_STATE_SCHEMA_VERSION),
        payload.get("revision", 0),
    )
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
            tool_fingerprint=item.get("tool_fingerprint"),
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
        schema_version=schema_version,
        revision=revision,
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


def _claim_pending_approval(
    state: AgentRunState,
    approval_id: str,
    *,
    claim_token: str,
    claimed_at_ms: int,
) -> bool:
    if state.status != "suspended" or state.metadata.get("resume_claim") is not None:
        return False
    if not any(pending.id == approval_id for pending in state.pending_approvals):
        return False
    state.status = "running"
    state.updated_at_ms = claimed_at_ms
    state.metadata = {
        **state.metadata,
        "resume_claim": {
            "approval_id": approval_id,
            "claim_token": claim_token,
            "claimed_at_ms": claimed_at_ms,
        },
    }
    return True


class InMemoryAgentRunStore:
    def __init__(self) -> None:
        self._states: dict[str, AgentRunState] = {}
        self._lock = asyncio.Lock()

    async def load(self, run_id: str) -> AgentRunState | None:
        async with self._lock:
            state = self._states.get(run_id)
            return _clone_agent_run_state(state) if state is not None else None

    async def find_by_idempotency_key(self, idempotency_key: str) -> AgentRunState | None:
        async with self._lock:
            for state in self._states.values():
                if state.idempotency_key == idempotency_key:
                    return _clone_agent_run_state(state)
        return None

    async def find_by_parent_run_id(self, parent_run_id: str) -> list[AgentRunState]:
        async with self._lock:
            return [
                _clone_agent_run_state(state)
                for state in self._states.values()
                if state.parent_run_id == parent_run_id
            ]

    async def save(self, state: AgentRunState) -> AgentRunState:
        async with self._lock:
            current = self._states.get(state.run_id)
            persisted = _prepare_new_state(state) if current is None else _prepare_state_update(current, state)
            self._states[state.run_id] = persisted
            return _copy_persisted_revision(state, persisted)

    async def claim_idempotency_key(self, state: AgentRunState) -> AgentRunState:
        if not state.idempotency_key:
            raise ValidationError("claim_idempotency_key(...) requires state.idempotency_key.")
        async with self._lock:
            existing = next(
                (item for item in self._states.values() if item.idempotency_key == state.idempotency_key),
                None,
            )
            if existing is not None:
                return _clone_agent_run_state(existing)
            stored = _prepare_new_state(state)
            self._states[state.run_id] = stored
            return _copy_persisted_revision(state, stored)

    async def claim_pending_approval(
        self,
        run_id: str,
        approval_id: str,
        *,
        claim_token: str,
        claimed_at_ms: int,
    ) -> AgentRunState | None:
        async with self._lock:
            state = self._states.get(run_id)
            if state is None:
                return None
            claimed = _clone_agent_run_state(state)
            if not _claim_pending_approval(
                claimed,
                approval_id,
                claim_token=claim_token,
                claimed_at_ms=claimed_at_ms,
            ):
                return None
            persisted = _prepare_state_update(state, claimed)
            self._states[run_id] = persisted
            return _clone_agent_run_state(persisted)

    async def cancel_run(
        self,
        run_id: str,
        *,
        reason: str | None = None,
        cancelled_at_ms: int | None = None,
    ) -> AgentRunState | None:
        async with self._lock:
            current = self._states.get(run_id)
            if current is None:
                return None
            cancelled = _cancel_state(current, reason=reason, cancelled_at_ms=cancelled_at_ms)
            if cancelled.revision != current.revision:
                self._states[run_id] = cancelled
            return _clone_agent_run_state(cancelled)

    async def fail_resume_claim(
        self,
        run_id: str,
        *,
        claim_token: str,
        reason: str,
        failed_at_ms: int | None = None,
    ) -> AgentRunState | None:
        async with self._lock:
            current = self._states.get(run_id)
            if current is None:
                return None
            failed = _fail_resume_claim_state(
                current,
                claim_token=claim_token,
                reason=reason,
                failed_at_ms=failed_at_ms,
            )
            if failed is None:
                return None
            self._states[run_id] = failed
            return _clone_agent_run_state(failed)


def _state_from_storage_row(row: Any) -> AgentRunState:
    state = agent_run_state_from_json(row[0])
    if len(row) >= 3 and (state.schema_version != int(row[1]) or state.revision != int(row[2])):
        raise ValidationError(
            f'Agent run "{state.run_id}" has inconsistent schema_version or revision columns.'
        )
    return state


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
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    revision INTEGER NOT NULL DEFAULT 0,
                    updated_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (namespace, run_id)
                )
                """
            )
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(zhivex_agent_runs)").fetchall()}
            if "schema_version" not in columns:
                connection.execute(
                    "ALTER TABLE zhivex_agent_runs ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 1"
                )
            if "revision" not in columns:
                connection.execute("ALTER TABLE zhivex_agent_runs ADD COLUMN revision INTEGER NOT NULL DEFAULT 0")
            duplicates = connection.execute(
                """
                SELECT namespace, idempotency_key
                FROM zhivex_agent_runs
                WHERE idempotency_key IS NOT NULL
                GROUP BY namespace, idempotency_key
                HAVING COUNT(*) > 1
                LIMIT 1
                """
            ).fetchone()
            if duplicates is not None:
                raise ValidationError(
                    "Cannot migrate the SQLite agent run store: duplicate idempotency keys exist "
                    f'in namespace "{duplicates[0]}". Resolve duplicates before reopening the store.'
                )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS zhivex_agent_runs_idempotency_unique
                ON zhivex_agent_runs (namespace, idempotency_key)
                WHERE idempotency_key IS NOT NULL
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS zhivex_agent_runs_parent_idx ON zhivex_agent_runs (namespace, parent_run_id)"
            )
            connection.commit()
        finally:
            connection.close()

    async def fail_resume_claim(
        self,
        run_id: str,
        *,
        claim_token: str,
        reason: str,
        failed_at_ms: int | None = None,
    ) -> AgentRunState | None:
        connection = sqlite3.connect(self._path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT state_json, schema_version, revision
                FROM zhivex_agent_runs
                WHERE namespace = ? AND run_id = ?
                """,
                (self._namespace, run_id),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            current = _state_from_storage_row(row)
            failed = _fail_resume_claim_state(
                current,
                claim_token=claim_token,
                reason=reason,
                failed_at_ms=failed_at_ms,
            )
            if failed is None:
                connection.rollback()
                return None
            cursor = connection.execute(
                """
                UPDATE zhivex_agent_runs
                SET state_json = ?, schema_version = ?, revision = ?, updated_at_ms = ?
                WHERE namespace = ? AND run_id = ? AND revision = ?
                """,
                (
                    agent_run_state_to_json(failed),
                    failed.schema_version,
                    failed.revision,
                    failed.updated_at_ms or 0,
                    self._namespace,
                    run_id,
                    current.revision,
                ),
            )
            if cursor.rowcount != 1:
                raise ValidationError(f'Agent run "{run_id}" changed while failing its resume claim.')
            connection.commit()
            return failed
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def load(self, run_id: str) -> AgentRunState | None:
        connection = sqlite3.connect(self._path)
        try:
            row = connection.execute(
                "SELECT state_json, schema_version, revision FROM zhivex_agent_runs WHERE namespace = ? AND run_id = ?",
                (self._namespace, run_id),
            ).fetchone()
        finally:
            connection.close()
        return _state_from_storage_row(row) if row else None

    async def find_by_idempotency_key(self, idempotency_key: str) -> AgentRunState | None:
        connection = sqlite3.connect(self._path)
        try:
            row = connection.execute(
                """
                SELECT state_json, schema_version, revision
                FROM zhivex_agent_runs
                WHERE namespace = ? AND idempotency_key = ?
                """,
                (self._namespace, idempotency_key),
            ).fetchone()
        finally:
            connection.close()
        return _state_from_storage_row(row) if row else None

    async def find_by_parent_run_id(self, parent_run_id: str) -> list[AgentRunState]:
        connection = sqlite3.connect(self._path)
        try:
            rows = connection.execute(
                """
                SELECT state_json, schema_version, revision
                FROM zhivex_agent_runs
                WHERE namespace = ? AND parent_run_id = ?
                ORDER BY updated_at_ms
                """,
                (self._namespace, parent_run_id),
            ).fetchall()
        finally:
            connection.close()
        return [_state_from_storage_row(row) for row in rows]

    async def save(self, state: AgentRunState) -> AgentRunState:
        connection = sqlite3.connect(self._path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT state_json, schema_version, revision
                FROM zhivex_agent_runs
                WHERE namespace = ? AND run_id = ?
                """,
                (self._namespace, state.run_id),
            ).fetchone()
            if row is None:
                persisted = _prepare_new_state(state)
                connection.execute(
                    """
                    INSERT INTO zhivex_agent_runs
                        (namespace, run_id, idempotency_key, parent_run_id, state_json,
                         schema_version, revision, updated_at_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._namespace,
                        persisted.run_id,
                        persisted.idempotency_key,
                        persisted.parent_run_id,
                        agent_run_state_to_json(persisted),
                        persisted.schema_version,
                        persisted.revision,
                        persisted.updated_at_ms or 0,
                    ),
                )
            else:
                current = _state_from_storage_row(row)
                persisted = _prepare_state_update(current, state)
                cursor = connection.execute(
                    """
                    UPDATE zhivex_agent_runs
                    SET idempotency_key = ?, parent_run_id = ?, state_json = ?,
                        schema_version = ?, revision = ?, updated_at_ms = ?
                    WHERE namespace = ? AND run_id = ? AND revision = ?
                    """,
                    (
                        persisted.idempotency_key,
                        persisted.parent_run_id,
                        agent_run_state_to_json(persisted),
                        persisted.schema_version,
                        persisted.revision,
                        persisted.updated_at_ms or 0,
                        self._namespace,
                        persisted.run_id,
                        current.revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValidationError(f'Agent run "{state.run_id}" changed during save.')
            connection.commit()
            return _copy_persisted_revision(state, persisted)
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def claim_idempotency_key(self, state: AgentRunState) -> AgentRunState:
        if not state.idempotency_key:
            raise ValidationError("claim_idempotency_key(...) requires state.idempotency_key.")
        connection = sqlite3.connect(self._path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT state_json, schema_version, revision
                FROM zhivex_agent_runs
                WHERE namespace = ? AND idempotency_key = ?
                """,
                (self._namespace, state.idempotency_key),
            ).fetchone()
            if row is not None:
                connection.commit()
                return _state_from_storage_row(row)
            persisted = _prepare_new_state(state)
            connection.execute(
                """
                INSERT INTO zhivex_agent_runs
                    (namespace, run_id, idempotency_key, parent_run_id, state_json,
                     schema_version, revision, updated_at_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._namespace,
                    persisted.run_id,
                    persisted.idempotency_key,
                    persisted.parent_run_id,
                    agent_run_state_to_json(persisted),
                    persisted.schema_version,
                    persisted.revision,
                    persisted.updated_at_ms or 0,
                ),
            )
            connection.commit()
            return _copy_persisted_revision(state, persisted)
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def claim_pending_approval(
        self,
        run_id: str,
        approval_id: str,
        *,
        claim_token: str,
        claimed_at_ms: int,
    ) -> AgentRunState | None:
        connection = sqlite3.connect(self._path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT state_json, schema_version, revision
                FROM zhivex_agent_runs
                WHERE namespace = ? AND run_id = ?
                """,
                (self._namespace, run_id),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            current = _state_from_storage_row(row)
            claimed = _clone_agent_run_state(current)
            if not _claim_pending_approval(
                claimed,
                approval_id,
                claim_token=claim_token,
                claimed_at_ms=claimed_at_ms,
            ):
                connection.rollback()
                return None
            persisted = _prepare_state_update(current, claimed)
            cursor = connection.execute(
                """
                UPDATE zhivex_agent_runs
                SET state_json = ?, schema_version = ?, revision = ?, updated_at_ms = ?
                WHERE namespace = ? AND run_id = ? AND revision = ?
                """,
                (
                    agent_run_state_to_json(persisted),
                    persisted.schema_version,
                    persisted.revision,
                    persisted.updated_at_ms or 0,
                    self._namespace,
                    run_id,
                    current.revision,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
            return persisted
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def cancel_run(
        self,
        run_id: str,
        *,
        reason: str | None = None,
        cancelled_at_ms: int | None = None,
    ) -> AgentRunState | None:
        connection = sqlite3.connect(self._path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT state_json, schema_version, revision
                FROM zhivex_agent_runs
                WHERE namespace = ? AND run_id = ?
                """,
                (self._namespace, run_id),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None
            current = _state_from_storage_row(row)
            cancelled = _cancel_state(current, reason=reason, cancelled_at_ms=cancelled_at_ms)
            if cancelled.revision == current.revision:
                connection.commit()
                return cancelled
            cursor = connection.execute(
                """
                UPDATE zhivex_agent_runs
                SET state_json = ?, schema_version = ?, revision = ?, updated_at_ms = ?
                WHERE namespace = ? AND run_id = ? AND revision = ?
                """,
                (
                    agent_run_state_to_json(cancelled),
                    cancelled.schema_version,
                    cancelled.revision,
                    cancelled.updated_at_ms or 0,
                    self._namespace,
                    run_id,
                    current.revision,
                ),
            )
            if cursor.rowcount != 1:
                raise ValidationError(f'Agent run "{run_id}" changed during cancellation.')
            connection.commit()
            return cancelled
        except BaseException:
            connection.rollback()
            raise
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
        try:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    f"zhivex-agent-run-schema:{self._table}",
                )
                await connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        run_id TEXT PRIMARY KEY,
                        idempotency_key TEXT,
                        parent_run_id TEXT,
                        state_json JSONB NOT NULL,
                        schema_version INTEGER NOT NULL DEFAULT 1,
                        revision BIGINT NOT NULL DEFAULT 0,
                        updated_at_ms BIGINT NOT NULL
                    )
                    """
                )
                await connection.execute(
                    f"ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS schema_version INTEGER NOT NULL DEFAULT 1"
                )
                await connection.execute(
                    f"ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS revision BIGINT NOT NULL DEFAULT 0"
                )
                duplicate = await connection.fetchrow(
                    f"""
                    SELECT idempotency_key
                    FROM {self._table}
                    WHERE idempotency_key IS NOT NULL
                    GROUP BY idempotency_key
                    HAVING COUNT(*) > 1
                    LIMIT 1
                    """
                )
                if duplicate is not None:
                    raise ValidationError(
                        "Cannot migrate the Postgres agent run store: duplicate idempotency keys exist. "
                        "Resolve duplicates before reconnecting."
                    )
                await connection.execute(
                    f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS {self._table}_idempotency_unique
                    ON {self._table} (idempotency_key)
                    WHERE idempotency_key IS NOT NULL
                    """
                )
                await connection.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._table}_parent_idx ON {self._table} (parent_run_id)"
                )
        except BaseException:
            await connection.close()
            raise
        return connection

    @staticmethod
    def _state_from_row(row: Any) -> AgentRunState:
        payload = row["state_json"]
        state = deserialize_agent_run_state(payload if isinstance(payload, dict) else json.loads(payload))
        try:
            schema_version = int(row["schema_version"])
            revision = int(row["revision"])
        except (KeyError, TypeError):
            return state
        if state.schema_version != schema_version or state.revision != revision:
            raise ValidationError(
                f'Agent run "{state.run_id}" has inconsistent schema_version or revision columns.'
            )
        return state

    async def load(self, run_id: str) -> AgentRunState | None:
        connection = await self._connect()
        try:
            row = await connection.fetchrow(
                f"SELECT state_json, schema_version, revision FROM {self._table} WHERE run_id = $1",
                run_id,
            )
            if row is None:
                return None
            return self._state_from_row(row)
        finally:
            await connection.close()

    async def find_by_idempotency_key(self, idempotency_key: str) -> AgentRunState | None:
        connection = await self._connect()
        try:
            row = await connection.fetchrow(
                f"SELECT state_json, schema_version, revision FROM {self._table} WHERE idempotency_key = $1",
                idempotency_key,
            )
            if row is None:
                return None
            return self._state_from_row(row)
        finally:
            await connection.close()

    async def find_by_parent_run_id(self, parent_run_id: str) -> list[AgentRunState]:
        connection = await self._connect()
        try:
            rows = await connection.fetch(
                f"""
                SELECT state_json, schema_version, revision
                FROM {self._table}
                WHERE parent_run_id = $1
                ORDER BY updated_at_ms
                """,
                parent_run_id,
            )
            return [self._state_from_row(row) for row in rows]
        finally:
            await connection.close()

    async def save(self, state: AgentRunState) -> AgentRunState:
        connection = await self._connect()
        try:
            async with connection.transaction():
                row = await connection.fetchrow(
                    f"""
                    SELECT state_json, schema_version, revision
                    FROM {self._table}
                    WHERE run_id = $1
                    FOR UPDATE
                    """,
                    state.run_id,
                )
                if row is None:
                    persisted = _prepare_new_state(state)
                    await connection.execute(
                        f"""
                        INSERT INTO {self._table}
                            (run_id, idempotency_key, parent_run_id, state_json,
                             schema_version, revision, updated_at_ms)
                        VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
                        """,
                        persisted.run_id,
                        persisted.idempotency_key,
                        persisted.parent_run_id,
                        agent_run_state_to_json(persisted),
                        persisted.schema_version,
                        persisted.revision,
                        persisted.updated_at_ms or 0,
                    )
                else:
                    current = self._state_from_row(row)
                    persisted = _prepare_state_update(current, state)
                    status = await connection.execute(
                        f"""
                        UPDATE {self._table}
                        SET idempotency_key = $1, parent_run_id = $2, state_json = $3::jsonb,
                            schema_version = $4, revision = $5, updated_at_ms = $6
                        WHERE run_id = $7 AND revision = $8
                        """,
                        persisted.idempotency_key,
                        persisted.parent_run_id,
                        agent_run_state_to_json(persisted),
                        persisted.schema_version,
                        persisted.revision,
                        persisted.updated_at_ms or 0,
                        persisted.run_id,
                        current.revision,
                    )
                    if status != "UPDATE 1":
                        raise ValidationError(f'Agent run "{state.run_id}" changed during save.')
            return _copy_persisted_revision(state, persisted)
        finally:
            await connection.close()

    async def claim_idempotency_key(self, state: AgentRunState) -> AgentRunState:
        if not state.idempotency_key:
            raise ValidationError("claim_idempotency_key(...) requires state.idempotency_key.")
        connection = await self._connect()
        try:
            async with connection.transaction():
                # Transaction-scoped advisory locks avoid a schema migration
                # while making the SDK claim path atomic across workers.
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    state.idempotency_key,
                )
                row = await connection.fetchrow(
                    f"""
                    SELECT state_json, schema_version, revision
                    FROM {self._table}
                    WHERE idempotency_key = $1
                    LIMIT 1
                    """,
                    state.idempotency_key,
                )
                if row is not None:
                    return self._state_from_row(row)
                persisted = _prepare_new_state(state)
                await connection.execute(
                    f"""
                    INSERT INTO {self._table}
                        (run_id, idempotency_key, parent_run_id, state_json,
                         schema_version, revision, updated_at_ms)
                    VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
                    """,
                    persisted.run_id,
                    persisted.idempotency_key,
                    persisted.parent_run_id,
                    agent_run_state_to_json(persisted),
                    persisted.schema_version,
                    persisted.revision,
                    persisted.updated_at_ms or 0,
                )
            return _copy_persisted_revision(state, persisted)
        finally:
            await connection.close()

    async def claim_pending_approval(
        self,
        run_id: str,
        approval_id: str,
        *,
        claim_token: str,
        claimed_at_ms: int,
    ) -> AgentRunState | None:
        connection = await self._connect()
        try:
            async with connection.transaction():
                row = await connection.fetchrow(
                    f"""
                    SELECT state_json, schema_version, revision
                    FROM {self._table}
                    WHERE run_id = $1
                    FOR UPDATE
                    """,
                    run_id,
                )
                if row is None:
                    return None
                current = self._state_from_row(row)
                claimed = _clone_agent_run_state(current)
                if not _claim_pending_approval(
                    claimed,
                    approval_id,
                    claim_token=claim_token,
                    claimed_at_ms=claimed_at_ms,
                ):
                    return None
                persisted = _prepare_state_update(current, claimed)
                status = await connection.execute(
                    f"""
                    UPDATE {self._table}
                    SET state_json = $1::jsonb, schema_version = $2, revision = $3, updated_at_ms = $4
                    WHERE run_id = $5 AND revision = $6
                    """,
                    agent_run_state_to_json(persisted),
                    persisted.schema_version,
                    persisted.revision,
                    persisted.updated_at_ms or 0,
                    run_id,
                    current.revision,
                )
                if status != "UPDATE 1":
                    return None
                return persisted
        finally:
            await connection.close()

    async def cancel_run(
        self,
        run_id: str,
        *,
        reason: str | None = None,
        cancelled_at_ms: int | None = None,
    ) -> AgentRunState | None:
        connection = await self._connect()
        try:
            async with connection.transaction():
                row = await connection.fetchrow(
                    f"""
                    SELECT state_json, schema_version, revision
                    FROM {self._table}
                    WHERE run_id = $1
                    FOR UPDATE
                    """,
                    run_id,
                )
                if row is None:
                    return None
                current = self._state_from_row(row)
                cancelled = _cancel_state(current, reason=reason, cancelled_at_ms=cancelled_at_ms)
                if cancelled.revision == current.revision:
                    return cancelled
                status = await connection.execute(
                    f"""
                    UPDATE {self._table}
                    SET state_json = $1::jsonb, schema_version = $2, revision = $3, updated_at_ms = $4
                    WHERE run_id = $5 AND revision = $6
                    """,
                    agent_run_state_to_json(cancelled),
                    cancelled.schema_version,
                    cancelled.revision,
                    cancelled.updated_at_ms or 0,
                    run_id,
                    current.revision,
                )
                if status != "UPDATE 1":
                    raise ValidationError(f'Agent run "{run_id}" changed during cancellation.')
                return cancelled
        finally:
            await connection.close()

    async def fail_resume_claim(
        self,
        run_id: str,
        *,
        claim_token: str,
        reason: str,
        failed_at_ms: int | None = None,
    ) -> AgentRunState | None:
        connection = await self._connect()
        try:
            async with connection.transaction():
                row = await connection.fetchrow(
                    f"""
                    SELECT state_json, schema_version, revision
                    FROM {self._table}
                    WHERE run_id = $1
                    FOR UPDATE
                    """,
                    run_id,
                )
                if row is None:
                    return None
                current = self._state_from_row(row)
                failed = _fail_resume_claim_state(
                    current,
                    claim_token=claim_token,
                    reason=reason,
                    failed_at_ms=failed_at_ms,
                )
                if failed is None:
                    return None
                status = await connection.execute(
                    f"""
                    UPDATE {self._table}
                    SET state_json = $1::jsonb, schema_version = $2, revision = $3, updated_at_ms = $4
                    WHERE run_id = $5 AND revision = $6
                    """,
                    agent_run_state_to_json(failed),
                    failed.schema_version,
                    failed.revision,
                    failed.updated_at_ms or 0,
                    run_id,
                    current.revision,
                )
                if status != "UPDATE 1":
                    raise ValidationError(f'Agent run "{run_id}" changed while failing its resume claim.')
                return failed
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


async def fail_agent_run_resume_claim(
    store: AgentRunStore,
    run_id: str,
    *,
    claim_token: str,
    reason: str,
    now_ms: int | None = None,
) -> AgentRunState | None:
    """Atomically fail one known approval-resume claim without retrying its tool."""

    fail_resume_claim = getattr(store, "fail_resume_claim", None)
    if not callable(fail_resume_claim):
        raise ValidationError(
            "Resume-claim reconciliation requires an AgentRunStore with fail_resume_claim(...). "
            "Use a built-in run store or implement the atomic reconciliation contract."
        )
    return await fail_resume_claim(
        run_id,
        claim_token=claim_token,
        reason=reason,
        failed_at_ms=now_ms,
    )


async def cancel_agent_run(
    store: AgentRunStore,
    run_id: str,
    *,
    reason: str | None = None,
    now_ms: int | None = None,
    cancellation_token: Any = None,
) -> AgentRunState | None:
    cancel_run = getattr(store, "cancel_run", None)
    if not callable(cancel_run):
        raise ValidationError(
            "Atomic cancellation requires an AgentRunStore with cancel_run(...). "
            "Use a built-in run store or implement the atomic cancellation contract."
        )
    cancelled = await cancel_run(run_id, reason=reason, cancelled_at_ms=now_ms)
    if cancelled is not None and cancelled.status == "cancelled" and cancellation_token is not None:
        cancel = getattr(cancellation_token, "cancel", None)
        if not callable(cancel):
            raise ValidationError("cancellation_token must expose cancel(reason).")
        cancel(reason)
    return cancelled


async def cancel_agent_run_tree(
    store: AgentRunStore,
    run_id: str,
    *,
    reason: str | None = None,
    now_ms: int | None = None,
    cancellation_token: Any = None,
) -> AgentRunTreeCancellationResult:
    root = await cancel_agent_run(
        store,
        run_id,
        reason=reason,
        now_ms=now_ms,
        cancellation_token=cancellation_token,
    )
    if root is None:
        return AgentRunTreeCancellationResult(root=None)
    cancelled = [root] if root.status == "cancelled" else []
    visited = {run_id}

    async def collect(parent_run_id: str) -> None:
        children = await store.find_by_parent_run_id(parent_run_id)
        for child in children:
            if child.run_id in visited:
                continue
            visited.add(child.run_id)
            cancelled_child = await cancel_agent_run(store, child.run_id, reason=reason, now_ms=now_ms)
            if cancelled_child is not None:
                if cancelled_child.status == "cancelled":
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
