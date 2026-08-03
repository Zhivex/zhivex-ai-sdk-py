from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from .errors import ValidationError
from .types import JsonValue


WORKFLOW_CHECKPOINT_SCHEMA_VERSION = 1

WorkflowNodeStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "suspended",
    "cancelled",
    "skipped",
]
WorkflowCheckpointStatus = Literal["running", "completed", "failed", "suspended", "cancelled"]
WorkflowInterruptPhase = Literal["before", "after"]

_WORKFLOW_NODE_STATUSES: frozenset[str] = frozenset(
    {"pending", "running", "completed", "failed", "suspended", "cancelled", "skipped"}
)
_WORKFLOW_CHECKPOINT_STATUSES: frozenset[str] = frozenset(
    {"running", "completed", "failed", "suspended", "cancelled"}
)
_TERMINAL_WORKFLOW_CHECKPOINT_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})
_WORKFLOW_INTERRUPT_PHASES: frozenset[str] = frozenset({"before", "after"})
_POSTGRES_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _validate_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"Workflow checkpoint {field_name} must be a non-empty string.")
    return value


def _validate_non_negative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"Workflow checkpoint {field_name} must be a non-negative integer.")
    return value


def _validate_optional_timestamp(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _validate_non_negative_integer(value, field_name)


def _validate_postgres_table_prefix(table_prefix: str) -> str:
    if not _POSTGRES_IDENTIFIER_RE.match(table_prefix):
        raise ValidationError(
            'The "table_prefix" field must match the SQL identifier pattern [A-Za-z_][A-Za-z0-9_]*.'
        )
    return table_prefix


def _validate_lease_request(run_id: str, owner_id: str, ttl_ms: int) -> tuple[str, str, int]:
    resolved_run_id = _validate_non_empty(run_id, "lease run_id")
    resolved_owner_id = _validate_non_empty(owner_id, "lease owner_id")
    if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms <= 0:
        raise ValidationError("Workflow execution lease ttl_ms must be a positive integer.")
    return resolved_run_id, resolved_owner_id, ttl_ms


def _validate_lease_token(run_id: str, token: str, ttl_ms: int | None = None) -> tuple[str, str]:
    resolved_run_id = _validate_non_empty(run_id, "lease run_id")
    resolved_token = _validate_non_empty(token, "lease token")
    if ttl_ms is not None and (isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms <= 0):
        raise ValidationError("Workflow execution lease ttl_ms must be a positive integer.")
    return resolved_run_id, resolved_token


def _validate_json_value(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f'Workflow checkpoint field "{path}" must not contain NaN or infinity.')
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f'Workflow checkpoint field "{path}" must contain string object keys.')
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ValidationError(
        f'Workflow checkpoint field "{path}" contains non-JSON value {type(value).__name__}.'
    )


@dataclass(slots=True)
class WorkflowNodeCheckpoint:
    node_name: str
    status: WorkflowNodeStatus = "pending"
    attempt: int = 0
    idempotency_key: str | None = None
    child_run_id: str | None = None
    output: JsonValue | None = None
    error: str | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    suspension: dict[str, JsonValue] | None = None


@dataclass(slots=True)
class WorkflowInterrupt:
    interrupt_id: str
    node_name: str
    reason: str | None = None
    payload: JsonValue | None = None
    created_at_ms: int | None = None
    phase: WorkflowInterruptPhase = "before"
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowTransition:
    type: str
    at_ms: int
    node_name: str | None = None
    from_status: str | None = None
    to_status: str | None = None
    detail: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowCheckpoint:
    checkpoint_id: str
    run_id: str
    workflow_name: str
    definition_version: str
    definition_digest: str
    sequence: int = 0
    schema_version: int = WORKFLOW_CHECKPOINT_SCHEMA_VERSION
    status: WorkflowCheckpointStatus = "running"
    session_id: str | None = None
    parent_run_id: str | None = None
    idempotency_key: str | None = None
    state: dict[str, JsonValue] = field(default_factory=dict)
    nodes: dict[str, WorkflowNodeCheckpoint] = field(default_factory=dict)
    edge_decisions: dict[str, bool] = field(default_factory=dict)
    ready_nodes: list[str] = field(default_factory=list)
    pending_interrupt: WorkflowInterrupt | None = None
    transition: WorkflowTransition | None = None
    forked_from_run_id: str | None = None
    forked_from_checkpoint_id: str | None = None
    resume_values: dict[str, JsonValue] = field(default_factory=dict)
    created_at_ms: int | None = None
    updated_at_ms: int | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


class WorkflowCheckpointStore(Protocol):
    async def append(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        expected_sequence: int | None = None,
    ) -> WorkflowCheckpoint: ...

    async def load_latest(self, run_id: str) -> WorkflowCheckpoint | None: ...

    async def load_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint | None: ...

    async def find_by_idempotency_key(self, idempotency_key: str) -> WorkflowCheckpoint | None: ...

    async def list_checkpoints(self, run_id: str) -> list[WorkflowCheckpoint]: ...


@dataclass(frozen=True, slots=True)
class WorkflowExecutionLease:
    run_id: str
    owner_id: str
    token: str
    fencing_token: int
    acquired_at_ms: int
    renewed_at_ms: int
    expires_at_ms: int

    def is_expired(self, *, now_ms: int | None = None) -> bool:
        effective_now = _now_ms() if now_ms is None else now_ms
        return self.expires_at_ms <= effective_now


class WorkflowLeaseManager(Protocol):
    async def acquire(
        self,
        run_id: str,
        *,
        owner_id: str,
        ttl_ms: int,
        now_ms: int | None = None,
    ) -> WorkflowExecutionLease | None: ...

    async def renew(
        self,
        run_id: str,
        *,
        token: str,
        ttl_ms: int,
        now_ms: int | None = None,
    ) -> WorkflowExecutionLease | None: ...

    async def release(self, run_id: str, *, token: str) -> bool: ...

    async def get(self, run_id: str) -> WorkflowExecutionLease | None: ...


def _validate_node_checkpoint(node: WorkflowNodeCheckpoint) -> None:
    _validate_non_empty(node.node_name, "node_name")
    if node.status not in _WORKFLOW_NODE_STATUSES:
        raise ValidationError(f'Unsupported workflow node status "{node.status}".')
    _validate_non_negative_integer(node.attempt, "node attempt")
    _validate_optional_timestamp(node.started_at_ms, "node started_at_ms")
    _validate_optional_timestamp(node.finished_at_ms, "node finished_at_ms")
    _validate_json_value(node.output, path=f"nodes.{node.node_name}.output")
    _validate_json_value(node.metadata, path=f"nodes.{node.node_name}.metadata")
    if node.suspension is not None:
        _validate_json_value(node.suspension, path=f"nodes.{node.node_name}.suspension")


def _validate_interrupt(interrupt: WorkflowInterrupt) -> None:
    _validate_non_empty(interrupt.interrupt_id, "interrupt_id")
    _validate_non_empty(interrupt.node_name, "interrupt node_name")
    if interrupt.phase not in _WORKFLOW_INTERRUPT_PHASES:
        raise ValidationError(f'Unsupported workflow interrupt phase "{interrupt.phase}".')
    _validate_optional_timestamp(interrupt.created_at_ms, "interrupt created_at_ms")
    _validate_json_value(interrupt.payload, path="pending_interrupt.payload")
    _validate_json_value(interrupt.metadata, path="pending_interrupt.metadata")


def _validate_transition(transition: WorkflowTransition) -> None:
    _validate_non_empty(transition.type, "transition type")
    _validate_non_negative_integer(transition.at_ms, "transition at_ms")
    _validate_json_value(transition.detail, path="transition.detail")


def _validate_checkpoint(checkpoint: WorkflowCheckpoint) -> None:
    _validate_non_empty(checkpoint.checkpoint_id, "checkpoint_id")
    _validate_non_empty(checkpoint.run_id, "run_id")
    _validate_non_empty(checkpoint.workflow_name, "workflow_name")
    _validate_non_empty(checkpoint.definition_version, "definition_version")
    _validate_non_empty(checkpoint.definition_digest, "definition_digest")
    _validate_non_negative_integer(checkpoint.sequence, "sequence")
    schema_version = _validate_non_negative_integer(checkpoint.schema_version, "schema_version")
    if schema_version < 1:
        raise ValidationError("Workflow checkpoint schema_version must be a positive integer.")
    if schema_version > WORKFLOW_CHECKPOINT_SCHEMA_VERSION:
        raise ValidationError(
            "Workflow checkpoint uses unsupported future schema_version "
            f"{schema_version}; this SDK supports up to {WORKFLOW_CHECKPOINT_SCHEMA_VERSION}."
        )
    if checkpoint.status not in _WORKFLOW_CHECKPOINT_STATUSES:
        raise ValidationError(f'Unsupported workflow checkpoint status "{checkpoint.status}".')
    for field_name, value in (
        ("session_id", checkpoint.session_id),
        ("parent_run_id", checkpoint.parent_run_id),
        ("idempotency_key", checkpoint.idempotency_key),
        ("forked_from_run_id", checkpoint.forked_from_run_id),
        ("forked_from_checkpoint_id", checkpoint.forked_from_checkpoint_id),
    ):
        if value is not None:
            _validate_non_empty(value, field_name)
    if (checkpoint.forked_from_run_id is None) != (checkpoint.forked_from_checkpoint_id is None):
        raise ValidationError(
            "Workflow checkpoint fork lineage requires both forked_from_run_id and forked_from_checkpoint_id."
        )
    _validate_optional_timestamp(checkpoint.created_at_ms, "created_at_ms")
    _validate_optional_timestamp(checkpoint.updated_at_ms, "updated_at_ms")
    for node_name, node in checkpoint.nodes.items():
        _validate_non_empty(node_name, "nodes key")
        _validate_node_checkpoint(node)
        if node.node_name != node_name:
            raise ValidationError(
                f'Workflow node checkpoint key "{node_name}" does not match node_name "{node.node_name}".'
            )
    for edge_name, decision in checkpoint.edge_decisions.items():
        _validate_non_empty(edge_name, "edge decision key")
        if not isinstance(decision, bool):
            raise ValidationError(f'Workflow edge decision "{edge_name}" must be a boolean.')
    if len(checkpoint.ready_nodes) != len(set(checkpoint.ready_nodes)):
        raise ValidationError("Workflow checkpoint ready_nodes must not contain duplicates.")
    if checkpoint.pending_interrupt is not None:
        _validate_interrupt(checkpoint.pending_interrupt)
    if checkpoint.transition is not None:
        _validate_transition(checkpoint.transition)
    _validate_json_value(checkpoint.state, path="state")
    _validate_json_value(checkpoint.resume_values, path="resume_values")
    _validate_json_value(checkpoint.metadata, path="metadata")


def _node_to_payload(node: WorkflowNodeCheckpoint) -> dict[str, Any]:
    return {
        "node_name": node.node_name,
        "status": node.status,
        "attempt": node.attempt,
        "idempotency_key": node.idempotency_key,
        "child_run_id": node.child_run_id,
        "output": node.output,
        "error": node.error,
        "started_at_ms": node.started_at_ms,
        "finished_at_ms": node.finished_at_ms,
        "metadata": dict(node.metadata),
        "suspension": dict(node.suspension) if node.suspension is not None else None,
    }


def _interrupt_to_payload(interrupt: WorkflowInterrupt) -> dict[str, Any]:
    return {
        "interrupt_id": interrupt.interrupt_id,
        "node_name": interrupt.node_name,
        "reason": interrupt.reason,
        "payload": interrupt.payload,
        "created_at_ms": interrupt.created_at_ms,
        "phase": interrupt.phase,
        "metadata": dict(interrupt.metadata),
    }


def _transition_to_payload(transition: WorkflowTransition) -> dict[str, Any]:
    return {
        "type": transition.type,
        "at_ms": transition.at_ms,
        "node_name": transition.node_name,
        "from_status": transition.from_status,
        "to_status": transition.to_status,
        "detail": dict(transition.detail),
    }


def serialize_workflow_checkpoint(checkpoint: WorkflowCheckpoint) -> dict[str, Any]:
    _validate_checkpoint(checkpoint)
    return {
        "checkpoint_id": checkpoint.checkpoint_id,
        "run_id": checkpoint.run_id,
        "workflow_name": checkpoint.workflow_name,
        "definition_version": checkpoint.definition_version,
        "definition_digest": checkpoint.definition_digest,
        "sequence": checkpoint.sequence,
        "schema_version": checkpoint.schema_version,
        "status": checkpoint.status,
        "session_id": checkpoint.session_id,
        "parent_run_id": checkpoint.parent_run_id,
        "idempotency_key": checkpoint.idempotency_key,
        "state": dict(checkpoint.state),
        "nodes": {name: _node_to_payload(node) for name, node in checkpoint.nodes.items()},
        "edge_decisions": dict(checkpoint.edge_decisions),
        "ready_nodes": list(checkpoint.ready_nodes),
        "pending_interrupt": (
            _interrupt_to_payload(checkpoint.pending_interrupt)
            if checkpoint.pending_interrupt is not None
            else None
        ),
        "transition": _transition_to_payload(checkpoint.transition) if checkpoint.transition is not None else None,
        "forked_from_run_id": checkpoint.forked_from_run_id,
        "forked_from_checkpoint_id": checkpoint.forked_from_checkpoint_id,
        "resume_values": dict(checkpoint.resume_values),
        "created_at_ms": checkpoint.created_at_ms,
        "updated_at_ms": checkpoint.updated_at_ms,
        "metadata": dict(checkpoint.metadata),
    }


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _node_from_payload(payload: Any, *, fallback_name: str) -> WorkflowNodeCheckpoint:
    item = _mapping(payload)
    return WorkflowNodeCheckpoint(
        node_name=str(item.get("node_name") or fallback_name),
        status=item.get("status", "pending"),
        attempt=item.get("attempt", 0),
        idempotency_key=item.get("idempotency_key"),
        child_run_id=item.get("child_run_id"),
        output=item.get("output"),
        error=item.get("error"),
        started_at_ms=item.get("started_at_ms"),
        finished_at_ms=item.get("finished_at_ms"),
        metadata=_mapping(item.get("metadata")),
        suspension=_mapping(item.get("suspension")) if item.get("suspension") is not None else None,
    )


def _interrupt_from_payload(payload: Any) -> WorkflowInterrupt | None:
    if not isinstance(payload, dict):
        return None
    return WorkflowInterrupt(
        interrupt_id=str(payload.get("interrupt_id", "")),
        node_name=str(payload.get("node_name", "")),
        reason=payload.get("reason"),
        payload=payload.get("payload"),
        created_at_ms=payload.get("created_at_ms"),
        phase=payload.get("phase", "before"),
        metadata=_mapping(payload.get("metadata")),
    )


def _transition_from_payload(payload: Any) -> WorkflowTransition | None:
    if not isinstance(payload, dict):
        return None
    return WorkflowTransition(
        type=str(payload.get("type", "")),
        at_ms=payload.get("at_ms", 0),
        node_name=payload.get("node_name"),
        from_status=payload.get("from_status"),
        to_status=payload.get("to_status"),
        detail=_mapping(payload.get("detail")),
    )


def deserialize_workflow_checkpoint(payload: dict[str, Any]) -> WorkflowCheckpoint:
    nodes_payload = _mapping(payload.get("nodes"))
    checkpoint = WorkflowCheckpoint(
        checkpoint_id=str(payload.get("checkpoint_id", "")),
        run_id=str(payload.get("run_id", "")),
        workflow_name=str(payload.get("workflow_name", "")),
        definition_version=str(payload.get("definition_version", "")),
        definition_digest=str(payload.get("definition_digest", "")),
        sequence=payload.get("sequence", 0),
        schema_version=payload.get("schema_version", WORKFLOW_CHECKPOINT_SCHEMA_VERSION),
        status=payload.get("status", "running"),
        session_id=payload.get("session_id"),
        parent_run_id=payload.get("parent_run_id"),
        idempotency_key=payload.get("idempotency_key"),
        state=_mapping(payload.get("state")),
        nodes={name: _node_from_payload(node, fallback_name=name) for name, node in nodes_payload.items()},
        edge_decisions={str(name): decision for name, decision in _mapping(payload.get("edge_decisions")).items()},
        ready_nodes=[str(name) for name in payload.get("ready_nodes") or []],
        pending_interrupt=_interrupt_from_payload(payload.get("pending_interrupt")),
        transition=_transition_from_payload(payload.get("transition")),
        forked_from_run_id=payload.get("forked_from_run_id"),
        forked_from_checkpoint_id=payload.get("forked_from_checkpoint_id"),
        resume_values=_mapping(payload.get("resume_values")),
        created_at_ms=payload.get("created_at_ms"),
        updated_at_ms=payload.get("updated_at_ms"),
        metadata=_mapping(payload.get("metadata")),
    )
    _validate_checkpoint(checkpoint)
    return checkpoint


def workflow_checkpoint_to_json(checkpoint: WorkflowCheckpoint) -> str:
    return json.dumps(
        serialize_workflow_checkpoint(checkpoint),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def workflow_checkpoint_from_json(value: str) -> WorkflowCheckpoint:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValidationError("Serialized workflow checkpoint must be valid JSON.") from error
    if not isinstance(payload, dict):
        raise ValidationError("Serialized workflow checkpoint must be a JSON object.")
    return deserialize_workflow_checkpoint(payload)


def _clone_checkpoint(checkpoint: WorkflowCheckpoint) -> WorkflowCheckpoint:
    return workflow_checkpoint_from_json(workflow_checkpoint_to_json(checkpoint))


def _prepare_append(
    checkpoint: WorkflowCheckpoint,
    *,
    current: WorkflowCheckpoint | None,
    expected_sequence: int | None,
) -> WorkflowCheckpoint:
    candidate = _clone_checkpoint(checkpoint)
    if current is None:
        if expected_sequence is not None:
            raise ValidationError(
                f'Workflow run "{candidate.run_id}" does not exist; expected_sequence must be None for its first checkpoint.'
            )
        if candidate.sequence != 0:
            raise ValidationError(f'Workflow run "{candidate.run_id}" must start at checkpoint sequence 0.')
    else:
        if expected_sequence is None:
            raise ValidationError(
                f'Workflow run "{candidate.run_id}" already exists; expected_sequence is required for append.'
            )
        _validate_non_negative_integer(expected_sequence, "expected_sequence")
        if expected_sequence != current.sequence:
            raise ValidationError(
                f'Workflow run "{candidate.run_id}" sequence conflict: expected {expected_sequence}, '
                f"stored sequence is {current.sequence}. Reload the latest checkpoint before retrying."
            )
        if current.status in _TERMINAL_WORKFLOW_CHECKPOINT_STATUSES:
            raise ValidationError(
                f'Workflow run "{candidate.run_id}" is terminal with status "{current.status}" and cannot be appended.'
            )
        if candidate.sequence != current.sequence + 1:
            raise ValidationError(
                f'Workflow run "{candidate.run_id}" next checkpoint sequence must be {current.sequence + 1}, '
                f"got {candidate.sequence}."
            )
        if candidate.workflow_name != current.workflow_name:
            raise ValidationError(f'Workflow run "{candidate.run_id}" cannot change workflow_name during append.')
        if candidate.definition_version != current.definition_version:
            raise ValidationError(f'Workflow run "{candidate.run_id}" cannot change definition_version during append.')
        if candidate.definition_digest != current.definition_digest:
            raise ValidationError(f'Workflow run "{candidate.run_id}" cannot change definition_digest during append.')
        if candidate.idempotency_key != current.idempotency_key:
            raise ValidationError(f'Workflow run "{candidate.run_id}" cannot change idempotency_key during append.')
    now_ms = _now_ms()
    if candidate.created_at_ms is None:
        candidate.created_at_ms = now_ms
    if candidate.updated_at_ms is None:
        candidate.updated_at_ms = now_ms
    _validate_checkpoint(candidate)
    return candidate


class InMemoryWorkflowCheckpointStore:
    def __init__(self) -> None:
        self._checkpoints: dict[str, list[WorkflowCheckpoint]] = {}
        self._checkpoint_ids: dict[str, WorkflowCheckpoint] = {}
        self._idempotency_runs: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def append(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        expected_sequence: int | None = None,
    ) -> WorkflowCheckpoint:
        async with self._lock:
            items = self._checkpoints.get(checkpoint.run_id, [])
            current = items[-1] if items else None
            candidate = _prepare_append(
                checkpoint,
                current=current,
                expected_sequence=expected_sequence,
            )
            if candidate.checkpoint_id in self._checkpoint_ids:
                raise ValidationError(f'Workflow checkpoint "{candidate.checkpoint_id}" already exists.')
            if candidate.idempotency_key is not None:
                claimed_run_id = self._idempotency_runs.get(candidate.idempotency_key)
                if claimed_run_id is not None and claimed_run_id != candidate.run_id:
                    raise ValidationError(
                        f'Workflow idempotency key "{candidate.idempotency_key}" is already claimed '
                        f'by run "{claimed_run_id}".'
                    )
                self._idempotency_runs[candidate.idempotency_key] = candidate.run_id
            stored = _clone_checkpoint(candidate)
            self._checkpoints.setdefault(candidate.run_id, []).append(stored)
            self._checkpoint_ids[candidate.checkpoint_id] = stored
            return _clone_checkpoint(stored)

    async def load_latest(self, run_id: str) -> WorkflowCheckpoint | None:
        async with self._lock:
            items = self._checkpoints.get(run_id, [])
            return _clone_checkpoint(items[-1]) if items else None

    async def load_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        async with self._lock:
            checkpoint = self._checkpoint_ids.get(checkpoint_id)
            return _clone_checkpoint(checkpoint) if checkpoint is not None else None

    async def find_by_idempotency_key(self, idempotency_key: str) -> WorkflowCheckpoint | None:
        async with self._lock:
            run_id = self._idempotency_runs.get(idempotency_key)
            if run_id is None:
                return None
            items = self._checkpoints.get(run_id, [])
            return _clone_checkpoint(items[-1]) if items else None

    async def list_checkpoints(self, run_id: str) -> list[WorkflowCheckpoint]:
        async with self._lock:
            return [_clone_checkpoint(item) for item in self._checkpoints.get(run_id, [])]


class InMemoryWorkflowLeaseManager:
    def __init__(self) -> None:
        self._leases: dict[str, WorkflowExecutionLease] = {}
        self._fencing_tokens: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        run_id: str,
        *,
        owner_id: str,
        ttl_ms: int,
        now_ms: int | None = None,
    ) -> WorkflowExecutionLease | None:
        resolved_run_id, resolved_owner_id, resolved_ttl = _validate_lease_request(
            run_id,
            owner_id,
            ttl_ms,
        )
        effective_now = _now_ms() if now_ms is None else _validate_non_negative_integer(now_ms, "lease now_ms")
        async with self._lock:
            current = self._leases.get(resolved_run_id)
            if current is not None and not current.is_expired(now_ms=effective_now):
                return None
            fencing_token = self._fencing_tokens.get(resolved_run_id, 0) + 1
            self._fencing_tokens[resolved_run_id] = fencing_token
            lease = WorkflowExecutionLease(
                run_id=resolved_run_id,
                owner_id=resolved_owner_id,
                token=f"wfl_{uuid4().hex}",
                fencing_token=fencing_token,
                acquired_at_ms=effective_now,
                renewed_at_ms=effective_now,
                expires_at_ms=effective_now + resolved_ttl,
            )
            self._leases[resolved_run_id] = lease
            return lease

    async def renew(
        self,
        run_id: str,
        *,
        token: str,
        ttl_ms: int,
        now_ms: int | None = None,
    ) -> WorkflowExecutionLease | None:
        resolved_run_id, resolved_token = _validate_lease_token(run_id, token, ttl_ms)
        effective_now = _now_ms() if now_ms is None else _validate_non_negative_integer(now_ms, "lease now_ms")
        async with self._lock:
            current = self._leases.get(resolved_run_id)
            if current is None or current.token != resolved_token or current.is_expired(now_ms=effective_now):
                return None
            renewed = WorkflowExecutionLease(
                run_id=current.run_id,
                owner_id=current.owner_id,
                token=current.token,
                fencing_token=current.fencing_token,
                acquired_at_ms=current.acquired_at_ms,
                renewed_at_ms=effective_now,
                expires_at_ms=effective_now + ttl_ms,
            )
            self._leases[resolved_run_id] = renewed
            return renewed

    async def release(self, run_id: str, *, token: str) -> bool:
        resolved_run_id, resolved_token = _validate_lease_token(run_id, token)
        async with self._lock:
            current = self._leases.get(resolved_run_id)
            if current is None or current.token != resolved_token:
                return False
            del self._leases[resolved_run_id]
            return True

    async def get(self, run_id: str) -> WorkflowExecutionLease | None:
        resolved_run_id = _validate_non_empty(run_id, "lease run_id")
        async with self._lock:
            return self._leases.get(resolved_run_id)


class SQLiteWorkflowCheckpointStore:
    def __init__(self, path: str, *, namespace: str = "default") -> None:
        if not namespace.strip():
            raise ValidationError("Workflow checkpoint namespace must be a non-empty string.")
        self._path = path
        self._namespace = namespace
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        connection = sqlite3.connect(self._path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS zhivex_workflow_runs (
                    namespace TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    idempotency_key TEXT,
                    workflow_name TEXT NOT NULL,
                    definition_version TEXT NOT NULL,
                    definition_digest TEXT NOT NULL,
                    latest_sequence INTEGER NOT NULL,
                    PRIMARY KEY (namespace, run_id),
                    UNIQUE (namespace, idempotency_key)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS zhivex_workflow_checkpoints (
                    namespace TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    checkpoint_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (namespace, checkpoint_id),
                    UNIQUE (namespace, run_id, sequence)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS zhivex_workflow_checkpoints_run_idx
                ON zhivex_workflow_checkpoints (namespace, run_id, sequence)
                """
            )
            connection.commit()
        finally:
            connection.close()

    async def append(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        expected_sequence: int | None = None,
    ) -> WorkflowCheckpoint:
        def runner() -> WorkflowCheckpoint:
            connection = sqlite3.connect(self._path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT latest_sequence
                    FROM zhivex_workflow_runs
                    WHERE namespace = ? AND run_id = ?
                    """,
                    (self._namespace, checkpoint.run_id),
                ).fetchone()
                current = self._load_latest_with_connection(connection, checkpoint.run_id) if row is not None else None
                candidate = _prepare_append(
                    checkpoint,
                    current=current,
                    expected_sequence=expected_sequence,
                )
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO zhivex_workflow_runs (
                            namespace, run_id, idempotency_key, workflow_name,
                            definition_version, definition_digest, latest_sequence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            self._namespace,
                            candidate.run_id,
                            candidate.idempotency_key,
                            candidate.workflow_name,
                            candidate.definition_version,
                            candidate.definition_digest,
                            candidate.sequence,
                        ),
                    )
                else:
                    cursor = connection.execute(
                        """
                        UPDATE zhivex_workflow_runs
                        SET latest_sequence = ?
                        WHERE namespace = ? AND run_id = ? AND latest_sequence = ?
                        """,
                        (candidate.sequence, self._namespace, candidate.run_id, expected_sequence),
                    )
                    if cursor.rowcount != 1:
                        raise ValidationError(f'Workflow run "{candidate.run_id}" changed during append.')
                connection.execute(
                    """
                    INSERT INTO zhivex_workflow_checkpoints (
                        namespace, checkpoint_id, run_id, sequence, checkpoint_json, created_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._namespace,
                        candidate.checkpoint_id,
                        candidate.run_id,
                        candidate.sequence,
                        workflow_checkpoint_to_json(candidate),
                        candidate.created_at_ms or 0,
                    ),
                )
                connection.commit()
                return _clone_checkpoint(candidate)
            except sqlite3.IntegrityError as error:
                connection.rollback()
                raise ValidationError(f"Workflow checkpoint append violated a uniqueness constraint: {error}.") from error
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        return await asyncio.to_thread(runner)

    def _load_latest_with_connection(
        self,
        connection: sqlite3.Connection,
        run_id: str,
    ) -> WorkflowCheckpoint | None:
        row = connection.execute(
            """
            SELECT checkpoint_json
            FROM zhivex_workflow_checkpoints
            WHERE namespace = ? AND run_id = ?
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (self._namespace, run_id),
        ).fetchone()
        return workflow_checkpoint_from_json(row[0]) if row is not None else None

    async def load_latest(self, run_id: str) -> WorkflowCheckpoint | None:
        def runner() -> WorkflowCheckpoint | None:
            connection = sqlite3.connect(self._path)
            try:
                return self._load_latest_with_connection(connection, run_id)
            finally:
                connection.close()

        return await asyncio.to_thread(runner)

    async def load_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        def runner() -> WorkflowCheckpoint | None:
            connection = sqlite3.connect(self._path)
            try:
                row = connection.execute(
                    """
                    SELECT checkpoint_json
                    FROM zhivex_workflow_checkpoints
                    WHERE namespace = ? AND checkpoint_id = ?
                    """,
                    (self._namespace, checkpoint_id),
                ).fetchone()
                return workflow_checkpoint_from_json(row[0]) if row is not None else None
            finally:
                connection.close()

        return await asyncio.to_thread(runner)

    async def find_by_idempotency_key(self, idempotency_key: str) -> WorkflowCheckpoint | None:
        def runner() -> WorkflowCheckpoint | None:
            connection = sqlite3.connect(self._path)
            try:
                row = connection.execute(
                    """
                    SELECT run_id
                    FROM zhivex_workflow_runs
                    WHERE namespace = ? AND idempotency_key = ?
                    """,
                    (self._namespace, idempotency_key),
                ).fetchone()
                return self._load_latest_with_connection(connection, row[0]) if row is not None else None
            finally:
                connection.close()

        return await asyncio.to_thread(runner)

    async def list_checkpoints(self, run_id: str) -> list[WorkflowCheckpoint]:
        def runner() -> list[WorkflowCheckpoint]:
            connection = sqlite3.connect(self._path)
            try:
                rows = connection.execute(
                    """
                    SELECT checkpoint_json
                    FROM zhivex_workflow_checkpoints
                    WHERE namespace = ? AND run_id = ?
                    ORDER BY sequence ASC
                    """,
                    (self._namespace, run_id),
                ).fetchall()
                return [workflow_checkpoint_from_json(row[0]) for row in rows]
            finally:
                connection.close()

        return await asyncio.to_thread(runner)


class SQLiteWorkflowLeaseManager:
    def __init__(self, path: str, *, namespace: str = "default") -> None:
        if not namespace.strip():
            raise ValidationError("Workflow execution lease namespace must be a non-empty string.")
        self._path = path
        self._namespace = namespace
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS zhivex_workflow_leases (
                    namespace TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    token TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    acquired_at_ms INTEGER NOT NULL,
                    renewed_at_ms INTEGER NOT NULL,
                    expires_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (namespace, run_id)
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _lease_from_row(row: tuple[Any, ...] | None) -> WorkflowExecutionLease | None:
        if row is None:
            return None
        return WorkflowExecutionLease(
            run_id=str(row[0]),
            owner_id=str(row[1]),
            token=str(row[2]),
            fencing_token=int(row[3]),
            acquired_at_ms=int(row[4]),
            renewed_at_ms=int(row[5]),
            expires_at_ms=int(row[6]),
        )

    async def acquire(
        self,
        run_id: str,
        *,
        owner_id: str,
        ttl_ms: int,
        now_ms: int | None = None,
    ) -> WorkflowExecutionLease | None:
        resolved_run_id, resolved_owner_id, resolved_ttl = _validate_lease_request(
            run_id,
            owner_id,
            ttl_ms,
        )
        effective_now = _now_ms() if now_ms is None else _validate_non_negative_integer(now_ms, "lease now_ms")

        def runner() -> WorkflowExecutionLease | None:
            connection = sqlite3.connect(self._path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT run_id, owner_id, token, fencing_token,
                           acquired_at_ms, renewed_at_ms, expires_at_ms
                    FROM zhivex_workflow_leases
                    WHERE namespace = ? AND run_id = ?
                    """,
                    (self._namespace, resolved_run_id),
                ).fetchone()
                current = self._lease_from_row(row)
                if current is not None and not current.is_expired(now_ms=effective_now):
                    connection.rollback()
                    return None
                fencing_token = (current.fencing_token if current is not None else 0) + 1
                lease = WorkflowExecutionLease(
                    run_id=resolved_run_id,
                    owner_id=resolved_owner_id,
                    token=f"wfl_{uuid4().hex}",
                    fencing_token=fencing_token,
                    acquired_at_ms=effective_now,
                    renewed_at_ms=effective_now,
                    expires_at_ms=effective_now + resolved_ttl,
                )
                connection.execute(
                    """
                    INSERT INTO zhivex_workflow_leases (
                        namespace, run_id, owner_id, token, fencing_token,
                        acquired_at_ms, renewed_at_ms, expires_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, run_id) DO UPDATE SET
                        owner_id = excluded.owner_id,
                        token = excluded.token,
                        fencing_token = excluded.fencing_token,
                        acquired_at_ms = excluded.acquired_at_ms,
                        renewed_at_ms = excluded.renewed_at_ms,
                        expires_at_ms = excluded.expires_at_ms
                    """,
                    (
                        self._namespace,
                        lease.run_id,
                        lease.owner_id,
                        lease.token,
                        lease.fencing_token,
                        lease.acquired_at_ms,
                        lease.renewed_at_ms,
                        lease.expires_at_ms,
                    ),
                )
                connection.commit()
                return lease
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        return await asyncio.to_thread(runner)

    async def renew(
        self,
        run_id: str,
        *,
        token: str,
        ttl_ms: int,
        now_ms: int | None = None,
    ) -> WorkflowExecutionLease | None:
        resolved_run_id, resolved_token = _validate_lease_token(run_id, token, ttl_ms)
        effective_now = _now_ms() if now_ms is None else _validate_non_negative_integer(now_ms, "lease now_ms")

        def runner() -> WorkflowExecutionLease | None:
            connection = sqlite3.connect(self._path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    """
                    UPDATE zhivex_workflow_leases
                    SET renewed_at_ms = ?, expires_at_ms = ?
                    WHERE namespace = ? AND run_id = ? AND token = ? AND expires_at_ms > ?
                    """,
                    (
                        effective_now,
                        effective_now + ttl_ms,
                        self._namespace,
                        resolved_run_id,
                        resolved_token,
                        effective_now,
                    ),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    return None
                row = connection.execute(
                    """
                    SELECT run_id, owner_id, token, fencing_token,
                           acquired_at_ms, renewed_at_ms, expires_at_ms
                    FROM zhivex_workflow_leases
                    WHERE namespace = ? AND run_id = ?
                    """,
                    (self._namespace, resolved_run_id),
                ).fetchone()
                connection.commit()
                return self._lease_from_row(row)
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        return await asyncio.to_thread(runner)

    async def release(self, run_id: str, *, token: str) -> bool:
        resolved_run_id, resolved_token = _validate_lease_token(run_id, token)

        def runner() -> bool:
            connection = sqlite3.connect(self._path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    """
                    UPDATE zhivex_workflow_leases
                    SET expires_at_ms = 0
                    WHERE namespace = ? AND run_id = ? AND token = ? AND expires_at_ms != 0
                    """,
                    (self._namespace, resolved_run_id, resolved_token),
                )
                connection.commit()
                return cursor.rowcount == 1
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        return await asyncio.to_thread(runner)

    async def get(self, run_id: str) -> WorkflowExecutionLease | None:
        resolved_run_id = _validate_non_empty(run_id, "lease run_id")

        def runner() -> WorkflowExecutionLease | None:
            connection = sqlite3.connect(self._path)
            try:
                row = connection.execute(
                    """
                    SELECT run_id, owner_id, token, fencing_token,
                           acquired_at_ms, renewed_at_ms, expires_at_ms
                    FROM zhivex_workflow_leases
                    WHERE namespace = ? AND run_id = ?
                    """,
                    (self._namespace, resolved_run_id),
                ).fetchone()
                return self._lease_from_row(row)
            finally:
                connection.close()

        return await asyncio.to_thread(runner)


class PostgresWorkflowCheckpointStore:
    def __init__(self, dsn: str, *, table_prefix: str = "zhivex_ai") -> None:
        self._dsn = dsn
        prefix = _validate_postgres_table_prefix(table_prefix)
        self._runs_table = f"{prefix}_workflow_runs"
        self._checkpoints_table = f"{prefix}_workflow_checkpoints"

    async def _connect(self) -> Any:
        try:
            import asyncpg  # type: ignore[import-not-found,import-untyped]
        except Exception as error:
            raise RuntimeError('Postgres support requires the optional dependency "asyncpg".') from error
        connection = await asyncpg.connect(self._dsn)
        try:
            await self._ensure_schema(connection)
        except BaseException:
            await connection.close()
            raise
        return connection

    async def _ensure_schema(self, connection: Any) -> None:
        async with connection.transaction():
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtext($1))",
                f"zhivex-workflow-schema:{self._runs_table}",
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._runs_table} (
                    run_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    workflow_name TEXT NOT NULL,
                    definition_version TEXT NOT NULL,
                    definition_digest TEXT NOT NULL,
                    latest_sequence BIGINT NOT NULL
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._checkpoints_table} (
                    checkpoint_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    sequence BIGINT NOT NULL,
                    checkpoint_json JSONB NOT NULL,
                    created_at_ms BIGINT NOT NULL,
                    UNIQUE (run_id, sequence)
                )
                """
            )
            await connection.execute(
                f"CREATE INDEX IF NOT EXISTS {self._checkpoints_table}_run_idx "
                f"ON {self._checkpoints_table} (run_id, sequence)"
            )

    @staticmethod
    def _payload_from_row(row: Any) -> dict[str, Any]:
        payload = row["checkpoint_json"]
        return payload if isinstance(payload, dict) else json.loads(payload)

    async def append(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        expected_sequence: int | None = None,
    ) -> WorkflowCheckpoint:
        connection = await self._connect()
        try:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    f"zhivex-workflow-run:{self._runs_table}:{checkpoint.run_id}",
                )
                if checkpoint.idempotency_key is not None:
                    await connection.execute(
                        "SELECT pg_advisory_xact_lock(hashtext($1))",
                        f"zhivex-workflow-idempotency:{self._runs_table}:{checkpoint.idempotency_key}",
                    )
                row = await connection.fetchrow(
                    f"SELECT latest_sequence FROM {self._runs_table} WHERE run_id = $1 FOR UPDATE",
                    checkpoint.run_id,
                )
                current = None
                if row is not None:
                    current_row = await connection.fetchrow(
                        f"""
                        SELECT checkpoint_json
                        FROM {self._checkpoints_table}
                        WHERE run_id = $1
                        ORDER BY sequence DESC
                        LIMIT 1
                        """,
                        checkpoint.run_id,
                    )
                    if current_row is not None:
                        current = deserialize_workflow_checkpoint(self._payload_from_row(current_row))
                candidate = _prepare_append(
                    checkpoint,
                    current=current,
                    expected_sequence=expected_sequence,
                )
                if row is None:
                    if candidate.idempotency_key is not None:
                        claimed = await connection.fetchrow(
                            f"SELECT run_id FROM {self._runs_table} WHERE idempotency_key = $1",
                            candidate.idempotency_key,
                        )
                        if claimed is not None:
                            raise ValidationError(
                                f'Workflow idempotency key "{candidate.idempotency_key}" is already claimed '
                                f'by run "{claimed["run_id"]}".'
                            )
                    await connection.execute(
                        f"""
                        INSERT INTO {self._runs_table} (
                            run_id, idempotency_key, workflow_name,
                            definition_version, definition_digest, latest_sequence
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        candidate.run_id,
                        candidate.idempotency_key,
                        candidate.workflow_name,
                        candidate.definition_version,
                        candidate.definition_digest,
                        candidate.sequence,
                    )
                else:
                    status = await connection.execute(
                        f"""
                        UPDATE {self._runs_table}
                        SET latest_sequence = $1
                        WHERE run_id = $2 AND latest_sequence = $3
                        """,
                        candidate.sequence,
                        candidate.run_id,
                        expected_sequence,
                    )
                    if status != "UPDATE 1":
                        raise ValidationError(f'Workflow run "{candidate.run_id}" changed during append.')
                await connection.execute(
                    f"""
                    INSERT INTO {self._checkpoints_table} (
                        checkpoint_id, run_id, sequence, checkpoint_json, created_at_ms
                    ) VALUES ($1, $2, $3, $4::jsonb, $5)
                    """,
                    candidate.checkpoint_id,
                    candidate.run_id,
                    candidate.sequence,
                    workflow_checkpoint_to_json(candidate),
                    candidate.created_at_ms or 0,
                )
                return _clone_checkpoint(candidate)
        finally:
            await connection.close()

    async def load_latest(self, run_id: str) -> WorkflowCheckpoint | None:
        connection = await self._connect()
        try:
            row = await connection.fetchrow(
                f"""
                SELECT checkpoint_json
                FROM {self._checkpoints_table}
                WHERE run_id = $1
                ORDER BY sequence DESC
                LIMIT 1
                """,
                run_id,
            )
            return deserialize_workflow_checkpoint(self._payload_from_row(row)) if row is not None else None
        finally:
            await connection.close()

    async def load_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpoint | None:
        connection = await self._connect()
        try:
            row = await connection.fetchrow(
                f"SELECT checkpoint_json FROM {self._checkpoints_table} WHERE checkpoint_id = $1",
                checkpoint_id,
            )
            return deserialize_workflow_checkpoint(self._payload_from_row(row)) if row is not None else None
        finally:
            await connection.close()

    async def find_by_idempotency_key(self, idempotency_key: str) -> WorkflowCheckpoint | None:
        connection = await self._connect()
        try:
            row = await connection.fetchrow(
                f"SELECT run_id FROM {self._runs_table} WHERE idempotency_key = $1",
                idempotency_key,
            )
            if row is None:
                return None
            checkpoint_row = await connection.fetchrow(
                f"""
                SELECT checkpoint_json
                FROM {self._checkpoints_table}
                WHERE run_id = $1
                ORDER BY sequence DESC
                LIMIT 1
                """,
                row["run_id"],
            )
            return (
                deserialize_workflow_checkpoint(self._payload_from_row(checkpoint_row))
                if checkpoint_row is not None
                else None
            )
        finally:
            await connection.close()

    async def list_checkpoints(self, run_id: str) -> list[WorkflowCheckpoint]:
        connection = await self._connect()
        try:
            rows = await connection.fetch(
                f"""
                SELECT checkpoint_json
                FROM {self._checkpoints_table}
                WHERE run_id = $1
                ORDER BY sequence ASC
                """,
                run_id,
            )
            return [deserialize_workflow_checkpoint(self._payload_from_row(row)) for row in rows]
        finally:
            await connection.close()


class PostgresWorkflowLeaseManager:
    def __init__(self, dsn: str, *, table_prefix: str = "zhivex_ai") -> None:
        self._dsn = dsn
        prefix = _validate_postgres_table_prefix(table_prefix)
        self._leases_table = f"{prefix}_workflow_leases"

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
                    f"zhivex-workflow-lease-schema:{self._leases_table}",
                )
                await connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._leases_table} (
                        run_id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        token TEXT NOT NULL,
                        fencing_token BIGINT NOT NULL,
                        acquired_at_ms BIGINT NOT NULL,
                        renewed_at_ms BIGINT NOT NULL,
                        expires_at_ms BIGINT NOT NULL
                    )
                    """
                )
        except BaseException:
            await connection.close()
            raise
        return connection

    @staticmethod
    def _lease_from_row(row: Any) -> WorkflowExecutionLease | None:
        if row is None:
            return None
        return WorkflowExecutionLease(
            run_id=str(row["run_id"]),
            owner_id=str(row["owner_id"]),
            token=str(row["token"]),
            fencing_token=int(row["fencing_token"]),
            acquired_at_ms=int(row["acquired_at_ms"]),
            renewed_at_ms=int(row["renewed_at_ms"]),
            expires_at_ms=int(row["expires_at_ms"]),
        )

    async def acquire(
        self,
        run_id: str,
        *,
        owner_id: str,
        ttl_ms: int,
        now_ms: int | None = None,
    ) -> WorkflowExecutionLease | None:
        resolved_run_id, resolved_owner_id, resolved_ttl = _validate_lease_request(
            run_id,
            owner_id,
            ttl_ms,
        )
        effective_now = _now_ms() if now_ms is None else _validate_non_negative_integer(now_ms, "lease now_ms")
        connection = await self._connect()
        try:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))",
                    f"zhivex-workflow-lease:{self._leases_table}:{resolved_run_id}",
                )
                row = await connection.fetchrow(
                    f"SELECT * FROM {self._leases_table} WHERE run_id = $1 FOR UPDATE",
                    resolved_run_id,
                )
                current = self._lease_from_row(row)
                if current is not None and not current.is_expired(now_ms=effective_now):
                    return None
                lease = WorkflowExecutionLease(
                    run_id=resolved_run_id,
                    owner_id=resolved_owner_id,
                    token=f"wfl_{uuid4().hex}",
                    fencing_token=(current.fencing_token if current is not None else 0) + 1,
                    acquired_at_ms=effective_now,
                    renewed_at_ms=effective_now,
                    expires_at_ms=effective_now + resolved_ttl,
                )
                await connection.execute(
                    f"""
                    INSERT INTO {self._leases_table} (
                        run_id, owner_id, token, fencing_token,
                        acquired_at_ms, renewed_at_ms, expires_at_ms
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (run_id) DO UPDATE SET
                        owner_id = EXCLUDED.owner_id,
                        token = EXCLUDED.token,
                        fencing_token = EXCLUDED.fencing_token,
                        acquired_at_ms = EXCLUDED.acquired_at_ms,
                        renewed_at_ms = EXCLUDED.renewed_at_ms,
                        expires_at_ms = EXCLUDED.expires_at_ms
                    """,
                    lease.run_id,
                    lease.owner_id,
                    lease.token,
                    lease.fencing_token,
                    lease.acquired_at_ms,
                    lease.renewed_at_ms,
                    lease.expires_at_ms,
                )
                return lease
        finally:
            await connection.close()

    async def renew(
        self,
        run_id: str,
        *,
        token: str,
        ttl_ms: int,
        now_ms: int | None = None,
    ) -> WorkflowExecutionLease | None:
        resolved_run_id, resolved_token = _validate_lease_token(run_id, token, ttl_ms)
        effective_now = _now_ms() if now_ms is None else _validate_non_negative_integer(now_ms, "lease now_ms")
        connection = await self._connect()
        try:
            row = await connection.fetchrow(
                f"""
                UPDATE {self._leases_table}
                SET renewed_at_ms = $1, expires_at_ms = $2
                WHERE run_id = $3 AND token = $4 AND expires_at_ms > $1
                RETURNING *
                """,
                effective_now,
                effective_now + ttl_ms,
                resolved_run_id,
                resolved_token,
            )
            return self._lease_from_row(row)
        finally:
            await connection.close()

    async def release(self, run_id: str, *, token: str) -> bool:
        resolved_run_id, resolved_token = _validate_lease_token(run_id, token)
        connection = await self._connect()
        try:
            status = await connection.execute(
                f"""
                UPDATE {self._leases_table}
                SET expires_at_ms = 0
                WHERE run_id = $1 AND token = $2 AND expires_at_ms != 0
                """,
                resolved_run_id,
                resolved_token,
            )
            return status == "UPDATE 1"
        finally:
            await connection.close()

    async def get(self, run_id: str) -> WorkflowExecutionLease | None:
        resolved_run_id = _validate_non_empty(run_id, "lease run_id")
        connection = await self._connect()
        try:
            row = await connection.fetchrow(
                f"SELECT * FROM {self._leases_table} WHERE run_id = $1",
                resolved_run_id,
            )
            return self._lease_from_row(row)
        finally:
            await connection.close()


def create_in_memory_workflow_checkpoint_store() -> InMemoryWorkflowCheckpointStore:
    return InMemoryWorkflowCheckpointStore()


def create_in_memory_workflow_lease_manager() -> InMemoryWorkflowLeaseManager:
    return InMemoryWorkflowLeaseManager()


def create_sqlite_workflow_checkpoint_store(
    path: str,
    *,
    namespace: str = "default",
) -> SQLiteWorkflowCheckpointStore:
    return SQLiteWorkflowCheckpointStore(path, namespace=namespace)


def create_sqlite_workflow_lease_manager(
    path: str,
    *,
    namespace: str = "default",
) -> SQLiteWorkflowLeaseManager:
    return SQLiteWorkflowLeaseManager(path, namespace=namespace)


def create_postgres_workflow_checkpoint_store(
    dsn: str,
    *,
    table_prefix: str = "zhivex_ai",
) -> PostgresWorkflowCheckpointStore:
    return PostgresWorkflowCheckpointStore(dsn, table_prefix=table_prefix)


def create_postgres_workflow_lease_manager(
    dsn: str,
    *,
    table_prefix: str = "zhivex_ai",
) -> PostgresWorkflowLeaseManager:
    return PostgresWorkflowLeaseManager(dsn, table_prefix=table_prefix)
