from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast, runtime_checkable

from .errors import ValidationError
from .types import JsonValue

WORKFLOW_ADAPTER_SCHEMA_VERSION = 1

WorkflowStepStatus = Literal["completed", "failed", "suspended", "cancelled"]


def _require_non_empty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f'Workflow adapter field "{name}" must be a non-empty string.')
    return value


def _require_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f'Workflow adapter field "{name}" must be an integer greater than or equal to {minimum}.')
    return value


def _validate_schema_version(value: object) -> int:
    version = _require_int("schema_version", value, minimum=1)
    if version != WORKFLOW_ADAPTER_SCHEMA_VERSION:
        raise ValidationError(
            "Workflow adapter payload uses unsupported schema_version "
            f"{version}; this SDK supports {WORKFLOW_ADAPTER_SCHEMA_VERSION}."
        )
    return version


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f'Workflow adapter field "{path}" must contain finite JSON numbers.')
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f'Workflow adapter field "{path}" must contain only string object keys.')
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ValidationError(
        f'Workflow adapter field "{path}" must be JSON serializable; runtime objects and callables are not allowed.'
    )


def _clone_json_value(value: JsonValue, *, path: str) -> JsonValue:
    _validate_json_value(value, path=path)
    return cast(JsonValue, json.loads(json.dumps(value, allow_nan=False, separators=(",", ":"))))


def _clone_json_object(value: Mapping[str, JsonValue], *, path: str) -> dict[str, JsonValue]:
    copied = _clone_json_value(dict(value), path=path)
    if not isinstance(copied, dict):  # pragma: no cover - guaranteed by the input shape
        raise ValidationError(f'Workflow adapter field "{path}" must be a JSON object.')
    return copied


def _mapping_field(payload: Mapping[str, Any], name: str) -> dict[str, JsonValue]:
    value = payload.get(name, {})
    if not isinstance(value, dict):
        raise ValidationError(f'Workflow adapter field "{name}" must be a JSON object.')
    _validate_json_value(value, path=name)
    return cast(dict[str, JsonValue], value)


def _optional_mapping_field(payload: Mapping[str, Any], name: str) -> dict[str, JsonValue] | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValidationError(f'Workflow adapter field "{name}" must be a JSON object or null.')
    _validate_json_value(value, path=name)
    return cast(dict[str, JsonValue], value)


def _json_value_field(payload: Mapping[str, Any], name: str) -> JsonValue:
    value = payload.get(name)
    _validate_json_value(value, path=name)
    return cast(JsonValue, value)


def _reject_unknown_fields(payload: Mapping[str, Any], allowed: frozenset[str], *, payload_name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValidationError(f"{payload_name} contains unknown fields: {', '.join(unknown)}.")


def _decode_json_object(value: str, *, payload_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{payload_name} must be valid JSON.") from error
    if not isinstance(payload, dict):
        raise ValidationError(f"{payload_name} must be a JSON object.")
    return cast(dict[str, Any], payload)


def _step_idempotency_key(
    *,
    definition_digest: str,
    workflow_run_id: str,
    node_id: str,
    activation_index: int,
) -> str:
    identity = json.dumps(
        [definition_digest, workflow_run_id, node_id, activation_index],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"wfstep_{hashlib.sha256(identity).hexdigest()}"


@dataclass(frozen=True, slots=True)
class WorkflowAdapterCapabilities:
    durable_steps: bool = False
    native_step_retries: bool = False
    signals: bool = False
    explicit_resume: bool = False
    fork: bool = False
    cancellation: bool = False
    durable_timers: bool = False


@dataclass(frozen=True, slots=True)
class WorkflowStepRequest:
    workflow_name: str
    definition_version: str
    definition_digest: str
    workflow_run_id: str
    node_id: str
    executor_ref: str
    activation_index: int = 0
    attempt: int = 1
    state_revision: int = 0
    input: JsonValue = None
    state: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    checkpoint_id: str | None = None
    correlation_ids: dict[str, str] = field(default_factory=dict)
    schema_version: int = WORKFLOW_ADAPTER_SCHEMA_VERSION
    step_idempotency_key: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        for name in (
            "workflow_name",
            "definition_version",
            "definition_digest",
            "workflow_run_id",
            "node_id",
            "executor_ref",
        ):
            _require_non_empty_string(name, getattr(self, name))
        _require_int("activation_index", self.activation_index, minimum=0)
        _require_int("attempt", self.attempt, minimum=1)
        _require_int("state_revision", self.state_revision, minimum=0)
        if self.checkpoint_id is not None:
            _require_non_empty_string("checkpoint_id", self.checkpoint_id)
        _validate_json_value(self.input, path="input")
        object.__setattr__(self, "input", _clone_json_value(self.input, path="input"))
        object.__setattr__(self, "state", _clone_json_object(self.state, path="state"))
        object.__setattr__(self, "metadata", _clone_json_object(self.metadata, path="metadata"))
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.correlation_ids.items()):
            raise ValidationError('Workflow adapter field "correlation_ids" must contain string keys and values.')
        object.__setattr__(self, "correlation_ids", dict(self.correlation_ids))
        object.__setattr__(
            self,
            "step_idempotency_key",
            _step_idempotency_key(
                definition_digest=self.definition_digest,
                workflow_run_id=self.workflow_run_id,
                node_id=self.node_id,
                activation_index=self.activation_index,
            ),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "workflow_name": self.workflow_name,
            "definition_version": self.definition_version,
            "definition_digest": self.definition_digest,
            "workflow_run_id": self.workflow_run_id,
            "node_id": self.node_id,
            "executor_ref": self.executor_ref,
            "activation_index": self.activation_index,
            "attempt": self.attempt,
            "state_revision": self.state_revision,
            "step_idempotency_key": self.step_idempotency_key,
            "input": _clone_json_value(self.input, path="input"),
            "state": _clone_json_object(self.state, path="state"),
            "metadata": _clone_json_object(self.metadata, path="metadata"),
            "checkpoint_id": self.checkpoint_id,
            "correlation_ids": cast(JsonValue, dict(self.correlation_ids)),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkflowStepRequest:
        allowed = frozenset(
            {
                "schema_version",
                "workflow_name",
                "definition_version",
                "definition_digest",
                "workflow_run_id",
                "node_id",
                "executor_ref",
                "activation_index",
                "attempt",
                "state_revision",
                "step_idempotency_key",
                "input",
                "state",
                "metadata",
                "checkpoint_id",
                "correlation_ids",
            }
        )
        _reject_unknown_fields(payload, allowed, payload_name="WorkflowStepRequest")
        correlation_ids = payload.get("correlation_ids", {})
        if not isinstance(correlation_ids, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in correlation_ids.items()
        ):
            raise ValidationError('Workflow adapter field "correlation_ids" must contain string keys and values.')
        request = cls(
            schema_version=_validate_schema_version(payload.get("schema_version")),
            workflow_name=_require_non_empty_string("workflow_name", payload.get("workflow_name")),
            definition_version=_require_non_empty_string("definition_version", payload.get("definition_version")),
            definition_digest=_require_non_empty_string("definition_digest", payload.get("definition_digest")),
            workflow_run_id=_require_non_empty_string("workflow_run_id", payload.get("workflow_run_id")),
            node_id=_require_non_empty_string("node_id", payload.get("node_id")),
            executor_ref=_require_non_empty_string("executor_ref", payload.get("executor_ref")),
            activation_index=_require_int("activation_index", payload.get("activation_index"), minimum=0),
            attempt=_require_int("attempt", payload.get("attempt"), minimum=1),
            state_revision=_require_int("state_revision", payload.get("state_revision"), minimum=0),
            input=_json_value_field(payload, "input"),
            state=_mapping_field(payload, "state"),
            metadata=_mapping_field(payload, "metadata"),
            checkpoint_id=cast(str | None, payload.get("checkpoint_id")),
            correlation_ids=cast(dict[str, str], correlation_ids),
        )
        serialized_key = _require_non_empty_string("step_idempotency_key", payload.get("step_idempotency_key"))
        if serialized_key != request.step_idempotency_key:
            raise ValidationError("WorkflowStepRequest step_idempotency_key does not match its durable step identity.")
        return request

    @classmethod
    def from_json(cls, value: str) -> WorkflowStepRequest:
        return cls.from_dict(_decode_json_object(value, payload_name="WorkflowStepRequest"))


@dataclass(frozen=True, slots=True)
class WorkflowStepOutcome:
    workflow_run_id: str
    node_id: str
    activation_index: int
    step_idempotency_key: str
    status: WorkflowStepStatus
    output: JsonValue = None
    state_patch: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    error: dict[str, JsonValue] | None = None
    suspension: dict[str, JsonValue] | None = None
    child_run_id: str | None = None
    schema_version: int = WORKFLOW_ADAPTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        _require_non_empty_string("workflow_run_id", self.workflow_run_id)
        _require_non_empty_string("node_id", self.node_id)
        _require_non_empty_string("step_idempotency_key", self.step_idempotency_key)
        _require_int("activation_index", self.activation_index, minimum=0)
        if self.status not in {"completed", "failed", "suspended", "cancelled"}:
            raise ValidationError(f'Unsupported workflow step outcome status "{self.status}".')
        if self.child_run_id is not None:
            _require_non_empty_string("child_run_id", self.child_run_id)
        _validate_json_value(self.output, path="output")
        object.__setattr__(self, "output", _clone_json_value(self.output, path="output"))
        object.__setattr__(self, "state_patch", _clone_json_object(self.state_patch, path="state_patch"))
        object.__setattr__(self, "metadata", _clone_json_object(self.metadata, path="metadata"))
        if self.error is not None:
            object.__setattr__(self, "error", _clone_json_object(self.error, path="error"))
        if self.suspension is not None:
            object.__setattr__(self, "suspension", _clone_json_object(self.suspension, path="suspension"))
        if self.status == "failed" and self.error is None:
            raise ValidationError('A failed WorkflowStepOutcome must include a serializable "error" payload.')
        if self.status != "failed" and self.error is not None:
            raise ValidationError('Only a failed WorkflowStepOutcome may include an "error" payload.')
        if self.status == "suspended" and self.suspension is None:
            raise ValidationError('A suspended WorkflowStepOutcome must include a serializable "suspension" payload.')
        if self.status != "suspended" and self.suspension is not None:
            raise ValidationError('Only a suspended WorkflowStepOutcome may include a "suspension" payload.')

    @classmethod
    def for_request(
        cls,
        request: WorkflowStepRequest,
        *,
        status: WorkflowStepStatus,
        output: JsonValue = None,
        state_patch: Mapping[str, JsonValue] | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        error: Mapping[str, JsonValue] | None = None,
        suspension: Mapping[str, JsonValue] | None = None,
        child_run_id: str | None = None,
    ) -> WorkflowStepOutcome:
        return cls(
            workflow_run_id=request.workflow_run_id,
            node_id=request.node_id,
            activation_index=request.activation_index,
            step_idempotency_key=request.step_idempotency_key,
            status=status,
            output=output,
            state_patch=dict(state_patch or {}),
            metadata=dict(metadata or {}),
            error=dict(error) if error is not None else None,
            suspension=dict(suspension) if suspension is not None else None,
            child_run_id=child_run_id,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "workflow_run_id": self.workflow_run_id,
            "node_id": self.node_id,
            "activation_index": self.activation_index,
            "step_idempotency_key": self.step_idempotency_key,
            "status": self.status,
            "output": _clone_json_value(self.output, path="output"),
            "state_patch": _clone_json_object(self.state_patch, path="state_patch"),
            "metadata": _clone_json_object(self.metadata, path="metadata"),
            "error": _clone_json_object(self.error, path="error") if self.error is not None else None,
            "suspension": (
                _clone_json_object(self.suspension, path="suspension") if self.suspension is not None else None
            ),
            "child_run_id": self.child_run_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> WorkflowStepOutcome:
        allowed = frozenset(
            {
                "schema_version",
                "workflow_run_id",
                "node_id",
                "activation_index",
                "step_idempotency_key",
                "status",
                "output",
                "state_patch",
                "metadata",
                "error",
                "suspension",
                "child_run_id",
            }
        )
        _reject_unknown_fields(payload, allowed, payload_name="WorkflowStepOutcome")
        status = payload.get("status")
        if status not in {"completed", "failed", "suspended", "cancelled"}:
            raise ValidationError(f'Unsupported workflow step outcome status "{status}".')
        child_run_id = payload.get("child_run_id")
        if child_run_id is not None and not isinstance(child_run_id, str):
            raise ValidationError('Workflow adapter field "child_run_id" must be a string or null.')
        return cls(
            schema_version=_validate_schema_version(payload.get("schema_version")),
            workflow_run_id=_require_non_empty_string("workflow_run_id", payload.get("workflow_run_id")),
            node_id=_require_non_empty_string("node_id", payload.get("node_id")),
            activation_index=_require_int("activation_index", payload.get("activation_index"), minimum=0),
            step_idempotency_key=_require_non_empty_string(
                "step_idempotency_key", payload.get("step_idempotency_key")
            ),
            status=cast(WorkflowStepStatus, status),
            output=_json_value_field(payload, "output"),
            state_patch=_mapping_field(payload, "state_patch"),
            metadata=_mapping_field(payload, "metadata"),
            error=_optional_mapping_field(payload, "error"),
            suspension=_optional_mapping_field(payload, "suspension"),
            child_run_id=child_run_id,
        )

    @classmethod
    def from_json(cls, value: str) -> WorkflowStepOutcome:
        return cls.from_dict(_decode_json_object(value, payload_name="WorkflowStepOutcome"))


WorkflowStepExecutor = Callable[
    [WorkflowStepRequest],
    WorkflowStepOutcome | Awaitable[WorkflowStepOutcome],
]


def _validate_outcome_identity(request: WorkflowStepRequest, outcome: WorkflowStepOutcome) -> None:
    if outcome.workflow_run_id != request.workflow_run_id:
        raise ValidationError("Workflow step outcome workflow_run_id does not match its request.")
    if outcome.node_id != request.node_id or outcome.activation_index != request.activation_index:
        raise ValidationError("Workflow step outcome node identity does not match its request.")
    if outcome.step_idempotency_key != request.step_idempotency_key:
        raise ValidationError("Workflow step outcome idempotency key does not match its request.")


async def _invoke_executor(executor: WorkflowStepExecutor, request: WorkflowStepRequest) -> WorkflowStepOutcome:
    result = executor(request)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, WorkflowStepOutcome):
        raise ValidationError("Workflow step executors must return WorkflowStepOutcome.")
    _validate_outcome_identity(request, result)
    return result


@runtime_checkable
class WorkflowAdapter(Protocol):
    backend: str
    capabilities: WorkflowAdapterCapabilities

    async def dispatch(self, request: WorkflowStepRequest) -> WorkflowStepOutcome: ...


class WorkflowStepExecutorRegistry:
    def __init__(self) -> None:
        self._executors: dict[tuple[str, str], WorkflowStepExecutor] = {}
        self._executor_refs: set[str] = set()

    def register(
        self,
        *,
        executor_ref: str,
        definition_digest: str,
        executor: WorkflowStepExecutor,
    ) -> None:
        resolved_ref = _require_non_empty_string("executor_ref", executor_ref)
        resolved_digest = _require_non_empty_string("definition_digest", definition_digest)
        if not callable(executor):
            raise ValidationError("Workflow step executor must be callable.")
        key = (resolved_ref, resolved_digest)
        if key in self._executors:
            raise ValidationError(
                f'Workflow executor "{resolved_ref}" is already registered for this definition digest.'
            )
        self._executors[key] = executor
        self._executor_refs.add(resolved_ref)

    def resolve(self, request: WorkflowStepRequest) -> WorkflowStepExecutor:
        if request.executor_ref not in self._executor_refs:
            raise ValidationError(f'Unknown workflow executor_ref "{request.executor_ref}".')
        executor = self._executors.get((request.executor_ref, request.definition_digest))
        if executor is None:
            raise ValidationError(
                f'Workflow definition digest mismatch for executor_ref "{request.executor_ref}"; refusing dispatch.'
            )
        return executor

    async def dispatch(self, request: WorkflowStepRequest) -> WorkflowStepOutcome:
        return await _invoke_executor(self.resolve(request), request)


@dataclass(slots=True)
class CallbackWorkflowAdapter:
    backend: str
    callback: WorkflowStepExecutor = field(repr=False)
    capabilities: WorkflowAdapterCapabilities = field(default_factory=WorkflowAdapterCapabilities)

    def __post_init__(self) -> None:
        _require_non_empty_string("backend", self.backend)
        if not callable(self.callback):
            raise ValidationError("Workflow adapter callback must be callable.")

    async def dispatch(self, request: WorkflowStepRequest) -> WorkflowStepOutcome:
        return await _invoke_executor(self.callback, request)


def create_dbos_workflow_adapter(callback: WorkflowStepExecutor) -> CallbackWorkflowAdapter:
    return CallbackWorkflowAdapter(
        backend="dbos",
        callback=callback,
        capabilities=WorkflowAdapterCapabilities(
            durable_steps=True,
            native_step_retries=True,
            explicit_resume=True,
            fork=True,
            cancellation=True,
            durable_timers=True,
        ),
    )


def create_temporal_workflow_adapter(callback: WorkflowStepExecutor) -> CallbackWorkflowAdapter:
    return CallbackWorkflowAdapter(
        backend="temporal",
        callback=callback,
        capabilities=WorkflowAdapterCapabilities(
            durable_steps=True,
            native_step_retries=True,
            signals=True,
            cancellation=True,
            durable_timers=True,
        ),
    )


def create_prefect_workflow_adapter(callback: WorkflowStepExecutor) -> CallbackWorkflowAdapter:
    return CallbackWorkflowAdapter(
        backend="prefect",
        callback=callback,
        capabilities=WorkflowAdapterCapabilities(native_step_retries=True),
    )


def create_restate_workflow_adapter(callback: WorkflowStepExecutor) -> CallbackWorkflowAdapter:
    return CallbackWorkflowAdapter(
        backend="restate",
        callback=callback,
        capabilities=WorkflowAdapterCapabilities(
            durable_steps=True,
            native_step_retries=True,
            signals=True,
            durable_timers=True,
        ),
    )
