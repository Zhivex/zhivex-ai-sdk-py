from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast
from uuid import uuid4

from .agent import AgentObserver, AgentRunResult, AgentSession, create_agent_session, resume_agent_run, run_agent
from .agent_state import AgentChildRun, AgentRunState, AgentRunStep, AgentRunStore
from .errors import ProviderHTTPError, ValidationError
from .types import JsonValue
from .workflow import (
    WorkflowRetryPolicy,
    WorkflowFunctionContext,
    WorkflowFunctionResult,
    WorkflowRunResult,
    WorkflowStep,
    WorkflowStepResult,
    WorkflowTraceEvent,
)
from .workflow_adapters import WorkflowAdapter, WorkflowStepOutcome, WorkflowStepRequest
from .workflow_state import (
    WorkflowCheckpoint,
    WorkflowCheckpointStore,
    WorkflowExecutionLease,
    WorkflowInterrupt,
    WorkflowLeaseManager,
    WorkflowNodeCheckpoint,
    WorkflowTransition,
    create_in_memory_workflow_checkpoint_store,
)

WorkflowEdgeCondition = Callable[["WorkflowContext"], bool | Awaitable[bool]]
WorkflowInterruptPhase = Literal["before", "after"]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _json_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _callable_identity(value: Callable[..., Any] | None) -> str | None:
    if value is None:
        return None
    module = getattr(value, "__module__", "")
    qualname = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
    try:
        source = inspect.getsource(value).strip()
    except (OSError, TypeError):
        source = ""
    return f"{module}:{qualname}:{source}"


@dataclass(slots=True, frozen=True)
class WorkflowContext:
    run_id: str
    workflow_name: str
    source: str
    target: str
    state: Mapping[str, JsonValue]
    source_status: str
    source_output: JsonValue | None = None
    resume_values: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class WorkflowEdge:
    source: str
    target: str
    condition: WorkflowEdgeCondition | None = None
    name: str | None = None

    @property
    def id(self) -> str:
        return self.name or f"{self.source}->{self.target}"


@dataclass(slots=True)
class _NodeExecution:
    status: Literal["completed", "failed", "suspended", "cancelled"]
    output: JsonValue | None = None
    child_run_id: str | None = None
    agent_result: AgentRunResult[Any] | None = None
    error: Exception | None = None
    state_patch: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    suspension: dict[str, JsonValue] | None = None


@dataclass(slots=True)
class _WorkflowLeaseGuard:
    lease: WorkflowExecutionLease
    graph_identity: int
    lost: asyncio.Event = field(default_factory=asyncio.Event)


_ACTIVE_WORKFLOW_LEASE: ContextVar[_WorkflowLeaseGuard | None] = ContextVar(
    "zhivex_active_workflow_lease",
    default=None,
)


class _WorkflowCancellationObserved(Exception):
    def __init__(self, checkpoint: WorkflowCheckpoint) -> None:
        self.checkpoint = checkpoint
        super().__init__(f'Workflow run "{checkpoint.run_id}" was cancelled.')


class WorkflowBuilder:
    """Build a validated acyclic workflow graph without mutating prior builders."""

    def __init__(self, name: str, *, definition_version: str = "1") -> None:
        self.name = name
        self.definition_version = definition_version
        self._steps: list[WorkflowStep] = []
        self._edges: list[WorkflowEdge] = []
        self._entrypoints: list[str] = []
        self._interrupt_before: dict[str, str | None] = {}
        self._interrupt_after: dict[str, str | None] = {}

    def add_step(self, step: WorkflowStep, *, entrypoint: bool = False) -> WorkflowBuilder:
        self._steps.append(step)
        if entrypoint:
            self._entrypoints.append(step.name)
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        condition: WorkflowEdgeCondition | None = None,
        name: str | None = None,
    ) -> WorkflowBuilder:
        self._edges.append(WorkflowEdge(source=source, target=target, condition=condition, name=name))
        return self

    def interrupt_before(self, step_name: str, *, reason: str | None = None) -> WorkflowBuilder:
        self._interrupt_before[step_name] = reason
        return self

    def interrupt_after(self, step_name: str, *, reason: str | None = None) -> WorkflowBuilder:
        self._interrupt_after[step_name] = reason
        return self

    def build(
        self,
        *,
        checkpoint_store: WorkflowCheckpointStore | None = None,
        run_store: AgentRunStore | None = None,
        adapter: WorkflowAdapter | None = None,
        max_concurrency: int | None = None,
        lease_manager: WorkflowLeaseManager | None = None,
        lease_ttl_ms: int = 30_000,
        lease_heartbeat_ms: int | None = None,
        observer: AgentObserver | None = None,
    ) -> WorkflowGraph:
        return WorkflowGraph(
            name=self.name,
            steps=self._steps,
            edges=self._edges,
            definition_version=self.definition_version,
            entrypoints=self._entrypoints or None,
            checkpoint_store=checkpoint_store,
            run_store=run_store,
            adapter=adapter,
            interrupt_before=self._interrupt_before,
            interrupt_after=self._interrupt_after,
            max_concurrency=max_concurrency,
            lease_manager=lease_manager,
            lease_ttl_ms=lease_ttl_ms,
            lease_heartbeat_ms=lease_heartbeat_ms,
            observer=observer,
        )


class WorkflowGraph:
    """Durable DAG workflow with append-only checkpoints and explicit resume."""

    def __init__(
        self,
        *,
        name: str,
        steps: Sequence[WorkflowStep],
        edges: Sequence[WorkflowEdge],
        definition_version: str = "1",
        entrypoints: Sequence[str] | None = None,
        checkpoint_store: WorkflowCheckpointStore | None = None,
        run_store: AgentRunStore | None = None,
        adapter: WorkflowAdapter | None = None,
        interrupt_before: Mapping[str, str | None] | Sequence[str] = (),
        interrupt_after: Mapping[str, str | None] | Sequence[str] = (),
        max_concurrency: int | None = None,
        lease_manager: WorkflowLeaseManager | None = None,
        lease_ttl_ms: int = 30_000,
        lease_heartbeat_ms: int | None = None,
        observer: AgentObserver | None = None,
    ) -> None:
        self.name = name
        self.definition_version = str(definition_version).strip()
        self.steps = list(steps)
        self.edges = list(edges)
        self.checkpoint_store = checkpoint_store or create_in_memory_workflow_checkpoint_store()
        self.run_store = run_store
        self.adapter = adapter
        self.max_concurrency = max_concurrency
        self.lease_manager = lease_manager
        self.observer = observer
        self.lease_ttl_ms = lease_ttl_ms
        self.lease_heartbeat_ms = (
            max(1, lease_ttl_ms // 3)
            if lease_heartbeat_ms is None and isinstance(lease_ttl_ms, int) and not isinstance(lease_ttl_ms, bool)
            else 1
            if lease_heartbeat_ms is None
            else lease_heartbeat_ms
        )
        self._steps = {step.name: step for step in self.steps}
        self._interrupt_before = self._normalize_interrupts(interrupt_before)
        self._interrupt_after = self._normalize_interrupts(interrupt_after)
        self._validate(entrypoints)
        incoming_targets = {edge.target for edge in self.edges}
        self.entrypoints = list(entrypoints or [step.name for step in self.steps if step.name not in incoming_targets])
        self._validate_reachability()
        self.definition_digest = self._definition_digest()

    @staticmethod
    def _normalize_interrupts(
        value: Mapping[str, str | None] | Sequence[str],
    ) -> dict[str, str | None]:
        if isinstance(value, Mapping):
            return {str(key): str(reason) if reason is not None else None for key, reason in value.items()}
        return {str(item): None for item in value}

    def _validate(self, entrypoints: Sequence[str] | None) -> None:
        if not self.name.strip():
            raise ValidationError("WorkflowGraph.name must not be empty.")
        if not self.definition_version:
            raise ValidationError("WorkflowGraph.definition_version must not be empty.")
        if not self.steps:
            raise ValidationError("WorkflowGraph requires at least one step.")
        names = [step.name for step in self.steps]
        if any(not name.strip() for name in names):
            raise ValidationError("Workflow step names must not be empty.")
        if len(set(names)) != len(names):
            raise ValidationError("WorkflowGraph step names must be unique.")
        edge_ids = [edge.id for edge in self.edges]
        if len(set(edge_ids)) != len(edge_ids):
            raise ValidationError("WorkflowGraph edge names must be unique.")
        for edge in self.edges:
            if edge.source not in self._steps or edge.target not in self._steps:
                raise ValidationError(
                    f'Workflow edge "{edge.id}" references an unknown source or target step.'
                )
            if edge.source == edge.target:
                raise ValidationError(f'Workflow edge "{edge.id}" cannot target its own source.')
        selected_entrypoints = list(entrypoints or [])
        if selected_entrypoints and any(item not in self._steps for item in selected_entrypoints):
            raise ValidationError("WorkflowGraph entrypoints must reference existing steps.")
        if len(selected_entrypoints) != len(set(selected_entrypoints)):
            raise ValidationError("WorkflowGraph entrypoints must be unique.")
        interrupt_nodes = {*self._interrupt_before, *self._interrupt_after}
        if any(item not in self._steps for item in interrupt_nodes):
            raise ValidationError("WorkflowGraph interrupt points must reference existing steps.")
        if self.max_concurrency is not None and (
            isinstance(self.max_concurrency, bool)
            or not isinstance(self.max_concurrency, int)
            or self.max_concurrency <= 0
        ):
            raise ValidationError("WorkflowGraph.max_concurrency must be a positive integer.")
        if self.lease_manager is not None:
            if (
                isinstance(self.lease_ttl_ms, bool)
                or not isinstance(self.lease_ttl_ms, int)
                or self.lease_ttl_ms <= 0
            ):
                raise ValidationError("WorkflowGraph.lease_ttl_ms must be a positive integer.")
            if (
                isinstance(self.lease_heartbeat_ms, bool)
                or not isinstance(self.lease_heartbeat_ms, int)
                or self.lease_heartbeat_ms <= 0
                or self.lease_heartbeat_ms >= self.lease_ttl_ms
            ):
                raise ValidationError(
                    "WorkflowGraph.lease_heartbeat_ms must be a positive integer smaller than lease_ttl_ms."
                )
        for step in self.steps:
            self._validate_durable_payload(step.metadata)
            if self.adapter is not None:
                if not step.executor_ref:
                    raise ValidationError(
                        f'Workflow step "{step.name}" requires executor_ref when a workflow adapter is configured.'
                    )
            elif (step.agent is None) == (step.executor is None):
                raise ValidationError(
                    f'Workflow step "{step.name}" must define exactly one of agent or executor.'
                )
        output_keys = [step.output_key for step in self.steps if step.output_key]
        metadata_keys = [step.metadata_key for step in self.steps if step.metadata_key]
        duplicates = {
            key
            for key in [*output_keys, *metadata_keys]
            if [*output_keys, *metadata_keys].count(key) > 1
        }
        if duplicates:
            raise ValidationError(
                "WorkflowGraph output_key and metadata_key values must be globally unique: "
                + ", ".join(sorted(cast(set[str], duplicates)))
                + "."
            )
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        outgoing: dict[str, list[str]] = {name: [] for name in self._steps}
        indegree = {name: 0 for name in self._steps}
        for edge in self.edges:
            outgoing[edge.source].append(edge.target)
            indegree[edge.target] += 1
        queue = [name for name in self._steps if indegree[name] == 0]
        visited = 0
        while queue:
            source = queue.pop(0)
            visited += 1
            for target in outgoing[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if visited != len(self._steps):
            raise ValidationError("WorkflowGraph edges must form an acyclic graph.")

    def _validate_reachability(self) -> None:
        if not self.entrypoints:
            raise ValidationError("WorkflowGraph requires at least one entrypoint.")
        reachable = set(self.entrypoints)
        queue = list(self.entrypoints)
        while queue:
            source = queue.pop(0)
            for edge in self._outgoing(source):
                if edge.target not in reachable:
                    reachable.add(edge.target)
                    queue.append(edge.target)
        unreachable = sorted(set(self._steps) - reachable)
        if unreachable:
            raise ValidationError(
                "WorkflowGraph contains steps that cannot be reached from its entrypoints: "
                + ", ".join(unreachable)
                + "."
            )

    def _definition_digest(self) -> str:
        payload = {
            "name": self.name,
            "version": self.definition_version,
            "steps": [
                {
                    "name": step.name,
                    "agent": step.agent.name if step.agent is not None else None,
                    "executor": _callable_identity(step.executor),
                    "prompt": step.prompt,
                    "input_template": step.input_template,
                    "output_key": step.output_key,
                    "metadata_key": step.metadata_key,
                    "error_policy": step.error_policy,
                    "timeout_ms": step.timeout_ms,
                    "max_retries": step.max_retries,
                    "executor_ref": step.executor_ref,
                    "idempotency_key": step.idempotency_key,
                    "metadata": step.metadata,
                    "retry": {
                        "max_attempts": step.retry_policy.max_attempts,
                        "backoff_ms": step.retry_policy.backoff_ms,
                        "max_backoff_ms": step.retry_policy.max_backoff_ms,
                        "retry_if": _callable_identity(step.retry_policy.retry_if),
                    }
                    if step.retry_policy is not None
                    else None,
                }
                for step in self.steps
            ],
            "edges": [
                {
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "condition": _callable_identity(edge.condition),
                }
                for edge in self.edges
            ],
            "entrypoints": self.entrypoints,
            "interrupt_before": self._interrupt_before,
            "interrupt_after": self._interrupt_after,
            "adapter": getattr(self.adapter, "backend", None),
            "max_concurrency": self.max_concurrency,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    async def _assert_execution_lease(self) -> None:
        guard = _ACTIVE_WORKFLOW_LEASE.get()
        if guard is None or guard.graph_identity != id(self):
            return
        if guard.lost.is_set() or self.lease_manager is None:
            raise ValidationError(
                f'Workflow execution lease for run "{guard.lease.run_id}" was lost; refusing stale worker progress.'
            )
        current = await self.lease_manager.get(guard.lease.run_id)
        if (
            current is None
            or current.token != guard.lease.token
            or current.fencing_token != guard.lease.fencing_token
            or current.is_expired()
        ):
            guard.lost.set()
            raise ValidationError(
                f'Workflow execution lease for run "{guard.lease.run_id}" was lost; refusing stale worker progress.'
            )

    async def _with_execution_lease(
        self,
        run_id: str,
        operation: Callable[[], Awaitable[WorkflowRunResult]],
    ) -> WorkflowRunResult:
        started_at = time.perf_counter()
        span = self._start_span(
            "zhivex.workflow.run",
            {
                "zhivex.workflow.name": self.name,
                "zhivex.workflow.run_id": run_id,
                "zhivex.workflow.definition_version": self.definition_version,
                "zhivex.workflow.definition_digest": self.definition_digest,
                "zhivex.workflow.leased": self.lease_manager is not None,
            },
        )
        try:
            result = await self._with_execution_lease_impl(run_id, operation)
        except BaseException as error:
            attributes = {
                "zhivex.workflow.status": (
                    "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
                ),
                "zhivex.duration_ms": (time.perf_counter() - started_at) * 1_000,
            }
            self._finish_span(
                span,
                attributes=attributes,
                error=error if isinstance(error, Exception) else None,
            )
            raise
        self._finish_span(
            span,
            attributes={
                "zhivex.workflow.status": result.status,
                "zhivex.workflow.checkpoint_sequence": (
                    result.checkpoint.sequence if result.checkpoint is not None else -1
                ),
                "zhivex.duration_ms": (time.perf_counter() - started_at) * 1_000,
            },
        )
        return result

    async def _with_execution_lease_impl(
        self,
        run_id: str,
        operation: Callable[[], Awaitable[WorkflowRunResult]],
    ) -> WorkflowRunResult:
        lease_manager = self.lease_manager
        if lease_manager is None:
            return await operation()
        lease = await lease_manager.acquire(
            run_id,
            owner_id=f"workflow-worker-{uuid4().hex}",
            ttl_ms=self.lease_ttl_ms,
        )
        if lease is None:
            current = await lease_manager.get(run_id)
            owner = current.owner_id if current is not None and not current.is_expired() else "another worker"
            raise ValidationError(
                f'Workflow run "{run_id}" has an active execution lease owned by "{owner}".'
            )

        guard = _WorkflowLeaseGuard(lease=lease, graph_identity=id(self))
        context_token = _ACTIVE_WORKFLOW_LEASE.set(guard)

        async def heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(self.lease_heartbeat_ms / 1_000)
                    renewed = await lease_manager.renew(
                        run_id,
                        token=lease.token,
                        ttl_ms=self.lease_ttl_ms,
                    )
                    if renewed is None:
                        guard.lost.set()
                        return
            except asyncio.CancelledError:
                raise
            except Exception:
                guard.lost.set()

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            try:
                return await operation()
            except _WorkflowCancellationObserved as cancellation:
                return await self._result(cancellation.checkpoint, session=None)
        finally:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
            with suppress(Exception):
                await lease_manager.release(run_id, token=lease.token)
            _ACTIVE_WORKFLOW_LEASE.reset(context_token)

    async def _execute_with_optional_lease(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        session: AgentSession | None,
        deps: Any,
    ) -> WorkflowRunResult:
        return await self._with_execution_lease(
            checkpoint.run_id,
            lambda: self._execute(checkpoint, session=session, deps=deps),
        )

    async def run(
        self,
        *,
        session: AgentSession | None = None,
        prompt: str | None = None,
        parent_run_id: str | None = None,
        deps: Any = None,
        idempotency_key: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        recover_running: bool = False,
    ) -> WorkflowRunResult:
        if idempotency_key:
            existing = await self.checkpoint_store.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                self._validate_checkpoint(existing)
                if existing.status in {"completed", "failed", "suspended", "cancelled"}:
                    return await self._result(existing, session=session)
                if not recover_running:
                    raise ValidationError(
                        f'Workflow run "{existing.run_id}" is still running. Pass recover_running=True only '
                        "after reconciling that the prior worker is no longer active."
                    )
                async def recover_and_execute() -> WorkflowRunResult:
                    current = await self.checkpoint_store.load_latest(existing.run_id)
                    if current is None:
                        raise ValidationError(f'Workflow run "{existing.run_id}" was not found during recovery.')
                    self._validate_checkpoint(current)
                    recovered = self._recover_running_nodes(current)
                    if recovered is not current:
                        current = await self._append(
                            current,
                            recovered,
                            WorkflowTransition(
                                type="workflow-recovered",
                                at_ms=_now_ms(),
                                detail={"reason": "expired lease" if self.lease_manager is not None else "idempotent re-entry"},
                            ),
                        )
                    return await self._execute(current, session=session, deps=deps)

                return await self._with_execution_lease(existing.run_id, recover_and_execute)

        now = _now_ms()
        run_id = _new_id("wf")
        resolved_session = session or create_agent_session()
        checkpoint = WorkflowCheckpoint(
            checkpoint_id=_new_id("wfc"),
            run_id=run_id,
            workflow_name=self.name,
            definition_version=self.definition_version,
            definition_digest=self.definition_digest,
            status="running",
            session_id=resolved_session.id,
            parent_run_id=parent_run_id,
            idempotency_key=idempotency_key,
            state=dict(resolved_session.state),
            nodes={name: WorkflowNodeCheckpoint(node_name=name) for name in self._steps},
            ready_nodes=list(self.entrypoints),
            transition=WorkflowTransition(type="workflow-start", at_ms=now),
            created_at_ms=now,
            updated_at_ms=now,
            metadata={
                **dict(metadata or {}),
                "prompt": prompt,
                "completed_order": [],
                "resolved_interrupts": [],
            },
        )
        try:
            checkpoint = await self.checkpoint_store.append(checkpoint)
        except ValidationError:
            if idempotency_key is None:
                raise
            winner = await self.checkpoint_store.find_by_idempotency_key(idempotency_key)
            if winner is None or winner.run_id == run_id:
                raise
            self._validate_checkpoint(winner)
            return await self._result(winner, session=session)
        return await self._execute_with_optional_lease(checkpoint, session=resolved_session, deps=deps)

    def _validate_checkpoint(self, checkpoint: WorkflowCheckpoint) -> None:
        if checkpoint.workflow_name != self.name:
            raise ValidationError(
                f'Workflow checkpoint belongs to "{checkpoint.workflow_name}", not "{self.name}".'
            )
        if checkpoint.definition_version != self.definition_version:
            raise ValidationError(
                "Workflow definition version changed; migrate the checkpoint explicitly before resume."
            )
        if checkpoint.definition_digest != self.definition_digest:
            raise ValidationError(
                "Workflow definition digest changed; resume and fork fail closed to avoid choosing a different path."
            )

    def _recover_running_nodes(self, checkpoint: WorkflowCheckpoint) -> WorkflowCheckpoint:
        if not any(node.status == "running" for node in checkpoint.nodes.values()):
            return checkpoint
        candidate = copy.deepcopy(checkpoint)
        for node in candidate.nodes.values():
            if node.status == "running":
                node.status = "pending"
                node.error = "Recovered after an incomplete worker execution."
        candidate.status = "running"
        return candidate

    async def _append(
        self,
        current: WorkflowCheckpoint,
        candidate: WorkflowCheckpoint,
        transition: WorkflowTransition,
    ) -> WorkflowCheckpoint:
        await self._assert_execution_lease()
        candidate = copy.deepcopy(candidate)
        guard = _ACTIVE_WORKFLOW_LEASE.get()
        if guard is not None and guard.graph_identity == id(self):
            candidate.metadata["execution_lease"] = {
                "owner_id": guard.lease.owner_id,
                "fencing_token": guard.lease.fencing_token,
            }
        candidate.sequence = current.sequence + 1
        candidate.checkpoint_id = _new_id("wfc")
        candidate.updated_at_ms = transition.at_ms
        candidate.transition = transition
        try:
            return await self.checkpoint_store.append(candidate, expected_sequence=current.sequence)
        except ValidationError:
            guard = _ACTIVE_WORKFLOW_LEASE.get()
            if guard is not None and guard.graph_identity == id(self):
                latest = await self.checkpoint_store.load_latest(current.run_id)
                if latest is not None and latest.status == "cancelled":
                    raise _WorkflowCancellationObserved(latest) from None
            raise

    async def _update(
        self,
        current: WorkflowCheckpoint,
        transition: WorkflowTransition,
        mutate: Callable[[WorkflowCheckpoint], None],
    ) -> WorkflowCheckpoint:
        candidate = copy.deepcopy(current)
        mutate(candidate)
        return await self._append(current, candidate, transition)

    def _incoming(self, node_name: str) -> list[WorkflowEdge]:
        return [edge for edge in self.edges if edge.target == node_name]

    def _outgoing(self, node_name: str) -> list[WorkflowEdge]:
        return [edge for edge in self.edges if edge.source == node_name]

    def _ready(self, checkpoint: WorkflowCheckpoint) -> list[str]:
        ready: list[str] = []
        for name in self._steps:
            node = checkpoint.nodes[name]
            if node.status != "pending":
                continue
            incoming = self._incoming(name)
            if not incoming:
                if name in self.entrypoints:
                    ready.append(name)
                continue
            if all(edge.id in checkpoint.edge_decisions for edge in incoming) and any(
                checkpoint.edge_decisions[edge.id] for edge in incoming
            ):
                ready.append(name)
        return ready

    async def _resolve_skipped(self, checkpoint: WorkflowCheckpoint) -> WorkflowCheckpoint:
        changed = True
        while changed:
            changed = False
            for name in self._steps:
                node = checkpoint.nodes[name]
                incoming = self._incoming(name)
                if node.status != "pending" or not incoming:
                    continue
                if not all(edge.id in checkpoint.edge_decisions for edge in incoming):
                    continue
                if any(checkpoint.edge_decisions[edge.id] for edge in incoming):
                    continue

                def mark_skipped(candidate: WorkflowCheckpoint, node_name: str = name) -> None:
                    candidate.nodes[node_name].status = "skipped"
                    candidate.nodes[node_name].finished_at_ms = _now_ms()
                    for edge in self._outgoing(node_name):
                        candidate.edge_decisions.setdefault(edge.id, False)
                    candidate.ready_nodes = self._ready(candidate)

                checkpoint = await self._update(
                    checkpoint,
                    WorkflowTransition(
                        type="workflow-step-skipped",
                        at_ms=_now_ms(),
                        node_name=name,
                        from_status="pending",
                        to_status="skipped",
                    ),
                    mark_skipped,
                )
                changed = True
        return checkpoint

    def _interrupt_resolved(self, checkpoint: WorkflowCheckpoint, node_name: str, phase: str) -> bool:
        raw = checkpoint.metadata.get("resolved_interrupts")
        return f"{phase}:{node_name}" in raw if isinstance(raw, list) else False

    async def _suspend_for_interrupt(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        node_name: str,
        phase: WorkflowInterruptPhase,
        reason: str | None,
    ) -> WorkflowCheckpoint:
        now = _now_ms()
        interrupt = WorkflowInterrupt(
            interrupt_id=_new_id("wfi"),
            node_name=node_name,
            phase=phase,
            reason=reason,
            created_at_ms=now,
        )

        def suspend(candidate: WorkflowCheckpoint) -> None:
            candidate.status = "suspended"
            candidate.pending_interrupt = interrupt

        return await self._update(
            checkpoint,
            WorkflowTransition(
                type="workflow-interrupt",
                at_ms=now,
                node_name=node_name,
                from_status=checkpoint.nodes[node_name].status,
                to_status="suspended",
                detail={"interrupt_id": interrupt.interrupt_id, "phase": phase},
            ),
            suspend,
        )

    async def _execute(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        session: AgentSession | None,
        deps: Any,
    ) -> WorkflowRunResult:
        resolved_session = session or create_agent_session(id=checkpoint.session_id, state=dict(checkpoint.state))
        resolved_session.state = dict(checkpoint.state)
        while True:
            await self._assert_execution_lease()
            latest = await self.checkpoint_store.load_latest(checkpoint.run_id)
            if latest is not None and latest.status == "cancelled":
                return await self._result(latest, session=resolved_session)
            checkpoint = await self._resolve_skipped(checkpoint)
            if checkpoint.pending_interrupt is not None:
                checkpoint.status = "suspended"
                return await self._result(checkpoint, session=resolved_session)

            ready = self._ready(checkpoint)
            before = next(
                (
                    name
                    for name in ready
                    if name in self._interrupt_before and not self._interrupt_resolved(checkpoint, name, "before")
                ),
                None,
            )
            if before is not None:
                checkpoint = await self._suspend_for_interrupt(
                    checkpoint,
                    node_name=before,
                    phase="before",
                    reason=self._interrupt_before[before],
                )
                return await self._result(checkpoint, session=resolved_session)

            if not ready:
                statuses = {node.status for node in checkpoint.nodes.values()}
                if "cancelled" in statuses:
                    checkpoint = await self._finish(checkpoint, status="cancelled")
                elif "suspended" in statuses:
                    checkpoint = await self._finish(checkpoint, status="suspended")
                elif "failed" in statuses:
                    checkpoint = await self._finish(checkpoint, status="failed")
                elif statuses.issubset({"completed", "skipped"}):
                    checkpoint = await self._finish(checkpoint, status="completed")
                else:
                    checkpoint = await self._finish(
                        checkpoint,
                        status="failed",
                        error="Workflow graph has no runnable nodes; definition or state is inconsistent.",
                    )
                return await self._result(checkpoint, session=resolved_session)

            attempts: dict[str, int] = {}
            for node_name in ready:
                now = _now_ms()

                def start(candidate: WorkflowCheckpoint, name: str = node_name, at_ms: int = now) -> None:
                    node = candidate.nodes[name]
                    node.status = "running"
                    node.attempt += 1
                    node.idempotency_key = self._step_idempotency_key(candidate, self._steps[name])
                    node.started_at_ms = at_ms
                    node.error = None
                    candidate.status = "running"
                    candidate.ready_nodes = self._ready(candidate)

                checkpoint = await self._update(
                    checkpoint,
                    WorkflowTransition(
                        type="workflow-step-start",
                        at_ms=now,
                        node_name=node_name,
                        from_status="pending",
                        to_status="running",
                    ),
                    start,
                )
                attempts[node_name] = checkpoint.nodes[node_name].attempt

            semaphore = asyncio.Semaphore(self.max_concurrency or max(1, len(ready)))

            async def execute_one(node_name: str) -> _NodeExecution:
                async with semaphore:
                    started_at = time.perf_counter()
                    span = self._start_span(
                        "zhivex.workflow.step",
                        {
                            "zhivex.workflow.name": self.name,
                            "zhivex.workflow.run_id": checkpoint.run_id,
                            "zhivex.workflow.step_name": node_name,
                            "zhivex.workflow.step_attempt": attempts[node_name],
                        },
                    )
                    try:
                        outcome = await self._execute_node(
                            checkpoint,
                            node_name=node_name,
                            attempt=attempts[node_name],
                            deps=deps,
                        )
                    except BaseException as error:
                        self._finish_span(
                            span,
                            attributes={
                                "zhivex.workflow.step_status": "cancelled",
                                "zhivex.duration_ms": (time.perf_counter() - started_at) * 1_000,
                            },
                            error=error if isinstance(error, Exception) else None,
                        )
                        raise
                    attributes: dict[str, Any] = {
                        "zhivex.workflow.step_status": outcome.status,
                        "zhivex.duration_ms": (time.perf_counter() - started_at) * 1_000,
                    }
                    if outcome.child_run_id is not None:
                        attributes["zhivex.agent.run_id"] = outcome.child_run_id
                    self._finish_span(span, attributes=attributes, error=outcome.error)
                    return outcome

            outcomes = await asyncio.gather(*(execute_one(name) for name in ready))
            latest = await self.checkpoint_store.load_latest(checkpoint.run_id)
            if latest is not None and latest.status == "cancelled":
                return await self._result(latest, session=resolved_session)
            fail_fast = False
            suspended = False
            cancelled = False
            after_interrupt: str | None = None
            edge_sources: list[str] = []
            for node_name, outcome in zip(ready, outcomes, strict=True):
                step = self._steps[node_name]
                node = checkpoint.nodes[node_name]
                retry = False
                if outcome.error is not None:
                    try:
                        retry = await self._should_retry(step, outcome.error, node.attempt)
                    except Exception as retry_error:
                        outcome.error = retry_error
                if retry and outcome.error is not None:
                    delay_ms = self._retry_delay(step, node.attempt)

                    def schedule_retry(
                        candidate: WorkflowCheckpoint,
                        name: str = node_name,
                        error: Exception = outcome.error,
                        retry_delay_ms: int = delay_ms,
                    ) -> None:
                        retry_node = candidate.nodes[name]
                        retry_node.status = "pending"
                        retry_node.error = str(error)
                        retry_node.metadata["retry_not_before_ms"] = _now_ms() + retry_delay_ms
                        candidate.ready_nodes = [*candidate.ready_nodes, name]

                    checkpoint = await self._update(
                        checkpoint,
                        WorkflowTransition(
                            type="workflow-step-retry",
                            at_ms=_now_ms(),
                            node_name=node_name,
                            from_status="running",
                            to_status="pending",
                            detail={"attempt": node.attempt, "delay_ms": delay_ms, "error": str(outcome.error)},
                        ),
                        schedule_retry,
                    )
                    continue

                finished_at = _now_ms()

                def finish_node(
                    candidate: WorkflowCheckpoint,
                    name: str = node_name,
                    execution: _NodeExecution = outcome,
                    at_ms: int = finished_at,
                    workflow_step: WorkflowStep = step,
                ) -> None:
                    result_node = candidate.nodes[name]
                    result_node.status = execution.status
                    result_node.output = execution.output
                    result_node.child_run_id = execution.child_run_id
                    result_node.error = str(execution.error) if execution.error is not None else None
                    result_node.finished_at_ms = at_ms
                    result_node.metadata = {**result_node.metadata, **execution.metadata}
                    result_node.suspension = (
                        copy.deepcopy(execution.suspension)
                        if execution.status == "suspended" and execution.suspension is not None
                        else None
                    )
                    if execution.status == "completed":
                        if workflow_step.output_key is not None:
                            candidate.state[workflow_step.output_key] = execution.output
                        if workflow_step.metadata_key is not None:
                            candidate.state[workflow_step.metadata_key] = {
                                "name": name,
                                "status": execution.status,
                                "run_id": execution.child_run_id,
                                "agent_name": (
                                    workflow_step.agent.name
                                    if workflow_step.agent is not None
                                    else workflow_step.executor_ref
                                ),
                                "text": execution.output,
                                "attempts": result_node.attempt,
                                "error": None,
                            }
                        candidate.state.update(execution.state_patch)
                        completed_order = candidate.metadata.setdefault("completed_order", [])
                        if isinstance(completed_order, list) and name not in completed_order:
                            completed_order.append(name)
                    elif workflow_step.error_policy == "capture" and workflow_step.output_key is not None:
                        candidate.state[workflow_step.output_key] = {"error": result_node.error, "step": name}

                checkpoint = await self._update(
                    checkpoint,
                    WorkflowTransition(
                        type="workflow-step-finish",
                        at_ms=finished_at,
                        node_name=node_name,
                        from_status="running",
                        to_status=outcome.status,
                        detail={"attempt": node.attempt, "child_run_id": outcome.child_run_id},
                    ),
                    finish_node,
                )
                resolved_session.state = dict(checkpoint.state)
                if outcome.status == "suspended":
                    suspended = True
                    continue
                if outcome.status == "cancelled":
                    cancelled = True
                    continue
                if outcome.status == "failed" and step.error_policy == "fail_fast":
                    fail_fast = True
                    continue
                if (
                    node_name in self._interrupt_after
                    and not self._interrupt_resolved(checkpoint, node_name, "after")
                    and after_interrupt is None
                ):
                    after_interrupt = node_name
                    continue
                edge_sources.append(node_name)

            if cancelled:
                checkpoint = await self._finish(checkpoint, status="cancelled")
                return await self._result(checkpoint, session=resolved_session)
            if suspended:
                checkpoint = await self._finish(checkpoint, status="suspended")
                return await self._result(checkpoint, session=resolved_session)
            if fail_fast:
                checkpoint = await self._finish(checkpoint, status="failed")
                return await self._result(checkpoint, session=resolved_session)
            try:
                for node_name in edge_sources:
                    checkpoint = await self._evaluate_edges(checkpoint, node_name)
            except Exception as error:
                checkpoint = await self._finish(
                    checkpoint,
                    status="failed",
                    error=f"Workflow edge evaluation failed: {error}",
                )
                return await self._result(checkpoint, session=resolved_session)
            if after_interrupt is not None:
                checkpoint = await self._suspend_for_interrupt(
                    checkpoint,
                    node_name=after_interrupt,
                    phase="after",
                    reason=self._interrupt_after[after_interrupt],
                )
                return await self._result(checkpoint, session=resolved_session)

    async def _execute_node(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        node_name: str,
        attempt: int,
        deps: Any,
    ) -> _NodeExecution:
        try:
            await self._assert_execution_lease()
            step = self._steps[node_name]
            retry_not_before = checkpoint.nodes[node_name].metadata.get("retry_not_before_ms")
            if isinstance(retry_not_before, int):
                remaining_ms = retry_not_before - _now_ms()
                if remaining_ms > 0:
                    await asyncio.sleep(remaining_ms / 1_000)
            step_key = self._step_idempotency_key(checkpoint, step)
            prompt = self._render_prompt(
                step,
                checkpoint.state,
                cast(str | None, checkpoint.metadata.get("prompt")),
            )
            if self.adapter is not None:
                guard = _ACTIVE_WORKFLOW_LEASE.get()
                correlation_ids = {"session_id": checkpoint.session_id or ""}
                if guard is not None and guard.graph_identity == id(self):
                    correlation_ids["workflow_fencing_token"] = str(guard.lease.fencing_token)
                request = WorkflowStepRequest(
                    workflow_name=self.name,
                    definition_version=self.definition_version,
                    definition_digest=self.definition_digest,
                    workflow_run_id=checkpoint.run_id,
                    node_id=node_name,
                    executor_ref=step.executor_ref or node_name,
                    attempt=attempt,
                    state_revision=checkpoint.sequence,
                    input=prompt,
                    state=dict(checkpoint.state),
                    metadata={
                        **dict(step.metadata),
                        "workflow_resume_values": dict(checkpoint.resume_values),
                    },
                    checkpoint_id=checkpoint.checkpoint_id,
                    correlation_ids=correlation_ids,
                )
                outcome = await self.adapter.dispatch(request)
                if (
                    outcome.workflow_run_id != request.workflow_run_id
                    or outcome.node_id != request.node_id
                    or outcome.activation_index != request.activation_index
                    or outcome.step_idempotency_key != request.step_idempotency_key
                ):
                    raise ValidationError("Workflow adapter returned an outcome for a different durable step identity.")
                return self._execution_from_adapter(outcome)

            if step.executor is not None:
                function_result = step.executor(
                    WorkflowFunctionContext(
                        run_id=checkpoint.run_id,
                        workflow_name=self.name,
                        step_name=node_name,
                        attempt=attempt,
                        idempotency_key=step_key,
                        input=cast(JsonValue, prompt),
                        state=copy.deepcopy(checkpoint.state),
                        resume_values=copy.deepcopy(checkpoint.resume_values),
                        deps=deps,
                    )
                )
                if inspect.isawaitable(function_result):
                    function_result = await function_result
                if isinstance(function_result, WorkflowFunctionResult):
                    output = function_result.output
                    state_patch = dict(function_result.state_patch)
                    result_metadata = dict(function_result.metadata)
                else:
                    output = cast(JsonValue, function_result)
                    state_patch = {}
                    result_metadata = {}
                self._validate_durable_payload(
                    {"output": output, "state_patch": state_patch, "metadata": result_metadata}
                )
                return _NodeExecution(
                    status="completed",
                    output=output,
                    state_patch=state_patch,
                    metadata=result_metadata,
                )

            if step.agent is None:
                raise ValidationError(f'Workflow step "{node_name}" has no local executor.')
            isolated_session = create_agent_session(
                id=checkpoint.session_id,
                state=copy.deepcopy(checkpoint.state),
            )
            step_agent = replace(
                step.agent,
                metadata={
                    **step.agent.metadata,
                    "zhivex_workflow_step_idempotency_key": step_key,
                    "workflow_run_id": checkpoint.run_id,
                    "workflow_step": node_name,
                },
            )
            result = await run_agent(
                agent=step_agent,
                session=isolated_session,
                prompt=prompt,
                deps=deps,
                parent_run_id=checkpoint.run_id,
                timeout_ms=step.timeout_ms,
                max_retries=step.max_retries,
                idempotency_key=f"{step_key}:attempt:{attempt}",
                observer=self.observer,
            )
        except Exception as error:
            return _NodeExecution(status="failed", error=error)
        status: Literal["completed", "suspended"] = (
            "suspended" if result.state is not None and result.state.status == "suspended" else "completed"
        )
        return _NodeExecution(
            status=status,
            output=cast(JsonValue, result.text),
            child_run_id=result.run_id,
            agent_result=result,
        )

    @staticmethod
    def _validate_durable_payload(value: Any) -> None:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValidationError("Functional workflow results must contain only finite JSON values.") from error

    @staticmethod
    def _execution_from_adapter(outcome: WorkflowStepOutcome) -> _NodeExecution:
        error = None
        if outcome.error is not None:
            message = outcome.error.get("message") or outcome.error.get("code") or "Workflow adapter step failed."
            error = RuntimeError(str(message))
        status = outcome.status
        if status not in {"completed", "failed", "suspended", "cancelled"}:
            return _NodeExecution(status="failed", error=ValidationError("Workflow adapter returned invalid status."))
        return _NodeExecution(
            status=cast(Literal["completed", "failed", "suspended", "cancelled"], status),
            output=outcome.output,
            child_run_id=outcome.child_run_id,
            error=error,
            state_patch=dict(outcome.state_patch),
            metadata=dict(outcome.metadata),
            suspension=dict(outcome.suspension) if outcome.suspension is not None else None,
        )

    def _start_span(self, name: str, attributes: dict[str, Any]) -> Any:
        if self.observer is None:
            return None
        return self.observer.start_span(name, attributes)

    @staticmethod
    def _finish_span(
        span: Any,
        *,
        attributes: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        if span is not None:
            span.end(attributes=attributes, error=error)

    @staticmethod
    def _render_prompt(step: WorkflowStep, state: Mapping[str, JsonValue], fallback: str | None) -> str | None:
        if step.input_template is not None:
            try:
                return step.input_template.format(**state)
            except KeyError as error:
                raise ValidationError(
                    f"Workflow input_template is missing state key: {error.args[0]}."
                ) from error
            except (IndexError, ValueError, AttributeError) as error:
                raise ValidationError(f"Workflow input_template is invalid: {error}.") from error
        return step.prompt if step.prompt is not None else fallback

    def _step_idempotency_key(self, checkpoint: WorkflowCheckpoint, step: WorkflowStep) -> str:
        namespace = step.idempotency_key or step.name
        raw = f"{self.definition_digest}:{checkpoint.run_id}:{namespace}:1"
        return "wfs_" + hashlib.sha256(raw.encode()).hexdigest()

    async def _should_retry(self, step: WorkflowStep, error: Exception, attempt: int) -> bool:
        policy = step.retry_policy
        if policy is None or attempt >= policy.max_attempts:
            return False
        if policy.retry_if is not None:
            decision = policy.retry_if(error)
            return bool(await decision if inspect.isawaitable(decision) else decision)
        return isinstance(error, ProviderHTTPError) and error.retryable

    @staticmethod
    def _retry_delay(step: WorkflowStep, attempt: int) -> int:
        policy = cast(WorkflowRetryPolicy, step.retry_policy)
        return min(policy.max_backoff_ms, policy.backoff_ms * (2 ** max(0, attempt - 1)))

    async def _evaluate_edges(self, checkpoint: WorkflowCheckpoint, node_name: str) -> WorkflowCheckpoint:
        outgoing = [edge for edge in self._outgoing(node_name) if edge.id not in checkpoint.edge_decisions]
        if not outgoing:
            return checkpoint
        decisions: dict[str, bool] = {}
        source_node = checkpoint.nodes[node_name]
        for edge in outgoing:
            if source_node.status not in {"completed", "failed"}:
                decision = False
            elif edge.condition is None:
                decision = True
            else:
                context = WorkflowContext(
                    run_id=checkpoint.run_id,
                    workflow_name=self.name,
                    source=edge.source,
                    target=edge.target,
                    state=copy.deepcopy(checkpoint.state),
                    source_status=source_node.status,
                    source_output=source_node.output,
                    resume_values=copy.deepcopy(checkpoint.resume_values),
                )
                value = edge.condition(context)
                decision = bool(await value if inspect.isawaitable(value) else value)
            decisions[edge.id] = decision

        def record(candidate: WorkflowCheckpoint) -> None:
            candidate.edge_decisions.update(decisions)
            candidate.ready_nodes = self._ready(candidate)

        return await self._update(
            checkpoint,
            WorkflowTransition(
                type="workflow-edge-decisions",
                at_ms=_now_ms(),
                node_name=node_name,
                detail={"decisions": cast(JsonValue, decisions)},
            ),
            record,
        )

    async def _finish(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        status: Literal["completed", "failed", "suspended", "cancelled"],
        error: str | None = None,
    ) -> WorkflowCheckpoint:
        if checkpoint.status == status and checkpoint.transition and checkpoint.transition.type == "workflow-finish":
            return checkpoint

        def finish(candidate: WorkflowCheckpoint) -> None:
            candidate.status = status
            if error:
                candidate.metadata["error"] = error

        return await self._update(
            checkpoint,
            WorkflowTransition(
                type="workflow-finish",
                at_ms=_now_ms(),
                from_status=checkpoint.status,
                to_status=status,
                detail={"error": error},
            ),
            finish,
        )

    def _projection(self, checkpoint: WorkflowCheckpoint) -> AgentRunState:
        status = checkpoint.status if checkpoint.status in {"running", "completed", "failed", "cancelled", "suspended"} else "failed"
        completed_order = checkpoint.metadata.get("completed_order")
        ordered = completed_order if isinstance(completed_order, list) else []
        last_output: JsonValue | None = None
        for name in reversed(ordered):
            if isinstance(name, str) and name in checkpoint.nodes:
                last_output = checkpoint.nodes[name].output
                break
        steps: list[AgentRunStep] = []
        child_runs: list[AgentChildRun] = []
        for index, step in enumerate(self.steps, start=1):
            node = checkpoint.nodes[step.name]
            projected_status = node.status if node.status in {"running", "completed", "failed", "cancelled", "suspended"} else "completed"
            steps.append(
                AgentRunStep(
                    index=index,
                    status=cast(Any, projected_status),
                    error=node.error,
                    started_at_ms=node.started_at_ms,
                    finished_at_ms=node.finished_at_ms,
                )
            )
            if node.child_run_id:
                child_runs.append(
                    AgentChildRun(
                        run_id=node.child_run_id,
                        agent_name=(
                            step.agent.name
                            if step.agent is not None
                            else step.executor_ref or step.name
                        ),
                        parent_run_id=checkpoint.run_id,
                        status=cast(Any, projected_status),
                        output_text=str(node.output or ""),
                        tool_name=step.name,
                        error=node.error,
                        steps=node.attempt,
                    )
                )
        return AgentRunState(
            run_id=checkpoint.run_id,
            agent_name=self.name,
            provider="workflow",
            model_id=self.definition_version,
            status=cast(Any, status),
            session_id=checkpoint.session_id,
            parent_run_id=checkpoint.parent_run_id,
            idempotency_key=checkpoint.idempotency_key,
            started_at_ms=checkpoint.created_at_ms,
            updated_at_ms=checkpoint.updated_at_ms,
            finished_at_ms=checkpoint.updated_at_ms if status in {"completed", "failed", "cancelled"} else None,
            current_step=sum(node.status in {"completed", "failed", "skipped"} for node in checkpoint.nodes.values()),
            steps=steps,
            child_runs=child_runs,
            output_text=str(last_output or ""),
            error=str(checkpoint.metadata.get("error") or "") or None,
            cancellation_reason=str(checkpoint.metadata.get("cancellation_reason") or "") or None,
            metadata={
                "workflow_name": self.name,
                "workflow_definition_version": self.definition_version,
                "workflow_definition_digest": self.definition_digest,
                "workflow_checkpoint_id": checkpoint.checkpoint_id,
                "workflow_checkpoint_sequence": checkpoint.sequence,
                "workflow_state": dict(checkpoint.state),
                "workflow_edge_decisions": dict(checkpoint.edge_decisions),
                "forked_from_run_id": checkpoint.forked_from_run_id,
                "forked_from_checkpoint_id": checkpoint.forked_from_checkpoint_id,
            },
        )

    async def _persist_projection(self, snapshot: AgentRunState) -> AgentRunState:
        if self.run_store is None:
            return snapshot
        current = await self.run_store.load(snapshot.run_id)
        if current is not None:
            if current.status in {"completed", "failed", "cancelled"}:
                return current
            snapshot.revision = current.revision
        persisted = await self.run_store.save(snapshot)
        return persisted or snapshot

    async def _result(
        self,
        checkpoint: WorkflowCheckpoint,
        *,
        session: AgentSession | None,
    ) -> WorkflowRunResult:
        resolved_session = session or create_agent_session(id=checkpoint.session_id, state=dict(checkpoint.state))
        resolved_session.state = dict(checkpoint.state)
        history = await self.checkpoint_store.list_checkpoints(checkpoint.run_id)
        trace = [
            WorkflowTraceEvent(
                type=item.transition.type,
                workflow_name=self.name,
                step_name=item.transition.node_name,
                status=item.transition.to_status,
                run_id=checkpoint.run_id,
                error=str(item.transition.detail.get("error") or "") or None,
            )
            for item in history
            if item.transition is not None
        ]
        step_results: list[WorkflowStepResult] = []
        for step in self.steps:
            node = checkpoint.nodes[step.name]
            if node.status == "pending":
                continue
            persisted_error = RuntimeError(node.error) if node.error else None
            output_text = node.output if isinstance(node.output, str) else json.dumps(node.output, sort_keys=True)
            step_results.append(
                WorkflowStepResult(
                    name=step.name,
                    status=cast(Any, node.status),
                    error=persisted_error,
                    output_text=output_text or "",
                    agent_run_id=node.child_run_id,
                    attempts=node.attempt,
                )
            )
        snapshot = await self._persist_projection(self._projection(checkpoint))
        text = snapshot.output_text
        return WorkflowRunResult(
            run_id=checkpoint.run_id,
            name=self.name,
            session=resolved_session,
            state=dict(checkpoint.state),
            step_results=step_results,
            text=text,
            status=cast(Any, checkpoint.status),
            trace=trace,
            state_snapshot=snapshot,
            checkpoint=checkpoint,
            forked_from_run_id=checkpoint.forked_from_run_id,
        )


GraphWorkflow = WorkflowGraph


async def resume_workflow(
    workflow: WorkflowGraph,
    run_id: str,
    *,
    interrupt_id: str | None = None,
    resume_value: JsonValue | None = None,
    state_updates: Mapping[str, JsonValue] | None = None,
    approval_id: str | None = None,
    approved: bool = True,
    reason: str | None = None,
    node_name: str | None = None,
    deps: Any = None,
    session: AgentSession | None = None,
) -> WorkflowRunResult:
    return await workflow._with_execution_lease(
        run_id,
        lambda: _resume_workflow_claimed(
            workflow,
            run_id,
            interrupt_id=interrupt_id,
            resume_value=resume_value,
            state_updates=state_updates,
            approval_id=approval_id,
            approved=approved,
            reason=reason,
            node_name=node_name,
            deps=deps,
            session=session,
        ),
    )


async def _resume_workflow_claimed(
    workflow: WorkflowGraph,
    run_id: str,
    *,
    interrupt_id: str | None = None,
    resume_value: JsonValue | None = None,
    state_updates: Mapping[str, JsonValue] | None = None,
    approval_id: str | None = None,
    approved: bool = True,
    reason: str | None = None,
    node_name: str | None = None,
    deps: Any = None,
    session: AgentSession | None = None,
) -> WorkflowRunResult:
    checkpoint = await workflow.checkpoint_store.load_latest(run_id)
    if checkpoint is None:
        raise ValidationError(f'Workflow run "{run_id}" was not found.')
    workflow._validate_checkpoint(checkpoint)
    if checkpoint.status != "suspended":
        raise ValidationError(f'Workflow run "{run_id}" is not suspended.')

    candidate = copy.deepcopy(checkpoint)
    transition_detail: dict[str, JsonValue] = {}
    if candidate.pending_interrupt is not None:
        pending = candidate.pending_interrupt
        if interrupt_id != pending.interrupt_id:
            raise ValidationError(
                f'Workflow interrupt "{pending.interrupt_id}" must be acknowledged explicitly before resume.'
            )
        candidate.resume_values[pending.interrupt_id] = resume_value
        resolved = candidate.metadata.setdefault("resolved_interrupts", [])
        marker = f"{pending.phase}:{pending.node_name}"
        if isinstance(resolved, list) and marker not in resolved:
            resolved.append(marker)
        transition_detail = {
            "interrupt_id": pending.interrupt_id,
            "phase": pending.phase,
            "resume_value": resume_value,
        }
        candidate.pending_interrupt = None
        candidate.status = "running"
    else:
        suspended_nodes = [node for node in candidate.nodes.values() if node.status == "suspended"]
        if node_name is not None:
            suspended_nodes = [node for node in suspended_nodes if node.node_name == node_name]
        if len(suspended_nodes) != 1:
            raise ValidationError(
                "Suspended workflow must identify exactly one suspended step; pass node_name when needed."
            )
        node = suspended_nodes[0]
        if workflow.adapter is not None:
            key = f"step:{node.node_name}"
            candidate.resume_values[key] = resume_value
            node.status = "pending"
            node.error = None
            node.finished_at_ms = None
            node.suspension = None
            candidate.status = "running"
            transition_detail = {
                "node_name": node.node_name,
                "adapter": workflow.adapter.backend,
                "resume_value": resume_value,
            }
            if state_updates:
                candidate.state.update(dict(state_updates))
            checkpoint = await workflow._append(
                checkpoint,
                candidate,
                WorkflowTransition(
                    type="workflow-resume",
                    at_ms=_now_ms(),
                    detail=transition_detail,
                ),
            )
            return await workflow._execute(checkpoint, session=session, deps=deps)
        if not node.child_run_id:
            raise ValidationError(f'Suspended workflow step "{node.node_name}" has no child run id.')
        step = workflow._steps[node.node_name]
        if step.agent is None:
            raise ValidationError(f'Suspended workflow step "{node.node_name}" has no Agent to resume.')
        step_key = node.idempotency_key or workflow._step_idempotency_key(candidate, step)
        step_agent = replace(
            step.agent,
            metadata={
                **step.agent.metadata,
                "zhivex_workflow_step_idempotency_key": step_key,
                "workflow_run_id": candidate.run_id,
                "workflow_step": node.node_name,
            },
        )
        result = await resume_agent_run(
            agent=step_agent,
            run_id=node.child_run_id,
            approval_id=approval_id,
            approved=approved,
            reason=reason,
            deps=deps,
            idempotency_key=f"{step_key}:resume",
        )
        node.child_run_id = result.run_id
        node.output = cast(JsonValue, result.text)
        node.error = None
        node.status = "suspended" if result.state is not None and result.state.status == "suspended" else "completed"
        node.finished_at_ms = _now_ms()
        if node.status == "completed" and step.output_key is not None:
            candidate.state[step.output_key] = cast(JsonValue, result.text)
        if node.status == "completed" and step.metadata_key is not None:
            candidate.state[step.metadata_key] = {
                "name": node.node_name,
                "status": node.status,
                "run_id": node.child_run_id,
                "agent_name": step.agent.name,
                "text": cast(JsonValue, result.text),
                "attempts": node.attempt,
                "error": None,
            }
        completed_order = candidate.metadata.setdefault("completed_order", [])
        if node.status == "completed" and isinstance(completed_order, list) and node.node_name not in completed_order:
            completed_order.append(node.node_name)
        candidate.status = "suspended" if node.status == "suspended" else "running"
        transition_detail = {
            "node_name": node.node_name,
            "child_run_id": node.child_run_id,
            "approved": approved,
        }

    if state_updates:
        candidate.state.update(dict(state_updates))
    checkpoint = await workflow._append(
        checkpoint,
        candidate,
        WorkflowTransition(
            type="workflow-resume",
            at_ms=_now_ms(),
            detail=transition_detail,
        ),
    )
    if checkpoint.status == "suspended":
        return await workflow._result(checkpoint, session=session)
    completed_without_edges = [
        name
        for name, node in checkpoint.nodes.items()
        if node.status == "completed"
        and any(edge.id not in checkpoint.edge_decisions for edge in workflow._outgoing(name))
    ]
    for name in completed_without_edges:
        checkpoint = await workflow._evaluate_edges(checkpoint, name)
    return await workflow._execute(checkpoint, session=session, deps=deps)


async def cancel_workflow(
    workflow: WorkflowGraph,
    run_id: str,
    *,
    reason: str | None = None,
    session: AgentSession | None = None,
) -> WorkflowRunResult:
    checkpoint = await workflow.checkpoint_store.load_latest(run_id)
    if checkpoint is None:
        raise ValidationError(f'Workflow run "{run_id}" was not found.')
    workflow._validate_checkpoint(checkpoint)
    if checkpoint.status in {"completed", "failed", "cancelled"}:
        return await workflow._result(checkpoint, session=session)

    now = _now_ms()
    candidate = copy.deepcopy(checkpoint)
    candidate.status = "cancelled"
    candidate.pending_interrupt = None
    candidate.metadata["cancellation_reason"] = reason
    for node in candidate.nodes.values():
        if node.status in {"running", "suspended"}:
            node.status = "cancelled"
            node.finished_at_ms = now
            node.suspension = None
        elif node.status == "pending":
            node.status = "skipped"
            node.finished_at_ms = now
    try:
        checkpoint = await workflow._append(
            checkpoint,
            candidate,
            WorkflowTransition(
                type="workflow-cancelled",
                at_ms=now,
                from_status=checkpoint.status,
                to_status="cancelled",
                detail={"reason": reason},
            ),
        )
    except ValidationError:
        latest = await workflow.checkpoint_store.load_latest(run_id)
        if latest is None or latest.status != "cancelled":
            raise
        checkpoint = latest
    return await workflow._result(checkpoint, session=session)


async def fork_workflow(
    workflow: WorkflowGraph,
    run_id: str,
    *,
    checkpoint_id: str | None = None,
    state_updates: Mapping[str, JsonValue] | None = None,
    idempotency_key: str | None = None,
    deps: Any = None,
    session: AgentSession | None = None,
) -> WorkflowRunResult:
    source = (
        await workflow.checkpoint_store.load_checkpoint(checkpoint_id)
        if checkpoint_id is not None
        else await workflow.checkpoint_store.load_latest(run_id)
    )
    if source is None or source.run_id != run_id:
        raise ValidationError("Workflow fork checkpoint was not found on the requested run.")
    workflow._validate_checkpoint(source)
    if idempotency_key:
        existing = await workflow.checkpoint_store.find_by_idempotency_key(idempotency_key)
        if existing is not None:
            workflow._validate_checkpoint(existing)
            if existing.status == "running":
                raise ValidationError(
                    f'Workflow fork run "{existing.run_id}" is still active for this idempotency key.'
                )
            return await workflow._result(existing, session=session)

    now = _now_ms()
    candidate = copy.deepcopy(source)
    candidate.checkpoint_id = _new_id("wfc")
    candidate.run_id = _new_id("wf")
    candidate.sequence = 0
    candidate.status = "running"
    candidate.parent_run_id = source.parent_run_id
    candidate.idempotency_key = idempotency_key
    candidate.forked_from_run_id = source.run_id
    candidate.forked_from_checkpoint_id = source.checkpoint_id
    candidate.created_at_ms = now
    candidate.updated_at_ms = now
    candidate.pending_interrupt = None
    candidate.resume_values = {}
    candidate.metadata = {
        **candidate.metadata,
        "resolved_interrupts": [],
        "forked_from_sequence": source.sequence,
    }
    for node in candidate.nodes.values():
        if node.status in {"running", "suspended"}:
            node.status = "pending"
            node.child_run_id = None
            node.error = None
            node.started_at_ms = None
            node.finished_at_ms = None
    if state_updates:
        candidate.state.update(dict(state_updates))
    candidate.transition = WorkflowTransition(
        type="workflow-fork",
        at_ms=now,
        detail={
            "forked_from_run_id": source.run_id,
            "forked_from_checkpoint_id": source.checkpoint_id,
        },
    )
    candidate = await workflow.checkpoint_store.append(candidate)
    return await workflow._execute_with_optional_lease(candidate, session=session, deps=deps)
