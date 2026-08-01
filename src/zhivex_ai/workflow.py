from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from string import Formatter
from typing import TYPE_CHECKING, Any, Literal, Protocol

from .agent import Agent, AgentRunResult, AgentSession, create_agent_session, run_agent
from .agent_state import AgentChildRun, AgentRunState, AgentRunStep, AgentRunStore
from .errors import ValidationError
from .types import JsonValue

if TYPE_CHECKING:
    from .workflow_state import WorkflowCheckpoint

WorkflowState = dict[str, JsonValue]
WorkflowErrorPolicy = Literal["fail_fast", "continue", "capture"]
WorkflowRunStatus = Literal["completed", "failed", "suspended", "cancelled"]
WorkflowStepStatus = Literal["completed", "failed", "suspended", "cancelled", "skipped"]
WorkflowStopCondition = Callable[["WorkflowRunResult"], bool | Awaitable[bool]]
WorkflowRetryPredicate = Callable[[Exception], bool | Awaitable[bool]]


@dataclass(slots=True, frozen=True)
class WorkflowFunctionContext:
    run_id: str
    workflow_name: str
    step_name: str
    attempt: int
    idempotency_key: str
    input: JsonValue
    state: Mapping[str, JsonValue]
    resume_values: Mapping[str, JsonValue] = field(default_factory=dict)
    deps: Any = field(default=None, repr=False, compare=False)


@dataclass(slots=True, frozen=True)
class WorkflowFunctionResult:
    output: JsonValue = None
    state_patch: Mapping[str, JsonValue] = field(default_factory=dict)
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)


WorkflowFunctionExecutor = Callable[
    [WorkflowFunctionContext],
    JsonValue | WorkflowFunctionResult | Awaitable[JsonValue | WorkflowFunctionResult],
]


@dataclass(slots=True, frozen=True)
class WorkflowRetryPolicy:
    """Retry policy for a complete logical workflow step.

    ``WorkflowStep.max_retries`` remains the provider/model retry setting passed
    to ``run_agent``. This policy is deliberately separate because retrying a
    complete step can repeat tools or external side effects.
    """

    max_attempts: int = 1
    backoff_ms: int = 250
    max_backoff_ms: int = 5_000
    retry_if: WorkflowRetryPredicate | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts <= 0:
            raise ValidationError("WorkflowRetryPolicy.max_attempts must be greater than zero.")
        if isinstance(self.backoff_ms, bool) or not isinstance(self.backoff_ms, int) or self.backoff_ms < 0:
            raise ValidationError("WorkflowRetryPolicy.backoff_ms must be zero or greater.")
        if (
            isinstance(self.max_backoff_ms, bool)
            or not isinstance(self.max_backoff_ms, int)
            or self.max_backoff_ms < self.backoff_ms
        ):
            raise ValidationError(
                "WorkflowRetryPolicy.max_backoff_ms must be greater than or equal to backoff_ms."
            )
        if self.retry_if is not None and not callable(self.retry_if):
            raise ValidationError("WorkflowRetryPolicy.retry_if must be callable.")


@dataclass(slots=True)
class WorkflowStep:
    name: str
    agent: Agent | None = None
    prompt: str | None = None
    input_template: str | None = None
    output_key: str | None = None
    metadata_key: str | None = None
    max_retries: int | None = None
    timeout_ms: int | None = None
    error_policy: WorkflowErrorPolicy = "fail_fast"
    retry_policy: WorkflowRetryPolicy | None = None
    idempotency_key: str | None = None
    executor_ref: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    executor: WorkflowFunctionExecutor | None = None


@dataclass(slots=True)
class WorkflowStepResult:
    name: str
    status: WorkflowStepStatus
    output: AgentRunResult | None = None
    error: Exception | None = None
    iteration: int | None = None
    output_text: str = ""
    agent_run_id: str | None = None
    attempts: int = 1


@dataclass(slots=True)
class WorkflowTraceEvent:
    type: str
    workflow_name: str
    step_name: str | None = None
    status: str | None = None
    iteration: int | None = None
    run_id: str | None = None
    error: str | None = None


@dataclass(slots=True)
class WorkflowRunResult:
    run_id: str
    name: str
    session: AgentSession
    state: WorkflowState
    step_results: list[WorkflowStepResult]
    text: str = ""
    status: WorkflowRunStatus = "completed"
    trace: list[WorkflowTraceEvent] = field(default_factory=list)
    state_snapshot: AgentRunState | None = None
    checkpoint: WorkflowCheckpoint | None = None
    forked_from_run_id: str | None = None


class WorkflowAgent(Protocol):
    name: str

    async def run(
        self,
        *,
        session: AgentSession | None = None,
        prompt: str | None = None,
        parent_run_id: str | None = None,
    ) -> WorkflowRunResult: ...


def _workflow_run_id(name: str) -> str:
    from uuid import uuid4

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower() or "workflow"
    return f"wf_{slug}_{uuid4().hex[:10]}"


def _template_fields(template: str) -> list[str]:
    fields: list[str] = []
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name:
            fields.append(field_name)
    return fields


def _render_template(template: str, state: WorkflowState) -> str:
    missing = [field for field in _template_fields(template) if field not in state]
    if missing:
        raise ValidationError(f"Workflow input_template is missing state keys: {', '.join(missing)}.")
    return template.format(**state)


def _step_prompt(step: WorkflowStep, state: WorkflowState, fallback_prompt: str | None) -> str | None:
    if step.input_template is not None:
        return _render_template(step.input_template, state)
    if step.prompt is not None:
        return step.prompt
    return fallback_prompt


def _step_metadata(result: WorkflowStepResult) -> dict[str, JsonValue]:
    output = result.output
    return {
        "name": result.name,
        "status": result.status,
        "run_id": output.run_id if output is not None else result.agent_run_id,
        "agent_name": output.agent_name if output is not None else None,
        "text": output.text if output is not None else result.output_text,
        "error": str(result.error) if result.error is not None else None,
        "attempts": result.attempts,
    }


def _validate_parallel_output_keys(steps: Sequence[WorkflowStep]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for step in steps:
        if not step.output_key:
            continue
        if step.output_key in seen:
            duplicates.add(step.output_key)
        seen.add(step.output_key)
    if duplicates:
        raise ValidationError(f"Parallel workflow output_key values must be unique: {', '.join(sorted(duplicates))}.")


def _failure_status(step_results: list[WorkflowStepResult]) -> WorkflowRunStatus:
    if any(result.status == "failed" for result in step_results):
        return "failed"
    if any(result.status == "suspended" for result in step_results):
        return "suspended"
    return "completed"


def _state_snapshot(
    *,
    run_id: str,
    name: str,
    status: WorkflowRunStatus,
    parent_run_id: str | None,
    step_results: list[WorkflowStepResult],
    state: WorkflowState,
) -> AgentRunState:
    def projected_status(value: WorkflowStepStatus) -> Literal[
        "running", "completed", "failed", "cancelled", "suspended"
    ]:
        return "completed" if value == "skipped" else value

    return AgentRunState(
        run_id=run_id,
        agent_name=name,
        provider="workflow",
        model_id=name,
        status=status,
        parent_run_id=parent_run_id,
        current_step=len(step_results),
        output_text=next((result.output.text for result in reversed(step_results) if result.output is not None), ""),
        steps=[
            AgentRunStep(
                index=index,
                status=projected_status(result.status),
                error=str(result.error) if result.error is not None else None,
            )
            for index, result in enumerate(step_results, start=1)
        ],
        child_runs=[
            AgentChildRun(
                run_id=result.output.run_id,
                agent_name=result.output.agent_name,
                parent_run_id=run_id,
                status=(
                    result.output.state.status
                    if result.output.state is not None
                    else projected_status(result.status)
                ),
                output_text=result.output.text,
                tool_name=result.name,
                error=str(result.error) if result.error is not None else None,
                steps=len(result.output.steps),
                tool_calls=len(result.output.tool_results),
                tool_errors=sum(1 for item in result.output.tool_results if item.is_error),
                usage=result.output.usage,
            )
            for result in step_results
            if result.output is not None
        ],
        error="; ".join(str(result.error) for result in step_results if result.error is not None) or None,
        metadata={
            "workflow_name": name,
            "workflow_steps": [result.name for result in step_results],
            "workflow_step_runs": [result.output.run_id for result in step_results if result.output is not None],
            "failed_steps": [result.name for result in step_results if result.status == "failed"],
            "suspended_steps": [result.name for result in step_results if result.status == "suspended"],
            "state": dict(state),
        },
    )


class _BaseWorkflow:
    def __init__(self, *, name: str, run_store: AgentRunStore | None = None) -> None:
        self.name = name
        self.run_store = run_store

    async def _run_step(
        self,
        step: WorkflowStep,
        *,
        session: AgentSession,
        fallback_prompt: str | None,
        parent_run_id: str | None,
        iteration: int | None = None,
    ) -> WorkflowStepResult:
        try:
            if step.agent is None:
                raise ValidationError(
                    "SequentialAgent, ParallelAgent, and LoopAgent require an Agent on every WorkflowStep. "
                    "Use WorkflowGraph for functional steps."
                )
            output = await run_agent(
                agent=step.agent,
                session=session,
                prompt=_step_prompt(step, session.state, fallback_prompt),
                parent_run_id=parent_run_id,
                timeout_ms=step.timeout_ms,
                max_retries=step.max_retries,
            )
            status: WorkflowRunStatus = (
                "suspended" if output.state is not None and output.state.status == "suspended" else "completed"
            )
            if status == "completed" and step.output_key is not None:
                session.state[step.output_key] = output.text
            result = WorkflowStepResult(name=step.name, status=status, output=output, iteration=iteration)
            if step.metadata_key is not None:
                session.state[step.metadata_key] = _step_metadata(result)
            return result
        except Exception as error:
            if step.error_policy == "fail_fast":
                raise
            result = WorkflowStepResult(name=step.name, status="failed", error=error, iteration=iteration)
            if step.metadata_key is not None:
                session.state[step.metadata_key] = _step_metadata(result)
            if step.error_policy == "capture" and step.output_key is not None:
                session.state[step.output_key] = {"error": str(error), "step": step.name}
            return result

    def _result(
        self,
        *,
        run_id: str,
        session: AgentSession,
        step_results: list[WorkflowStepResult],
        trace: list[WorkflowTraceEvent],
        parent_run_id: str | None,
    ) -> WorkflowRunResult:
        status = _failure_status(step_results)
        text = next((result.output.text for result in reversed(step_results) if result.output is not None), "")
        snapshot = _state_snapshot(
            run_id=run_id,
            name=self.name,
            status=status,
            parent_run_id=parent_run_id,
            step_results=step_results,
            state=session.state,
        )
        trace.append(WorkflowTraceEvent("workflow-finish", self.name, status=status, run_id=run_id))
        return WorkflowRunResult(
            run_id=run_id,
            name=self.name,
            session=session,
            state=dict(session.state),
            step_results=step_results,
            text=text,
            status=status,
            trace=trace,
            state_snapshot=snapshot,
        )

    async def _finish(
        self,
        *,
        run_id: str,
        session: AgentSession,
        step_results: list[WorkflowStepResult],
        trace: list[WorkflowTraceEvent],
        parent_run_id: str | None,
    ) -> WorkflowRunResult:
        result = self._result(
            run_id=run_id,
            session=session,
            step_results=step_results,
            trace=trace,
            parent_run_id=parent_run_id,
        )
        if self.run_store is not None and result.state_snapshot is not None:
            await self.run_store.save(result.state_snapshot)
        return result


class SequentialAgent(_BaseWorkflow):
    def __init__(self, *, name: str, steps: Sequence[WorkflowStep], run_store: AgentRunStore | None = None) -> None:
        super().__init__(name=name, run_store=run_store)
        self.steps = list(steps)

    async def run(
        self,
        *,
        session: AgentSession | None = None,
        prompt: str | None = None,
        parent_run_id: str | None = None,
    ) -> WorkflowRunResult:
        resolved_session = session or create_agent_session()
        run_id = _workflow_run_id(self.name)
        trace = [WorkflowTraceEvent("workflow-start", self.name, status="running", run_id=run_id)]
        results: list[WorkflowStepResult] = []
        for step in self.steps:
            trace.append(WorkflowTraceEvent("workflow-step-start", self.name, step.name, "running", run_id=run_id))
            try:
                result = await self._run_step(step, session=resolved_session, fallback_prompt=prompt, parent_run_id=run_id)
            except Exception as error:
                result = WorkflowStepResult(name=step.name, status="failed", error=error)
                results.append(result)
                trace.append(WorkflowTraceEvent("workflow-step-finish", self.name, step.name, "failed", run_id=run_id, error=str(error)))
                return await self._finish(run_id=run_id, session=resolved_session, step_results=results, trace=trace, parent_run_id=parent_run_id)
            results.append(result)
            trace.append(WorkflowTraceEvent("workflow-step-finish", self.name, step.name, result.status, run_id=run_id, error=str(result.error) if result.error else None))
            if result.status == "suspended":
                return await self._finish(run_id=run_id, session=resolved_session, step_results=results, trace=trace, parent_run_id=parent_run_id)
        return await self._finish(run_id=run_id, session=resolved_session, step_results=results, trace=trace, parent_run_id=parent_run_id)


class ParallelAgent(_BaseWorkflow):
    def __init__(self, *, name: str, steps: Sequence[WorkflowStep], run_store: AgentRunStore | None = None) -> None:
        super().__init__(name=name, run_store=run_store)
        _validate_parallel_output_keys(steps)
        self.steps = list(steps)

    async def run(
        self,
        *,
        session: AgentSession | None = None,
        prompt: str | None = None,
        parent_run_id: str | None = None,
    ) -> WorkflowRunResult:
        resolved_session = session or create_agent_session()
        run_id = _workflow_run_id(self.name)
        trace = [WorkflowTraceEvent("workflow-start", self.name, status="running", run_id=run_id)]
        base_state = dict(resolved_session.state)

        async def run_isolated(step: WorkflowStep) -> tuple[WorkflowStepResult, WorkflowState]:
            isolated_session = create_agent_session(
                id=resolved_session.id,
                messages=list(resolved_session.messages),
                summary=resolved_session.summary,
                state=dict(base_state),
                metadata=dict(resolved_session.metadata),
            )
            result = await self._run_step(step, session=isolated_session, fallback_prompt=prompt, parent_run_id=run_id)
            return result, dict(isolated_session.state)

        trace.extend(WorkflowTraceEvent("workflow-step-start", self.name, step.name, "running", run_id=run_id) for step in self.steps)
        tasks = [asyncio.create_task(run_isolated(step)) for step in self.steps]
        task_indexes = {task: index for index, task in enumerate(tasks)}
        resolved: dict[int, tuple[WorkflowStepResult, WorkflowState]] = {}
        pending = set(tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                fail_fast_triggered = False
                for task in done:
                    index = task_indexes[task]
                    step = self.steps[index]
                    try:
                        resolved[index] = task.result()
                    except Exception as error:
                        resolved[index] = (WorkflowStepResult(name=step.name, status="failed", error=error), base_state)
                        fail_fast_triggered = fail_fast_triggered or step.error_policy == "fail_fast"
                if not fail_fast_triggered:
                    continue
                cancelled_indexes = [task_indexes[task] for task in pending]
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for index in cancelled_indexes:
                    step = self.steps[index]
                    resolved[index] = (
                        WorkflowStepResult(
                            name=step.name,
                            status="failed",
                            error=RuntimeError("Cancelled because another parallel step failed fast."),
                        ),
                        base_state,
                    )
                pending.clear()
        finally:
            unfinished = [task for task in tasks if not task.done()]
            for task in unfinished:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        results: list[WorkflowStepResult] = []
        for index, step in enumerate(self.steps):
            result, isolated_state = resolved[index]
            if step.output_key is not None and step.output_key in isolated_state:
                resolved_session.state[step.output_key] = isolated_state[step.output_key]
            if step.metadata_key is not None and step.metadata_key in isolated_state:
                resolved_session.state[step.metadata_key] = isolated_state[step.metadata_key]
            results.append(result)
            trace.append(WorkflowTraceEvent("workflow-step-finish", self.name, step.name, result.status, run_id=run_id, error=str(result.error) if result.error else None))
        return await self._finish(run_id=run_id, session=resolved_session, step_results=results, trace=trace, parent_run_id=parent_run_id)


class LoopAgent(_BaseWorkflow):
    def __init__(
        self,
        *,
        name: str,
        steps: Sequence[WorkflowStep],
        max_iterations: int,
        stop_condition: WorkflowStopCondition | None = None,
        run_store: AgentRunStore | None = None,
    ) -> None:
        super().__init__(name=name, run_store=run_store)
        if max_iterations <= 0:
            raise ValidationError("LoopAgent max_iterations must be greater than zero.")
        self.steps = list(steps)
        self.max_iterations = max_iterations
        self.stop_condition = stop_condition

    async def run(
        self,
        *,
        session: AgentSession | None = None,
        prompt: str | None = None,
        parent_run_id: str | None = None,
    ) -> WorkflowRunResult:
        resolved_session = session or create_agent_session()
        run_id = _workflow_run_id(self.name)
        trace = [WorkflowTraceEvent("workflow-start", self.name, status="running", run_id=run_id)]
        results: list[WorkflowStepResult] = []
        for iteration in range(1, self.max_iterations + 1):
            for step in self.steps:
                trace.append(WorkflowTraceEvent("workflow-step-start", self.name, step.name, "running", iteration, run_id))
                try:
                    result = await self._run_step(
                        step,
                        session=resolved_session,
                        fallback_prompt=prompt,
                        parent_run_id=run_id,
                        iteration=iteration,
                    )
                except Exception as error:
                    result = WorkflowStepResult(name=step.name, status="failed", error=error, iteration=iteration)
                    results.append(result)
                    trace.append(WorkflowTraceEvent("workflow-step-finish", self.name, step.name, "failed", iteration, run_id, str(error)))
                    return await self._finish(run_id=run_id, session=resolved_session, step_results=results, trace=trace, parent_run_id=parent_run_id)
                results.append(result)
                trace.append(WorkflowTraceEvent("workflow-step-finish", self.name, step.name, result.status, iteration, run_id, str(result.error) if result.error else None))
                if result.status == "suspended":
                    return await self._finish(run_id=run_id, session=resolved_session, step_results=results, trace=trace, parent_run_id=parent_run_id)
            partial = self._result(run_id=run_id, session=resolved_session, step_results=list(results), trace=list(trace), parent_run_id=parent_run_id)
            if self.stop_condition is not None:
                decision = self.stop_condition(partial)
                if await decision if isinstance(decision, Awaitable) else decision:
                    if self.run_store is not None and partial.state_snapshot is not None:
                        await self.run_store.save(partial.state_snapshot)
                    return partial
        return await self._finish(run_id=run_id, session=resolved_session, step_results=results, trace=trace, parent_run_id=parent_run_id)


def workflow_step(
    name: str,
    agent: Agent | None = None,
    *,
    prompt: str | None = None,
    input_template: str | None = None,
    output_key: str | None = None,
    metadata_key: str | None = None,
    max_retries: int | None = None,
    timeout_ms: int | None = None,
    error_policy: WorkflowErrorPolicy = "fail_fast",
    retry_policy: WorkflowRetryPolicy | None = None,
    idempotency_key: str | None = None,
    executor_ref: str | None = None,
    metadata: dict[str, JsonValue] | None = None,
    executor: WorkflowFunctionExecutor | None = None,
) -> WorkflowStep:
    return WorkflowStep(
        name=name,
        agent=agent,
        prompt=prompt,
        input_template=input_template,
        output_key=output_key,
        metadata_key=metadata_key,
        max_retries=max_retries,
        timeout_ms=timeout_ms,
        error_policy=error_policy,
        retry_policy=retry_policy,
        idempotency_key=idempotency_key,
        executor_ref=executor_ref,
        metadata=dict(metadata or {}),
        executor=executor,
    )


def validate_workflow_expectations(result: WorkflowRunResult, expectations: object) -> list[str]:
    workflow_steps = getattr(expectations, "workflow_steps", [])
    state_contains = getattr(expectations, "state_contains", [])
    state_equals = getattr(expectations, "state_equals", {})
    failed_steps = getattr(expectations, "failed_steps", [])
    failures: list[str] = []
    actual_steps = [item.name for item in result.step_results]
    actual_failed_steps = [item.name for item in result.step_results if item.status == "failed"]
    if workflow_steps and actual_steps != list(workflow_steps):
        failures.append(f"Expected workflow steps {list(workflow_steps)}, got {actual_steps}.")
    if failed_steps and actual_failed_steps != list(failed_steps):
        failures.append(f"Expected failed workflow steps {list(failed_steps)}, got {actual_failed_steps}.")
    for key in list(state_contains or []):
        if key not in result.state:
            failures.append(f'Expected workflow state key "{key}".')
    for key, expected in dict(state_equals or {}).items():
        if result.state.get(key) != expected:
            failures.append(f'Expected workflow state "{key}" to equal {expected!r}.')
    return failures


async def run_workflow(
    workflow: WorkflowAgent,
    *,
    session: AgentSession | None = None,
    prompt: str | None = None,
    parent_run_id: str | None = None,
) -> WorkflowRunResult:
    return await workflow.run(session=session, prompt=prompt, parent_run_id=parent_run_id)
