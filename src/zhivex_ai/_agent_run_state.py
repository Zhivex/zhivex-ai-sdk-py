"""Agent run state implementation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from ._agent_context import _text_from_message as _text_from_message
from ._agent_contracts import HANDOFF_MARKER
from ._agent_contracts import AgentCheckpoint as AgentCheckpoint
from ._agent_contracts import AgentCheckpointEvent as AgentCheckpointEvent
from ._agent_contracts import AgentCheckpointStore as AgentCheckpointStore
from ._agent_contracts import AgentEvent as AgentEvent
from ._agent_contracts import AgentHandoff as AgentHandoff
from ._agent_contracts import AgentRunResult as AgentRunResult
from ._agent_contracts import AgentSession as AgentSession
from ._agent_contracts import AgentTrace as AgentTrace
from ._agent_execution import _int_from_json as _int_from_json
from ._agent_execution import _now_ms as _now_ms
from ._agent_execution import _parse_agent_output as _parse_agent_output
from ._agent_execution import _response_messages as _response_messages
from ._agent_execution import create_agent_session
from ._agent_persistence import (
    _deserialize_agent_checkpoint as _deserialize_agent_checkpoint,
)
from ._agent_persistence import (
    _serialize_agent_checkpoint as _serialize_agent_checkpoint,
)
from ._serde import deserialize_messages, serialize_messages
from .agent_state import (
    AgentChildRun,
    AgentRunState,
    AgentRunStatus,
    AgentRunStep,
    PendingApproval,
)
from .errors import ToolExecutionSuspended
from .messages import create_text_message
from .types import (
    FinishReason,
    GenerateTextOutput,
    GenerateTextStep,
    JsonValue,
    ModelMessage,
    ToolCall,
    ToolCallPart,
    ToolExecutionResult,
    ToolResultPart,
)

if TYPE_CHECKING:
    from .agent import Agent


def _detect_handoff(tool_results: list[ToolExecutionResult]) -> AgentHandoff | None:
    for result in reversed(tool_results):
        if isinstance(result.output, dict) and result.output.get(HANDOFF_MARKER):
            handoff_input = result.output.get("input")
            handoff_metadata = result.output.get("metadata")
            return AgentHandoff(
                target_agent=str(result.output.get("target_agent")),
                input=handoff_input if isinstance(handoff_input, str) else None,
                metadata=dict(handoff_metadata)
                if isinstance(handoff_metadata, dict)
                else {},
            )
    return None


def _assistant_messages_from_result(result: GenerateTextOutput) -> list[ModelMessage]:
    messages = [message for message in result.messages if message.role == "assistant"]
    if messages:
        return messages
    if result.steps:
        collected: list[ModelMessage] = []
        for step in result.steps:
            collected.extend(
                message
                for message in _response_messages(step)
                if message.role == "assistant"
            )
        if collected:
            return collected
    if result.text:
        return [create_text_message("assistant", result.text)]
    return []


def _replace_assistant_messages(
    messages: list[ModelMessage],
    replacements: list[ModelMessage],
) -> list[ModelMessage]:
    resolved = list(messages)
    positions = [
        index for index, message in enumerate(resolved) if message.role == "assistant"
    ]
    if not positions or not replacements:
        return resolved
    positions = positions[-len(replacements) :]
    replacements = replacements[-len(positions) :]
    for index, replacement in zip(positions, replacements, strict=True):
        resolved[index] = replacement
    return resolved


def _apply_guarded_output(
    result: GenerateTextOutput,
    *,
    text: str,
    messages: list[ModelMessage],
) -> None:
    result.text = text
    result.messages = _replace_assistant_messages(result.messages, messages)
    cursor = 0
    for step in result.steps:
        response = step.response
        response_messages = _response_messages(step)
        assistant_count = sum(
            message.role == "assistant" for message in response_messages
        )
        step_replacements = messages[cursor : cursor + assistant_count]
        cursor += assistant_count
        if response.messages is not None:
            response.messages = _replace_assistant_messages(
                response.messages, step_replacements
            )
        elif (
            response.message is not None
            and response.message.role == "assistant"
            and step_replacements
        ):
            response.message = step_replacements[-1]
        if step_replacements:
            response.text = "".join(
                _text_from_message(message) for message in step_replacements
            )
    if result.steps:
        result.steps[-1].response.text = text


def _extract_tool_calls_from_steps(steps: list[GenerateTextStep]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for step in steps:
        response_messages = step.response.messages or (
            [step.response.message] if step.response.message else []
        )
        for message in response_messages:
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    calls.append(part.tool_call)
    return calls


def _extract_provider_tool_results_from_steps(
    steps: list[GenerateTextStep],
) -> list[ToolExecutionResult]:
    results: list[ToolExecutionResult] = []
    for step in steps:
        response_messages = step.response.messages or (
            [step.response.message] if step.response.message else []
        )
        for message in response_messages:
            for part in message.parts:
                if isinstance(part, ToolResultPart):
                    results.append(part.tool_result)
    return results


def _step_status_from_result(result: GenerateTextOutput) -> AgentRunStatus:
    return "failed" if result.finish_reason == "error" else "completed"


def _child_runs_from_tool_results(
    results: list[ToolExecutionResult], parent_run_id: str
) -> list[AgentChildRun]:
    children: list[AgentChildRun] = []
    for result in results:
        output = result.output
        if not isinstance(output, dict):
            continue
        raw_child = output.get("child_run")
        if not isinstance(raw_child, dict):
            continue
        children.append(
            AgentChildRun(
                run_id=str(raw_child.get("run_id", "")),
                agent_name=str(raw_child.get("agent_name", "")),
                parent_run_id=str(raw_child.get("parent_run_id") or parent_run_id),
                status=cast(AgentRunStatus, raw_child.get("status", "completed")),
                output_text=str(raw_child.get("output_text", "")),
                tool_name=result.tool_name,
                error=str(raw_child.get("error"))
                if raw_child.get("error") is not None
                else None,
                steps=_int_from_json(raw_child.get("steps")),
                tool_calls=_int_from_json(raw_child.get("tool_calls")),
                tool_errors=_int_from_json(raw_child.get("tool_errors")),
            )
        )
    return children


def _agent_run_state_from_result(
    *,
    result: AgentRunResult,
    agent: Agent,
    parent_run_id: str | None,
    idempotency_key: str | None,
    started_at_ms: int,
    finished_at_ms: int,
    error: str | None = None,
) -> AgentRunState:
    steps = [
        AgentRunStep(
            index=index,
            status=_step_status_from_result(
                GenerateTextOutput(
                    text=step.response.text or "",
                    finish_reason=step.response.finish_reason,
                )
            ),
            tool_calls=_extract_tool_calls_from_steps([step]),
            tool_results=[],
            usage=step.response.usage,
            messages=_response_messages(step),
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
        )
        for index, step in enumerate(result.steps, start=1)
    ]
    status: AgentRunStatus = (
        "failed" if error or result.finish_reason == "error" else "completed"
    )
    state = AgentRunState(
        run_id=result.run_id,
        agent_name=result.agent_name,
        provider=str(getattr(agent.model, "provider", "")),
        model_id=str(getattr(agent.model, "model_id", "")),
        status=status,
        session_id=result.session.id,
        parent_run_id=parent_run_id,
        idempotency_key=idempotency_key,
        started_at_ms=started_at_ms,
        updated_at_ms=finished_at_ms,
        finished_at_ms=finished_at_ms,
        current_step=len(result.steps),
        steps=steps,
        child_runs=_child_runs_from_tool_results(result.tool_results, result.run_id),
        tool_results=list(result.tool_results),
        usage=result.usage,
        output_text=result.text,
        finish_reason=result.finish_reason,
        error=error,
        metadata={
            "orchestration_path": list(result.orchestration_path),
            "session_messages": cast(
                JsonValue, serialize_messages(result.session.messages)
            ),
        },
    )
    return state


def _agent_run_state_from_suspension(
    *,
    run_id: str,
    agent_name: str,
    session_id: str,
    agent: Agent,
    parent_run_id: str | None,
    idempotency_key: str | None,
    started_at_ms: int,
    suspended_at_ms: int,
    suspended: ToolExecutionSuspended,
    orchestration_path: list[str],
) -> AgentRunState:
    steps = [
        AgentRunStep(
            index=index,
            status="suspended" if index == len(suspended.steps) else "completed",
            tool_calls=_extract_tool_calls_from_steps([step]),
            tool_results=[],
            usage=step.response.usage,
            messages=_response_messages(step),
            started_at_ms=started_at_ms,
            finished_at_ms=suspended_at_ms if index == len(suspended.steps) else None,
        )
        for index, step in enumerate(
            cast(list[GenerateTextStep], suspended.steps), start=1
        )
    ]
    pending = cast(PendingApproval, suspended.pending_approval)
    return AgentRunState(
        run_id=run_id,
        agent_name=agent_name,
        provider=str(getattr(agent.model, "provider", "")),
        model_id=str(getattr(agent.model, "model_id", "")),
        status="suspended",
        session_id=session_id,
        parent_run_id=parent_run_id,
        idempotency_key=idempotency_key,
        started_at_ms=started_at_ms,
        updated_at_ms=suspended_at_ms,
        current_step=len(steps),
        steps=steps,
        pending_approvals=[pending],
        tool_results=list(cast(list[ToolExecutionResult], suspended.tool_results)),
        finish_reason="tool-calls",
        metadata={
            "orchestration_path": list(orchestration_path),
            "resume_messages": cast(
                JsonValue,
                serialize_messages(cast(list[ModelMessage], suspended.messages)),
            ),
        },
    )


def _segment_text(result: GenerateTextOutput) -> str:
    if result.steps and result.steps[-1].response.text:
        return result.steps[-1].response.text or ""
    return result.text


def _segment_finish_reason(result: GenerateTextOutput) -> FinishReason | None:
    if result.steps:
        return result.steps[-1].response.finish_reason
    return result.finish_reason


def _segment_provider_finish_reason(result: GenerateTextOutput) -> str | None:
    if result.steps:
        return result.steps[-1].response.provider_finish_reason
    return result.provider_finish_reason


async def _save_checkpoints(
    *,
    checkpoint_store: AgentCheckpointStore | None,
    result: GenerateTextOutput,
    run_id: str,
    session_id: str,
    agent_name: str,
    emit: Callable[[AgentEvent], Awaitable[None]] | None = None,
    trace: AgentTrace | None = None,
) -> None:
    if checkpoint_store is None:
        return
    for index, step in enumerate(result.steps, start=1):
        checkpoint = AgentCheckpoint(
            run_id=run_id,
            session_id=session_id,
            agent_name=agent_name,
            step_index=index,
            request=step.request,
            response=step.response,
            saved_at_ms=_now_ms(),
            is_final=index == len(result.steps),
        )
        # Emit and persist the same sanitized checkpoint so credentials and
        # raw provider payloads cannot leak through observers or trace events.
        checkpoint = _deserialize_agent_checkpoint(
            _serialize_agent_checkpoint(checkpoint)
        )
        await checkpoint_store.save(checkpoint)
        if trace is not None:
            trace.checkpoint_count += 1
        if emit is not None:
            await emit(AgentCheckpointEvent(checkpoint=checkpoint))


def _agent_run_result_from_state(
    state: AgentRunState,
    fallback_session: AgentSession,
    *,
    agent: Agent[Any, Any],
) -> AgentRunResult[Any]:
    raw_path = state.metadata.get("orchestration_path")
    orchestration_path = (
        [str(item) for item in raw_path]
        if isinstance(raw_path, list) and raw_path
        else [state.agent_name]
    )
    raw_session_messages = state.metadata.get("session_messages")
    if isinstance(raw_session_messages, list):
        session_messages = deserialize_messages(
            cast(list[dict[str, Any]], raw_session_messages)
        )
    else:
        session_messages = [
            message for step in state.steps for message in step.messages
        ]
    session = create_agent_session(
        id=state.session_id or fallback_session.id,
        messages=session_messages,
        summary=fallback_session.summary,
        metadata={**fallback_session.metadata, "idempotency_reused": True},
    )
    return AgentRunResult(
        run_id=state.run_id,
        agent_name=state.agent_name,
        session=session,
        text=state.output_text,
        finish_reason=state.finish_reason,
        usage=state.usage,
        tool_results=list(state.tool_results),
        orchestration_path=orchestration_path,
        state=state,
        output=_parse_agent_output(agent, state.output_text)
        if state.status == "completed"
        else None,
    )
