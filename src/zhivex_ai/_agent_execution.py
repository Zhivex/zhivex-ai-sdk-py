"""Agent execution helpers."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from ._agent_context import _text_from_message as _text_from_message
from ._agent_contracts import AgentCancellationToken as AgentCancellationToken
from ._agent_contracts import AgentContext as AgentContext
from ._agent_contracts import AgentEvent as AgentEvent
from ._agent_contracts import AgentHooks as AgentHooks
from ._agent_contracts import AgentMemory as AgentMemory
from ._agent_contracts import AgentSession as AgentSession
from ._agent_contracts import AgentTrace as AgentTrace
from ._agent_contracts import GuardrailResult as GuardrailResult
from ._agent_contracts import RunLimits as RunLimits
from ._agent_skills import _skill_system_message
from ._agent_tools import ToolRegistry as ToolRegistry
from .agent_state import AgentRunState, AgentRunStore
from .errors import ToolExecutionOutcomeUnknown, AgentRunCancelled, ValidationError
from .generate_object import _parse_object, _resolve_object_mode
from .messages import create_text_message
from .runtime import current_execution_budget
from .schema import create_schema_adapter
from .skills import SkillDefinition
from .types import (
    GenerateTextStep,
    JsonValue,
    LanguageModel,
    ModelMessage,
    RealtimeModel,
    StructuredOutputConfig,
    TokenUsage,
    ToolSet,
)

if TYPE_CHECKING:
    from .agent import Agent


SUMMARY_MARKER = "Conversation summary:\n"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


async def _persist_agent_run_state(
    store: AgentRunStore, state: AgentRunState
) -> AgentRunState:
    """Persist one CAS revision and surface a winning cancellation explicitly."""

    try:
        persisted = await store.save(state)
    except ValidationError as error:
        current = await store.load(state.run_id)
        if current is not None and current.status == "cancelled":
            raise AgentRunCancelled(
                state.run_id,
                reason=current.cancellation_reason,
            ) from error
        raise
    return persisted if isinstance(persisted, AgentRunState) else state


async def _raise_if_agent_run_cancelled(
    *,
    run_store: AgentRunStore | None,
    run_id: str,
    cancellation_token: AgentCancellationToken | None,
) -> None:
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled(run_id)
    if run_store is None:
        return
    current = await run_store.load(run_id)
    if current is not None and current.status == "cancelled":
        if cancellation_token is not None:
            cancellation_token.cancel(current.cancellation_reason)
        raise AgentRunCancelled(run_id, reason=current.cancellation_reason)


async def _await_with_agent_cancellation(
    awaitable: Awaitable[Any],
    *,
    cancellation_token: AgentCancellationToken | None,
    run_id: str,
) -> Any:
    if cancellation_token is None:
        return await awaitable
    operation_task = asyncio.ensure_future(awaitable)
    try:
        cancellation_token.raise_if_cancelled(run_id)
    except BaseException:
        operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        raise
    cancellation_task = asyncio.create_task(cancellation_token.wait())
    try:
        done, _ = await asyncio.wait(
            {operation_task, cancellation_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancellation_task in done:
            operation_task.cancel()
            outcomes = await asyncio.gather(operation_task, return_exceptions=True)
            if isinstance(outcomes[0], ToolExecutionOutcomeUnknown):
                raise outcomes[0]
            cancellation_token.raise_if_cancelled(run_id)
        return await operation_task
    finally:
        if not operation_task.done():
            operation_task.cancel()
        await asyncio.gather(operation_task, return_exceptions=True)
        if not cancellation_task.done():
            cancellation_task.cancel()
        await asyncio.gather(cancellation_task, return_exceptions=True)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _resolve_agent_instructions(
    agent: Agent[Any, Any],
    context: AgentContext[Any],
) -> str | None:
    instructions = agent.instructions
    if instructions is None or isinstance(instructions, str):
        return instructions
    dynamic = cast(Callable[..., Any], instructions)
    try:
        signature = inspect.signature(dynamic)
    except (TypeError, ValueError):
        value = dynamic(context)
    else:
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        accepts_varargs = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        value = (
            dynamic(context, agent)
            if accepts_varargs or len(positional) >= 2
            else dynamic(context)
        )
    resolved = await _maybe_await(value)
    if resolved is not None and not isinstance(resolved, str):
        raise TypeError("Dynamic agent instructions must return str or None.")
    return cast(str | None, resolved)


def _resolve_agent_structured_output(
    agent: Agent[Any, Any],
    *,
    model: LanguageModel | RealtimeModel | None = None,
) -> tuple[StructuredOutputConfig | None, str | None]:
    if agent.output_type is None:
        return None, None
    output_model = model or agent.model
    if not hasattr(output_model, "capabilities"):
        raise ValidationError(
            "Typed outputs require a language model with declared capabilities."
        )
    output_mode = _resolve_object_mode(
        agent.output_mode,
        bool(output_model.capabilities.structured_output),
    )
    if output_mode == "native":
        return (
            StructuredOutputConfig(
                schema=agent.output_type,
                mode="native",
                name=agent.output_name,
                description=agent.output_description,
            ),
            None,
        )
    schema = create_schema_adapter(agent.output_type).json_schema()
    details = [
        "Return only valid JSON matching this JSON Schema:",
        json.dumps(schema, sort_keys=True, separators=(",", ":")),
    ]
    if agent.output_name:
        details.insert(0, f"Structured output name: {agent.output_name}.")
    if agent.output_description:
        details.insert(0, f"Structured output description: {agent.output_description}")
    return None, "\n".join(details)


def _parse_agent_output(agent: Agent[Any, Any], text: str) -> Any:
    if agent.output_type is None:
        return text
    return _parse_object(text, agent.output_type)


async def _call_agent_hooks(
    hooks: Iterable[AgentHooks],
    method_name: str,
    *args: Any,
    reverse: bool = False,
) -> None:
    ordered = list(hooks)
    if reverse:
        ordered.reverse()
    for hooks_instance in ordered:
        method = getattr(hooks_instance, method_name)
        await _maybe_await(method(*args))


async def _call_error_hooks_preserving(
    hooks: Iterable[AgentHooks],
    context: AgentContext[Any],
    agent: Agent[Any, Any],
    error: Exception,
) -> None:
    try:
        await _call_agent_hooks(hooks, "on_error", context, agent, error, reverse=True)
    except Exception as hook_error:
        error.add_note(f"Agent on_error hook also failed: {hook_error}")


def _effective_max_steps(limits: RunLimits, requested: int | None) -> int | None:
    if limits.max_steps is None:
        return requested
    if requested is None:
        return limits.max_steps
    return min(limits.max_steps, requested)


def _effective_timeout_ms(limits: RunLimits, requested: int | None) -> int | None:
    if limits.max_wall_time_ms is None:
        return requested
    if requested is None:
        return limits.max_wall_time_ms
    return min(limits.max_wall_time_ms, requested)


def _int_from_json(value: JsonValue | None, default: int = 0) -> int:
    if isinstance(value, (str, int, float)):
        return int(value)
    return default


def _should_refresh_summary(memory: AgentMemory, session: AgentSession) -> bool:
    config = memory.summary_config
    if len(session.messages) > config.max_messages:
        return True
    text_length = sum(len(_text_from_message(message)) for message in session.messages)
    return text_length > config.max_summary_chars * 2


def _context_messages(
    session: AgentSession, memory: AgentMemory | None
) -> list[ModelMessage]:
    if memory is None:
        return list(session.messages)
    preserve = max(0, memory.summary_config.preserve_recent_messages)
    if preserve == 0:
        return []
    return list(session.messages[-preserve:])


def _build_run_messages(
    *,
    agent: Agent,
    session: AgentSession,
    prompt: str | None,
    messages: list[ModelMessage] | None,
    active_skills: list[SkillDefinition] | None = None,
    instructions: str | None = None,
    structured_output_instructions: str | None = None,
) -> list[ModelMessage]:
    if prompt is not None and messages is not None:
        raise ValidationError('Pass either "prompt" or "messages", but not both.')

    built: list[ModelMessage] = []
    if instructions:
        built.append(create_text_message("system", instructions))
    for active_skill in active_skills or []:
        built.append(create_text_message("system", _skill_system_message(active_skill)))
    if structured_output_instructions:
        built.append(create_text_message("system", structured_output_instructions))
    if session.summary:
        built.append(
            create_text_message("system", f"{SUMMARY_MARKER}{session.summary}")
        )
    built.extend(_context_messages(session, agent.memory))
    if messages is not None:
        built.extend(messages)
    elif prompt is not None:
        built.append(create_text_message("user", prompt))
    return built


def _guardrail_name(guardrail: Any) -> str:
    if hasattr(guardrail, "__name__"):
        return str(getattr(guardrail, "__name__"))
    return guardrail.__class__.__name__


def _normalize_guardrail_result(
    value: GuardrailResult | bool | None,
) -> GuardrailResult:
    if isinstance(value, GuardrailResult):
        return value
    if isinstance(value, bool):
        return GuardrailResult(tripwire_triggered=value)
    if value is None:
        return GuardrailResult(tripwire_triggered=False)
    raise TypeError("Guardrails must return GuardrailResult, bool, or None.")


def _resolve_tool_registry(
    agent: Agent, extra_tools: ToolSet | ToolRegistry | None
) -> ToolRegistry:
    base = (
        agent.tools
        if isinstance(agent.tools, ToolRegistry)
        else ToolRegistry(agent.tools)
    )
    return base.merge(extra_tools)


def _merge_usage(usages: list[TokenUsage | None]) -> TokenUsage | None:
    present = [usage for usage in usages if usage is not None]
    if not present:
        return None
    input_tokens = (
        sum(usage.input_tokens for usage in present if usage.input_tokens is not None)
        if all(usage.input_tokens is not None for usage in present)
        else None
    )
    output_tokens = (
        sum(usage.output_tokens for usage in present if usage.output_tokens is not None)
        if all(usage.output_tokens is not None for usage in present)
        else None
    )
    total_tokens = (
        sum(usage.total_tokens for usage in present if usage.total_tokens is not None)
        if all(usage.total_tokens is not None for usage in present)
        else None
    )
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _response_messages(step: GenerateTextStep) -> list[ModelMessage]:
    if step.response.messages:
        return list(step.response.messages)
    if step.response.message is not None:
        return [step.response.message]
    if step.response.text:
        return [create_text_message("assistant", step.response.text)]
    return []


def _strip_runtime_system_messages(
    messages: list[ModelMessage], instructions: str | None
) -> list[ModelMessage]:
    stripped = list(messages)
    while stripped and stripped[0].role == "system":
        text = _text_from_message(stripped[0])
        if instructions and text == instructions:
            stripped.pop(0)
            continue
        if text.startswith(SUMMARY_MARKER):
            stripped.pop(0)
            continue
        break
    return stripped


def create_agent_session(
    *,
    id: str | None = None,
    messages: list[ModelMessage] | None = None,
    summary: str | None = None,
    state: dict[str, JsonValue] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentSession:
    return AgentSession(
        id=id or _new_id("session"),
        messages=list(messages or []),
        summary=summary,
        state=dict(state or {}),
        metadata=dict(metadata or {}),
    )


def _record_trace_event(
    trace: AgentTrace, event: AgentEvent, limit: int | None
) -> None:
    trace.events.append(event)
    if limit is not None and len(trace.events) > limit:
        dropped = len(trace.events) - limit
        del trace.events[:dropped]
        trace.events_dropped += dropped


def _execution_error(error: BaseException) -> Exception:
    budget = current_execution_budget()
    if (
        isinstance(error, asyncio.CancelledError)
        and budget is not None
        and time.monotonic() >= budget.deadline
    ):
        return TimeoutError("The total execution budget was exhausted.")
    if not isinstance(error, Exception):
        raise error
    return error
