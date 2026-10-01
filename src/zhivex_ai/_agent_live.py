"""Agent live implementation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Iterable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from ._agent_context import _text_from_message as _text_from_message
from ._agent_contracts import AgentCancellationToken as AgentCancellationToken
from ._agent_contracts import AgentContext as AgentContext
from ._agent_contracts import AgentDepsT as AgentDepsT
from ._agent_contracts import AgentErrorEvent as AgentErrorEvent
from ._agent_contracts import AgentEvent as AgentEvent
from ._agent_contracts import AgentFinishEvent as AgentFinishEvent
from ._agent_contracts import AgentHooks as AgentHooks
from ._agent_contracts import AgentLiveEvent as AgentLiveEvent
from ._agent_contracts import AgentMemoryState as AgentMemoryState
from ._agent_contracts import AgentMiddleware as AgentMiddleware
from ._agent_contracts import AgentOutputT as AgentOutputT
from ._agent_contracts import AgentRunRequest as AgentRunRequest
from ._agent_contracts import AgentRunResult as AgentRunResult
from ._agent_contracts import AgentRunStartEvent as AgentRunStartEvent
from ._agent_contracts import AgentSession as AgentSession
from ._agent_contracts import AgentSummaryUpdateEvent as AgentSummaryUpdateEvent
from ._agent_contracts import AgentTextDeltaEvent as AgentTextDeltaEvent
from ._agent_contracts import AgentToolCallEvent as AgentToolCallEvent
from ._agent_contracts import AgentToolResultEvent as AgentToolResultEvent
from ._agent_contracts import AgentTrace as AgentTrace
from ._agent_execution import (
    _await_with_agent_cancellation as _await_with_agent_cancellation,
)
from ._agent_execution import _build_run_messages as _build_run_messages
from ._agent_execution import _call_agent_hooks as _call_agent_hooks
from ._agent_execution import (
    _call_error_hooks_preserving as _call_error_hooks_preserving,
)
from ._agent_execution import (
    _execution_error,
    _record_trace_event,
    _strip_runtime_system_messages,
    create_agent_session,
)
from ._agent_execution import _maybe_await as _maybe_await
from ._agent_execution import _new_id as _new_id
from ._agent_execution import _now_ms as _now_ms
from ._agent_execution import _parse_agent_output as _parse_agent_output
from ._agent_execution import _persist_agent_run_state as _persist_agent_run_state
from ._agent_execution import (
    _raise_if_agent_run_cancelled as _raise_if_agent_run_cancelled,
)
from ._agent_execution import _resolve_agent_instructions as _resolve_agent_instructions
from ._agent_execution import (
    _resolve_agent_structured_output as _resolve_agent_structured_output,
)
from ._agent_execution import _resolve_tool_registry as _resolve_tool_registry
from ._agent_execution import _should_refresh_summary as _should_refresh_summary
from ._agent_run_state import (
    _agent_run_result_from_state,
    _agent_run_state_from_result,
    _agent_run_state_from_suspension,
    _replace_assistant_messages,
)
from ._agent_skills import _emit_skill_events as _emit_skill_events
from ._agent_skills import _persist_active_skills as _persist_active_skills
from ._agent_skills import _resolve_skill_registry as _resolve_skill_registry
from ._agent_skills import _select_active_skills as _select_active_skills
from ._agent_skills import _skill_system_message as _skill_system_message
from ._agent_streams import LiveAgentStreamResult
from ._agent_tools import ToolRegistry as ToolRegistry
from ._streaming import DEFAULT_STREAM_BUFFER_SIZE, Broadcast
from .agent_state import AgentRunState, PendingApproval
from .errors import AgentRunCancelled, ToolExecutionSuspended, ValidationError
from .messages import create_text_message, is_callable_tool_definition, tool_result_part
from .runtime import current_execution_budget, execution_scope
from .skills import SkillRegistry, SkillSet
from .types import (
    GenerateTextStep,
    ModelMessage,
    RealtimeAudioOutputEvent,
    RealtimeConnectOptions,
    RealtimeErrorEvent,
    RealtimeEvent,
    RealtimeModel,
    RealtimeResponseCompletedEvent,
    RealtimeSessionConfig,
    RealtimeSessionEndedEvent,
    RealtimeTextDeltaEvent,
    RealtimeToolCallEvent,
    RealtimeToolResultEvent,
    RealtimeTranscriptEvent,
    ToolCall,
    ToolCallPart,
    ToolChoiceName,
    ToolExecutionError,
    ToolExecutionOptions,
    ToolExecutionResult,
    ToolSet,
)

if TYPE_CHECKING:
    from .agent import Agent, AgentObserver, AgentRegistry, AgentRuntime


def stream_live_agent_impl(
    *,
    agent: Agent[AgentDepsT, AgentOutputT],
    session: AgentSession | None = None,
    deps: AgentDepsT | None = None,
    tools: ToolSet | ToolRegistry | None = None,
    skills: SkillSet | SkillRegistry | None = None,
    tool_choice: str | ToolChoiceName | None = None,
    tool_execution: ToolExecutionOptions | None = None,
    connect_options: RealtimeConnectOptions | None = None,
    realtime_config: RealtimeSessionConfig | None = None,
    provider_options: dict[str, Any] | None = None,
    runtime: AgentRuntime | None = None,
    registry: AgentRegistry | None = None,
    observer: AgentObserver | None = None,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    parent_run_id: str | None = None,
    idempotency_key: str | None = None,
    cancellation_token: AgentCancellationToken | None = None,
    hooks: Iterable[AgentHooks] | None = None,
    middleware: Iterable[AgentMiddleware] | None = None,
    total_timeout_ms: int | None = None,
    retry_jitter: float | None = None,
    stream_buffer_size: int | None = DEFAULT_STREAM_BUFFER_SIZE,
) -> LiveAgentStreamResult[AgentOutputT]:
    if not hasattr(agent.model, "connect"):
        raise ValidationError(
            "stream_live_agent() requires an agent.model that supports realtime sessions."
        )

    assert runtime is not None
    resolved_runtime = runtime
    broadcast = Broadcast[AgentLiveEvent](
        max_events=stream_buffer_size
        if stream_buffer_size is not None
        else agent.trace_event_limit
    )
    live_session_future: asyncio.Future[Any] = (
        asyncio.get_running_loop().create_future()
    )
    # collect() callers may never send input; retrieve failures without consuming
    # their availability to callers awaiting the connection future.
    live_session_future.add_done_callback(
        lambda future: None if future.cancelled() else future.exception()
    )

    async def emit_live(event: RealtimeEvent) -> None:
        await broadcast.publish(event)

    if tool_execution is not None and tool_execution.timeout_ms is not None:
        if tool_execution.timeout_ms <= 0:
            raise ValidationError(
                'The "tool_execution.timeout_ms" field must be greater than zero.'
            )

    async def run_live_impl(
        request: AgentRunRequest[Any, Any],
    ) -> AgentRunResult[AgentOutputT]:
        live_agent = cast("Agent[AgentDepsT, AgentOutputT]", request.agent)
        resolved_session = request.session or create_agent_session()
        live_prompt = request.prompt
        live_messages = request.messages
        live_deps = cast(AgentDepsT | None, request.deps)
        live_cancellation_token = request.cancellation_token
        live_tool_execution = (
            tool_execution if tool_execution is not None else live_agent.tool_execution
        )
        if (
            live_tool_execution is not None
            and live_tool_execution.timeout_ms is not None
        ):
            if live_tool_execution.timeout_ms <= 0:
                raise ValidationError(
                    'The "tool_execution.timeout_ms" field must be greater than zero.'
                )
        if (
            live_agent.memory is not None
            and not resolved_session.messages
            and resolved_session.summary is None
        ):
            state = await live_agent.memory.load(resolved_session.id)
            resolved_session.messages = list(state.messages)
            resolved_session.summary = state.summary
            resolved_session.metadata = {**state.metadata, **resolved_session.metadata}

        run_id = _new_id("run")
        started_at_ms = _now_ms()
        live_started = time.monotonic()
        trace = AgentTrace(
            run_id=run_id,
            session_id=resolved_session.id,
            agent_name=live_agent.name,
            started_at_ms=started_at_ms,
            orchestration_path=[live_agent.name],
        )

        async def emit_agent(event: AgentEvent) -> None:
            _record_trace_event(trace, event, live_agent.trace_event_limit)
            await broadcast.publish(event)

        initial_state = AgentRunState(
            run_id=run_id,
            agent_name=live_agent.name,
            provider=str(getattr(live_agent.model, "provider", "")),
            model_id=str(getattr(live_agent.model, "model_id", "")),
            session_id=resolved_session.id,
            parent_run_id=parent_run_id,
            idempotency_key=idempotency_key,
            started_at_ms=started_at_ms,
            updated_at_ms=started_at_ms,
            metadata={"orchestration_path": [live_agent.name], "realtime": True},
        )
        transcript = list(resolved_session.messages)
        tool_results: list[ToolExecutionResult] = []
        assistant_buffer: list[str] = []
        last_assistant_text = ""
        input_messages: list[ModelMessage] = []
        context: AgentContext[AgentDepsT] | None = None
        effective_hooks = [
            *resolved_runtime._hooks,
            *list(hooks or []),
            *live_agent.hooks,
        ]
        resolved_instructions: str | None = None
        live_session: Any = None

        async def check_cancelled() -> None:
            await _raise_if_agent_run_cancelled(
                run_store=live_agent.run_store,
                run_id=run_id,
                cancellation_token=live_cancellation_token,
            )

        def remaining_wall_seconds() -> float | None:
            max_wall_time_ms = live_agent.run_limits.max_wall_time_ms
            if max_wall_time_ms is None:
                return None
            return max_wall_time_ms / 1000 - (time.monotonic() - live_started)

        async def await_boundary(awaitable: Awaitable[Any]) -> Any:
            task = asyncio.ensure_future(
                _await_with_agent_cancellation(
                    awaitable,
                    cancellation_token=live_cancellation_token,
                    run_id=run_id,
                )
            )
            try:
                while not task.done():
                    await check_cancelled()
                    remaining = remaining_wall_seconds()
                    if remaining is not None and remaining <= 0:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        raise RuntimeError("Agent run exceeded max wall time.")
                    await asyncio.wait(
                        {task},
                        timeout=min(0.25, remaining) if remaining is not None else 0.25,
                    )
                return task.result()
            except BaseException:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                raise

        async def persist_cancelled(reason: str | None) -> AgentRunState | None:
            if live_agent.run_store is None:
                return None
            current = await live_agent.run_store.load(run_id)
            if current is None or current.status == "cancelled":
                return current
            cancelled = replace(
                current,
                status="cancelled",
                cancellation_reason=reason,
                updated_at_ms=_now_ms(),
                finished_at_ms=_now_ms(),
            )
            try:
                return await _persist_agent_run_state(live_agent.run_store, cancelled)
            except AgentRunCancelled:
                return await live_agent.run_store.load(run_id)

        try:
            if live_agent.run_store is not None:
                if idempotency_key:
                    claim_idempotency_key = getattr(
                        live_agent.run_store, "claim_idempotency_key", None
                    )
                    if not callable(claim_idempotency_key):
                        raise ValidationError(
                            "Idempotent realtime agent runs require an AgentRunStore with atomic "
                            "claim_idempotency_key(...)."
                        )
                    claimed_state = await claim_idempotency_key(initial_state)
                    if claimed_state.run_id != run_id:
                        if not live_session_future.done():
                            live_session_future.cancel()
                        return cast(
                            AgentRunResult[AgentOutputT],
                            _agent_run_result_from_state(
                                claimed_state, resolved_session, agent=live_agent
                            ),
                        )
                else:
                    await live_agent.run_store.save(initial_state)

            await check_cancelled()
            if (
                live_agent.run_limits.max_steps is not None
                and live_agent.run_limits.max_steps < 1
            ):
                raise RuntimeError(
                    f"Agent exceeded max steps ({live_agent.run_limits.max_steps})."
                )
            await emit_agent(
                AgentRunStartEvent(
                    run_id=run_id,
                    session_id=resolved_session.id,
                    agent_name=live_agent.name,
                )
            )
            (
                active_skill_activations,
                skipped_skills,
                skill_tools,
            ) = await _select_active_skills(
                _resolve_skill_registry(live_agent, skills),
                agent=live_agent,
                session=resolved_session,
                prompt=live_prompt,
                messages=live_messages,
            )
            await _emit_skill_events(
                active_skills=active_skill_activations,
                skipped_skills=skipped_skills,
                emit=emit_agent,
            )
            registry_instance = _resolve_tool_registry(live_agent, tools)
            if skill_tools:
                registry_instance = registry_instance.merge(skill_tools)
            context = AgentContext(
                run_id=run_id,
                session_id=resolved_session.id,
                agent_name=live_agent.name,
                memory_summary=resolved_session.summary,
                metadata={
                    **dict(live_agent.metadata),
                    "skills": [item.skill.name for item in active_skill_activations],
                },
                handoff_path=list(trace.orchestration_path),
                deps=live_deps,
                session=resolved_session,
                cancellation_token=live_cancellation_token,
            )
            await _call_agent_hooks(
                effective_hooks, "on_agent_start", context, live_agent
            )
            resolved_instructions = await _resolve_agent_instructions(
                live_agent, context
            )
            if (
                live_agent.output_type is not None
                and live_agent.output_mode == "native"
            ):
                raise ValidationError(
                    "Realtime agents do not support native typed outputs; use output_mode='prompted'."
                )
            realtime_output_agent = (
                replace(live_agent, output_mode="prompted")
                if live_agent.output_type is not None
                else live_agent
            )
            _, structured_output_instructions = _resolve_agent_structured_output(
                realtime_output_agent
            )
            input_messages = _build_run_messages(
                agent=live_agent,
                session=resolved_session,
                prompt=live_prompt,
                messages=live_messages,
                active_skills=[item.skill for item in active_skill_activations],
                instructions=resolved_instructions,
                structured_output_instructions=structured_output_instructions,
            )
            _persist_active_skills(resolved_session, active_skill_activations)
            guarded_input = await resolved_runtime._run_input_guardrails(
                agent=live_agent,
                run_id=run_id,
                session_id=resolved_session.id,
                prompt=live_prompt,
                messages=input_messages,
                context=context,
                trace=trace,
                emit=emit_agent,
            )
            input_messages = guarded_input.messages
            guarded_new_messages = (
                input_messages[-len(live_messages) :] if live_messages else []
            )
            wrapped_tools = resolved_runtime._wrap_agent_tools(
                agent=live_agent,
                registry=registry_instance,
                run_id=run_id,
                session_id=resolved_session.id,
                trace=trace,
                started_at_ms=trace.started_at_ms,
                context=context,
                emit=emit_agent,
                hooks=effective_hooks,
            )
            instruction_base = (
                realtime_config.instructions
                if realtime_config is not None
                and realtime_config.instructions is not None
                else resolved_instructions
            )
            combined_instructions = instruction_base
            supplemental_instructions = [
                *(
                    _skill_system_message(item.skill)
                    for item in active_skill_activations
                ),
                *(
                    [structured_output_instructions]
                    if structured_output_instructions
                    else []
                ),
            ]
            if supplemental_instructions:
                combined_instructions = "\n\n".join(
                    [
                        text
                        for text in [instruction_base, *supplemental_instructions]
                        if text
                    ]
                )
            live_config = realtime_config or RealtimeSessionConfig(
                instructions=combined_instructions,
                tools=wrapped_tools or None,
                tool_choice=cast(Any, tool_choice),
                provider_options=provider_options,
            )
            if realtime_config is not None:
                live_config = replace(
                    live_config,
                    instructions=combined_instructions,
                    tools=live_config.tools
                    if live_config.tools is not None
                    else (wrapped_tools or None),
                    tool_choice=(
                        live_config.tool_choice
                        if live_config.tool_choice is not None
                        else cast(Any, tool_choice)
                    ),
                    provider_options=(
                        live_config.provider_options
                        if live_config.provider_options is not None
                        else provider_options
                    ),
                )

            live_model = cast(RealtimeModel, live_agent.model)
            live_session = await await_boundary(
                live_model.connect(config=live_config, options=connect_options)
            )
            live_session_future.set_result(live_session)
            resolved_session.metadata = {
                **resolved_session.metadata,
                "realtime": {
                    "provider": getattr(live_agent.model, "provider", ""),
                    "model_id": getattr(live_agent.model, "model_id", ""),
                },
            }
            await check_cancelled()
            if live_messages is not None:
                for message in guarded_new_messages:
                    text = _text_from_message(message)
                    if message.role == "user" and text:
                        await await_boundary(live_session.send_text(text))
                        transcript.append(create_text_message("user", text))
            elif guarded_input.prompt is not None:
                await await_boundary(live_session.send_text(guarded_input.prompt))
                transcript.append(create_text_message("user", guarded_input.prompt))

            event_iterator = live_session.event_stream().__aiter__()
            buffer_realtime_output = bool(live_agent.output_guardrails)
            seen_tool_calls: dict[str, ToolCall] = {}
            awaiting_tool_response = False
            transcript_buffer: list[str] = []
            completed_response_steps = 0
            while True:
                try:
                    event = await await_boundary(event_iterator.__anext__())
                except StopAsyncIteration:
                    raise RuntimeError(
                        "Realtime stream ended before response completion."
                    ) from None
                await check_cancelled()
                if isinstance(event, RealtimeErrorEvent):
                    raise RuntimeError(
                        "Realtime provider reported an error."
                    ) from event.error
                if isinstance(event, RealtimeToolResultEvent):
                    # Tool results are emitted once by the runtime after delivery.
                    continue
                if isinstance(event, RealtimeAudioOutputEvent) and event.audio:
                    awaiting_tool_response = False
                contains_unredacted_assistant_text = buffer_realtime_output and (
                    isinstance(event, RealtimeTextDeltaEvent)
                    or (
                        isinstance(event, RealtimeTranscriptEvent)
                        and event.role == "assistant"
                    )
                )
                if not contains_unredacted_assistant_text:
                    await emit_live(event)
                if isinstance(event, RealtimeTextDeltaEvent):
                    if event.text_delta:
                        awaiting_tool_response = False
                    assistant_buffer.append(event.text_delta)
                    if not buffer_realtime_output:
                        await emit_agent(
                            AgentTextDeltaEvent(text_delta=event.text_delta)
                        )
                    continue
                if isinstance(event, RealtimeTranscriptEvent):
                    if event.role == "assistant" and event.text:
                        awaiting_tool_response = False
                        if not event.is_final:
                            transcript_buffer.append(event.text)
                    if event.role == "user" and event.is_final and event.text:
                        transcript.append(create_text_message("user", event.text))
                    if event.role == "assistant" and event.is_final:
                        text = event.text or "".join(assistant_buffer)
                        if text:
                            last_assistant_text = text
                            transcript.append(create_text_message("assistant", text))
                            assistant_buffer.clear()
                            transcript_buffer.clear()
                    continue
                if isinstance(event, RealtimeToolCallEvent):
                    previous_call = seen_tool_calls.get(event.tool_call.id)
                    if previous_call is not None:
                        if previous_call != event.tool_call:
                            raise ValidationError(
                                "Realtime tool call ID was reused with different arguments."
                            )
                        continue
                    if not event.tool_call.id:
                        raise ValidationError(
                            "Realtime tool calls require a non-empty ID."
                        )
                    seen_tool_calls[event.tool_call.id] = event.tool_call
                    awaiting_tool_response = True
                    await emit_agent(AgentToolCallEvent(tool_call=event.tool_call))
                    definition = (
                        wrapped_tools.get(event.tool_call.name)
                        if wrapped_tools
                        else None
                    )
                    if (
                        definition is None
                        or not is_callable_tool_definition(definition)
                        or definition.execute is None
                    ):
                        tool_result = ToolExecutionResult(
                            tool_call_id=event.tool_call.id,
                            tool_name=event.tool_call.name,
                            error=ToolExecutionError(
                                message=f'Unknown realtime tool "{event.tool_call.name}".'
                            ),
                            is_error=True,
                        )
                    else:
                        from .generate_text import _execute_tool

                        await check_cancelled()
                        tool_timeout_ms = (
                            live_tool_execution.timeout_ms
                            if live_tool_execution is not None
                            else None
                        )
                        remaining = remaining_wall_seconds()
                        if remaining is not None:
                            if remaining <= 0:
                                raise RuntimeError("Agent run exceeded max wall time.")
                            remaining_ms = max(1, int(remaining * 1000))
                            tool_timeout_ms = (
                                remaining_ms
                                if tool_timeout_ms is None
                                else min(tool_timeout_ms, remaining_ms)
                            )
                        try:
                            tool_result = await _execute_tool(
                                event.tool_call,
                                wrapped_tools,
                                timeout_ms=tool_timeout_ms,
                            )
                        except ToolExecutionSuspended as suspended:
                            resume_messages = _strip_runtime_system_messages(
                                input_messages, resolved_instructions
                            )
                            transcript_offset = len(resolved_session.messages)
                            resume_messages.extend(transcript[transcript_offset:])
                            resume_messages.append(
                                ModelMessage(
                                    role="assistant",
                                    parts=[ToolCallPart(tool_call=event.tool_call)],
                                )
                            )
                            suspended.messages = resume_messages
                            suspended.tool_results = [
                                *tool_results,
                                *suspended.tool_results,
                            ]
                            raise
                        await check_cancelled()
                        if (
                            live_tool_execution is not None
                            and live_tool_execution.stop_on_error
                            and tool_result.is_error
                        ):
                            raise RuntimeError(
                                f'Tool "{tool_result.tool_name}" failed: '
                                f"{tool_result.error.message if tool_result.error else 'Unknown tool error.'}"
                            )
                    tool_results.append(tool_result)
                    transcript.append(
                        ModelMessage(role="tool", parts=[tool_result_part(tool_result)])
                    )
                    await await_boundary(live_session.send_tool_result(tool_result))
                    await emit_agent(AgentToolResultEvent(tool_result=tool_result))
                    await emit_live(RealtimeToolResultEvent(tool_result=tool_result))
                    continue
                if isinstance(event, RealtimeSessionEndedEvent):
                    raise RuntimeError(
                        "Realtime session ended before response completion."
                    )
                if isinstance(event, RealtimeResponseCompletedEvent):
                    if event.reason == "generation-complete":
                        continue
                    if event.reason in {"failed", "cancelled", "incomplete", "error"}:
                        raise RuntimeError(
                            "Realtime response did not complete successfully."
                        )
                    completed_response_steps += 1
                    if event.reason == "tool-calls" or awaiting_tool_response:
                        if (
                            live_agent.run_limits.max_steps is not None
                            and completed_response_steps
                            >= live_agent.run_limits.max_steps
                        ):
                            raise RuntimeError(
                                f"Agent exceeded max steps ({live_agent.run_limits.max_steps})."
                            )
                        continue
                    break
            if transcript_buffer and not assistant_buffer:
                assistant_buffer = transcript_buffer
            if assistant_buffer:
                last_assistant_text = "".join(assistant_buffer)
                if last_assistant_text:
                    transcript.append(
                        create_text_message("assistant", last_assistant_text)
                    )
            guarded_output = await resolved_runtime._run_output_guardrails(
                agent=live_agent,
                run_id=run_id,
                session_id=resolved_session.id,
                result=None,
                text=last_assistant_text,
                messages=[create_text_message("assistant", last_assistant_text)]
                if last_assistant_text
                else [],
                context=context,
                trace=trace,
                emit=emit_agent,
            )
            last_assistant_text = guarded_output.text
            transcript = _replace_assistant_messages(
                transcript, guarded_output.messages
            )
            if buffer_realtime_output and last_assistant_text:
                await emit_agent(AgentTextDeltaEvent(text_delta=last_assistant_text))
            await check_cancelled()
            parsed_output = _parse_agent_output(live_agent, last_assistant_text)
            resolved_session.messages = _strip_runtime_system_messages(
                transcript, resolved_instructions
            )
            if live_agent.memory is not None and _should_refresh_summary(
                live_agent.memory, resolved_session
            ):
                resolved_session.summary = await live_agent.memory.summarize(
                    session_id=resolved_session.id,
                    state=AgentMemoryState(
                        messages=list(resolved_session.messages),
                        summary=resolved_session.summary,
                        metadata={
                            **dict(resolved_session.metadata),
                            "state": dict(resolved_session.state),
                        },
                    ),
                    agent=live_agent,
                )
                await emit_agent(
                    AgentSummaryUpdateEvent(summary=resolved_session.summary)
                )
            if live_agent.memory is not None:
                await live_agent.memory.save(
                    resolved_session.id,
                    AgentMemoryState(
                        messages=list(resolved_session.messages),
                        summary=resolved_session.summary,
                        metadata={
                            **dict(resolved_session.metadata),
                            "state": dict(resolved_session.state),
                        },
                    ),
                )

            trace.finished_at_ms = _now_ms()
            result: AgentRunResult[AgentOutputT] = AgentRunResult(
                run_id=run_id,
                agent_name=live_agent.name,
                session=resolved_session,
                text=last_assistant_text,
                finish_reason="stop",
                steps=[],
                messages=list(resolved_session.messages),
                tool_results=tool_results,
                trace=trace,
                orchestration_path=list(trace.orchestration_path),
                output=cast(AgentOutputT, parsed_output),
            )
            await _call_agent_hooks(
                effective_hooks,
                "on_agent_end",
                context,
                live_agent,
                result,
                reverse=True,
            )
            run_state = _agent_run_state_from_result(
                result=result,
                agent=live_agent,
                parent_run_id=parent_run_id,
                idempotency_key=idempotency_key,
                started_at_ms=started_at_ms,
                finished_at_ms=trace.finished_at_ms,
            )
            if live_agent.run_store is not None:
                run_state = await _persist_agent_run_state(
                    live_agent.run_store, run_state
                )
            result.state = run_state
            await emit_agent(
                AgentFinishEvent(
                    run_id=run_id,
                    session_id=resolved_session.id,
                    text=last_assistant_text,
                    finish_reason="stop",
                )
            )
            return result
        except ToolExecutionSuspended as suspended:
            suspended_at_ms = _now_ms()
            trace.finished_at_ms = suspended_at_ms
            pending = cast(PendingApproval, suspended.pending_approval)
            run_state = _agent_run_state_from_suspension(
                run_id=run_id,
                agent_name=live_agent.name,
                session_id=resolved_session.id,
                agent=live_agent,
                parent_run_id=parent_run_id,
                idempotency_key=idempotency_key,
                started_at_ms=started_at_ms,
                suspended_at_ms=suspended_at_ms,
                suspended=suspended,
                orchestration_path=list(trace.orchestration_path),
            )
            suspended_result: AgentRunResult[AgentOutputT] = AgentRunResult(
                run_id=run_id,
                agent_name=live_agent.name,
                session=resolved_session,
                text="",
                finish_reason="tool-calls",
                steps=list(cast(list[GenerateTextStep], suspended.steps)),
                messages=list(cast(list[ModelMessage], suspended.messages)),
                tool_results=list(
                    cast(list[ToolExecutionResult], suspended.tool_results)
                ),
                trace=trace,
                orchestration_path=list(trace.orchestration_path),
                state=run_state,
                provider_finish_reason=pending.reason,
            )
            if context is not None:
                await _call_agent_hooks(
                    effective_hooks,
                    "on_agent_end",
                    context,
                    live_agent,
                    suspended_result,
                    reverse=True,
                )
            if live_agent.run_store is not None:
                run_state = await _persist_agent_run_state(
                    live_agent.run_store, run_state
                )
                suspended_result.state = run_state
            await emit_agent(
                AgentFinishEvent(
                    run_id=run_id,
                    session_id=resolved_session.id,
                    text="",
                    finish_reason="tool-calls",
                )
            )
            return suspended_result
        except AgentRunCancelled as error:
            trace.finished_at_ms = _now_ms()
            await persist_cancelled(error.reason)
            error_context = context or AgentContext(
                run_id=run_id,
                session_id=resolved_session.id,
                agent_name=live_agent.name,
                deps=live_deps,
                session=resolved_session,
                cancellation_token=live_cancellation_token,
            )
            await _call_error_hooks_preserving(
                effective_hooks, error_context, live_agent, error
            )
            await emit_agent(AgentErrorEvent(error=error))
            raise
        except BaseException as caught:
            if isinstance(caught, asyncio.CancelledError):
                budget = current_execution_budget()
                if budget is None or time.monotonic() < budget.deadline:
                    trace.finished_at_ms = _now_ms()
                    await persist_cancelled("Realtime agent task cancelled.")
                    raise
            failure_error = _execution_error(caught)
            trace.finished_at_ms = _now_ms()
            resolved_session.messages = _strip_runtime_system_messages(
                transcript, resolved_instructions
            )
            if live_agent.run_store is not None:
                failed_result: AgentRunResult[Any] = AgentRunResult(
                    run_id=run_id,
                    agent_name=live_agent.name,
                    session=resolved_session,
                    text=last_assistant_text,
                    finish_reason="error",
                    steps=[],
                    messages=list(resolved_session.messages),
                    tool_results=list(tool_results),
                    trace=trace,
                    orchestration_path=list(trace.orchestration_path),
                )
                try:
                    await _persist_agent_run_state(
                        live_agent.run_store,
                        _agent_run_state_from_result(
                            result=failed_result,
                            agent=live_agent,
                            parent_run_id=parent_run_id,
                            idempotency_key=idempotency_key,
                            started_at_ms=started_at_ms,
                            finished_at_ms=trace.finished_at_ms,
                            error=str(failure_error),
                        ),
                    )
                except AgentRunCancelled as cancelled:
                    await emit_agent(AgentErrorEvent(error=cancelled))
                    raise cancelled from failure_error
            if not live_session_future.done():
                live_session_future.set_exception(failure_error)
            error_context = context or AgentContext(
                run_id=run_id,
                session_id=resolved_session.id,
                agent_name=live_agent.name,
                deps=live_deps,
                session=resolved_session,
                cancellation_token=live_cancellation_token,
            )
            await _call_error_hooks_preserving(
                effective_hooks, error_context, live_agent, failure_error
            )
            await emit_agent(AgentErrorEvent(error=failure_error))
            raise failure_error

        finally:
            if live_session is not None:
                await live_session.aclose()

    request = AgentRunRequest(
        agent=agent,
        session=session,
        prompt=prompt,
        messages=messages,
        deps=deps,
        cancellation_token=cancellation_token,
        metadata={"parent_run_id": parent_run_id, "idempotency_key": idempotency_key},
    )
    resolved_middleware = [
        *resolved_runtime._middleware,
        *list(middleware or []),
        *agent.middleware,
    ]

    async def call_at(
        index: int, current: AgentRunRequest[Any, Any]
    ) -> AgentRunResult[Any]:
        if index >= len(resolved_middleware):
            return await run_live_impl(current)

        async def call_next(
            next_request: AgentRunRequest[Any, Any],
        ) -> AgentRunResult[Any]:
            return await call_at(index + 1, next_request)

        result = await _maybe_await(resolved_middleware[index](current, call_next))
        if not isinstance(result, AgentRunResult):
            raise TypeError("Agent middleware must return AgentRunResult.")
        return result

    async def runner() -> AgentRunResult[AgentOutputT]:
        try:
            async with execution_scope(total_timeout_ms, retry_jitter=retry_jitter):
                result = cast(AgentRunResult[AgentOutputT], await call_at(0, request))
        finally:
            if not live_session_future.done():
                live_session_future.cancel()
        return result

    async def managed_runner() -> AgentRunResult[AgentOutputT]:
        try:
            return await runner()
        finally:
            await broadcast.close()

    task = asyncio.create_task(managed_runner())
    return LiveAgentStreamResult(task, broadcast, live_session_future)
