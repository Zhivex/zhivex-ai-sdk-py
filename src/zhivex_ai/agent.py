from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias
from uuid import uuid4

from .errors import ValidationError
from .generate_text import generate_text, stream_text
from .messages import create_text_message, tool_result_part
from .types import (
    FinishReason,
    GenerateTextOutput,
    GenerateTextStep,
    LanguageModel,
    ModelGenerateInput,
    ModelMessage,
    ReasoningConfig,
    StreamErrorEvent,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    StreamToolResultEvent,
    ToolChoiceName,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionOptions,
    ToolExecutionResult,
    ToolSet,
    TokenUsage,
    ToolCall,
)

HANDOFF_MARKER = "__zhivex_agent_handoff__"
SUMMARY_MARKER = "Conversation summary:\n"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _text_from_message(message: ModelMessage) -> str:
    return "".join(part.text for part in message.parts if getattr(part, "type", None) == "text")


def _message_text(messages: Iterable[ModelMessage]) -> str:
    chunks: list[str] = []
    for message in messages:
        text = _text_from_message(message).strip()
        if text:
            chunks.append(f"{message.role}: {text}")
    return "\n".join(chunks)


def _strip_runtime_system_messages(messages: list[ModelMessage], instructions: str | None) -> list[ModelMessage]:
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


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


def _invoke_tool_callable(execute: Any, parsed: Any, context: ToolExecutionContext) -> Any:
    signature = inspect.signature(execute)
    parameters = list(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return execute(parsed, context=context)
    if any(parameter.name == "context" for parameter in parameters):
        return execute(parsed, context=context)
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        return execute(parsed, context)
    return execute(parsed)


def handoff_to(target_agent: str, *, input: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        HANDOFF_MARKER: True,
        "target_agent": target_agent,
        "input": input,
        "metadata": metadata or {},
    }


@dataclass(slots=True)
class AgentHandoff:
    target_agent: str
    input: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentContext:
    run_id: str
    session_id: str
    agent_name: str
    memory_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    handoff_path: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ApprovalDecision:
    approved: bool
    reason: str | None = None


@dataclass(slots=True)
class ToolApprovalRequest:
    run_id: str
    session_id: str
    agent_name: str
    tool_name: str
    tool_input: Any
    tool_permissions: list[str] = field(default_factory=list)
    tool_source: str = "local"
    tool_metadata: dict[str, Any] = field(default_factory=dict)
    context: AgentContext | None = None
    handoff_path: list[str] = field(default_factory=list)


class ApprovalPolicy(Protocol):
    async def __call__(self, request: ToolApprovalRequest) -> ApprovalDecision | bool: ...


@dataclass(slots=True)
class SummaryConfig:
    max_messages: int = 12
    preserve_recent_messages: int = 8
    max_summary_chars: int = 1200


@dataclass(slots=True)
class AgentMemoryState:
    messages: list[ModelMessage] = field(default_factory=list)
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentMemory(Protocol):
    summary_config: SummaryConfig

    async def load(self, session_id: str) -> AgentMemoryState: ...

    async def save(self, session_id: str, state: AgentMemoryState) -> None: ...

    async def summarize(
        self,
        *,
        session_id: str,
        state: AgentMemoryState,
        agent: "Agent",
    ) -> str | None: ...


class AgentCheckpointStore(Protocol):
    async def save(self, checkpoint: "AgentCheckpoint") -> None: ...

    async def list(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> list["AgentCheckpoint"]: ...


@dataclass(slots=True)
class RunLimits:
    max_steps: int | None = 8
    max_tool_calls: int | None = 32
    max_wall_time_ms: int | None = 120_000
    max_handoffs: int | None = 1


@dataclass(slots=True)
class AgentSession:
    id: str = field(default_factory=lambda: _new_id("session"))
    messages: list[ModelMessage] = field(default_factory=list)
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRuntime(Protocol):
    async def execute(self, definition: ToolDefinition, input: Any, context: ToolExecutionContext) -> Any: ...


class LocalToolRuntime:
    async def execute(self, definition: ToolDefinition, input: Any, context: ToolExecutionContext) -> Any:
        if definition.execute is None:
            raise RuntimeError(f'Tool "{definition.name}" does not define a local executor.')
        result = _invoke_tool_callable(definition.execute, input, context)
        return await _maybe_await(result)


class UnsupportedToolRuntime:
    def __init__(self, source: str) -> None:
        self._source = source

    async def execute(self, definition: ToolDefinition, input: Any, context: ToolExecutionContext) -> Any:
        raise RuntimeError(
            f'Tool "{definition.name}" uses source "{self._source}", but no runtime is configured for that source.'
        )


class ToolRegistry:
    def __init__(
        self,
        tools: ToolSet | None = None,
        *,
        runtimes: dict[str, ToolRuntime] | None = None,
    ) -> None:
        self._tools: ToolSet = dict(tools or {})
        self._runtimes: dict[str, ToolRuntime] = {
            "local": LocalToolRuntime(),
            "remote": UnsupportedToolRuntime("remote"),
            "mcp": UnsupportedToolRuntime("mcp"),
        }
        self._runtimes.update(runtimes or {})

    def register(self, definition: ToolDefinition) -> ToolDefinition:
        self._tools[definition.name] = definition
        return definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def items(self) -> list[tuple[str, ToolDefinition]]:
        return list(self._tools.items())

    def merge(self, tools: ToolSet | "ToolRegistry" | None) -> "ToolRegistry":
        merged = ToolRegistry(self._tools, runtimes=self._runtimes)
        if isinstance(tools, ToolRegistry):
            for definition in tools._tools.values():
                merged.register(definition)
            merged._runtimes.update(tools._runtimes)
            return merged
        for definition in dict(tools or {}).values():
            merged.register(definition)
        return merged

    async def execute(self, definition: ToolDefinition, input: Any, context: ToolExecutionContext) -> Any:
        runtime = self._runtimes.get(definition.source, UnsupportedToolRuntime(definition.source))
        return await runtime.execute(definition, input, context)


@dataclass(slots=True)
class Agent:
    name: str
    model: LanguageModel
    instructions: str | None = None
    tools: ToolSet | ToolRegistry = field(default_factory=dict)
    subagents: dict[str, "Agent"] = field(default_factory=dict)
    memory: AgentMemory | None = None
    checkpoint_store: AgentCheckpointStore | None = None
    approval_policy: ApprovalPolicy | None = None
    run_limits: RunLimits = field(default_factory=RunLimits)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentCheckpoint:
    run_id: str
    session_id: str
    agent_name: str
    step_index: int
    request: ModelGenerateInput
    response: Any
    saved_at_ms: int
    is_final: bool = False


@dataclass(slots=True)
class AgentRunStartEvent:
    type: str = "run-start"
    run_id: str = ""
    session_id: str = ""
    agent_name: str = ""


@dataclass(slots=True)
class AgentDelegationStartEvent:
    type: str = "delegation-start"
    agent_name: str = ""
    handoff_depth: int = 0


@dataclass(slots=True)
class AgentDelegationFinishEvent:
    type: str = "delegation-finish"
    agent_name: str = ""
    handoff_depth: int = 0
    finish_reason: FinishReason | None = None


@dataclass(slots=True)
class AgentTextDeltaEvent:
    type: str = "text-delta"
    text_delta: str = ""


@dataclass(slots=True)
class AgentToolCallEvent:
    type: str = "tool-call"
    tool_call: ToolCall = field(default_factory=lambda: ToolCall(id="", name="", input={}))


@dataclass(slots=True)
class AgentToolApprovalEvent:
    type: str = "tool-approval"
    tool_name: str = ""
    tool_input: Any = None
    approved: bool = True
    reason: str | None = None


@dataclass(slots=True)
class AgentToolResultEvent:
    type: str = "tool-result"
    tool_result: ToolExecutionResult = field(
        default_factory=lambda: ToolExecutionResult(tool_call_id="", tool_name="", is_error=False)
    )


@dataclass(slots=True)
class AgentCheckpointEvent:
    type: str = "checkpoint"
    checkpoint: AgentCheckpoint | None = None


@dataclass(slots=True)
class AgentSummaryUpdateEvent:
    type: str = "summary-update"
    summary: str | None = None


@dataclass(slots=True)
class AgentHandoffRequestedEvent:
    type: str = "handoff-requested"
    handoff: AgentHandoff | None = None


@dataclass(slots=True)
class AgentHandoffResolvedEvent:
    type: str = "handoff-resolved"
    source_agent: str = ""
    target_agent: str = ""


@dataclass(slots=True)
class AgentHandoffFailedEvent:
    type: str = "handoff-failed"
    source_agent: str = ""
    target_agent: str = ""
    reason: str | None = None


@dataclass(slots=True)
class AgentHandoffEvent:
    type: str = "handoff"
    handoff: AgentHandoff | None = None


@dataclass(slots=True)
class AgentFinishEvent:
    type: str = "finish"
    run_id: str = ""
    session_id: str = ""
    text: str = ""
    finish_reason: FinishReason | None = None


@dataclass(slots=True)
class AgentErrorEvent:
    type: str = "error"
    error: Exception | None = None


AgentEvent: TypeAlias = (
    AgentRunStartEvent
    | AgentDelegationStartEvent
    | AgentDelegationFinishEvent
    | AgentTextDeltaEvent
    | AgentToolCallEvent
    | AgentToolApprovalEvent
    | AgentToolResultEvent
    | AgentCheckpointEvent
    | AgentSummaryUpdateEvent
    | AgentHandoffRequestedEvent
    | AgentHandoffResolvedEvent
    | AgentHandoffFailedEvent
    | AgentHandoffEvent
    | AgentFinishEvent
    | AgentErrorEvent
)


@dataclass(slots=True)
class AgentTraceSegment:
    agent_name: str
    started_at_ms: int
    finished_at_ms: int | None = None


@dataclass(slots=True)
class AgentTrace:
    run_id: str
    session_id: str
    agent_name: str
    started_at_ms: int
    finished_at_ms: int | None = None
    events: list[AgentEvent] = field(default_factory=list)
    orchestration_path: list[str] = field(default_factory=list)
    segments: list[AgentTraceSegment] = field(default_factory=list)
    tool_call_count: int = 0
    approval_count: int = 0
    checkpoint_count: int = 0
    handoff_count: int = 0


@dataclass(slots=True)
class AgentRunResult:
    run_id: str
    agent_name: str
    session: AgentSession
    text: str
    finish_reason: FinishReason | None = None
    provider_finish_reason: str | None = None
    usage: TokenUsage | None = None
    steps: list[GenerateTextStep] = field(default_factory=list)
    messages: list[ModelMessage] = field(default_factory=list)
    tool_results: list[ToolExecutionResult] = field(default_factory=list)
    trace: AgentTrace | None = None
    handoff: AgentHandoff | None = None
    orchestration_path: list[str] = field(default_factory=list)


class InMemoryAgentMemory:
    def __init__(self, *, summary_config: SummaryConfig | None = None) -> None:
        self.summary_config = summary_config or SummaryConfig()
        self._store: dict[str, AgentMemoryState] = {}

    async def load(self, session_id: str) -> AgentMemoryState:
        state = self._store.get(session_id)
        if state is None:
            return AgentMemoryState()
        return AgentMemoryState(messages=list(state.messages), summary=state.summary, metadata=dict(state.metadata))

    async def save(self, session_id: str, state: AgentMemoryState) -> None:
        self._store[session_id] = AgentMemoryState(
            messages=list(state.messages),
            summary=state.summary,
            metadata=dict(state.metadata),
        )

    async def summarize(
        self,
        *,
        session_id: str,
        state: AgentMemoryState,
        agent: Agent,
    ) -> str | None:
        if not state.messages:
            return state.summary
        older_messages = state.messages[:-self.summary_config.preserve_recent_messages]
        if not older_messages:
            return state.summary
        existing = state.summary.strip() if state.summary else ""
        transcript = _message_text(older_messages)
        parts = [part for part in [existing, transcript] if part]
        if not parts:
            return state.summary
        summary = "\n".join(parts)
        if len(summary) > self.summary_config.max_summary_chars:
            summary = summary[: self.summary_config.max_summary_chars]
        return summary.strip() or None


class InMemoryAgentCheckpointStore:
    def __init__(self) -> None:
        self._items: list[AgentCheckpoint] = []

    async def save(self, checkpoint: AgentCheckpoint) -> None:
        self._items.append(checkpoint)

    async def list(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> list[AgentCheckpoint]:
        items = list(self._items)
        if session_id is not None:
            items = [item for item in items if item.session_id == session_id]
        if run_id is not None:
            items = [item for item in items if item.run_id == run_id]
        return items


def create_agent_session(
    *,
    id: str | None = None,
    messages: list[ModelMessage] | None = None,
    summary: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentSession:
    return AgentSession(
        id=id or _new_id("session"),
        messages=list(messages or []),
        summary=summary,
        metadata=dict(metadata or {}),
    )


def create_in_memory_agent_memory_store(*, summary_config: SummaryConfig | None = None) -> InMemoryAgentMemory:
    return InMemoryAgentMemory(summary_config=summary_config)


def create_in_memory_checkpoint_store() -> InMemoryAgentCheckpointStore:
    return InMemoryAgentCheckpointStore()


async def allow_all_approval_policy(request: ToolApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(approved=True)


async def deny_all_approval_policy(request: ToolApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(approved=False, reason="Tool execution denied by policy.")


def permission_allowlist_approval_policy(*allowed_permissions: str) -> ApprovalPolicy:
    allowed = set(allowed_permissions)

    async def policy(request: ToolApprovalRequest) -> ApprovalDecision:
        if not request.tool_permissions:
            return ApprovalDecision(approved=True)
        missing = [permission for permission in request.tool_permissions if permission not in allowed]
        if missing:
            return ApprovalDecision(approved=False, reason=f"Missing permissions: {', '.join(missing)}")
        return ApprovalDecision(approved=True)

    return policy


async def load_agent_session(agent: Agent, session_id: str, *, metadata: dict[str, Any] | None = None) -> AgentSession:
    messages: list[ModelMessage] = []
    summary: str | None = None
    merged_metadata = dict(metadata or {})
    if agent.memory is not None:
        state = await agent.memory.load(session_id)
        messages = list(state.messages)
        summary = state.summary
        merged_metadata = {**state.metadata, **merged_metadata}
    return AgentSession(id=session_id, messages=messages, summary=summary, metadata=merged_metadata)


def _normalize_approval_decision(value: ApprovalDecision | bool | None) -> ApprovalDecision:
    if isinstance(value, ApprovalDecision):
        return value
    if value is False:
        return ApprovalDecision(approved=False)
    return ApprovalDecision(approved=True)


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


def _detect_handoff(tool_results: list[ToolExecutionResult]) -> AgentHandoff | None:
    for result in reversed(tool_results):
        if isinstance(result.output, dict) and result.output.get(HANDOFF_MARKER):
            return AgentHandoff(
                target_agent=str(result.output.get("target_agent")),
                input=result.output.get("input"),
                metadata=dict(result.output.get("metadata") or {}),
            )
    return None


def _should_refresh_summary(memory: AgentMemory, session: AgentSession) -> bool:
    config = memory.summary_config
    if len(session.messages) > config.max_messages:
        return True
    text_length = sum(len(_text_from_message(message)) for message in session.messages)
    return text_length > config.max_summary_chars * 2


def _context_messages(session: AgentSession, memory: AgentMemory | None) -> list[ModelMessage]:
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
) -> list[ModelMessage]:
    if prompt is not None and messages is not None:
        raise ValidationError('Pass either "prompt" or "messages", but not both.')

    built: list[ModelMessage] = []
    if agent.instructions:
        built.append(create_text_message("system", agent.instructions))
    if session.summary:
        built.append(create_text_message("system", f"{SUMMARY_MARKER}{session.summary}"))
    built.extend(_context_messages(session, agent.memory))
    if messages is not None:
        built.extend(messages)
    elif prompt is not None:
        built.append(create_text_message("user", prompt))
    return built


def _resolve_tool_registry(agent: Agent, extra_tools: ToolSet | ToolRegistry | None) -> ToolRegistry:
    base = agent.tools if isinstance(agent.tools, ToolRegistry) else ToolRegistry(agent.tools)
    return base.merge(extra_tools)


def _extract_tool_calls_from_steps(steps: list[GenerateTextStep]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for step in steps:
        response_messages = step.response.messages or ([step.response.message] if step.response.message else [])
        for message in response_messages:
            for part in message.parts:
                if getattr(part, "type", None) == "tool-call":
                    calls.append(part.tool_call)
    return calls


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


def _response_messages(step: GenerateTextStep) -> list[ModelMessage]:
    if step.response.messages:
        return list(step.response.messages)
    if step.response.message is not None:
        return [step.response.message]
    if step.response.text:
        return [create_text_message("assistant", step.response.text)]
    return []


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
        await checkpoint_store.save(checkpoint)
        if trace is not None:
            trace.checkpoint_count += 1
        if emit is not None:
            await emit(AgentCheckpointEvent(checkpoint=checkpoint))


class AgentRegistry:
    def __init__(self, agents: dict[str, Agent] | None = None) -> None:
        self._agents: dict[str, Agent] = dict(agents or {})

    def register(self, agent: Agent) -> Agent:
        self._agents[agent.name] = agent
        return agent

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)


@dataclass
class _Broadcast:
    history: list[AgentEvent]
    done: bool = False
    subscribers: list[asyncio.Queue[AgentEvent | None]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.subscribers = []

    async def publish(self, event: AgentEvent) -> None:
        self.history.append(event)
        for queue in list(self.subscribers):
            await queue.put(event)

    async def close(self) -> None:
        self.done = True
        for queue in list(self.subscribers):
            await queue.put(None)

    def stream(self) -> AsyncIterable[AgentEvent]:
        async def generator() -> AsyncIterable[AgentEvent]:
            queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
            cursor = 0
            self.subscribers.append(queue)
            try:
                while True:
                    while cursor < len(self.history):
                        event = self.history[cursor]
                        cursor += 1
                        yield event
                    if self.done:
                        return
                    item = await queue.get()
                    if item is None:
                        return
            finally:
                if queue in self.subscribers:
                    self.subscribers.remove(queue)

        return generator()


class AgentObserver(Protocol):
    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Any: ...


class AgentRuntime:
    def __init__(
        self,
        *,
        registry: AgentRegistry | None = None,
        observer: AgentObserver | None = None,
    ) -> None:
        self._registry = registry or AgentRegistry()
        self._observer = observer

    async def run(
        self,
        *,
        agent: Agent,
        session: AgentSession | None = None,
        prompt: str | None = None,
        messages: list[ModelMessage] | None = None,
        tools: ToolSet | ToolRegistry | None = None,
        tool_choice: str | ToolChoiceName | None = None,
        tool_execution: ToolExecutionOptions | None = None,
        max_steps: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: ReasoningConfig | None = None,
        provider_options: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
        max_retries: int | None = None,
        retry_backoff_ms: int | None = None,
        stop_on_handoff: bool = False,
        emit: Callable[[AgentEvent], Awaitable[None]] | None = None,
    ) -> AgentRunResult:
        resolved_session = session or create_agent_session()
        if agent.memory is not None and not resolved_session.messages and resolved_session.summary is None:
            state = await agent.memory.load(resolved_session.id)
            resolved_session.messages = list(state.messages)
            resolved_session.summary = state.summary
            resolved_session.metadata = {**state.metadata, **resolved_session.metadata}

        run_id = _new_id("run")
        started_at_ms = _now_ms()
        trace = AgentTrace(
            run_id=run_id,
            session_id=resolved_session.id,
            agent_name=agent.name,
            started_at_ms=started_at_ms,
            orchestration_path=[agent.name],
        )

        async def publish(event: AgentEvent) -> None:
            trace.events.append(event)
            if emit is not None:
                await emit(event)

        await publish(AgentRunStartEvent(run_id=run_id, session_id=resolved_session.id, agent_name=agent.name))

        current_agent = agent
        current_prompt = prompt
        current_messages = messages
        handoff_depth = 0
        last_result: AgentRunResult | None = None

        try:
            while True:
                trace.segments.append(AgentTraceSegment(agent_name=current_agent.name, started_at_ms=_now_ms()))
                await publish(AgentDelegationStartEvent(agent_name=current_agent.name, handoff_depth=handoff_depth))
                segment_result = await self._run_single(
                    agent=current_agent,
                    session=resolved_session,
                    run_id=run_id,
                    trace=trace,
                    prompt=current_prompt,
                    messages=current_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    tool_execution=tool_execution,
                    max_steps=max_steps,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    reasoning=reasoning,
                    provider_options=provider_options,
                    timeout_ms=timeout_ms,
                    max_retries=max_retries,
                    retry_backoff_ms=retry_backoff_ms,
                    emit=publish,
                )
                last_result = segment_result
                trace.segments[-1].finished_at_ms = _now_ms()
                await publish(
                    AgentDelegationFinishEvent(
                        agent_name=current_agent.name,
                        handoff_depth=handoff_depth,
                        finish_reason=segment_result.finish_reason,
                    )
                )
                if segment_result.handoff is None or stop_on_handoff:
                    final_handoff = segment_result.handoff if stop_on_handoff else None
                    await publish(
                        AgentFinishEvent(
                            run_id=run_id,
                            session_id=resolved_session.id,
                            text=segment_result.text,
                            finish_reason=segment_result.finish_reason,
                        )
                    )
                    trace.finished_at_ms = _now_ms()
                    return AgentRunResult(
                        run_id=run_id,
                        agent_name=segment_result.agent_name,
                        session=resolved_session,
                        text=segment_result.text,
                        finish_reason=segment_result.finish_reason,
                        provider_finish_reason=segment_result.provider_finish_reason,
                        usage=segment_result.usage,
                        steps=segment_result.steps,
                        messages=segment_result.messages,
                        tool_results=segment_result.tool_results,
                        trace=trace,
                        handoff=final_handoff,
                        orchestration_path=list(trace.orchestration_path),
                    )

                handoff = segment_result.handoff
                await publish(AgentHandoffRequestedEvent(handoff=handoff))
                trace.handoff_count += 1
                if current_agent.run_limits.max_handoffs is not None and trace.handoff_count > current_agent.run_limits.max_handoffs:
                    raise RuntimeError(f'Agent exceeded max handoffs ({current_agent.run_limits.max_handoffs}).')
                next_agent = current_agent.subagents.get(handoff.target_agent) or self._registry.get(handoff.target_agent)
                if next_agent is None:
                    await publish(
                        AgentHandoffFailedEvent(
                            source_agent=current_agent.name,
                            target_agent=handoff.target_agent,
                            reason="Unknown handoff target.",
                        )
                    )
                    raise RuntimeError(f'Unknown handoff target "{handoff.target_agent}".')
                await publish(
                    AgentHandoffResolvedEvent(
                        source_agent=current_agent.name,
                        target_agent=next_agent.name,
                    )
                )
                await publish(AgentHandoffEvent(handoff=handoff))
                current_agent = next_agent
                trace.orchestration_path.append(next_agent.name)
                current_prompt = handoff.input or f"Continue the delegated task from {trace.orchestration_path[-2]}."
                current_messages = None
                handoff_depth += 1
        except Exception as error:
            await publish(AgentErrorEvent(error=error))
            trace.finished_at_ms = _now_ms()
            raise

    async def _run_single(
        self,
        *,
        agent: Agent,
        session: AgentSession,
        run_id: str,
        trace: AgentTrace,
        prompt: str | None,
        messages: list[ModelMessage] | None,
        tools: ToolSet | ToolRegistry | None,
        tool_choice: str | ToolChoiceName | None,
        tool_execution: ToolExecutionOptions | None,
        max_steps: int | None,
        temperature: float | None,
        max_tokens: int | None,
        reasoning: ReasoningConfig | None,
        provider_options: dict[str, Any] | None,
        timeout_ms: int | None,
        max_retries: int | None,
        retry_backoff_ms: int | None,
        emit: Callable[[AgentEvent], Awaitable[None]],
    ) -> AgentRunResult:
        built_messages = _build_run_messages(
            agent=agent,
            session=session,
            prompt=prompt,
            messages=messages,
        )
        context = AgentContext(
            run_id=run_id,
            session_id=session.id,
            agent_name=agent.name,
            memory_summary=session.summary,
            metadata=dict(agent.metadata),
            handoff_path=list(trace.orchestration_path),
        )
        registry = _resolve_tool_registry(agent, tools)
        merged_tools = self._wrap_agent_tools(
            agent=agent,
            registry=registry,
            run_id=run_id,
            session_id=session.id,
            trace=trace,
            started_at_ms=trace.started_at_ms,
            context=context,
            emit=emit,
        )
        span = self._start_span(
            "zhivex.agent.model",
            {
                "agent.name": agent.name,
                "run.id": run_id,
                "session.id": session.id,
                "orchestration.depth": len(trace.orchestration_path) - 1,
            },
        )
        try:
            result = await generate_text(
                model=agent.model,
                messages=built_messages,
                tools=merged_tools or None,
                tool_choice=tool_choice,
                tool_execution=tool_execution,
                max_steps=_effective_max_steps(agent.run_limits, max_steps),
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
                provider_options=provider_options,
                timeout_ms=_effective_timeout_ms(agent.run_limits, timeout_ms),
                max_retries=max_retries,
                retry_backoff_ms=retry_backoff_ms,
            )
        except Exception as error:
            self._finish_span(span, error=error)
            raise
        self._finish_span(span, attributes={"finish.reason": result.finish_reason})

        for tool_call in _extract_tool_calls_from_steps(result.steps):
            await emit(AgentToolCallEvent(tool_call=tool_call))
        segment_text = _segment_text(result)
        segment_finish_reason = _segment_finish_reason(result)
        segment_provider_finish_reason = _segment_provider_finish_reason(result)
        if segment_text:
            await emit(AgentTextDeltaEvent(text_delta=segment_text))
        for tool_result in result.tool_results:
            await emit(AgentToolResultEvent(tool_result=tool_result))

        transcript = list(session.messages)
        if messages is not None:
            transcript.extend(messages)
        elif prompt is not None:
            transcript.append(create_text_message("user", prompt))
        for step in result.steps:
            transcript.extend(_response_messages(step))
        for tool_result in result.tool_results:
            transcript.append(ModelMessage(role="tool", parts=[tool_result_part(tool_result)]))
        session.messages = _strip_runtime_system_messages(transcript, agent.instructions)
        if agent.memory is not None and _should_refresh_summary(agent.memory, session):
            summary_span = self._start_span(
                "zhivex.agent.summary",
                {"agent.name": agent.name, "run.id": run_id, "session.id": session.id},
            )
            session.summary = await agent.memory.summarize(
                session_id=session.id,
                state=AgentMemoryState(
                    messages=list(session.messages),
                    summary=session.summary,
                    metadata=dict(session.metadata),
                ),
                agent=agent,
            )
            self._finish_span(summary_span, attributes={"summary.updated": bool(session.summary)})
            await emit(AgentSummaryUpdateEvent(summary=session.summary))
        if agent.memory is not None:
            await agent.memory.save(
                session.id,
                AgentMemoryState(
                    messages=list(session.messages),
                    summary=session.summary,
                    metadata=dict(session.metadata),
                ),
            )

        await _save_checkpoints(
            checkpoint_store=agent.checkpoint_store,
            result=result,
            run_id=run_id,
            session_id=session.id,
            agent_name=agent.name,
            emit=emit,
            trace=trace,
        )
        handoff = _detect_handoff(result.tool_results)
        return AgentRunResult(
            run_id=run_id,
            agent_name=agent.name,
            session=session,
            text=segment_text,
            finish_reason=segment_finish_reason,
            provider_finish_reason=segment_provider_finish_reason,
            usage=result.usage,
            steps=result.steps,
            messages=result.messages,
            tool_results=result.tool_results,
            trace=trace,
            handoff=handoff,
            orchestration_path=list(trace.orchestration_path),
        )

    def _wrap_agent_tools(
        self,
        *,
        agent: Agent,
        registry: ToolRegistry,
        run_id: str,
        session_id: str,
        trace: AgentTrace,
        started_at_ms: int,
        context: AgentContext,
        emit: Callable[[AgentEvent], Awaitable[None]],
    ) -> ToolSet:
        wrapped: ToolSet = {}
        tool_limit = agent.run_limits.max_tool_calls

        for tool_name, definition in registry.items():

            async def execute(
                input: Any,
                *,
                _tool_name: str = tool_name,
                _definition: ToolDefinition = definition,
            ) -> Any:
                if agent.run_limits.max_wall_time_ms is not None and _now_ms() - started_at_ms > agent.run_limits.max_wall_time_ms:
                    raise RuntimeError("Agent run exceeded max wall time.")

                trace.tool_call_count += 1
                if tool_limit is not None and trace.tool_call_count > tool_limit:
                    raise RuntimeError(f'Agent exceeded max tool calls ({tool_limit}).')

                request = ToolApprovalRequest(
                    run_id=run_id,
                    session_id=session_id,
                    agent_name=agent.name,
                    tool_name=_tool_name,
                    tool_input=input,
                    tool_permissions=list(_definition.permissions),
                    tool_source=_definition.source,
                    tool_metadata=dict(_definition.metadata),
                    context=context,
                    handoff_path=list(trace.orchestration_path),
                )
                decision = ApprovalDecision(approved=True)
                if _definition.requires_approval or (
                    _definition.requires_approval is None and agent.approval_policy is not None
                ):
                    trace.approval_count += 1
                    if agent.approval_policy is not None:
                        decision = _normalize_approval_decision(await _maybe_await(agent.approval_policy(request)))
                    await emit(
                        AgentToolApprovalEvent(
                            tool_name=_tool_name,
                            tool_input=input,
                            approved=decision.approved,
                            reason=decision.reason,
                        )
                    )
                if not decision.approved:
                    raise RuntimeError(decision.reason or f'Tool "{_tool_name}" denied by approval policy.')

                tool_context = ToolExecutionContext(
                    tool_name=_tool_name,
                    run_id=run_id,
                    session_id=session_id,
                    agent_name=agent.name,
                    memory_summary=context.memory_summary,
                    permissions=list(_definition.permissions),
                    source=_definition.source,
                    metadata={**context.metadata, **_definition.metadata},
                    handoff_path=list(trace.orchestration_path),
                )
                span = self._start_span(
                    "zhivex.agent.tool",
                    {
                        "tool.name": _tool_name,
                        "tool.source": _definition.source,
                        "agent.name": agent.name,
                        "run.id": run_id,
                    },
                )
                try:
                    result = await registry.execute(_definition, input, tool_context)
                except Exception as error:
                    self._finish_span(span, error=error)
                    raise
                self._finish_span(span)
                return result

            wrapped[tool_name] = ToolDefinition(
                name=definition.name,
                description=definition.description,
                schema=definition.schema,
                execute=execute,
                tags=list(definition.tags),
                requires_approval=definition.requires_approval,
                permissions=list(definition.permissions),
                source=definition.source,
                metadata=dict(definition.metadata),
                supports_streaming=definition.supports_streaming,
            )

        return wrapped

    def _start_span(self, name: str, attributes: dict[str, Any]) -> Any:
        if self._observer is None:
            return None
        return self._observer.start_span(name, attributes)

    def _finish_span(self, span: Any, *, attributes: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        if span is None:
            return
        span.end(attributes=attributes, error=error)


class AgentStreamResult:
    def __init__(self, runner: asyncio.Task[AgentRunResult], broadcast: _Broadcast) -> None:
        self._runner = runner
        self._broadcast = broadcast

    def event_stream(self) -> AsyncIterable[AgentEvent]:
        return self._broadcast.stream()

    def text_stream(self) -> AsyncIterable[str]:
        async def generator() -> AsyncIterable[str]:
            async for event in self._broadcast.stream():
                if isinstance(event, AgentTextDeltaEvent):
                    yield event.text_delta

        return generator()

    async def collect(self) -> AgentRunResult:
        return await self._runner


def run_agent(
    *,
    agent: Agent,
    session: AgentSession | None = None,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    tools: ToolSet | ToolRegistry | None = None,
    tool_choice: str | ToolChoiceName | None = None,
    tool_execution: ToolExecutionOptions | None = None,
    max_steps: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning: ReasoningConfig | None = None,
    provider_options: dict[str, Any] | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
    stop_on_handoff: bool = False,
    runtime: AgentRuntime | None = None,
    registry: AgentRegistry | None = None,
    observer: AgentObserver | None = None,
) -> Awaitable[AgentRunResult]:
    resolved_runtime = runtime or AgentRuntime(registry=registry, observer=observer)
    return resolved_runtime.run(
        agent=agent,
        session=session,
        prompt=prompt,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        tool_execution=tool_execution,
        max_steps=max_steps,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning=reasoning,
        provider_options=provider_options,
        timeout_ms=timeout_ms,
        max_retries=max_retries,
        retry_backoff_ms=retry_backoff_ms,
        stop_on_handoff=stop_on_handoff,
    )


def stream_agent(
    *,
    agent: Agent,
    session: AgentSession | None = None,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    tools: ToolSet | ToolRegistry | None = None,
    tool_choice: str | ToolChoiceName | None = None,
    tool_execution: ToolExecutionOptions | None = None,
    max_steps: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning: ReasoningConfig | None = None,
    provider_options: dict[str, Any] | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
    stop_on_handoff: bool = False,
    runtime: AgentRuntime | None = None,
    registry: AgentRegistry | None = None,
    observer: AgentObserver | None = None,
) -> AgentStreamResult:
    resolved_runtime = runtime or AgentRuntime(registry=registry, observer=observer)
    broadcast = _Broadcast(history=[])

    async def emit(event: AgentEvent) -> None:
        await broadcast.publish(event)

    async def runner() -> AgentRunResult:
        try:
            return await resolved_runtime.run(
                agent=agent,
                session=session,
                prompt=prompt,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                tool_execution=tool_execution,
                max_steps=max_steps,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
                provider_options=provider_options,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                retry_backoff_ms=retry_backoff_ms,
                stop_on_handoff=stop_on_handoff,
                emit=emit,
            )
        finally:
            await broadcast.close()

    return AgentStreamResult(asyncio.create_task(runner()), broadcast)
