from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, cast
from uuid import uuid4

from ._agent_budget import enter_budget_scope, exit_budget_scope
from ._agent_guardrail_messages import _GuardrailMessages
from ._agent_memory import InMemoryAgentMemory as InMemoryAgentMemory, InMemoryAgentCheckpointStore as InMemoryAgentCheckpointStore
from .runtime import ExecutionBudget, budgeted, execution_scope
from ._agent_run_state import (
    _detect_handoff as _detect_handoff,
    _assistant_messages_from_result as _assistant_messages_from_result,
    _replace_assistant_messages as _replace_assistant_messages,
    _replace_messages_by_identity as _replace_messages_by_identity,
    _apply_guarded_output as _apply_guarded_output,
    _extract_tool_calls_from_steps as _extract_tool_calls_from_steps,
    _extract_provider_tool_results_from_steps as _extract_provider_tool_results_from_steps,
    _step_status_from_result as _step_status_from_result,
    _child_runs_from_tool_results as _child_runs_from_tool_results,
    _agent_run_state_from_result as _agent_run_state_from_result,
    _agent_run_state_from_suspension as _agent_run_state_from_suspension,
    _segment_text as _segment_text,
    _segment_finish_reason as _segment_finish_reason,
    _segment_provider_finish_reason as _segment_provider_finish_reason,
    _save_checkpoints as _save_checkpoints,
    _agent_run_result_from_state as _agent_run_result_from_state,
)
from ._agent_streams import (
    AgentStreamResult as AgentStreamResult,
    LiveAgentStreamResult as LiveAgentStreamResult,
)
from ._agent_execution import (
    _strip_runtime_system_messages as _strip_runtime_system_messages,
    create_agent_session as create_agent_session,
    _record_trace_event as _record_trace_event,
    _execution_error as _execution_error,
)

from ._agent_context import _message_text as _message_text, _text_from_message as _text_from_message
from ._agent_execution import (
    _now_ms as _now_ms,
    _new_id as _new_id,
    _persist_agent_run_state as _persist_agent_run_state,
    _raise_if_agent_run_cancelled as _raise_if_agent_run_cancelled,
    _await_with_agent_cancellation as _await_with_agent_cancellation,
    _maybe_await as _maybe_await,
    _resolve_agent_instructions as _resolve_agent_instructions,
    _resolve_agent_structured_output as _resolve_agent_structured_output,
    _parse_agent_output as _parse_agent_output,
    _call_agent_hooks as _call_agent_hooks,
    _call_error_hooks_preserving as _call_error_hooks_preserving,
    _effective_max_steps as _effective_max_steps,
    _effective_timeout_ms as _effective_timeout_ms,
    _int_from_json as _int_from_json,
    _should_refresh_summary as _should_refresh_summary,
    _context_messages as _context_messages,
    _build_run_messages as _build_run_messages,
    _guardrail_name as _guardrail_name,
    _normalize_guardrail_result as _normalize_guardrail_result,
    _resolve_tool_registry as _resolve_tool_registry,
    _merge_usage as _merge_usage,
    _response_messages as _response_messages,
)
from ._agent_approvals import (
    allow_all_approval_policy as allow_all_approval_policy,
    deny_all_approval_policy as deny_all_approval_policy,
    permission_allowlist_approval_policy as permission_allowlist_approval_policy,
    _normalize_approval_decision as _normalize_approval_decision,
    _pending_approval_from_request as _pending_approval_from_request,
    _execute_resolved_approval_tool as _execute_resolved_approval_tool,
    _validate_resolved_approval_tool as _validate_resolved_approval_tool,
)
from ._agent_skills import (
    _skill_reference_name as _skill_reference_name,
    _SkillActivation as _SkillActivation,
    _SkillSkip as _SkillSkip,
    _normalize_skill_names as _normalize_skill_names,
    _tokenize_skill_text as _tokenize_skill_text,
    _matches_any_phrase as _matches_any_phrase,
    _explicit_skill_requested as _explicit_skill_requested,
    _should_activate_skill_implicitly as _should_activate_skill_implicitly,
    _skill_allowed_for_agent as _skill_allowed_for_agent,
    _skill_activation_sort_key as _skill_activation_sort_key,
    _skill_system_message as _skill_system_message,
    _resolve_skill_registry as _resolve_skill_registry,
    _sticky_skill_names as _sticky_skill_names,
    _resolve_skill_tools as _resolve_skill_tools,
    _select_active_skills as _select_active_skills,
    _persist_active_skills as _persist_active_skills,
    _emit_skill_events as _emit_skill_events,
    _extract_skill_artifacts as _extract_skill_artifacts,
)

from ._agent_contracts import (
    HANDOFF_MARKER as HANDOFF_MARKER,
    AgentDepsT as AgentDepsT,
    AgentOutputT as AgentOutputT,
    SkillActivationMode as SkillActivationMode,
    AgentHandoff as AgentHandoff,
    AgentCancellationToken as AgentCancellationToken,
    AgentContext as AgentContext,
    DynamicInstructions as DynamicInstructions,
    AgentHooks as AgentHooks,
    AgentRunRequest as AgentRunRequest,
    AgentMiddlewareNext as AgentMiddlewareNext,
    AgentMiddleware as AgentMiddleware,
    ApprovalDecision as ApprovalDecision,
    ToolApprovalRequest as ToolApprovalRequest,
    ApprovalPolicy as ApprovalPolicy,
    GuardrailResult as GuardrailResult,
    InputGuardrailRequest as InputGuardrailRequest,
    OutputGuardrailRequest as OutputGuardrailRequest,
    InputGuardrail as InputGuardrail,
    OutputGuardrail as OutputGuardrail,
    GuardrailTripwireTriggered as GuardrailTripwireTriggered,
    SummaryConfig as SummaryConfig,
    AgentMemoryState as AgentMemoryState,
    AgentMemory as AgentMemory,
    AgentCheckpointStore as AgentCheckpointStore,
    RunLimits as RunLimits,
    AgentSession as AgentSession,
    ToolRuntime as ToolRuntime,
    AgentCheckpoint as AgentCheckpoint,
    AgentRunStartEvent as AgentRunStartEvent,
    AgentDelegationStartEvent as AgentDelegationStartEvent,
    AgentDelegationFinishEvent as AgentDelegationFinishEvent,
    AgentTextDeltaEvent as AgentTextDeltaEvent,
    AgentToolCallEvent as AgentToolCallEvent,
    AgentToolApprovalEvent as AgentToolApprovalEvent,
    AgentToolResultEvent as AgentToolResultEvent,
    AgentSkillActivatedEvent as AgentSkillActivatedEvent,
    AgentSkillResolvedEvent as AgentSkillResolvedEvent,
    AgentSkillDependencyCheckEvent as AgentSkillDependencyCheckEvent,
    AgentSkillSkippedEvent as AgentSkillSkippedEvent,
    AgentSkillExecutionStartEvent as AgentSkillExecutionStartEvent,
    AgentSkillExecutionFinishEvent as AgentSkillExecutionFinishEvent,
    AgentSkillArtifactCreatedEvent as AgentSkillArtifactCreatedEvent,
    AgentGuardrailEvent as AgentGuardrailEvent,
    AgentCheckpointEvent as AgentCheckpointEvent,
    AgentSummaryUpdateEvent as AgentSummaryUpdateEvent,
    AgentHandoffRequestedEvent as AgentHandoffRequestedEvent,
    AgentHandoffResolvedEvent as AgentHandoffResolvedEvent,
    AgentHandoffFailedEvent as AgentHandoffFailedEvent,
    AgentHandoffEvent as AgentHandoffEvent,
    AgentFinishEvent as AgentFinishEvent,
    AgentErrorEvent as AgentErrorEvent,
    AgentEvent as AgentEvent,
    AgentTraceSegment as AgentTraceSegment,
    AgentTrace as AgentTrace,
    AgentRunResult as AgentRunResult,
    AgentLiveEvent as AgentLiveEvent,
)
from ._agent_tools import (
    _invoke_tool_callable as _invoke_tool_callable,
    _stable_fingerprint_value as _stable_fingerprint_value,
    _callable_fingerprint as _callable_fingerprint,
    _tool_definition_fingerprint as _tool_definition_fingerprint,
    _tool_callable_mode as _tool_callable_mode,
    LocalToolRuntime as LocalToolRuntime,
    UnsupportedToolRuntime as UnsupportedToolRuntime,
    HTTPRemoteToolRuntime as HTTPRemoteToolRuntime,
    _normalize_mcp_content_item as _normalize_mcp_content_item,
    _normalize_mcp_result as _normalize_mcp_result,
    _mcp_result_is_error as _mcp_result_is_error,
    MCPToolRuntime as MCPToolRuntime,
    ToolRegistry as ToolRegistry,
    mcp_stdio_server as mcp_stdio_server,
    mcp_http_server as mcp_http_server,
    _sanitize_tool_name as _sanitize_tool_name,
    _build_mcp_local_tool_name as _build_mcp_local_tool_name,
    _mcp_tool_annotations as _mcp_tool_annotations,
    _mcp_tool_security_classification as _mcp_tool_security_classification,
    _load_mcp_tool_definitions as _load_mcp_tool_definitions,
    discover_mcp_tools as discover_mcp_tools,
    create_mcp_tool_registry as create_mcp_tool_registry,
)

from ._agent_persistence import (
    _json_dumps as _json_dumps,
    _json_loads as _json_loads,
    _coerce_json_payload as _coerce_json_payload,
    _serialize_agent_memory_state as _serialize_agent_memory_state,
    _deserialize_agent_memory_state as _deserialize_agent_memory_state,
    _serialize_agent_checkpoint as _serialize_agent_checkpoint,
    _deserialize_agent_checkpoint as _deserialize_agent_checkpoint,
    SQLiteAgentMemoryStore as SQLiteAgentMemoryStore,
    SQLiteAgentCheckpointStore as SQLiteAgentCheckpointStore,
    PostgresAgentMemoryStore as PostgresAgentMemoryStore,
    PostgresAgentCheckpointStore as PostgresAgentCheckpointStore,
    create_sqlite_agent_memory_store as create_sqlite_agent_memory_store,
    create_sqlite_checkpoint_store as create_sqlite_checkpoint_store,
    create_postgres_agent_memory_store as create_postgres_agent_memory_store,
    create_postgres_checkpoint_store as create_postgres_checkpoint_store,
)

from ._streaming import Broadcast, DEFAULT_STREAM_BUFFER_SIZE
from ._serde import (
    deserialize_messages,
)
from .errors import (
    AgentEventDeliveryError,
    AgentRunCancelled,
    ToolExecutionSuspended,
    ValidationError,
)
from .agent_state import (
    AgentRunState,
    AgentRunStore,
    PendingApproval,
    agent_child_run_from_state,
    fail_agent_run_resume_claim,
)
from .generate_text import generate_text, stream_text
from .messages import is_callable_tool_definition, provider_data_part, tool_result_part
from .schema import create_schema_adapter
from .skills import SkillArtifact, SkillRegistry, SkillSet
from .types import (
    GenerateResult,
    GenerateTextOutput,
    GenerateTextStep,
    JsonValue,
    LanguageModel,
    ModelGenerateInput,
    ModelMessage,
    ReasoningConfig,
    RealtimeConnectOptions,
    RealtimeModel,
    RealtimeSessionConfig,
    StreamProviderDataEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    StreamToolResultEvent,
    ToolChoiceName,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionOptions,
    ToolExecutionResult,
    ToolGuardrailResult,
    ToolGuardrailTripwireTriggered,
    ToolInputGuardrailRequest,
    ToolOutputGuardrailRequest,
    ToolSet,
    TokenUsage,
    ToolCall,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


SUMMARY_MARKER = "Conversation summary:\n"
_POSTGRES_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def handoff_to(target_agent: str, *, input: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        HANDOFF_MARKER: True,
        "target_agent": target_agent,
        "input": input,
        "metadata": metadata or {},
    }


def _validate_postgres_table_prefix(table_prefix: str) -> str:
    if not _POSTGRES_IDENTIFIER_RE.match(table_prefix):
        raise ValidationError(
            'The "table_prefix" field must match the SQL identifier pattern [A-Za-z_][A-Za-z0-9_]*.'
        )
    return table_prefix


@dataclass(slots=True)
class Agent(Generic[AgentDepsT, AgentOutputT]):
    name: str
    model: LanguageModel | RealtimeModel
    instructions: (
        str
        | Callable[[AgentContext[AgentDepsT]], str | None | Awaitable[str | None]]
        | Callable[[AgentContext[AgentDepsT], Agent[AgentDepsT, AgentOutputT]], str | None | Awaitable[str | None]]
        | None
    ) = None
    tools: ToolSet | ToolRegistry = field(default_factory=dict)
    skills: SkillSet | SkillRegistry = field(default_factory=dict)
    subagents: dict[str, "Agent[AgentDepsT, Any]"] = field(default_factory=dict)
    memory: AgentMemory | None = None
    checkpoint_store: AgentCheckpointStore | None = None
    run_store: AgentRunStore | None = None
    approval_policy: ApprovalPolicy | None = None
    input_guardrails: list[InputGuardrail] = field(default_factory=list)
    output_guardrails: list[OutputGuardrail] = field(default_factory=list)
    tool_execution: ToolExecutionOptions | None = None
    run_limits: RunLimits = field(default_factory=RunLimits)
    metadata: dict[str, Any] = field(default_factory=dict)
    output_type: type[AgentOutputT] | None = None
    output_mode: Literal["auto", "native", "prompted"] = "auto"
    output_name: str | None = None
    output_description: str | None = None
    hooks: list[AgentHooks] = field(default_factory=list)
    middleware: list[AgentMiddleware] = field(default_factory=list)
    trace_event_limit: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.trace_event_limit is not None and (type(self.trace_event_limit) is not int or self.trace_event_limit < 1):
            raise ValidationError("Agent.trace_event_limit must be a positive integer or None.")
        if self.output_mode not in {"auto", "native", "prompted"}:
            raise ValidationError('Agent.output_mode must be "auto", "native", or "prompted".')


class _LifecycleLanguageModel:
    def __init__(
        self,
        model: LanguageModel,
        *,
        agent: Agent[Any, Any],
        context: AgentContext[Any],
        hooks: list[AgentHooks],
    ) -> None:
        self._model = model
        self._agent = agent
        self._context = context
        self._hooks = hooks
        self.provider = model.provider
        self.model_id = model.model_id
        self.capabilities = model.capabilities
        from .safety import RedactionPolicy

        self._redactions = [
            owner for guardrail in agent.input_guardrails
            if isinstance(owner := getattr(guardrail, "__self__", None), RedactionPolicy)
        ]

    def _redact_input(self, input: ModelGenerateInput) -> None:
        for policy in self._redactions:
            input.messages = policy.redact_messages(input.messages)

    def _raise_if_cancelled(self) -> None:
        if self._context.cancellation_token is not None:
            self._context.cancellation_token.raise_if_cancelled(self._context.run_id)

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        self._raise_if_cancelled()
        self._redact_input(input)
        await _call_agent_hooks(self._hooks, "on_model_start", self._context, self._agent, input)
        result = await _await_with_agent_cancellation(
            self._model.generate(input),
            cancellation_token=self._context.cancellation_token,
            run_id=self._context.run_id,
        )
        self._raise_if_cancelled()
        await _call_agent_hooks(
            self._hooks,
            "on_model_end",
            self._context,
            self._agent,
            result,
            reverse=True,
        )
        return result

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[Any]:
        self._raise_if_cancelled()
        self._redact_input(input)
        await _call_agent_hooks(self._hooks, "on_model_start", self._context, self._agent, input)
        events = await _await_with_agent_cancellation(
            self._model.stream(input),
            cancellation_token=self._context.cancellation_token,
            run_id=self._context.run_id,
        )

        async def generator() -> AsyncIterable[Any]:
            iterator = events.__aiter__()
            try:
                while True:
                    try:
                        event = await _await_with_agent_cancellation(
                            iterator.__anext__(),
                            cancellation_token=self._context.cancellation_token,
                            run_id=self._context.run_id,
                        )
                    except StopAsyncIteration:
                        break
                    self._raise_if_cancelled()
                    yield event
                self._raise_if_cancelled()
                await _call_agent_hooks(
                    self._hooks,
                    "on_model_end",
                    self._context,
                    self._agent,
                    None,
                    reverse=True,
                )
            finally:
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    await close()

        return generator()


def set_agent_session_skills(session: AgentSession, *skill_names: str) -> AgentSession:
    names = _normalize_skill_names(skill_names)
    session.metadata = {
        **session.metadata,
        "sticky_skills": names,
        "active_skills": [],
    }
    return session


def get_agent_session_skills(session: AgentSession) -> list[str]:
    return _sticky_skill_names(session)


def clear_agent_session_skills(session: AgentSession) -> AgentSession:
    session.metadata = {
        **session.metadata,
        "sticky_skills": [],
        "active_skills": [],
    }
    return session


def create_in_memory_agent_memory_store(*, summary_config: SummaryConfig | None = None) -> InMemoryAgentMemory:
    return InMemoryAgentMemory(summary_config=summary_config)


def create_in_memory_checkpoint_store() -> InMemoryAgentCheckpointStore:
    return InMemoryAgentCheckpointStore()


async def load_agent_session(agent: Agent, session_id: str, *, metadata: dict[str, Any] | None = None) -> AgentSession:
    messages: list[ModelMessage] = []
    summary: str | None = None
    workflow_state: dict[str, JsonValue] = {}
    merged_metadata = dict(metadata or {})
    if agent.memory is not None:
        state = await agent.memory.load(session_id)
        messages = list(state.messages)
        summary = state.summary
        raw_workflow_state = state.metadata.get("state") or state.metadata.get("workflow_state")
        if isinstance(raw_workflow_state, dict):
            workflow_state = dict(raw_workflow_state)
        merged_metadata = {**state.metadata, **merged_metadata}
    return AgentSession(id=session_id, messages=messages, summary=summary, state=workflow_state, metadata=merged_metadata)


@dataclass(slots=True)
class _ProviderManagedApproval:
    provider: str
    approval_request_id: str
    tool_name: str
    tool_input: Any
    server_label: str | None
    raw_payload: Any


@dataclass(slots=True)
class _ProviderManagedToolTraceEvent:
    provider: str
    event_key: str
    tool_call: ToolCall
    raw_payload: Any


def _decode_provider_managed_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments


def _parse_provider_managed_approval_part(part: Any) -> _ProviderManagedApproval | None:
    if getattr(part, "type", None) != "provider-data":
        return None
    provider = str(getattr(part, "provider", "") or "")
    parsed: Any
    if provider == "openai":
        from .providers.openai import parse_openai_provider_data_part

        parsed = parse_openai_provider_data_part(part)
    elif provider == "azure-openai":
        from .providers.azure_openai import parse_azure_openai_provider_data_part

        parsed = parse_azure_openai_provider_data_part(part)
    else:
        return None
    if parsed is None or getattr(parsed, "type", None) != "mcp_approval_request":
        return None
    return _ProviderManagedApproval(
        provider=provider,
        approval_request_id=str(getattr(parsed, "id", "") or ""),
        tool_name=str(getattr(parsed, "name", "") or ""),
        tool_input=_decode_provider_managed_arguments(getattr(parsed, "arguments", "")),
        server_label=str(getattr(parsed, "server_label", "") or "") or None,
        raw_payload=parsed,
    )


def _provider_managed_event_key(provider: str, payload: Any) -> str:
    event_type = str(getattr(payload, "type", "") or payload.__class__.__name__)
    identifier = getattr(payload, "id", None) or getattr(payload, "approval_request_id", None) or getattr(payload, "response_id", None)
    if identifier:
        return f"{provider}:{event_type}:{identifier}"
    try:
        stable_payload = json.dumps(
            asdict(cast("DataclassInstance", payload)) if is_dataclass(payload) and not isinstance(payload, type) else payload,
            sort_keys=True,
            default=str,
        )
    except TypeError:
        stable_payload = repr(payload)
    return f"{provider}:{event_type}:{stable_payload}"


def _parse_provider_managed_tool_trace_part(part: Any) -> _ProviderManagedToolTraceEvent | None:
    if getattr(part, "type", None) != "provider-data":
        return None
    provider = str(getattr(part, "provider", "") or "")
    parsed: Any
    tool_call: ToolCall | None
    if provider == "openai":
        from .providers.openai import openai_provider_data_tool_call, parse_openai_provider_data_part

        parsed = parse_openai_provider_data_part(part)
        tool_call = openai_provider_data_tool_call(parsed) if parsed is not None else None
    elif provider == "azure-openai":
        from .providers.azure_openai import azure_openai_provider_data_tool_call, parse_azure_openai_provider_data_part

        parsed = parse_azure_openai_provider_data_part(part)
        tool_call = azure_openai_provider_data_tool_call(parsed) if parsed is not None else None
    else:
        return None
    if parsed is None or tool_call is None:
        return None
    return _ProviderManagedToolTraceEvent(
        provider=provider,
        event_key=_provider_managed_event_key(provider, parsed),
        tool_call=tool_call,
        raw_payload=parsed,
    )


def _provider_managed_approval_response_message(
    approval: _ProviderManagedApproval,
    *,
    approved: bool,
    reason: str | None = None,
) -> ModelMessage:
    if approval.provider == "openai":
        from .providers.openai import openai_mcp_approval_response

        part = openai_mcp_approval_response(
            approval_request_id=approval.approval_request_id,
            approve=approved,
            reason=reason,
        )
    else:
        from .providers.azure_openai import azure_openai_mcp_approval_response

        part = azure_openai_mcp_approval_response(
            approval_request_id=approval.approval_request_id,
            approve=approved,
            reason=reason,
        )
    return ModelMessage(role="assistant", parts=[part])


class AgentRegistry:
    def __init__(self, agents: dict[str, Agent] | None = None) -> None:
        self._agents: dict[str, Agent] = dict(agents or {})

    def register(self, agent: Agent) -> Agent:
        self._agents[agent.name] = agent
        return agent

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)


class AgentObserver(Protocol):
    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Any: ...


class AgentRuntime:
    def __init__(
        self,
        *,
        registry: AgentRegistry | None = None,
        observer: AgentObserver | None = None,
        hooks: Iterable[AgentHooks] | None = None,
        middleware: Iterable[AgentMiddleware] | None = None,
    ) -> None:
        self._registry = registry or AgentRegistry()
        self._observer = observer
        self._hooks = list(hooks or [])
        self._middleware = list(middleware or [])

    @budgeted
    async def run(
        self,
        *,
        agent: Agent[AgentDepsT, AgentOutputT],
        session: AgentSession | None = None,
        prompt: str | None = None,
        messages: list[ModelMessage] | None = None,
        deps: AgentDepsT | None = None,
        tools: ToolSet | ToolRegistry | None = None,
        skills: SkillSet | SkillRegistry | None = None,
        tool_choice: str | ToolChoiceName | None = None,
        tool_execution: ToolExecutionOptions | None = None,
        max_steps: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: ReasoningConfig | None = None,
        provider_options: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
        total_timeout_ms: int | None = None,
        retry_jitter: float | None = None,
        max_retries: int | None = None,
        retry_backoff_ms: int | None = None,
        stop_on_handoff: bool = False,
        emit: Callable[[AgentEvent], Awaitable[None]] | None = None,
        resumed_from_checkpoint: AgentCheckpoint | None = None,
        live_stream: bool = False,
        parent_run_id: str | None = None,
        idempotency_key: str | None = None,
        cancellation_token: AgentCancellationToken | None = None,
        hooks: Iterable[AgentHooks] | None = None,
        middleware: Iterable[AgentMiddleware] | None = None,
    ) -> AgentRunResult[AgentOutputT]:
        request = AgentRunRequest(
            agent=agent,
            session=session,
            prompt=prompt,
            messages=messages,
            deps=deps,
            cancellation_token=cancellation_token,
            metadata={"parent_run_id": parent_run_id, "idempotency_key": idempotency_key},
        )
        resolved_middleware = [*self._middleware, *list(middleware or []), *agent.middleware]

        async def call_at(
            index: int,
            current: AgentRunRequest[Any, Any],
        ) -> AgentRunResult[Any]:
            if index >= len(resolved_middleware):
                return await self._run_impl(
                    agent=current.agent,
                    session=current.session,
                    prompt=current.prompt,
                    messages=current.messages,
                    deps=current.deps,
                    tools=tools,
                    skills=skills,
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
                    resumed_from_checkpoint=resumed_from_checkpoint,
                    live_stream=live_stream,
                    parent_run_id=parent_run_id,
                    idempotency_key=idempotency_key,
                    cancellation_token=current.cancellation_token,
                    hooks=hooks,
                )

            async def call_next(next_request: AgentRunRequest[Any, Any]) -> AgentRunResult[Any]:
                return await call_at(index + 1, next_request)

            result = await _maybe_await(resolved_middleware[index](current, call_next))
            if not isinstance(result, AgentRunResult):
                raise TypeError("Agent middleware must return AgentRunResult.")
            return result

        started_at_ms = _now_ms()
        root_span = self._start_span(
            "zhivex.agent.run",
            {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": agent.name,
                "gen_ai.provider.name": str(getattr(agent.model, "provider", "")),
                "gen_ai.request.model": str(getattr(agent.model, "model_id", "")),
                "session.id": session.id if session is not None else "",
                "run.parent_id": parent_run_id or "",
                "run.idempotency_key": idempotency_key or "",
            },
        )
        try:
            result = cast(AgentRunResult[AgentOutputT], await call_at(0, request))
        except BaseException as error:
            error_attributes = {
                "zhivex.duration_ms": max(0, _now_ms() - started_at_ms),
                "zhivex.run.status": (
                    "cancelled"
                    if isinstance(error, (AgentRunCancelled, asyncio.CancelledError))
                    else "failed"
                ),
            }
            if isinstance(error, Exception):
                self._finish_span(root_span, attributes=error_attributes, error=error)
            else:
                self._finish_span(root_span, attributes=error_attributes)
            raise
        result_attributes: dict[str, Any] = {
            "run.id": result.run_id,
            "session.id": result.session.id,
            "zhivex.duration_ms": max(0, _now_ms() - started_at_ms),
            "zhivex.run.status": (
                result.state.status
                if result.state is not None
                else ("failed" if result.finish_reason == "error" else "completed")
            ),
            "gen_ai.response.finish_reasons": [result.finish_reason or "unknown"],
        }
        if result.usage is not None:
            if result.usage.input_tokens is not None:
                result_attributes["gen_ai.usage.input_tokens"] = result.usage.input_tokens
            if result.usage.output_tokens is not None:
                result_attributes["gen_ai.usage.output_tokens"] = result.usage.output_tokens
            if result.usage.total_tokens is not None:
                result_attributes["gen_ai.usage.total_tokens"] = result.usage.total_tokens
        self._finish_span(root_span, attributes=result_attributes)
        return result

    async def _run_impl(
        self,
        *,
        agent: Agent[Any, Any],
        session: AgentSession | None = None,
        prompt: str | None = None,
        messages: list[ModelMessage] | None = None,
        deps: Any = None,
        tools: ToolSet | ToolRegistry | None = None,
        skills: SkillSet | SkillRegistry | None = None,
        tool_choice: str | ToolChoiceName | None = None,
        tool_execution: ToolExecutionOptions | None = None,
        max_steps: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: ReasoningConfig | None = None,
        provider_options: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
        total_timeout_ms: int | None = None,
        retry_jitter: float | None = None,
        max_retries: int | None = None,
        retry_backoff_ms: int | None = None,
        stop_on_handoff: bool = False,
        emit: Callable[[AgentEvent], Awaitable[None]] | None = None,
        resumed_from_checkpoint: AgentCheckpoint | None = None,
        live_stream: bool = False,
        parent_run_id: str | None = None,
        idempotency_key: str | None = None,
        cancellation_token: AgentCancellationToken | None = None,
        hooks: Iterable[AgentHooks] | None = None,
    ) -> AgentRunResult[Any]:
        resolved_session = session or create_agent_session()
        if agent.memory is not None and not resolved_session.messages and resolved_session.summary is None:
            state = await agent.memory.load(resolved_session.id)
            resolved_session.messages = list(state.messages)
            resolved_session.summary = state.summary
            resolved_session.metadata = {**state.metadata, **resolved_session.metadata}

        run_id = _new_id("run")
        started_at_ms = _now_ms()
        initial_state = AgentRunState(
            run_id=run_id,
            agent_name=agent.name,
            provider=str(getattr(agent.model, "provider", "")),
            model_id=str(getattr(agent.model, "model_id", "")),
            session_id=resolved_session.id,
            parent_run_id=parent_run_id,
            idempotency_key=idempotency_key,
            started_at_ms=started_at_ms,
            updated_at_ms=started_at_ms,
            metadata={"orchestration_path": [agent.name]},
        )
        if agent.run_store is not None:
            if idempotency_key:
                claim_idempotency_key = getattr(agent.run_store, "claim_idempotency_key", None)
                if not callable(claim_idempotency_key):
                    raise ValidationError(
                        "Idempotent agent runs require an AgentRunStore with atomic "
                        "claim_idempotency_key(...)."
                    )
                claimed_state = await claim_idempotency_key(initial_state)
                if claimed_state.run_id != run_id:
                    return _agent_run_result_from_state(
                        claimed_state,
                        resolved_session,
                        agent=agent,
                    )
            else:
                await agent.run_store.save(initial_state)
        trace = AgentTrace(
            run_id=run_id,
            session_id=resolved_session.id,
            agent_name=agent.name,
            started_at_ms=started_at_ms,
            orchestration_path=[agent.name],
        )

        async def publish(event: AgentEvent, *, durable_state_committed: bool = False) -> None:
            _record_trace_event(trace, event, agent.trace_event_limit)
            if emit is not None:
                try:
                    await emit(event)
                except Exception as error:
                    raise AgentEventDeliveryError(
                        run_id,
                        event_type=event.type,
                        durable_state_committed=durable_state_committed,
                    ) from error

        current_agent = agent
        current_prompt = prompt
        current_messages = messages
        handoff_depth = 0
        accumulated_steps: list[GenerateTextStep] = []
        accumulated_tool_results: list[ToolExecutionResult] = []
        accumulated_artifacts: list[SkillArtifact] = []
        accumulated_usages: list[TokenUsage | None] = []
        call_hooks = list(hooks or [])
        run_hooks = [*self._hooks, *call_hooks]
        run_budget = (
            ExecutionBudget(time.monotonic() + agent.run_limits.max_wall_time_ms / 1000)
            if agent.run_limits.max_wall_time_ms is not None
            else None
        )
        budget_scope, budget_token = enter_budget_scope(agent.output_guardrails)
        try:
            await _raise_if_agent_run_cancelled(
                run_store=agent.run_store,
                run_id=run_id,
                cancellation_token=cancellation_token,
            )
            await publish(AgentRunStartEvent(run_id=run_id, session_id=resolved_session.id, agent_name=agent.name))
            while True:
                await _raise_if_agent_run_cancelled(
                    run_store=agent.run_store,
                    run_id=run_id,
                    cancellation_token=cancellation_token,
                )
                budget_scope.add_guardrails(current_agent.output_guardrails)
                effective_hooks = [*run_hooks, *current_agent.hooks]
                trace.segments.append(AgentTraceSegment(agent_name=current_agent.name, started_at_ms=_now_ms()))
                await publish(AgentDelegationStartEvent(agent_name=current_agent.name, handoff_depth=handoff_depth))

                async def run_segment() -> AgentRunResult:
                    return await self._run_single(
                        agent=current_agent,
                        output_agent=agent,
                        session=resolved_session,
                        run_id=run_id,
                        trace=trace,
                        prompt=current_prompt,
                        messages=current_messages,
                        tools=tools,
                        skills=skills,
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
                        live_stream=live_stream,
                        deps=deps,
                        cancellation_token=cancellation_token,
                        hooks=effective_hooks,
                        child_hooks=call_hooks,
                    )

                if run_budget is None:
                    segment_result = await run_segment()
                else:
                    try:
                        remaining_seconds = run_budget.remaining_seconds()
                    except TimeoutError as error:
                        raise RuntimeError("Agent run exceeded max wall time.") from error
                    try:
                        async with asyncio.timeout(remaining_seconds):
                            segment_result = await run_segment()
                    except TimeoutError as error:
                        raise RuntimeError("Agent run exceeded max wall time.") from error
                await _raise_if_agent_run_cancelled(
                    run_store=agent.run_store,
                    run_id=run_id,
                    cancellation_token=cancellation_token,
                )
                accumulated_steps.extend(segment_result.steps)
                accumulated_tool_results.extend(segment_result.tool_results)
                accumulated_artifacts.extend(segment_result.artifacts)
                accumulated_usages.append(segment_result.usage)
                trace.segments[-1].finished_at_ms = _now_ms()
                await publish(
                    AgentDelegationFinishEvent(
                        agent_name=current_agent.name,
                        handoff_depth=handoff_depth,
                        finish_reason=segment_result.finish_reason,
                    )
                )
                if segment_result.handoff is None or stop_on_handoff:
                    segment_context = AgentContext(
                        run_id=run_id,
                        session_id=resolved_session.id,
                        agent_name=current_agent.name,
                        memory_summary=resolved_session.summary,
                        metadata=dict(current_agent.metadata),
                        handoff_path=list(trace.orchestration_path),
                        deps=deps,
                        session=resolved_session,
                        cancellation_token=cancellation_token,
                    )
                    await _call_agent_hooks(
                        effective_hooks,
                        "on_agent_end",
                        segment_context,
                        current_agent,
                        segment_result,
                        reverse=True,
                    )
                    final_handoff = segment_result.handoff if stop_on_handoff else None
                    trace.finished_at_ms = _now_ms()
                    output = AgentRunResult(
                        run_id=run_id,
                        agent_name=segment_result.agent_name,
                        session=resolved_session,
                        text=segment_result.text,
                        output=segment_result.output,
                        finish_reason=segment_result.finish_reason,
                        provider_finish_reason=segment_result.provider_finish_reason,
                        usage=_merge_usage(accumulated_usages),
                        steps=list(accumulated_steps),
                        messages=segment_result.messages,
                        tool_results=list(accumulated_tool_results),
                        artifacts=list(accumulated_artifacts),
                        trace=trace,
                        handoff=final_handoff,
                        orchestration_path=list(trace.orchestration_path),
                        resumed_from_checkpoint=resumed_from_checkpoint,
                    )
                    run_state = _agent_run_state_from_result(
                        result=output,
                        agent=current_agent,
                        parent_run_id=parent_run_id,
                        idempotency_key=idempotency_key,
                        started_at_ms=started_at_ms,
                        finished_at_ms=trace.finished_at_ms,
                    )
                    if agent.run_store is not None:
                        run_state = await _persist_agent_run_state(agent.run_store, run_state)
                    output.state = run_state
                    await publish(
                        AgentFinishEvent(
                            run_id=run_id,
                            session_id=resolved_session.id,
                            text=segment_result.text,
                            finish_reason=segment_result.finish_reason,
                        ),
                        durable_state_committed=agent.run_store is not None,
                    )
                    return output

                handoff = segment_result.handoff
                await publish(AgentHandoffRequestedEvent(handoff=handoff))
                trace.handoff_count += 1
                handoff_span = self._start_span(
                    "zhivex.agent.handoff",
                    {
                        "gen_ai.operation.name": "handoff",
                        "gen_ai.agent.name": current_agent.name,
                        "zhivex.handoff.source_agent": current_agent.name,
                        "zhivex.handoff.target_agent": handoff.target_agent,
                        "run.id": run_id,
                        "session.id": resolved_session.id,
                        "orchestration.depth": handoff_depth,
                    },
                )
                if current_agent.run_limits.max_handoffs is not None and trace.handoff_count > current_agent.run_limits.max_handoffs:
                    handoff_error = RuntimeError(
                        f'Agent exceeded max handoffs ({current_agent.run_limits.max_handoffs}).'
                    )
                    self._finish_span(handoff_span, error=handoff_error)
                    raise handoff_error
                next_agent = current_agent.subagents.get(handoff.target_agent) or self._registry.get(handoff.target_agent)
                if next_agent is None:
                    await publish(
                        AgentHandoffFailedEvent(
                            source_agent=current_agent.name,
                            target_agent=handoff.target_agent,
                            reason="Unknown handoff target.",
                        )
                    )
                    handoff_error = RuntimeError(f'Unknown handoff target "{handoff.target_agent}".')
                    self._finish_span(handoff_span, error=handoff_error)
                    raise handoff_error
                self._finish_span(
                    handoff_span,
                    attributes={
                        "zhivex.handoff.resolved": True,
                        "zhivex.handoff.target_agent": next_agent.name,
                    },
                )
                await publish(
                    AgentHandoffResolvedEvent(
                        source_agent=current_agent.name,
                        target_agent=next_agent.name,
                    )
                )
                handoff_context = AgentContext(
                    run_id=run_id,
                    session_id=resolved_session.id,
                    agent_name=current_agent.name,
                    memory_summary=resolved_session.summary,
                    metadata=dict(current_agent.metadata),
                    handoff_path=list(trace.orchestration_path),
                    deps=deps,
                    session=resolved_session,
                    cancellation_token=cancellation_token,
                )
                await _call_agent_hooks(
                    effective_hooks,
                    "on_handoff",
                    handoff_context,
                    current_agent,
                    next_agent,
                    handoff,
                )
                await _call_agent_hooks(
                    effective_hooks,
                    "on_agent_end",
                    handoff_context,
                    current_agent,
                    segment_result,
                    reverse=True,
                )
                await publish(AgentHandoffEvent(handoff=handoff))
                current_agent = next_agent
                trace.orchestration_path.append(next_agent.name)
                current_prompt = handoff.input or f"Continue the delegated task from {trace.orchestration_path[-2]}."
                current_messages = None
                handoff_depth += 1
        except ToolExecutionSuspended as suspended:
            suspended_at_ms = _now_ms()
            trace.finished_at_ms = suspended_at_ms
            pending = cast(PendingApproval, suspended.pending_approval)
            suspended.steps = [*accumulated_steps, *suspended.steps]
            suspended.tool_results = [*accumulated_tool_results, *suspended.tool_results]
            run_state = _agent_run_state_from_suspension(
                run_id=run_id,
                agent_name=current_agent.name,
                session_id=resolved_session.id,
                agent=current_agent,
                parent_run_id=parent_run_id,
                idempotency_key=idempotency_key,
                started_at_ms=started_at_ms,
                suspended_at_ms=suspended_at_ms,
                suspended=suspended,
                orchestration_path=list(trace.orchestration_path),
            )
            suspended_result: AgentRunResult[Any] = AgentRunResult(
                run_id=run_id,
                agent_name=current_agent.name,
                session=resolved_session,
                text="",
                finish_reason="tool-calls",
                steps=list(cast(list[GenerateTextStep], suspended.steps)),
                messages=list(cast(list[ModelMessage], suspended.messages)),
                tool_results=list(cast(list[ToolExecutionResult], suspended.tool_results)),
                trace=trace,
                orchestration_path=list(trace.orchestration_path),
                resumed_from_checkpoint=resumed_from_checkpoint,
                state=run_state,
                provider_finish_reason=pending.reason,
            )
            suspended_context = AgentContext(
                run_id=run_id,
                session_id=resolved_session.id,
                agent_name=current_agent.name,
                memory_summary=resolved_session.summary,
                metadata=dict(current_agent.metadata),
                handoff_path=list(trace.orchestration_path),
                deps=deps,
                session=resolved_session,
                cancellation_token=cancellation_token,
            )
            await _call_agent_hooks(
                [*run_hooks, *current_agent.hooks],
                "on_agent_end",
                suspended_context,
                current_agent,
                suspended_result,
                reverse=True,
            )
            if agent.run_store is not None:
                try:
                    run_state = await _persist_agent_run_state(agent.run_store, run_state)
                except AgentRunCancelled as error:
                    await publish(AgentErrorEvent(error=error), durable_state_committed=True)
                    raise
                suspended_result.state = run_state
            await publish(
                AgentFinishEvent(
                    run_id=run_id,
                    session_id=resolved_session.id,
                    text="",
                    finish_reason="tool-calls",
                ),
                durable_state_committed=agent.run_store is not None,
            )
            return suspended_result
        except AgentRunCancelled as error:
            trace.finished_at_ms = _now_ms()
            if agent.run_store is not None:
                current_state = await agent.run_store.load(run_id)
                if current_state is not None and current_state.status != "cancelled":
                    cancel_run = getattr(agent.run_store, "cancel_run", None)
                    if callable(cancel_run):
                        await cancel_run(
                            run_id,
                            reason=error.reason,
                            cancelled_at_ms=trace.finished_at_ms,
                        )
            cancelled_context = AgentContext(
                run_id=run_id,
                session_id=resolved_session.id,
                agent_name=current_agent.name,
                memory_summary=resolved_session.summary,
                metadata=dict(current_agent.metadata),
                handoff_path=list(trace.orchestration_path),
                deps=deps,
                session=resolved_session,
                cancellation_token=cancellation_token,
            )
            await _call_error_hooks_preserving(
                [*run_hooks, *current_agent.hooks],
                cancelled_context,
                current_agent,
                error,
            )
            await publish(AgentErrorEvent(error=error), durable_state_committed=True)
            raise
        except BaseException as caught:
            failure_error = _execution_error(caught)
            trace.finished_at_ms = _now_ms()
            if isinstance(failure_error, AgentEventDeliveryError) and failure_error.durable_state_committed:
                raise
            if agent.run_store is not None:
                failed_result: AgentRunResult[Any] = AgentRunResult(
                    run_id=run_id,
                    agent_name=current_agent.name,
                    session=resolved_session,
                    text="",
                    finish_reason="error",
                    usage=_merge_usage(accumulated_usages),
                    steps=list(accumulated_steps),
                    tool_results=list(accumulated_tool_results),
                    artifacts=list(accumulated_artifacts),
                    trace=trace,
                    orchestration_path=list(trace.orchestration_path),
                )
                try:
                    await _persist_agent_run_state(
                        agent.run_store,
                        _agent_run_state_from_result(
                            result=failed_result,
                            agent=current_agent,
                            parent_run_id=parent_run_id,
                            idempotency_key=idempotency_key,
                            started_at_ms=started_at_ms,
                            finished_at_ms=trace.finished_at_ms,
                            error=str(failure_error),
                        ),
                    )
                except AgentRunCancelled as cancelled:
                    await publish(AgentErrorEvent(error=cancelled), durable_state_committed=True)
                    raise cancelled from failure_error
            if isinstance(failure_error, AgentEventDeliveryError):
                raise
            error_context = AgentContext(
                run_id=run_id,
                session_id=resolved_session.id,
                agent_name=current_agent.name,
                memory_summary=resolved_session.summary,
                metadata=dict(current_agent.metadata),
                handoff_path=list(trace.orchestration_path),
                deps=deps,
                session=resolved_session,
                cancellation_token=cancellation_token,
            )
            await _call_error_hooks_preserving(
                [*run_hooks, *current_agent.hooks],
                error_context,
                current_agent,
                failure_error,
            )
            await publish(
                AgentErrorEvent(error=failure_error),
                durable_state_committed=agent.run_store is not None,
            )
            raise failure_error
        finally:
            exit_budget_scope(budget_token)

    async def _run_input_guardrails(
        self,
        *,
        agent: Agent,
        run_id: str,
        session_id: str,
        prompt: str | None,
        messages: list[ModelMessage],
        context: AgentContext,
        trace: AgentTrace,
        emit: Callable[[AgentEvent], Awaitable[None]],
        caller_message_count: int = 0,
        caller_messages: list[ModelMessage] | None = None,
    ) -> InputGuardrailRequest:
        tracked = _GuardrailMessages(messages, caller_message_count) if caller_messages is not None else None
        request = InputGuardrailRequest(
            run_id=run_id,
            session_id=session_id,
            agent_name=agent.name,
            prompt=prompt,
            messages=tracked if tracked is not None else list(messages),
            context=context,
        )
        for guardrail in agent.input_guardrails:
            name = _guardrail_name(guardrail)
            span = self._start_span(
                "zhivex.agent.guardrail",
                {
                    "guardrail.name": name,
                    "guardrail.stage": "input",
                    "agent.name": agent.name,
                    "run.id": run_id,
                },
            )
            try:
                outcome = _normalize_guardrail_result(await _maybe_await(guardrail(request)))
                if tracked is not None and request.messages is not tracked:
                    tracked.replace_all(request.messages)
                    request.messages = tracked
            except Exception as error:
                self._finish_span(span, error=error)
                raise
            self._finish_span(
                span,
                attributes={
                    "guardrail.triggered": outcome.tripwire_triggered,
                },
            )
            await emit(
                AgentGuardrailEvent(
                    stage="input",
                    guardrail_name=name,
                    triggered=outcome.tripwire_triggered,
                    reason=outcome.reason,
                    metadata=dict(outcome.metadata),
                )
            )
            if outcome.tripwire_triggered:
                trace.guardrail_trigger_count += 1
                raise GuardrailTripwireTriggered(
                    stage="input",
                    guardrail_name=name,
                    reason=outcome.reason,
                    metadata=outcome.metadata,
                )
        if tracked is not None and caller_messages is not None:
            caller_messages[:] = tracked.caller_messages()
            request.messages = list(tracked)
        return request

    async def _run_output_guardrails(
        self,
        *,
        agent: Agent,
        run_id: str,
        session_id: str,
        result: GenerateTextOutput | None,
        text: str,
        messages: list[ModelMessage],
        context: AgentContext,
        trace: AgentTrace,
        emit: Callable[[AgentEvent], Awaitable[None]],
    ) -> OutputGuardrailRequest:
        request = OutputGuardrailRequest(
            run_id=run_id,
            session_id=session_id,
            agent_name=agent.name,
            text=text,
            messages=list(messages),
            result=result,
            context=context,
        )
        for guardrail in agent.output_guardrails:
            name = _guardrail_name(guardrail)
            span = self._start_span(
                "zhivex.agent.guardrail",
                {
                    "guardrail.name": name,
                    "guardrail.stage": "output",
                    "agent.name": agent.name,
                    "run.id": run_id,
                },
            )
            try:
                outcome = _normalize_guardrail_result(await _maybe_await(guardrail(request)))
            except Exception as error:
                self._finish_span(span, error=error)
                raise
            self._finish_span(
                span,
                attributes={
                    "guardrail.triggered": outcome.tripwire_triggered,
                },
            )
            await emit(
                AgentGuardrailEvent(
                    stage="output",
                    guardrail_name=name,
                    triggered=outcome.tripwire_triggered,
                    reason=outcome.reason,
                    metadata=dict(outcome.metadata),
                )
            )
            if outcome.tripwire_triggered:
                trace.guardrail_trigger_count += 1
                raise GuardrailTripwireTriggered(
                    stage="output",
                    guardrail_name=name,
                    reason=outcome.reason,
                    metadata=outcome.metadata,
                )
        return request

    async def _run_single(
        self,
        *,
        agent: Agent,
        output_agent: Agent[Any, Any],
        session: AgentSession,
        run_id: str,
        trace: AgentTrace,
        prompt: str | None,
        messages: list[ModelMessage] | None,
        tools: ToolSet | ToolRegistry | None,
        skills: SkillSet | SkillRegistry | None,
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
        live_stream: bool,
        deps: Any,
        cancellation_token: AgentCancellationToken | None,
        hooks: list[AgentHooks],
        child_hooks: list[AgentHooks],
    ) -> AgentRunResult:
        active_skill_activations, skipped_skills, skill_tools = await _select_active_skills(
            _resolve_skill_registry(agent, skills),
            agent=agent,
            session=session,
            prompt=prompt,
            messages=messages,
        )
        await _emit_skill_events(active_skills=active_skill_activations, skipped_skills=skipped_skills, emit=emit)
        context = AgentContext(
            run_id=run_id,
            session_id=session.id,
            agent_name=agent.name,
            memory_summary=session.summary,
            metadata={
                **dict(agent.metadata),
                "skills": [item.skill.name for item in active_skill_activations],
            },
            handoff_path=list(trace.orchestration_path),
            deps=deps,
            session=session,
            cancellation_token=cancellation_token,
        )
        await _call_agent_hooks(hooks, "on_agent_start", context, agent)
        resolved_instructions = await _resolve_agent_instructions(agent, context)
        structured_output, structured_output_instructions = _resolve_agent_structured_output(
            output_agent,
            model=agent.model,
        )
        built_messages = _build_run_messages(
            agent=agent,
            session=session,
            prompt=prompt,
            messages=messages,
            active_skills=[item.skill for item in active_skill_activations],
            instructions=resolved_instructions,
            structured_output_instructions=structured_output_instructions,
        )
        input_count = len(messages) if messages is not None else int(prompt is not None)
        transcript_input: list[ModelMessage] = []
        _persist_active_skills(session, active_skill_activations)
        guarded_input = await self._run_input_guardrails(
            agent=agent,
            run_id=run_id,
            session_id=session.id,
            prompt=prompt,
            messages=built_messages,
            context=context,
            trace=trace,
            emit=emit,
            caller_message_count=input_count,
            caller_messages=transcript_input,
        )
        built_messages = guarded_input.messages
        registry = _resolve_tool_registry(agent, tools)
        if agent.subagents:
            registry = registry.merge(
                {
                    name: create_subagent_tool(
                        name=name,
                        agent=subagent,
                        parent_run_id=run_id,
                        runtime=self,
                        hooks=child_hooks,
                    )
                    for name, subagent in agent.subagents.items()
                }
            )
        if skill_tools:
            registry = registry.merge(skill_tools)
        merged_tools = self._wrap_agent_tools(
            agent=agent,
            registry=registry,
            run_id=run_id,
            session_id=session.id,
            trace=trace,
            started_at_ms=trace.started_at_ms,
            context=context,
            emit=emit,
            hooks=hooks,
        )
        lifecycle_model = _LifecycleLanguageModel(
            cast(LanguageModel, agent.model),
            agent=agent,
            context=context,
            hooks=hooks,
        )
        span = self._start_span(
            "zhivex.agent.model",
            {
                "agent.name": agent.name,
                "run.id": run_id,
                "session.id": session.id,
                "orchestration.depth": len(trace.orchestration_path) - 1,
                "gen_ai.operation.name": "chat",
                "gen_ai.agent.name": agent.name,
                "gen_ai.provider.name": str(getattr(agent.model, "provider", "")),
                "gen_ai.request.model": str(getattr(agent.model, "model_id", "")),
            },
        )
        buffer_live_text = live_stream and agent.model.capabilities.streaming and bool(agent.output_guardrails)
        buffered_text_deltas: list[str] = []
        accumulated_steps: list[GenerateTextStep] = []
        accumulated_tool_results: list[ToolExecutionResult] = []
        conversation_messages = list(built_messages)
        persisted_run_messages: list[ModelMessage] = []
        remaining_steps = _effective_max_steps(agent.run_limits, max_steps)
        resolved_tool_execution = tool_execution if tool_execution is not None else agent.tool_execution

        async def resolve_provider_managed_approval(
            approval: _ProviderManagedApproval,
        ) -> ModelMessage:
            if agent.approval_policy is None:
                raise RuntimeError(
                    "Provider-managed approvals require an approval_policy on the agent."
                )
            trace.approval_count += 1
            request = ToolApprovalRequest(
                run_id=run_id,
                session_id=session.id,
                agent_name=agent.name,
                tool_name=approval.tool_name,
                tool_input=approval.tool_input,
                tool_permissions=[],
                tool_source="hosted",
                tool_metadata={
                    "provider": approval.provider,
                    "server_label": approval.server_label,
                    "hosted_tool_class": "remote-mcp",
                    "provider_event_type": "mcp_approval_request",
                    "raw_provider_payload": approval.raw_payload,
                },
                context=context,
                handoff_path=list(trace.orchestration_path),
            )
            decision = _normalize_approval_decision(await _maybe_await(agent.approval_policy(request)))
            await _call_agent_hooks(hooks, "on_approval", context, agent, request, decision)
            await emit(
                AgentToolApprovalEvent(
                    tool_name=approval.tool_name,
                    tool_input=approval.tool_input,
                    approved=decision.approved,
                    reason=decision.reason,
                    provider=approval.provider,
                    provider_managed=True,
                    approval_request_id=approval.approval_request_id,
                    tool_source="hosted",
                    metadata={
                        "provider": approval.provider,
                        "server_label": approval.server_label,
                        "hosted_tool_class": "remote-mcp",
                        "provider_event_type": "mcp_approval_request",
                        "raw_provider_payload": approval.raw_payload,
                    },
                )
            )
            return _provider_managed_approval_response_message(
                approval,
                approved=decision.approved,
                reason=decision.reason,
            )

        try:
            emitted_live_text = False
            emitted_live_tool_events = False
            while True:
                if remaining_steps is not None and remaining_steps <= 0:
                    raise RuntimeError("Agent run exceeded max steps while handling provider-managed approvals.")
                pending_provider_responses: list[ModelMessage] = []
                handled_provider_approvals: set[str] = set()
                handled_provider_tool_events: set[str] = set()
                if live_stream and agent.model.capabilities.streaming:

                    async def handle_stream_event(event: Any) -> None:
                        nonlocal emitted_live_text, emitted_live_tool_events
                        if isinstance(event, StreamTextDeltaEvent):
                            if buffer_live_text:
                                buffered_text_deltas.append(event.text_delta)
                            else:
                                emitted_live_text = True
                                await emit(AgentTextDeltaEvent(text_delta=event.text_delta))
                        elif isinstance(event, StreamToolCallEvent):
                            emitted_live_tool_events = True
                            await emit(AgentToolCallEvent(tool_call=event.tool_call))
                        elif isinstance(event, StreamToolResultEvent):
                            emitted_live_tool_events = True
                            await emit(AgentToolResultEvent(tool_result=event.tool_result))
                        elif isinstance(event, StreamProviderDataEvent):
                            provider_part = provider_data_part(event.provider, event.data)
                            provider_tool_event = _parse_provider_managed_tool_trace_part(provider_part)
                            if (
                                provider_tool_event is not None
                                and provider_tool_event.event_key not in handled_provider_tool_events
                            ):
                                handled_provider_tool_events.add(provider_tool_event.event_key)
                                emitted_live_tool_events = True
                                await emit(AgentToolCallEvent(tool_call=provider_tool_event.tool_call))
                            approval = _parse_provider_managed_approval_part(provider_part)
                            if approval is None or approval.approval_request_id in handled_provider_approvals:
                                return
                            handled_provider_approvals.add(approval.approval_request_id)
                            pending_provider_responses.append(await resolve_provider_managed_approval(approval))

                    streamed = stream_text(
                        model=lifecycle_model,
                        messages=conversation_messages,
                        tools=merged_tools or None,
                        tool_choice=cast(Any, tool_choice),
                        tool_execution=resolved_tool_execution,
                        max_steps=remaining_steps,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        reasoning=reasoning,
                        provider_options=provider_options,
                        timeout_ms=_effective_timeout_ms(agent.run_limits, timeout_ms),
                        max_retries=max_retries,
                        retry_backoff_ms=retry_backoff_ms,
                        structured_output=structured_output,
                        on_event=handle_stream_event,
                        stream_buffer_size=1,
                    )
                    result = await streamed.collect()
                else:
                    result = await generate_text(
                        model=lifecycle_model,
                        messages=conversation_messages,
                        tools=merged_tools or None,
                        tool_choice=cast(Any, tool_choice),
                        tool_execution=resolved_tool_execution,
                        max_steps=remaining_steps,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        reasoning=reasoning,
                        provider_options=provider_options,
                        timeout_ms=_effective_timeout_ms(agent.run_limits, timeout_ms),
                        max_retries=max_retries,
                        retry_backoff_ms=retry_backoff_ms,
                        structured_output=structured_output,
                    )

                accumulated_steps.extend(result.steps)
                accumulated_tool_results.extend(result.tool_results)
                if remaining_steps is not None:
                    remaining_steps -= max(1, len(result.steps))

                new_response_messages: list[ModelMessage] = []
                for step in result.steps:
                    response_messages = _response_messages(step)
                    if response_messages:
                        new_response_messages.extend(response_messages)
                        conversation_messages.extend(response_messages)
                        persisted_run_messages.extend(response_messages)
                for tool_result in result.tool_results:
                    tool_message = ModelMessage(role="tool", parts=[tool_result_part(tool_result)])
                    conversation_messages.append(tool_message)
                    persisted_run_messages.append(tool_message)

                for message in new_response_messages:
                    for part in message.parts:
                        provider_tool_event = _parse_provider_managed_tool_trace_part(part)
                        if provider_tool_event is not None and provider_tool_event.event_key not in handled_provider_tool_events:
                            handled_provider_tool_events.add(provider_tool_event.event_key)
                            await emit(AgentToolCallEvent(tool_call=provider_tool_event.tool_call))
                        approval = _parse_provider_managed_approval_part(part)
                        if approval is None or approval.approval_request_id in handled_provider_approvals:
                            continue
                        handled_provider_approvals.add(approval.approval_request_id)
                        pending_provider_responses.append(await resolve_provider_managed_approval(approval))

                if pending_provider_responses:
                    conversation_messages.extend(pending_provider_responses)
                    persisted_run_messages.extend(pending_provider_responses)
                    continue
                break
        except Exception as error:
            self._finish_span(span, error=error)
            raise
        model_attributes: dict[str, Any] = {
            "finish.reason": result.finish_reason,
            "gen_ai.response.finish_reasons": [result.finish_reason or "unknown"],
        }
        if result.usage is not None:
            if result.usage.input_tokens is not None:
                model_attributes["gen_ai.usage.input_tokens"] = result.usage.input_tokens
            if result.usage.output_tokens is not None:
                model_attributes["gen_ai.usage.output_tokens"] = result.usage.output_tokens
            if result.usage.total_tokens is not None:
                model_attributes["gen_ai.usage.total_tokens"] = result.usage.total_tokens
        self._finish_span(span, attributes=model_attributes)

        if not emitted_live_tool_events:
            for tool_call in _extract_tool_calls_from_steps(accumulated_steps):
                await emit(AgentToolCallEvent(tool_call=tool_call))
            for tool_result in _extract_provider_tool_results_from_steps(accumulated_steps):
                await emit(AgentToolResultEvent(tool_result=tool_result))
        segment_text = _segment_text(result)
        segment_finish_reason = _segment_finish_reason(result)
        segment_provider_finish_reason = _segment_provider_finish_reason(result)
        if not emitted_live_tool_events:
            for tool_result in accumulated_tool_results:
                await emit(AgentToolResultEvent(tool_result=tool_result))
        # Every approval continuation belongs to this segment. Keep the
        # generated responses separate from old history and runtime approval
        # response messages when applying guardrail replacements.
        segment_generation = GenerateTextOutput(
            text=segment_text,
            finish_reason=segment_finish_reason,
            provider_finish_reason=segment_provider_finish_reason,
            usage=_merge_usage([step.response.usage for step in accumulated_steps]),
            steps=accumulated_steps,
            messages=conversation_messages,
            tool_results=accumulated_tool_results,
        )
        original_assistants = _assistant_messages_from_result(segment_generation)
        guarded_output = await self._run_output_guardrails(
            agent=agent,
            run_id=run_id,
            session_id=session.id,
            result=segment_generation,
            text=segment_text,
            messages=original_assistants,
            context=context,
            trace=trace,
            emit=emit,
        )
        segment_text = guarded_output.text
        _apply_guarded_output(segment_generation, text=segment_text, messages=guarded_output.messages)
        conversation_messages = _replace_messages_by_identity(conversation_messages, original_assistants, guarded_output.messages)
        persisted_run_messages = _replace_messages_by_identity(persisted_run_messages, original_assistants, guarded_output.messages)
        if segment_text and not emitted_live_text and not buffer_live_text:
            await emit(AgentTextDeltaEvent(text_delta=segment_text))
        if buffer_live_text:
            if buffered_text_deltas and segment_text == "".join(buffered_text_deltas):
                for text_delta in buffered_text_deltas:
                    await emit(AgentTextDeltaEvent(text_delta=text_delta))
            elif segment_text:
                await emit(AgentTextDeltaEvent(text_delta=segment_text))
        handoff = _detect_handoff(accumulated_tool_results)
        segment_output = None if handoff is not None else _parse_agent_output(output_agent, segment_text)
        transcript = list(session.messages)
        transcript.extend(transcript_input)
        transcript.extend(persisted_run_messages)
        session.messages = transcript
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
                    metadata={**dict(session.metadata), "state": dict(session.state)},
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
                    metadata={**dict(session.metadata), "state": dict(session.state)},
                ),
            )

        await _save_checkpoints(
            checkpoint_store=agent.checkpoint_store,
            result=GenerateTextOutput(
                text=segment_text,
                finish_reason=segment_finish_reason,
                provider_finish_reason=segment_provider_finish_reason,
                usage=_merge_usage([step.response.usage for step in accumulated_steps]),
                steps=accumulated_steps,
                messages=conversation_messages,
                tool_results=accumulated_tool_results,
            ),
            run_id=run_id,
            session_id=session.id,
            agent_name=agent.name,
            emit=emit,
            trace=trace,
        )
        artifacts = _extract_skill_artifacts(accumulated_tool_results)
        if artifacts:
            session.metadata = {
                **session.metadata,
                "skill_artifacts": [
                    {
                        "name": artifact.name,
                        "path": artifact.path,
                        "media_type": artifact.media_type,
                        "role": artifact.role,
                        "description": artifact.description,
                        "metadata": dict(artifact.metadata),
                    }
                    for artifact in artifacts
                ],
            }
        segment_result = AgentRunResult(
            run_id=run_id,
            agent_name=agent.name,
            session=session,
            text=segment_text,
            finish_reason=segment_finish_reason,
            provider_finish_reason=segment_provider_finish_reason,
            usage=_merge_usage([step.response.usage for step in accumulated_steps]),
            steps=accumulated_steps,
            messages=conversation_messages,
            tool_results=accumulated_tool_results,
            artifacts=artifacts,
            trace=trace,
            handoff=handoff,
            orchestration_path=list(trace.orchestration_path),
            output=segment_output,
        )
        return segment_result

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
        hooks: list[AgentHooks],
    ) -> ToolSet:
        wrapped: ToolSet = {}
        tool_limit = agent.run_limits.max_tool_calls
        tools_started = time.monotonic()

        async def run_tool_guardrails(
            *,
            stage: Literal["input", "output"],
            definition: ToolDefinition,
            tool_name: str,
            tool_input: Any,
            value: Any,
            tool_context: ToolExecutionContext,
        ) -> Any:
            guardrails = definition.input_guardrails if stage == "input" else definition.output_guardrails
            guarded_value = value
            for guardrail in guardrails:
                guardrail_name = _guardrail_name(guardrail)
                if stage == "input":
                    request: Any = ToolInputGuardrailRequest(
                        tool_name=tool_name,
                        input=guarded_value,
                        context=tool_context,
                    )
                else:
                    request = ToolOutputGuardrailRequest(
                        tool_name=tool_name,
                        input=tool_input,
                        output=guarded_value,
                        context=tool_context,
                    )
                try:
                    raw_outcome = await _maybe_await(guardrail(request))
                    if isinstance(raw_outcome, ToolGuardrailResult):
                        outcome = raw_outcome
                    elif isinstance(raw_outcome, bool):
                        outcome = ToolGuardrailResult(tripwire_triggered=raw_outcome)
                    elif raw_outcome is None:
                        outcome = ToolGuardrailResult()
                    else:
                        raise TypeError("Tool guardrails must return ToolGuardrailResult, bool, or None.")
                except Exception as error:
                    reason = "Guardrail evaluation failed."
                    trace.guardrail_trigger_count += 1
                    await emit(
                        AgentGuardrailEvent(
                            stage=stage,
                            guardrail_name=guardrail_name,
                            triggered=True,
                            reason=reason,
                            metadata={"scope": "tool", "tool_name": tool_name, "tool_stage": stage},
                        )
                    )
                    raise ToolGuardrailTripwireTriggered(
                        stage=stage,
                        tool_name=tool_name,
                        guardrail_name=guardrail_name,
                        reason=reason,
                    ) from error

                event_metadata = {
                    **dict(outcome.metadata),
                    "scope": "tool",
                    "tool_name": tool_name,
                    "tool_stage": stage,
                    "transformed": outcome.replace,
                }
                await emit(
                    AgentGuardrailEvent(
                        stage=stage,
                        guardrail_name=guardrail_name,
                        triggered=outcome.tripwire_triggered,
                        reason=outcome.reason,
                        metadata=event_metadata,
                    )
                )
                if outcome.tripwire_triggered:
                    trace.guardrail_trigger_count += 1
                    raise ToolGuardrailTripwireTriggered(
                        stage=stage,
                        tool_name=tool_name,
                        guardrail_name=guardrail_name,
                        reason=outcome.reason,
                        metadata=outcome.metadata,
                    )
                if outcome.replace:
                    guarded_value = outcome.replacement
            return guarded_value

        for tool_name, definition in registry.items():
            if not is_callable_tool_definition(definition):
                continue
            callable_definition = definition

            async def execute(
                input: Any,
                call_context: ToolExecutionContext | None = None,
                *,
                _tool_name: str = tool_name,
                _definition: ToolDefinition = callable_definition,
            ) -> Any:
                await _raise_if_agent_run_cancelled(
                    run_store=agent.run_store,
                    run_id=run_id,
                    cancellation_token=context.cancellation_token,
                )
                if agent.run_limits.max_wall_time_ms is not None and (time.monotonic() - tools_started) * 1000 > agent.run_limits.max_wall_time_ms:
                    raise RuntimeError("Agent run exceeded max wall time.")

                trace.tool_call_count += 1
                if tool_limit is not None and trace.tool_call_count > tool_limit:
                    raise RuntimeError(f'Agent exceeded max tool calls ({tool_limit}).')

                tool_context = ToolExecutionContext(
                    tool_name=_tool_name,
                    tool_call_id=call_context.tool_call_id if call_context is not None else "",
                    idempotency_key=(
                        call_context.idempotency_key
                        if call_context is not None and call_context.idempotency_key
                        else f"{run_id}:{call_context.tool_call_id if call_context is not None else _tool_name}"
                    ),
                    deadline_ms=call_context.deadline_ms if call_context is not None else None,
                    run_id=run_id,
                    session_id=session_id,
                    agent_name=agent.name,
                    memory_summary=context.memory_summary,
                    permissions=list(_definition.permissions),
                    source=_definition.source,
                    metadata={**context.metadata, **_definition.metadata},
                    handoff_path=list(trace.orchestration_path),
                    deps=context.deps,
                    cancellation_token=context.cancellation_token,
                )
                tool_context.raise_if_cancelled()
                guarded_input = await run_tool_guardrails(
                    stage="input",
                    definition=_definition,
                    tool_name=_tool_name,
                    tool_input=input,
                    value=input,
                    tool_context=tool_context,
                )
                try:
                    guarded_input = create_schema_adapter(_definition.schema).validate_python(guarded_input)
                except Exception as error:
                    raise ValidationError(f'Invalid guarded input for tool "{_tool_name}": {error}') from error

                request = ToolApprovalRequest(
                    run_id=run_id,
                    session_id=session_id,
                    agent_name=agent.name,
                    tool_name=_tool_name,
                    tool_input=guarded_input,
                    tool_permissions=list(_definition.permissions),
                    tool_source=_definition.source,
                    tool_metadata=dict(_definition.metadata),
                    context=context,
                    handoff_path=list(trace.orchestration_path),
                )
                decision = ApprovalDecision(approved=True)
                approval_required = _definition.requires_approval is True or (
                    _definition.requires_approval is None and _definition.source in {"remote", "mcp"}
                )
                if approval_required or (
                    _definition.requires_approval is None and agent.approval_policy is not None
                ):
                    if approval_required and agent.approval_policy is None:
                        raise RuntimeError(
                            f'Tool "{_tool_name}" requires an approval_policy on the agent.'
                        )
                    trace.approval_count += 1
                    if agent.approval_policy is not None:
                        decision = _normalize_approval_decision(await _maybe_await(agent.approval_policy(request)))
                    await _call_agent_hooks(hooks, "on_approval", context, agent, request, decision)
                    pending_approval = (
                        _pending_approval_from_request(
                            request,
                            decision,
                            tool_call_id=call_context.tool_call_id if call_context is not None else "",
                            tool_fingerprint=_tool_definition_fingerprint(_definition),
                        )
                        if decision.suspend
                        else None
                    )
                    await emit(
                        AgentToolApprovalEvent(
                            tool_name=_tool_name,
                            tool_input=guarded_input,
                            approved=decision.approved,
                            reason=decision.reason,
                            approval_request_id=pending_approval.id if pending_approval is not None else decision.approval_id,
                            tool_source=_definition.source,
                            metadata={"suspended": decision.suspend} if decision.suspend else {},
                        )
                    )
                    if pending_approval is not None:
                        raise ToolExecutionSuspended(
                            decision.reason or f'Tool "{_tool_name}" is waiting for human approval.',
                            pending_approval=pending_approval,
                        )
                if decision.suspend:
                    raise ToolExecutionSuspended(
                        decision.reason or f'Tool "{_tool_name}" is waiting for human approval.',
                        pending_approval=_pending_approval_from_request(
                            request,
                            decision,
                            tool_call_id=call_context.tool_call_id if call_context is not None else "",
                            tool_fingerprint=_tool_definition_fingerprint(_definition),
                        ),
                    )
                if not decision.approved:
                    raise RuntimeError(decision.reason or f'Tool "{_tool_name}" denied by approval policy.')

                span = self._start_span(
                    "zhivex.agent.tool",
                    {
                        "tool.name": _tool_name,
                        "tool.source": _definition.source,
                        "agent.name": agent.name,
                        "run.id": run_id,
                        "gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": _tool_name,
                        "gen_ai.tool.type": _definition.source,
                    },
                )
                skill_name = str(_definition.metadata.get("skill_name") or "")
                skill_entrypoint = str(_definition.metadata.get("skill_entrypoint") or "")
                try:
                    if skill_name and skill_entrypoint:
                        await emit(AgentSkillExecutionStartEvent(skill_name=skill_name, entrypoint=skill_entrypoint))
                    await _call_agent_hooks(
                        hooks,
                        "on_tool_start",
                        context,
                        agent,
                        _definition,
                        guarded_input,
                        tool_context,
                    )
                    tool_context.raise_if_cancelled()
                    result = await registry.execute(_definition, guarded_input, tool_context)
                    tool_context.raise_if_cancelled()
                    await _raise_if_agent_run_cancelled(
                        run_store=agent.run_store,
                        run_id=run_id,
                        cancellation_token=context.cancellation_token,
                    )
                    result = await run_tool_guardrails(
                        stage="output",
                        definition=_definition,
                        tool_name=_tool_name,
                        tool_input=guarded_input,
                        value=result,
                        tool_context=tool_context,
                    )
                except Exception as error:
                    if skill_name and skill_entrypoint:
                        await emit(AgentSkillExecutionFinishEvent(skill_name=skill_name, entrypoint=skill_entrypoint, ok=False))
                    self._finish_span(span, error=error)
                    try:
                        await _call_agent_hooks(
                            hooks,
                            "on_tool_error",
                            context,
                            agent,
                            _definition,
                            guarded_input,
                            tool_context,
                            error,
                            reverse=True,
                        )
                    except Exception as hook_error:
                        error.add_note(f"Agent on_tool_error hook also failed: {hook_error}")
                    raise
                if skill_name and skill_entrypoint:
                    await emit(AgentSkillExecutionFinishEvent(skill_name=skill_name, entrypoint=skill_entrypoint, ok=True))
                    if isinstance(result, dict):
                        for item in list(result.get("artifacts") or []):
                            if isinstance(item, dict):
                                artifact_path = str(item.get("path") or "").strip()
                                if artifact_path:
                                    artifact_metadata = item.get("metadata")
                                    await emit(
                                        AgentSkillArtifactCreatedEvent(
                                            skill_name=skill_name,
                                            entrypoint=skill_entrypoint,
                                            artifact=SkillArtifact(
                                                name=str(item.get("name") or Path(artifact_path).name),
                                                path=artifact_path,
                                                media_type=str(item.get("media_type") or "") or None,
                                                role=str(item.get("role") or "primary"),  # type: ignore[arg-type]
                                                description=str(item.get("description") or "") or None,
                                                metadata=cast(dict[str, Any], artifact_metadata)
                                                if isinstance(artifact_metadata, dict)
                                                else {},
                                            ),
                                        )
                                    )
                self._finish_span(span)
                await _call_agent_hooks(
                    hooks,
                    "on_tool_end",
                    context,
                    agent,
                    _definition,
                    guarded_input,
                    tool_context,
                    result,
                    reverse=True,
                )
                return result

            wrapped[tool_name] = ToolDefinition(
                name=definition.name,
                description=definition.description,
                schema=definition.schema,
                execute=execute,
                input_examples=list(definition.input_examples),
                strict=definition.strict,
                defer_loading=definition.defer_loading,
                eager_input_streaming=definition.eager_input_streaming,
                allowed_callers=list(definition.allowed_callers),
                output_schema=definition.output_schema,
                cache_control=dict(definition.cache_control) if definition.cache_control is not None else None,
                tags=list(definition.tags),
                requires_approval=definition.requires_approval,
                permissions=list(definition.permissions),
                source=definition.source,
                metadata={
                    **definition.metadata,
                    "zhivex_tool_idempotency_prefix": str(
                        agent.metadata.get("zhivex_workflow_step_idempotency_key") or run_id
                    ),
                    "zhivex_agent_approval_gated": bool(
                        definition.requires_approval is True
                        or (definition.requires_approval is None and definition.source in {"remote", "mcp"})
                        or (definition.requires_approval is None and agent.approval_policy is not None)
                    ),
                },
                supports_streaming=definition.supports_streaming,
                remote_config=definition.remote_config,
                mcp_config=definition.mcp_config,
                input_guardrails=list(definition.input_guardrails),
                output_guardrails=list(definition.output_guardrails),
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


def create_subagent_tool(
    *,
    name: str,
    agent: Agent,
    parent_run_id: str | None = None,
    description: str | None = None,
    runtime: AgentRuntime | None = None,
    hooks: Iterable[AgentHooks] | None = None,
) -> ToolDefinition:
    async def execute(
        input: Any,
        context: ToolExecutionContext[Any] | None = None,
    ) -> JsonValue:
        prompt = input.get("prompt") if isinstance(input, dict) else str(input)
        result = await run_agent(
            agent=agent,
            prompt=str(prompt or ""),
            deps=context.deps if context is not None else None,
            cancellation_token=getattr(context, "cancellation_token", None) if context is not None else None,
            parent_run_id=parent_run_id or (context.run_id if context is not None else None),
            runtime=runtime,
            hooks=hooks,
        )
        child_state = result.state
        if child_state is None:
            child_state = _agent_run_state_from_result(
                result=result,
                agent=agent,
                parent_run_id=parent_run_id,
                idempotency_key=None,
                started_at_ms=result.trace.started_at_ms if result.trace else _now_ms(),
                finished_at_ms=result.trace.finished_at_ms if result.trace and result.trace.finished_at_ms else _now_ms(),
            )
        return {
            "text": result.text,
            "child_run": asdict(agent_child_run_from_state(child_state, tool_name=name)),
        }

    return ToolDefinition(
        name=name,
        description=description or f"Run the {agent.name} subagent.",
        schema={
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
        execute=execute,
        metadata={"type": "subagent", "agent_name": agent.name, "parent_run_id": parent_run_id},
    )


def prepare_subagents_for_agent(agent: Agent) -> Agent:
    if not agent.subagents:
        return agent
    registry = _resolve_tool_registry(agent, None)
    registry = registry.merge(
        {name: create_subagent_tool(name=name, agent=subagent) for name, subagent in agent.subagents.items()}
    )
    return replace(agent, tools=registry)


@dataclass(slots=True)
class AgentGroupMember:
    name: str
    agent: Agent
    prompt: str | None = None
    session: AgentSession | None = None
    idempotency_key: str | None = None


@dataclass(slots=True)
class AgentGroupMemberResult:
    name: str
    output: AgentRunResult | None = None
    error: Exception | None = None


@dataclass(slots=True)
class AgentGroupRunResult:
    parent_run_id: str | None
    outputs: list[AgentGroupMemberResult]


@budgeted
async def run_agent_group(
    members: list[AgentGroupMember],
    *,
    prompt: str | None = None,
    parent_run_id: str | None = None,
    deps: Any = None,
    runtime: AgentRuntime | None = None,
    hooks: Iterable[AgentHooks] | None = None,
    middleware: Iterable[AgentMiddleware] | None = None,
    max_concurrency: int | None = None,
    timeout_ms: int | None = None,
    total_timeout_ms: int | None = None,
    retry_jitter: float | None = None,
    fail_fast: bool = False,
    idempotency_key: str | None = None,
    cancellation_token: AgentCancellationToken | None = None,
) -> AgentGroupRunResult:
    if max_concurrency is not None and (isinstance(max_concurrency, bool) or max_concurrency <= 0):
        raise ValidationError("Agent group max_concurrency must be a positive integer or None.")
    if timeout_ms is not None and (isinstance(timeout_ms, bool) or timeout_ms <= 0):
        raise ValidationError("Agent group timeout_ms must be a positive integer or None.")
    if not members:
        return AgentGroupRunResult(parent_run_id=parent_run_id, outputs=[])

    semaphore = asyncio.Semaphore(max_concurrency or len(members))
    resolved_runtime = runtime or AgentRuntime()

    async def run_member(member: AgentGroupMember) -> AgentGroupMemberResult:
        try:
            async with semaphore:
                async def execute() -> AgentRunResult:
                    return await run_agent(
                        agent=member.agent,
                        session=member.session,
                        prompt=member.prompt if member.prompt is not None else prompt,
                        deps=deps,
                        parent_run_id=parent_run_id,
                        runtime=resolved_runtime,
                        hooks=hooks,
                        middleware=middleware,
                        cancellation_token=cancellation_token,
                        idempotency_key=(
                            member.idempotency_key
                            or (f"{idempotency_key}:{member.name}" if idempotency_key is not None else None)
                        ),
                    )

                if timeout_ms is None:
                    output = await execute()
                else:
                    async with asyncio.timeout(timeout_ms / 1000):
                        output = await execute()
            return AgentGroupMemberResult(name=member.name, output=output)
        except Exception as error:
            return AgentGroupMemberResult(name=member.name, error=error)

    tasks = [asyncio.create_task(run_member(member)) for member in members]
    if not fail_fast:
        group_outputs = await asyncio.gather(*tasks)
        return AgentGroupRunResult(parent_run_id=parent_run_id, outputs=list(group_outputs))

    indexed_tasks = {task: index for index, task in enumerate(tasks)}
    pending: set[asyncio.Task[AgentGroupMemberResult]] = set(tasks)
    outputs: list[AgentGroupMemberResult | None] = [None] * len(members)
    failed_member: str | None = None
    while pending and failed_member is None:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            index = indexed_tasks[task]
            result = await task
            outputs[index] = result
            if result.error is not None and failed_member is None:
                failed_member = result.name

    if failed_member is not None and pending:
        cancelled_tasks = list(pending)
        for task in cancelled_tasks:
            task.cancel()
        await asyncio.gather(*cancelled_tasks, return_exceptions=True)
        for task in cancelled_tasks:
            index = indexed_tasks[task]
            outputs[index] = AgentGroupMemberResult(
                name=members[index].name,
                error=RuntimeError(f'Cancelled because agent group member "{failed_member}" failed fast.'),
            )

    return AgentGroupRunResult(
        parent_run_id=parent_run_id,
        outputs=[cast(AgentGroupMemberResult, item) for item in outputs],
    )


def run_agent(
    *,
    agent: Agent[AgentDepsT, AgentOutputT],
    session: AgentSession | None = None,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    deps: AgentDepsT | None = None,
    tools: ToolSet | ToolRegistry | None = None,
    skills: SkillSet | SkillRegistry | None = None,
    tool_choice: str | ToolChoiceName | None = None,
    tool_execution: ToolExecutionOptions | None = None,
    max_steps: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning: ReasoningConfig | None = None,
    provider_options: dict[str, Any] | None = None,
    timeout_ms: int | None = None,
    total_timeout_ms: int | None = None,
    retry_jitter: float | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
    stop_on_handoff: bool = False,
    runtime: AgentRuntime | None = None,
    registry: AgentRegistry | None = None,
    observer: AgentObserver | None = None,
    parent_run_id: str | None = None,
    idempotency_key: str | None = None,
    cancellation_token: AgentCancellationToken | None = None,
    hooks: Iterable[AgentHooks] | None = None,
    middleware: Iterable[AgentMiddleware] | None = None,
) -> Awaitable[AgentRunResult[AgentOutputT]]:
    resolved_runtime = runtime or AgentRuntime(registry=registry, observer=observer)
    return resolved_runtime.run(
        agent=agent,
        session=session,
        prompt=prompt,
        messages=messages,
        deps=deps,
        tools=tools,
        skills=skills,
        tool_choice=tool_choice,
        tool_execution=tool_execution,
        max_steps=max_steps,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning=reasoning,
        provider_options=provider_options,
        timeout_ms=timeout_ms,
        total_timeout_ms=total_timeout_ms,
        retry_jitter=retry_jitter,
        max_retries=max_retries,
        retry_backoff_ms=retry_backoff_ms,
        stop_on_handoff=stop_on_handoff,
        parent_run_id=parent_run_id,
        idempotency_key=idempotency_key,
        cancellation_token=cancellation_token,
        hooks=hooks,
        middleware=middleware,
    )


def resume_agent(
    *,
    agent: Agent[AgentDepsT, AgentOutputT],
    session_id: str,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    deps: AgentDepsT | None = None,
    tools: ToolSet | ToolRegistry | None = None,
    skills: SkillSet | SkillRegistry | None = None,
    tool_choice: str | ToolChoiceName | None = None,
    tool_execution: ToolExecutionOptions | None = None,
    max_steps: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning: ReasoningConfig | None = None,
    provider_options: dict[str, Any] | None = None,
    timeout_ms: int | None = None,
    total_timeout_ms: int | None = None,
    retry_jitter: float | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
    stop_on_handoff: bool = False,
    runtime: AgentRuntime | None = None,
    registry: AgentRegistry | None = None,
    observer: AgentObserver | None = None,
    cancellation_token: AgentCancellationToken | None = None,
    hooks: Iterable[AgentHooks] | None = None,
    middleware: Iterable[AgentMiddleware] | None = None,
) -> Awaitable[AgentRunResult[AgentOutputT]]:
    resolved_runtime = runtime or AgentRuntime(registry=registry, observer=observer)

    async def runner() -> AgentRunResult[AgentOutputT]:
        resumed_session = await load_agent_session(agent, session_id)
        latest_checkpoint: AgentCheckpoint | None = None
        if agent.checkpoint_store is not None:
            latest_checkpoint = await agent.checkpoint_store.get_latest(session_id=session_id)
            if latest_checkpoint is not None:
                resumed_session.metadata = {
                    **resumed_session.metadata,
                    "resumed_from_checkpoint": {
                        "run_id": latest_checkpoint.run_id,
                        "step_index": latest_checkpoint.step_index,
                        "saved_at_ms": latest_checkpoint.saved_at_ms,
                    },
                }
        return await resolved_runtime.run(
            agent=agent,
            session=resumed_session,
            prompt=prompt,
            messages=messages,
            deps=deps,
            tools=tools,
            skills=skills,
            tool_choice=tool_choice,
            tool_execution=tool_execution,
            max_steps=max_steps,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
            provider_options=provider_options,
            timeout_ms=timeout_ms,
            total_timeout_ms=total_timeout_ms,
            retry_jitter=retry_jitter,
            max_retries=max_retries,
            retry_backoff_ms=retry_backoff_ms,
            stop_on_handoff=stop_on_handoff,
            resumed_from_checkpoint=latest_checkpoint,
            cancellation_token=cancellation_token,
            hooks=hooks,
            middleware=middleware,
        )

    async def budget_runner() -> AgentRunResult[AgentOutputT]:
        async with execution_scope(total_timeout_ms, retry_jitter=retry_jitter):
            return await runner()

    return budget_runner()


def _resume_messages_from_state(state: AgentRunState) -> list[ModelMessage]:
    raw_messages = state.metadata.get("resume_messages")
    if not isinstance(raw_messages, list):
        raise ValidationError(
            'Suspended run state is missing "resume_messages"; it cannot be resumed from an approval decision.'
        )
    return deserialize_messages(cast(list[dict[str, Any]], raw_messages))


def _find_agent_for_resume(root: Agent, agent_name: str, registry: AgentRegistry) -> Agent | None:
    seen: set[int] = set()

    def visit(candidate: Agent) -> Agent | None:
        identity = id(candidate)
        if identity in seen:
            return None
        seen.add(identity)
        if candidate.name == agent_name:
            return candidate
        for subagent in candidate.subagents.values():
            match = visit(subagent)
            if match is not None:
                return match
        return None

    return visit(root) or registry.get(agent_name)


def resume_agent_run(
    *,
    agent: Agent[AgentDepsT, AgentOutputT],
    run_id: str,
    approval_id: str | None = None,
    approved: bool = True,
    reason: str | None = None,
    deps: AgentDepsT | None = None,
    tools: ToolSet | ToolRegistry | None = None,
    skills: SkillSet | SkillRegistry | None = None,
    tool_choice: str | ToolChoiceName | None = None,
    tool_execution: ToolExecutionOptions | None = None,
    max_steps: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning: ReasoningConfig | None = None,
    provider_options: dict[str, Any] | None = None,
    timeout_ms: int | None = None,
    total_timeout_ms: int | None = None,
    retry_jitter: float | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
    stop_on_handoff: bool = False,
    runtime: AgentRuntime | None = None,
    registry: AgentRegistry | None = None,
    observer: AgentObserver | None = None,
    idempotency_key: str | None = None,
    cancellation_token: AgentCancellationToken | None = None,
    hooks: Iterable[AgentHooks] | None = None,
    middleware: Iterable[AgentMiddleware] | None = None,
) -> Awaitable[AgentRunResult[AgentOutputT]]:
    resolved_runtime = runtime or AgentRuntime(registry=registry, observer=observer)

    async def runner() -> AgentRunResult[AgentOutputT]:
        run_store = agent.run_store
        if run_store is None:
            raise ValidationError("resume_agent_run(...) requires agent.run_store.")
        loaded_state = await run_store.load(run_id)
        if loaded_state is None:
            raise ValidationError(f'Agent run "{run_id}" was not found.')
        if loaded_state.status != "suspended":
            raise ValidationError(f'Agent run "{run_id}" is not suspended.')
        if not loaded_state.pending_approvals:
            raise ValidationError(f'Agent run "{run_id}" has no pending approvals.')
        selected_pending = (
            next((item for item in loaded_state.pending_approvals if item.id == approval_id), None)
            if approval_id is not None
            else loaded_state.pending_approvals[0]
        )
        if selected_pending is None:
            raise ValidationError(f'Pending approval "{approval_id}" was not found on run "{run_id}".')
        approval_agent = _find_agent_for_resume(agent, loaded_state.agent_name, resolved_runtime._registry)
        if approval_agent is None:
            raise ValidationError(
                f'Agent "{loaded_state.agent_name}" that suspended run "{run_id}" is not registered for resume.'
            )
        # Continue with the suspended agent's tools and instructions, while the
        # caller's root agent still owns the output contract across handoffs.
        resume_agent_instance = replace(
            approval_agent,
            run_store=run_store,
            output_type=agent.output_type,
            output_mode=agent.output_mode,
            output_name=agent.output_name,
            output_description=agent.output_description,
        )
        if not callable(getattr(resume_agent_instance.model, "generate", None)):
            raise ValidationError(
                "resume_agent_run(...) requires a language model with generate(). "
                "For a suspended realtime run, reconstruct the agent with a compatible "
                "language model and the same tools and run store before approving."
            )
        if approved:
            _validate_resolved_approval_tool(agent=resume_agent_instance, pending=selected_pending, tools=tools)
        claim_pending_approval = getattr(run_store, "claim_pending_approval", None)
        if not callable(claim_pending_approval):
            raise ValidationError(
                "resume_agent_run(...) requires a run store with atomic pending-approval claims. "
                "Use a built-in run store or implement claim_pending_approval(...)."
            )
        fail_resume_claim_capability = getattr(run_store, "fail_resume_claim", None)
        if not callable(fail_resume_claim_capability):
            raise ValidationError(
                "resume_agent_run(...) requires a run store with atomic resume-claim reconciliation. "
                "Use a built-in run store or implement fail_resume_claim(...)."
            )
        claim_token = str(uuid4())
        suspended_state = await claim_pending_approval(
            run_id,
            selected_pending.id,
            claim_token=claim_token,
            claimed_at_ms=_now_ms(),
        )
        if suspended_state is None:
            raise ValidationError(
                f'Pending approval "{selected_pending.id}" on run "{run_id}" is already being resumed or is no longer pending.'
            )
        raw_resume_claim = suspended_state.metadata.get("resume_claim")
        claim_is_valid = (
            suspended_state.status == "running"
            and isinstance(raw_resume_claim, dict)
            and raw_resume_claim.get("approval_id") == selected_pending.id
            and raw_resume_claim.get("claim_token") == claim_token
        )
        if not claim_is_valid:
            await fail_resume_claim_capability(
                run_id,
                claim_token=claim_token,
                reason="Run store returned an invalid pending-approval claim.",
                failed_at_ms=_now_ms(),
            )
            raise ValidationError(
                f'Run store returned an invalid claim for approval "{selected_pending.id}" on run "{run_id}".'
            )
        pending = next(
            (item for item in suspended_state.pending_approvals if item.id == selected_pending.id),
            None,
        )
        if pending is None:
            raise ValidationError(f'Pending approval "{selected_pending.id}" was not found on run "{run_id}" after claiming it.')

        async def resume_claimed_run() -> AgentRunResult[AgentOutputT]:
            approval_hooks = [*resolved_runtime._hooks, *list(hooks or []), *resume_agent_instance.hooks]
            approval_result = await _execute_resolved_approval_tool(
                agent=resume_agent_instance,
                state=suspended_state,
                pending=pending,
                approved=approved,
                reason=reason,
                tools=tools,
                tool_execution=tool_execution,
                deps=deps,
                cancellation_token=cancellation_token,
                hooks=approval_hooks,
            )
            if tool_execution is not None and tool_execution.stop_on_error and approval_result.is_error:
                raise RuntimeError(
                    f'Tool "{approval_result.tool_name}" failed: '
                    f'{approval_result.error.message if approval_result.error else "Unknown tool error."}'
                )
            resume_messages = _resume_messages_from_state(suspended_state)
            resume_messages.append(ModelMessage(role="tool", parts=[tool_result_part(approval_result)]))
            session = create_agent_session(
                id=suspended_state.session_id,
                messages=[],
                summary="",
                metadata={
                    "resumed_from_run_id": run_id,
                    "resolved_approval_id": pending.id,
                },
            )
            result = await resolved_runtime.run(
                agent=resume_agent_instance,
                session=session,
                messages=resume_messages,
                deps=deps,
                tools=tools,
                skills=skills,
                tool_choice=tool_choice,
                tool_execution=tool_execution,
                max_steps=max_steps,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
                provider_options=provider_options,
                timeout_ms=timeout_ms,
                total_timeout_ms=total_timeout_ms,
                retry_jitter=retry_jitter,
                max_retries=max_retries,
                retry_backoff_ms=retry_backoff_ms,
                stop_on_handoff=stop_on_handoff,
                parent_run_id=run_id,
                idempotency_key=idempotency_key or f"{run_id}:{pending.id}:resume",
                cancellation_token=cancellation_token,
                hooks=hooks,
                middleware=middleware,
            )
            resumed_at_ms = result.state.updated_at_ms if result.state is not None else _now_ms()
            suspended_state.pending_approvals = [item for item in suspended_state.pending_approvals if item.id != pending.id]
            suspended_state.status = result.state.status if result.state is not None else "completed"
            suspended_state.updated_at_ms = resumed_at_ms
            suspended_state.finished_at_ms = result.state.finished_at_ms if result.state is not None else resumed_at_ms
            suspended_state.output_text = result.text
            suspended_state.finish_reason = result.finish_reason
            suspended_state.tool_results = [*suspended_state.tool_results, approval_result, *result.tool_results]
            if result.state is not None:
                suspended_state.child_runs.append(agent_child_run_from_state(result.state))
            resolved_metadata = dict(suspended_state.metadata)
            resolved_metadata.pop("resume_claim", None)
            suspended_state.metadata = {
                **resolved_metadata,
                "resumed_by_run_id": result.run_id,
                "resolved_approval": {
                    "id": pending.id,
                    "approved": approved,
                    "reason": reason,
                    "tool_name": pending.name,
                    "resumed_at_ms": resumed_at_ms,
                },
            }
            await _persist_agent_run_state(run_store, suspended_state)
            result.resumed_from_checkpoint = result.resumed_from_checkpoint
            return result

        try:
            return await resume_claimed_run()
        except BaseException as error:
            failed_at_ms = _now_ms()
            reconciled = await fail_agent_run_resume_claim(
                run_store,
                run_id,
                claim_token=claim_token,
                reason=str(error) or type(error).__name__,
                now_ms=failed_at_ms,
            )
            if reconciled is None:
                current = await run_store.load(run_id)
                if current is not None and current.status == "cancelled":
                    raise AgentRunCancelled(
                        run_id,
                        reason=current.cancellation_reason,
                    ) from error
                raise ValidationError(
                    f'Agent run "{run_id}" resume claim changed before its failure could be reconciled.'
                ) from error
            raise

    async def budget_runner() -> AgentRunResult[AgentOutputT]:
        async with execution_scope(total_timeout_ms, retry_jitter=retry_jitter):
            return await runner()

    return budget_runner()


def stream_agent(
    *,
    agent: Agent[AgentDepsT, AgentOutputT],
    session: AgentSession | None = None,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    deps: AgentDepsT | None = None,
    tools: ToolSet | ToolRegistry | None = None,
    skills: SkillSet | SkillRegistry | None = None,
    tool_choice: str | ToolChoiceName | None = None,
    tool_execution: ToolExecutionOptions | None = None,
    max_steps: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning: ReasoningConfig | None = None,
    provider_options: dict[str, Any] | None = None,
    timeout_ms: int | None = None,
    total_timeout_ms: int | None = None,
    retry_jitter: float | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
    stop_on_handoff: bool = False,
    runtime: AgentRuntime | None = None,
    registry: AgentRegistry | None = None,
    observer: AgentObserver | None = None,
    idempotency_key: str | None = None,
    cancellation_token: AgentCancellationToken | None = None,
    hooks: Iterable[AgentHooks] | None = None,
    middleware: Iterable[AgentMiddleware] | None = None,
    stream_buffer_size: int | None = DEFAULT_STREAM_BUFFER_SIZE,
) -> AgentStreamResult[AgentOutputT]:
    resolved_runtime = runtime or AgentRuntime(registry=registry, observer=observer)
    broadcast = Broadcast[AgentEvent](max_events=stream_buffer_size if stream_buffer_size is not None else agent.trace_event_limit)

    async def emit(event: AgentEvent) -> None:
        await broadcast.publish(event)

    async def runner() -> AgentRunResult[AgentOutputT]:
        try:
            return await resolved_runtime.run(
                agent=agent,
                session=session,
                prompt=prompt,
                messages=messages,
                deps=deps,
                tools=tools,
                skills=skills,
                tool_choice=tool_choice,
                tool_execution=tool_execution,
                max_steps=max_steps,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
                provider_options=provider_options,
                timeout_ms=timeout_ms,
                total_timeout_ms=total_timeout_ms,
                retry_jitter=retry_jitter,
                max_retries=max_retries,
                retry_backoff_ms=retry_backoff_ms,
                stop_on_handoff=stop_on_handoff,
                emit=emit,
                live_stream=True,
                idempotency_key=idempotency_key,
                cancellation_token=cancellation_token,
                hooks=hooks,
                middleware=middleware,
            )
        finally:
            await broadcast.close()

    return AgentStreamResult(asyncio.create_task(runner()), broadcast)


def stream_live_agent(
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
    from ._agent_live import stream_live_agent_impl

    resolved_runtime = runtime or AgentRuntime(registry=registry, observer=observer)
    return stream_live_agent_impl(
        agent=agent,
        session=session,
        deps=deps,
        tools=tools,
        skills=skills,
        tool_choice=tool_choice,
        tool_execution=tool_execution,
        connect_options=connect_options,
        realtime_config=realtime_config,
        provider_options=provider_options,
        runtime=resolved_runtime,
        registry=registry,
        observer=observer,
        prompt=prompt,
        messages=messages,
        parent_run_id=parent_run_id,
        idempotency_key=idempotency_key,
        cancellation_token=cancellation_token,
        hooks=hooks,
        middleware=middleware,
        total_timeout_ms=total_timeout_ms,
        retry_jitter=retry_jitter,
        stream_buffer_size=stream_buffer_size,
    )
