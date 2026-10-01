"""Agent data contracts, independent of execution and storage implementations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeAlias, TypeVar
from uuid import uuid4

from .errors import AgentRunCancelled
from .skills import SkillArtifact
from .types import (
    AnyToolDefinition,
    FinishReason,
    GenerateResult,
    GenerateTextOutput,
    GenerateTextStep,
    JsonValue,
    ModelGenerateInput,
    ModelMessage,
    RealtimeEvent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionResult,
)

if TYPE_CHECKING:
    from .agent import Agent
    from .agent_state import AgentRunState


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


HANDOFF_MARKER = "__zhivex_agent_handoff__"

AgentDepsT = TypeVar("AgentDepsT")


AgentOutputT = TypeVar("AgentOutputT")


SkillActivationMode = Literal["explicit", "implicit", "sticky"]


@dataclass(slots=True)
class AgentHandoff:
    target_agent: str
    input: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentCancellationToken:
    """In-process cooperative cancellation signal for one agent run tree."""

    reason: str | None = None
    _event: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False, compare=False
    )

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str | None = None) -> None:
        if self._event.is_set():
            return
        self.reason = reason
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self, run_id: str = "") -> None:
        if self.cancelled:
            raise AgentRunCancelled(run_id, reason=self.reason)


@dataclass(slots=True)
class AgentContext(Generic[AgentDepsT]):
    run_id: str
    session_id: str
    agent_name: str
    memory_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    handoff_path: list[str] = field(default_factory=list)
    deps: AgentDepsT | None = field(default=None, repr=False, compare=False)
    session: AgentSession | None = field(default=None, repr=False, compare=False)
    cancellation_token: AgentCancellationToken | None = field(
        default=None, repr=False, compare=False
    )


DynamicInstructions: TypeAlias = Callable[..., str | None | Awaitable[str | None]]


class AgentHooks:
    """No-op lifecycle hooks that applications can override selectively."""

    async def on_agent_start(
        self, context: AgentContext[Any], agent: Agent[Any, Any]
    ) -> None:
        pass

    async def on_agent_end(
        self,
        context: AgentContext[Any],
        agent: Agent[Any, Any],
        result: AgentRunResult[Any],
    ) -> None:
        pass

    async def on_model_start(
        self,
        context: AgentContext[Any],
        agent: Agent[Any, Any],
        input: ModelGenerateInput,
    ) -> None:
        pass

    async def on_model_end(
        self,
        context: AgentContext[Any],
        agent: Agent[Any, Any],
        result: GenerateResult | None,
    ) -> None:
        pass

    async def on_tool_start(
        self,
        context: AgentContext[Any],
        agent: Agent[Any, Any],
        definition: ToolDefinition,
        input: Any,
        tool_context: ToolExecutionContext[Any],
    ) -> None:
        pass

    async def on_tool_end(
        self,
        context: AgentContext[Any],
        agent: Agent[Any, Any],
        definition: ToolDefinition,
        input: Any,
        tool_context: ToolExecutionContext[Any],
        output: Any,
    ) -> None:
        pass

    async def on_tool_error(
        self,
        context: AgentContext[Any],
        agent: Agent[Any, Any],
        definition: ToolDefinition,
        input: Any,
        tool_context: ToolExecutionContext[Any],
        error: Exception,
    ) -> None:
        pass

    async def on_handoff(
        self,
        context: AgentContext[Any],
        source_agent: Agent[Any, Any],
        target_agent: Agent[Any, Any],
        handoff: AgentHandoff,
    ) -> None:
        pass

    async def on_approval(
        self,
        context: AgentContext[Any],
        agent: Agent[Any, Any],
        request: ToolApprovalRequest,
        decision: ApprovalDecision,
    ) -> None:
        pass

    async def on_error(
        self,
        context: AgentContext[Any],
        agent: Agent[Any, Any],
        error: Exception,
    ) -> None:
        pass


@dataclass(slots=True)
class AgentRunRequest(Generic[AgentDepsT, AgentOutputT]):
    """Mutable request passed through agent run middleware."""

    agent: Agent[AgentDepsT, AgentOutputT]
    session: AgentSession | None = None
    prompt: str | None = None
    messages: list[ModelMessage] | None = None
    deps: AgentDepsT | None = field(default=None, repr=False, compare=False)
    cancellation_token: AgentCancellationToken | None = field(
        default=None, repr=False, compare=False
    )
    metadata: dict[str, Any] = field(default_factory=dict)


AgentMiddlewareNext: TypeAlias = Callable[
    [AgentRunRequest[Any, Any]],
    Awaitable["AgentRunResult[Any]"],
]


class AgentMiddleware(Protocol):
    def __call__(
        self,
        request: AgentRunRequest[Any, Any],
        call_next: AgentMiddlewareNext,
    ) -> AgentRunResult[Any] | Awaitable[AgentRunResult[Any]]: ...


@dataclass(slots=True)
class ApprovalDecision:
    approved: bool
    reason: str | None = None
    suspend: bool = False
    approval_id: str | None = None

    @classmethod
    def require_human(
        cls, reason: str | None = None, *, approval_id: str | None = None
    ) -> "ApprovalDecision":
        return cls(approved=False, reason=reason, suspend=True, approval_id=approval_id)


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
    async def __call__(
        self, request: ToolApprovalRequest
    ) -> ApprovalDecision | bool: ...


@dataclass(slots=True)
class GuardrailResult:
    tripwire_triggered: bool = False
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InputGuardrailRequest:
    run_id: str
    session_id: str
    agent_name: str
    prompt: str | None = None
    messages: list[ModelMessage] = field(default_factory=list)
    context: AgentContext | None = None


@dataclass(slots=True)
class OutputGuardrailRequest:
    run_id: str
    session_id: str
    agent_name: str
    text: str = ""
    messages: list[ModelMessage] = field(default_factory=list)
    result: GenerateTextOutput | None = None
    context: AgentContext | None = None


class InputGuardrail(Protocol):
    async def __call__(
        self, request: InputGuardrailRequest
    ) -> GuardrailResult | bool: ...


class OutputGuardrail(Protocol):
    async def __call__(
        self, request: OutputGuardrailRequest
    ) -> GuardrailResult | bool: ...


class GuardrailTripwireTriggered(RuntimeError):
    def __init__(
        self,
        *,
        stage: Literal["input", "output"],
        guardrail_name: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.stage = stage
        self.guardrail_name = guardrail_name
        self.reason = reason
        self.metadata = dict(metadata or {})
        message = f'Agent {stage} guardrail "{guardrail_name}" triggered.'
        if reason:
            message = f"{message} {reason}"
        super().__init__(message)


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

    async def get_latest(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> "AgentCheckpoint | None": ...

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
    state: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRuntime(Protocol):
    async def execute(
        self, definition: AnyToolDefinition, input: Any, context: ToolExecutionContext
    ) -> Any: ...

    async def aclose(self) -> None: ...


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
    tool_call: ToolCall = field(
        default_factory=lambda: ToolCall(id="", name="", input={})
    )


@dataclass(slots=True)
class AgentToolApprovalEvent:
    type: str = "tool-approval"
    tool_name: str = ""
    tool_input: Any = None
    approved: bool = True
    reason: str | None = None
    provider: str | None = None
    provider_managed: bool = False
    approval_request_id: str | None = None
    tool_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentToolResultEvent:
    type: str = "tool-result"
    tool_result: ToolExecutionResult = field(
        default_factory=lambda: ToolExecutionResult(
            tool_call_id="", tool_name="", is_error=False
        )
    )


@dataclass(slots=True)
class AgentSkillActivatedEvent:
    type: str = "skill-activated"
    skill_name: str = ""
    activation: SkillActivationMode = "explicit"
    path: str | None = None
    description: str | None = None


@dataclass(slots=True)
class AgentSkillResolvedEvent:
    type: str = "skill-resolved"
    skill_name: str = ""
    skill_version: str | None = None
    entrypoints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentSkillDependencyCheckEvent:
    type: str = "skill-dependency-check"
    skill_name: str = ""
    dependency_type: str = ""
    dependency_value: str = ""
    available: bool = True


@dataclass(slots=True)
class AgentSkillSkippedEvent:
    type: str = "skill-skipped"
    skill_name: str = ""
    activation: SkillActivationMode = "sticky"
    reason: str = ""
    path: str | None = None


@dataclass(slots=True)
class AgentSkillExecutionStartEvent:
    type: str = "skill-execution-start"
    skill_name: str = ""
    entrypoint: str = ""


@dataclass(slots=True)
class AgentSkillExecutionFinishEvent:
    type: str = "skill-execution-finish"
    skill_name: str = ""
    entrypoint: str = ""
    ok: bool = True


@dataclass(slots=True)
class AgentSkillArtifactCreatedEvent:
    type: str = "skill-artifact-created"
    skill_name: str = ""
    entrypoint: str = ""
    artifact: SkillArtifact | None = None


@dataclass(slots=True)
class AgentGuardrailEvent:
    type: str = "guardrail"
    stage: Literal["input", "output"] = "input"
    guardrail_name: str = ""
    triggered: bool = False
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
    | AgentSkillActivatedEvent
    | AgentSkillResolvedEvent
    | AgentSkillDependencyCheckEvent
    | AgentSkillSkippedEvent
    | AgentSkillExecutionStartEvent
    | AgentSkillExecutionFinishEvent
    | AgentSkillArtifactCreatedEvent
    | AgentGuardrailEvent
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
    guardrail_trigger_count: int = 0
    checkpoint_count: int = 0
    handoff_count: int = 0
    events_dropped: int = field(default=0, kw_only=True)


@dataclass(slots=True)
class AgentRunResult(Generic[AgentOutputT]):
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
    artifacts: list[SkillArtifact] = field(default_factory=list)
    trace: AgentTrace | None = None
    handoff: AgentHandoff | None = None
    orchestration_path: list[str] = field(default_factory=list)
    resumed_from_checkpoint: AgentCheckpoint | None = None
    state: AgentRunState | None = None
    output: AgentOutputT | None = None


AgentLiveEvent: TypeAlias = AgentEvent | RealtimeEvent


# Preserve historical pickle paths and annotation resolution through the public facade.
for _contract in (
    AgentHandoff,
    AgentCancellationToken,
    AgentContext,
    AgentHooks,
    AgentRunRequest,
    AgentMiddleware,
    ApprovalDecision,
    ToolApprovalRequest,
    ApprovalPolicy,
    GuardrailResult,
    InputGuardrailRequest,
    OutputGuardrailRequest,
    InputGuardrail,
    OutputGuardrail,
    GuardrailTripwireTriggered,
    SummaryConfig,
    AgentMemoryState,
    AgentMemory,
    AgentCheckpointStore,
    RunLimits,
    AgentSession,
    ToolRuntime,
    AgentCheckpoint,
    AgentRunStartEvent,
    AgentDelegationStartEvent,
    AgentDelegationFinishEvent,
    AgentTextDeltaEvent,
    AgentToolCallEvent,
    AgentToolApprovalEvent,
    AgentToolResultEvent,
    AgentSkillActivatedEvent,
    AgentSkillResolvedEvent,
    AgentSkillDependencyCheckEvent,
    AgentSkillSkippedEvent,
    AgentSkillExecutionStartEvent,
    AgentSkillExecutionFinishEvent,
    AgentSkillArtifactCreatedEvent,
    AgentGuardrailEvent,
    AgentCheckpointEvent,
    AgentSummaryUpdateEvent,
    AgentHandoffRequestedEvent,
    AgentHandoffResolvedEvent,
    AgentHandoffFailedEvent,
    AgentHandoffEvent,
    AgentFinishEvent,
    AgentErrorEvent,
    AgentTraceSegment,
    AgentTrace,
    AgentRunResult,
):
    _contract.__module__ = "zhivex_ai.agent"
