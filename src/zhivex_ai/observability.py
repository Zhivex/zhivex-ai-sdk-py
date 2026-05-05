from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .agent_evaluation import AgentRunSnapshot, create_agent_run_snapshot, replay_agent_run
from .agent_state import AgentRunState, AgentRunStore
from .catalog import ModelCatalog
from .types import TokenUsage


@dataclass(slots=True)
class _SpanHandle:
    manager: Any
    span: Any

    def end(self, *, attributes: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        if self.span is not None and attributes:
            for key, value in attributes.items():
                self.span.set_attribute(key, value)
        if self.span is not None and error is not None:
            self.span.record_exception(error)
            try:
                from opentelemetry.trace import Status, StatusCode

                self.span.set_status(Status(StatusCode.ERROR, str(error)))
            except Exception:
                pass
        self.manager.__exit__(type(error) if error is not None else None, error, getattr(error, "__traceback__", None))


class OTelAgentObserver:
    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> _SpanHandle:
        manager = self._tracer.start_as_current_span(name)
        span = manager.__enter__()
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)
        return _SpanHandle(manager=manager, span=span)


def create_otel_agent_observer(*, tracer_name: str = "zhivex_ai.agent", version: str | None = None) -> OTelAgentObserver:
    try:
        from opentelemetry import trace
    except Exception as error:
        raise RuntimeError("OpenTelemetry is not installed. Install opentelemetry-api/sdk to use OTEL observability.") from error
    tracer = trace.get_tracer(tracer_name, version)
    return OTelAgentObserver(tracer)


@dataclass(slots=True)
class TokenPricing:
    input_cost_per_1k_tokens: float | None = None
    output_cost_per_1k_tokens: float | None = None
    total_cost_per_1k_tokens: float | None = None
    currency: str = "USD"


@dataclass(slots=True)
class CostEstimate:
    input_cost: float | None = None
    output_cost: float | None = None
    total_cost: float | None = None
    currency: str = "USD"
    usage: TokenUsage | None = None


@dataclass(slots=True)
class AgentTraceStep:
    index: int
    status: str
    tool_calls: list[dict[str, Any]]
    tool_results: int
    usage: TokenUsage | None = None
    error: str | None = None


@dataclass(slots=True)
class AgentTraceArtifact:
    run_id: str
    agent_name: str
    provider: str
    model_id: str
    status: str
    steps: list[AgentTraceStep]
    child_runs: list[dict[str, Any]]
    events: list[dict[str, Any]]
    usage: TokenUsage | None = None
    output_preview: str = ""
    output_text: str | None = None
    error: str | None = None
    cancellation_reason: str | None = None


@dataclass(slots=True)
class AgentTraceSummary:
    run_id: str
    agent_name: str
    provider: str
    model_id: str
    status: str
    steps: int
    child_runs: int
    tool_calls: int
    tool_errors: int
    usage: TokenUsage | None = None
    cost: CostEstimate | None = None
    error: str | None = None


@dataclass(slots=True)
class AgentRunTreeNode:
    run_id: str
    agent_name: str
    parent_run_id: str | None
    status: str
    snapshot: AgentRunSnapshot
    children: list["AgentRunTreeNode"]


@dataclass(slots=True)
class AgentRunTreeSnapshot:
    root: AgentRunTreeNode
    total_runs: int


@dataclass(slots=True)
class HierarchicalAgentTraceNode:
    trace: AgentTraceArtifact
    children: list["HierarchicalAgentTraceNode"]


@dataclass(slots=True)
class HierarchicalAgentTrace:
    root: HierarchicalAgentTraceNode
    total_runs: int


class AgentTraceCollector:
    def __init__(self) -> None:
        self._events: dict[str, list[dict[str, Any]]] = {}

    async def __call__(self, event: Any) -> None:
        run_id = str(getattr(event, "run_id", "") or "unknown")
        self._events.setdefault(run_id, []).append({"type": getattr(event, "type", event.__class__.__name__), "event": event})

    def get_events(self, run_id: str | None = None) -> list[dict[str, Any]]:
        if run_id is not None:
            return list(self._events.get(run_id, []))
        return [event for events in self._events.values() for event in events]

    def reset(self, run_id: str | None = None) -> None:
        if run_id is None:
            self._events.clear()
            return
        self._events.pop(run_id, None)


def create_agent_trace_collector() -> AgentTraceCollector:
    return AgentTraceCollector()


def create_agent_trace_artifact(
    state: AgentRunState,
    *,
    include_messages: bool = False,
    include_tool_inputs: bool = False,
    output_preview_length: int = 500,
) -> AgentTraceArtifact:
    steps = [
        AgentTraceStep(
            index=step.index,
            status=step.status,
            tool_calls=[
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    **({"input": tool_call.input} if include_tool_inputs else {}),
                }
                for tool_call in step.tool_calls
            ],
            tool_results=len(step.tool_results),
            usage=step.usage,
            error=step.error,
        )
        for step in state.steps
    ]
    replay = replay_agent_run(state)
    return AgentTraceArtifact(
        run_id=state.run_id,
        agent_name=state.agent_name,
        provider=state.provider,
        model_id=state.model_id,
        status=state.status,
        steps=steps,
        child_runs=[
            {
                "run_id": child.run_id,
                "agent_name": child.agent_name,
                "parent_run_id": child.parent_run_id,
                "status": child.status,
                "tool_name": child.tool_name,
            }
            for child in state.child_runs
        ],
        events=[asdict(event) for event in replay.timeline],
        usage=state.usage,
        output_preview=state.output_text[:output_preview_length],
        output_text=state.output_text if include_messages else None,
        error=state.error,
        cancellation_reason=state.cancellation_reason,
    )


def estimate_token_cost(usage: TokenUsage | None, pricing: TokenPricing) -> CostEstimate:
    if usage is None:
        return CostEstimate(currency=pricing.currency, usage=None)
    input_cost = (
        (usage.input_tokens or 0) / 1000 * pricing.input_cost_per_1k_tokens
        if pricing.input_cost_per_1k_tokens is not None
        else None
    )
    output_cost = (
        (usage.output_tokens or 0) / 1000 * pricing.output_cost_per_1k_tokens
        if pricing.output_cost_per_1k_tokens is not None
        else None
    )
    total_cost = None
    if pricing.total_cost_per_1k_tokens is not None:
        total_cost = (usage.total_tokens or (usage.input_tokens or 0) + (usage.output_tokens or 0)) / 1000 * pricing.total_cost_per_1k_tokens
    elif input_cost is not None or output_cost is not None:
        total_cost = (input_cost or 0) + (output_cost or 0)
    return CostEstimate(input_cost=input_cost, output_cost=output_cost, total_cost=total_cost, currency=pricing.currency, usage=usage)


def estimate_agent_run_cost(state: AgentRunState, pricing: TokenPricing | ModelCatalog) -> CostEstimate:
    if isinstance(pricing, ModelCatalog):
        entry = pricing.find(state.provider, state.model_id)
        rate = entry.cost_per_1k_tokens if entry is not None else None
        return estimate_token_cost(state.usage, TokenPricing(total_cost_per_1k_tokens=rate))
    return estimate_token_cost(state.usage, pricing)


def summarize_agent_trace(state_or_trace: AgentRunState | AgentTraceArtifact, *, pricing: TokenPricing | ModelCatalog | None = None) -> AgentTraceSummary:
    if isinstance(state_or_trace, AgentTraceArtifact):
        usage = state_or_trace.usage
        cost = None
        if pricing is not None:
            synthetic = AgentRunState(
                run_id=state_or_trace.run_id,
                agent_name=state_or_trace.agent_name,
                provider=state_or_trace.provider,
                model_id=state_or_trace.model_id,
                usage=usage,
            )
            cost = estimate_agent_run_cost(synthetic, pricing)
        return AgentTraceSummary(
            run_id=state_or_trace.run_id,
            agent_name=state_or_trace.agent_name,
            provider=state_or_trace.provider,
            model_id=state_or_trace.model_id,
            status=state_or_trace.status,
            steps=len(state_or_trace.steps),
            child_runs=len(state_or_trace.child_runs),
            tool_calls=sum(len(step.tool_calls) for step in state_or_trace.steps),
            tool_errors=0,
            usage=usage,
            cost=cost,
            error=state_or_trace.error,
        )
    cost = estimate_agent_run_cost(state_or_trace, pricing) if pricing is not None else None
    return AgentTraceSummary(
        run_id=state_or_trace.run_id,
        agent_name=state_or_trace.agent_name,
        provider=state_or_trace.provider,
        model_id=state_or_trace.model_id,
        status=state_or_trace.status,
        steps=state_or_trace.current_step,
        child_runs=len(state_or_trace.child_runs),
        tool_calls=sum(len(step.tool_calls) for step in state_or_trace.steps),
        tool_errors=sum(1 for result in state_or_trace.tool_results if result.is_error),
        usage=state_or_trace.usage,
        cost=cost,
        error=state_or_trace.error,
    )


async def create_agent_run_tree_snapshot(store: AgentRunStore, run_id: str) -> AgentRunTreeSnapshot:
    root_state = await store.load(run_id)
    if root_state is None:
        raise ValueError(f'Agent run "{run_id}" was not found.')

    async def build(state: AgentRunState) -> AgentRunTreeNode:
        children = [await build(child) for child in await store.find_by_parent_run_id(state.run_id)]
        return AgentRunTreeNode(
            run_id=state.run_id,
            agent_name=state.agent_name,
            parent_run_id=state.parent_run_id,
            status=state.status,
            snapshot=create_agent_run_snapshot(state),
            children=children,
        )

    root = await build(root_state)

    def count(node: AgentRunTreeNode) -> int:
        return 1 + sum(count(child) for child in node.children)

    return AgentRunTreeSnapshot(root=root, total_runs=count(root))


async def create_hierarchical_agent_trace(store: AgentRunStore, run_id: str) -> HierarchicalAgentTrace:
    root_state = await store.load(run_id)
    if root_state is None:
        raise ValueError(f'Agent run "{run_id}" was not found.')

    async def build(state: AgentRunState) -> HierarchicalAgentTraceNode:
        children = [await build(child) for child in await store.find_by_parent_run_id(state.run_id)]
        return HierarchicalAgentTraceNode(trace=create_agent_trace_artifact(state), children=children)

    root = await build(root_state)

    def count(node: HierarchicalAgentTraceNode) -> int:
        return 1 + sum(count(child) for child in node.children)

    return HierarchicalAgentTrace(root=root, total_runs=count(root))
