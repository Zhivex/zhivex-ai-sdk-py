from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .agent_evaluation import AgentRunSnapshot, create_agent_run_snapshot, replay_agent_run
from .agent_state import AgentRunState, AgentRunStore
from .cache_usage import detail_token_sum, price_cache_write_lifetimes
from .catalog import ModelCatalog, ModelPricing
from .errors import ValidationError
from .types import TokenUsage


_OTEL_ATTRIBUTES = frozenset({
    "agent.name", "gen_ai.agent.name", "gen_ai.operation.name", "gen_ai.provider.name",
    "gen_ai.request.model", "gen_ai.response.finish_reasons",     "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens", "gen_ai.usage.total_tokens",
    "gen_ai.usage.cached_input_tokens", "gen_ai.usage.cache_write_tokens",
    "gen_ai.usage.cache_write_ttl_30m_tokens", "gen_ai.usage.cache_write_ttl_12h_tokens",
    "session.id", "run.id",
    "run.parent_id", "zhivex.duration_ms", "zhivex.run.status", "tool.name", "tool.source",
    "gen_ai.tool.name", "gen_ai.tool.type", "guardrail.name", "guardrail.stage",
    "guardrail.triggered", "orchestration.depth", "summary.updated", "finish.reason",
})


def _safe_otel_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    # Callers own the approved names/IDs. Never export arbitrary metadata or payloads.
    return {key: value for key, value in attributes.items()
            if key in _OTEL_ATTRIBUTES and isinstance(value, (str, bool, int, float))
            and (not isinstance(value, str) or len(value) <= 128)}


@dataclass(slots=True)
class _SpanHandle:
    manager: Any
    span: Any

    def end(self, *, attributes: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        if self.span is not None and attributes:
            for key, value in _safe_otel_attributes(attributes).items():
                self.span.set_attribute(key, value)
        if self.span is not None and error is not None:
            # Exception messages and stack traces can contain prompts, DSNs and keys.
            self.span.set_attribute("error.type", type(error).__name__)
            try:
                from opentelemetry.trace import Status, StatusCode  # type: ignore[import-not-found]

                self.span.set_status(Status(StatusCode.ERROR))
            except Exception:
                pass
        # Passing the exception to the OTel context manager would auto-record it again.
        self.manager.__exit__(None, None, None)


class OTelAgentObserver:
    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> _SpanHandle:
        manager = self._tracer.start_as_current_span(name)
        span = manager.__enter__()
        for key, value in _safe_otel_attributes(attributes or {}).items():
            span.set_attribute(key, value)
        return _SpanHandle(manager=manager, span=span)


def create_otel_agent_observer(*, tracer_name: str = "zhivex_ai.agent", version: str | None = None) -> OTelAgentObserver:
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
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
    cached_input_cost_per_1k_tokens: float | None = None
    cache_write_cost_per_1k_tokens: float | None = None


@dataclass(slots=True)
class CostEstimate:
    input_cost: float | None = None
    output_cost: float | None = None
    total_cost: float | None = None
    currency: str = "USD"
    usage: TokenUsage | None = None
    cached_input_cost: float | None = None
    cache_write_cost: float | None = None


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
    started_at_ms: int | None = None
    finished_at_ms: int | None = None
    duration_ms: int | None = None


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
    duration_ms: int | None = None


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
    duration_ms = (
        max(0, state.finished_at_ms - state.started_at_ms)
        if state.started_at_ms is not None and state.finished_at_ms is not None
        else None
    )
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
        started_at_ms=state.started_at_ms,
        finished_at_ms=state.finished_at_ms,
        duration_ms=duration_ms,
    )


def _component_cost(tokens: int | None, rate_per_1k: float | None) -> float | None:
    if tokens is None:
        return None
    if tokens == 0:
        return 0
    if rate_per_1k is None:
        return None
    if rate_per_1k < 0:
        raise ValidationError("Invalid catalog price.")
    return tokens * rate_per_1k / 1000


def _per_1k(rate_per_1m: float | None) -> float | None:
    return None if rate_per_1m is None else rate_per_1m / 1000


def token_pricing_from_model_pricing(pricing: ModelPricing, usage: TokenUsage | None) -> TokenPricing:
    """Convert catalog rates to per-1k prices, including long-context thresholds."""
    long_context = (
        pricing.long_context_threshold_tokens is not None
        and usage is not None
        and usage.input_tokens is not None
        and usage.input_tokens >= pricing.long_context_threshold_tokens
    )
    input_per_1m = (
        pricing.long_context_input_per_1m_tokens
        if long_context and pricing.long_context_input_per_1m_tokens is not None
        else pricing.input_per_1m_tokens
    )
    output_per_1m = (
        pricing.long_context_output_per_1m_tokens
        if long_context and pricing.long_context_output_per_1m_tokens is not None
        else pricing.output_per_1m_tokens
    )
    multiplier = 1.0
    if (
        long_context
        and pricing.input_per_1m_tokens
        and pricing.long_context_input_per_1m_tokens is not None
    ):
        multiplier = pricing.long_context_input_per_1m_tokens / pricing.input_per_1m_tokens
    cache_write = (
        None
        if pricing.cache_write_per_1m_tokens is None
        else pricing.cache_write_per_1m_tokens * multiplier / 1000
    )
    cached_input = (
        None
        if pricing.cached_input_per_1m_tokens is None
        else pricing.cached_input_per_1m_tokens * multiplier / 1000
    )
    return TokenPricing(
        input_cost_per_1k_tokens=_per_1k(input_per_1m),
        output_cost_per_1k_tokens=_per_1k(output_per_1m),
        currency=pricing.currency,
        cached_input_cost_per_1k_tokens=cached_input,
        cache_write_cost_per_1k_tokens=cache_write,
    )


def estimate_token_cost(usage: TokenUsage | None, pricing: TokenPricing) -> CostEstimate:
    if usage is None:
        return CostEstimate(currency=pricing.currency, usage=None)
    reports_cache = (
        usage.cached_input_tokens is not None
        or usage.cache_write_tokens is not None
        or usage.cache_write_tokens_details is not None
    )
    if pricing.total_cost_per_1k_tokens is not None and not reports_cache:
        total_cost = (
            usage.total_tokens or (usage.input_tokens or 0) + (usage.output_tokens or 0)
        ) / 1000 * pricing.total_cost_per_1k_tokens
        return CostEstimate(
            total_cost=total_cost,
            currency=pricing.currency,
            usage=usage,
        )
    cached = (usage.cached_input_tokens or 0) if reports_cache else None
    written = usage.cache_write_tokens
    details = usage.cache_write_tokens_details
    if written is None and details is not None:
        written = detail_token_sum(details)
    if reports_cache and written is None:
        written = 0
    if (
        usage.input_tokens is not None
        and cached is not None
        and written is not None
        and cached + written > usage.input_tokens
    ):
        raise ValidationError("Cache tokens exceed input_tokens.")
    if details is not None and written is not None and detail_token_sum(details) != written:
        raise ValidationError("Cache write lifetime tokens do not match cache_write_tokens.")
    uncached = (
        usage.input_tokens - cached - written
        if usage.input_tokens is not None and cached is not None and written is not None
        else usage.input_tokens
    )
    input_cost = _component_cost(uncached, pricing.input_cost_per_1k_tokens)
    output_cost = _component_cost(usage.output_tokens, pricing.output_cost_per_1k_tokens)
    cached_input_cost = _component_cost(cached, pricing.cached_input_cost_per_1k_tokens) if reports_cache else None
    cache_write_cost: float | None
    if details is not None and written is not None:
        priced, _reasons = price_cache_write_lifetimes(
            cache_write_tokens=written,
            details=details,
            cache_write_rate_per_1k_tokens=pricing.cache_write_cost_per_1k_tokens,
            input_rate_per_1k_tokens=pricing.input_cost_per_1k_tokens,
        )
        cache_write_cost = None if priced is None else priced.amount
    else:
        cache_write_cost = _component_cost(written, pricing.cache_write_cost_per_1k_tokens) if reports_cache else None
    parts = [input_cost, output_cost]
    if reports_cache:
        parts.extend([cached_input_cost, cache_write_cost])
    if any(part is None for part in parts):
        component_total: float | None = None
    else:
        component_total = float(sum(part or 0 for part in parts))
    if not reports_cache and pricing.total_cost_per_1k_tokens is None and (input_cost is not None or output_cost is not None):
        component_total = float((input_cost or 0) + (output_cost or 0))
    return CostEstimate(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=component_total,
        currency=pricing.currency,
        usage=usage,
        cached_input_cost=cached_input_cost,
        cache_write_cost=cache_write_cost,
    )


def estimate_agent_run_cost(state: AgentRunState, pricing: TokenPricing | ModelCatalog) -> CostEstimate:
    if isinstance(pricing, ModelCatalog):
        entry = pricing.find(state.provider, state.model_id)
        if entry is not None and entry.pricing is not None:
            return estimate_token_cost(state.usage, token_pricing_from_model_pricing(entry.pricing, state.usage))
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
            duration_ms=state_or_trace.duration_ms,
        )
    cost = estimate_agent_run_cost(state_or_trace, pricing) if pricing is not None else None
    duration_ms = (
        max(0, state_or_trace.finished_at_ms - state_or_trace.started_at_ms)
        if state_or_trace.started_at_ms is not None and state_or_trace.finished_at_ms is not None
        else None
    )
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
        duration_ms=duration_ms,
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
