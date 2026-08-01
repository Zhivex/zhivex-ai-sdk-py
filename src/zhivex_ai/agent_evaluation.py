from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent, AgentRunResult, run_agent
from .agent_state import AgentRunState
from .messages import create_text_message
from .types import FinishReason, GenerateResult, JsonValue, LanguageModel, ModelCapabilities, ModelGenerateInput, StreamEvent, ToolDefinition


@dataclass(slots=True)
class AgentRunSnapshot:
    run_id: str
    agent_name: str
    status: str
    provider: str
    model_id: str
    steps: int
    tool_calls: int
    child_runs: int
    output_text: str
    error: str | None = None


@dataclass(slots=True)
class AgentReplayEvent:
    type: str
    run_id: str
    step: int | None = None
    status: str | None = None
    name: str | None = None
    data: JsonValue | None = None


@dataclass(slots=True)
class AgentReplayResult:
    snapshot: AgentRunSnapshot
    timeline: list[AgentReplayEvent]


@dataclass(slots=True)
class AgentEvaluationExpectations:
    status: str | None = None
    output_contains: str | None = None
    output_equals: str | None = None
    tool_calls: list[str] = field(default_factory=list)
    child_run_count: int | None = None
    child_agents: list[str] = field(default_factory=list)
    child_statuses: list[str] = field(default_factory=list)
    finish_reason: FinishReason | None = None
    error_contains: str | None = None
    workflow_steps: list[str] = field(default_factory=list)
    state_contains: list[str] = field(default_factory=list)
    state_equals: dict[str, JsonValue] = field(default_factory=dict)
    failed_steps: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentEvaluationCase:
    name: str
    prompt: str | None = None
    expectations: AgentEvaluationExpectations | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(slots=True)
class AgentEvaluationCaseResult:
    name: str
    ok: bool
    output: AgentRunResult | None = None
    failures: list[str] = field(default_factory=list)
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(slots=True)
class AgentEvaluationResult:
    ok: bool
    cases: list[AgentEvaluationCaseResult]


@dataclass(slots=True)
class AgentEvaluationFixture:
    name: str
    dataset: list[AgentEvaluationCase]
    expected_ok: bool = True
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(slots=True)
class AgentEvaluationReport:
    ok: bool
    total: int
    passed: int
    failed: int
    pass_rate: float
    failures: list[dict[str, JsonValue]]
    cases: list[dict[str, JsonValue]]
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        return _finite_json_object(
            {
                "ok": self.ok,
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": self.pass_rate,
                "failures": self.failures,
                "cases": self.cases,
                "metadata": self.metadata,
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)


@dataclass(slots=True)
class AgentEvaluationJudgeResult:
    score: float
    feedback: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


AgentEvaluationScorer = Callable[[AgentEvaluationCaseResult], float | Awaitable[float]]
AgentEvaluationAgentFactory = Callable[[AgentEvaluationCase], Agent | Awaitable[Agent]]


@dataclass(slots=True)
class AgentEvaluationMetric:
    name: str
    scorer: AgentEvaluationScorer
    higher_is_better: bool = True


@dataclass(slots=True)
class AgentEvaluationGate:
    metric: str = "pass_rate"
    minimum: float | None = None
    maximum: float | None = None
    max_regression: float | None = 0.0


@dataclass(slots=True)
class AgentEvaluationVariant:
    name: str
    agent: Agent | AgentEvaluationAgentFactory
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(slots=True)
class AgentEvaluationVariantResult:
    name: str
    result: AgentEvaluationResult
    report: AgentEvaluationReport
    metrics: dict[str, float]
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(slots=True)
class AgentEvaluationGateResult:
    variant: str
    metric: str
    value: float
    baseline_value: float
    delta: float
    regression: float
    ok: bool
    minimum: float | None = None
    maximum: float | None = None
    max_regression: float | None = None
    failures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentEvaluationExperimentResult:
    ok: bool
    baseline: str
    variants: list[AgentEvaluationVariantResult]
    gates: list[AgentEvaluationGateResult]
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        return _finite_json_object(
            {
                "ok": self.ok,
                "baseline": self.baseline,
                "variants": [
                    {
                        "name": variant.name,
                        "ok": variant.result.ok,
                        "metrics": variant.metrics,
                        "report": variant.report.to_dict(),
                        "metadata": variant.metadata,
                    }
                    for variant in self.variants
                ],
                "gates": [
                    {
                        "variant": gate.variant,
                        "metric": gate.metric,
                        "value": gate.value,
                        "baseline_value": gate.baseline_value,
                        "delta": gate.delta,
                        "regression": gate.regression,
                        "ok": gate.ok,
                        "minimum": gate.minimum,
                        "maximum": gate.maximum,
                        "max_regression": gate.max_regression,
                        "failures": gate.failures,
                    }
                    for gate in self.gates
                ],
                "metadata": self.metadata,
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)


def _finite_json_value(value: Any, *, path: str = "$") -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Expected finite JSON number at {path}.")
        return value
    if isinstance(value, list):
        return [_finite_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Expected string JSON object key at {path}.")
            normalized[key] = _finite_json_value(item, path=f"{path}.{key}")
        return normalized
    raise TypeError(f"Expected JSON-compatible value at {path}, got {type(value).__name__}.")


def _finite_json_object(value: dict[str, Any]) -> dict[str, JsonValue]:
    normalized = _finite_json_value(value)
    if not isinstance(normalized, dict):
        raise TypeError("Expected a JSON object.")
    return normalized


def create_agent_run_snapshot(state: AgentRunState) -> AgentRunSnapshot:
    return AgentRunSnapshot(
        run_id=state.run_id,
        agent_name=state.agent_name,
        status=state.status,
        provider=state.provider,
        model_id=state.model_id,
        steps=state.current_step,
        tool_calls=sum(len(step.tool_calls) for step in state.steps),
        child_runs=len(state.child_runs),
        output_text=state.output_text,
        error=state.error,
    )


def replay_agent_run(state: AgentRunState) -> AgentReplayResult:
    timeline = [AgentReplayEvent("run-start", state.run_id, status=state.status)]
    workflow_name = state.metadata.get("workflow_name")
    workflow_steps = state.metadata.get("workflow_steps")
    failed_steps = state.metadata.get("failed_steps")
    if isinstance(workflow_name, str):
        timeline.append(AgentReplayEvent("workflow-start", state.run_id, name=workflow_name, status="running"))
        if isinstance(workflow_steps, list):
            for raw_step in workflow_steps:
                if not isinstance(raw_step, str):
                    continue
                status = "failed" if isinstance(failed_steps, list) and raw_step in failed_steps else "completed"
                timeline.append(AgentReplayEvent("workflow-step-start", state.run_id, name=raw_step, status="running"))
                timeline.append(AgentReplayEvent("workflow-step-finish", state.run_id, name=raw_step, status=status))
        timeline.append(AgentReplayEvent("workflow-finish", state.run_id, name=workflow_name, status=state.status))
    for step in state.steps:
        timeline.append(AgentReplayEvent("step-start", state.run_id, step=step.index, status=step.status))
        for tool_call in step.tool_calls:
            timeline.append(AgentReplayEvent("tool-call", state.run_id, step=step.index, name=tool_call.name, data=tool_call.input))
        for result in step.tool_results:
            timeline.append(AgentReplayEvent("tool-result", state.run_id, step=step.index, name=result.tool_name))
        timeline.append(AgentReplayEvent("step-finish", state.run_id, step=step.index, status=step.status, data={"error": step.error} if step.error else None))
    for child in state.child_runs:
        timeline.append(AgentReplayEvent("child-run", state.run_id, name=child.agent_name, status=child.status, data={"run_id": child.run_id, "tool_name": child.tool_name}))
    timeline.append(AgentReplayEvent("run-finish", state.run_id, status=state.status, data={"error": state.error} if state.error else None))
    return AgentReplayResult(snapshot=create_agent_run_snapshot(state), timeline=timeline)


class MockLanguageModel:
    def __init__(
        self,
        *,
        provider: str = "mock",
        model_id: str = "mock-model",
        responses: list[GenerateResult] | None = None,
        stream_events: list[list[StreamEvent]] | None = None,
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.capabilities = ModelCapabilities(
            streaming=True,
            tools=True,
            structured_output=True,
            json_mode=True,
            tool_choice=True,
            parallel_tool_calls=False,
            vision=False,
            files=False,
            audio_input=False,
            audio_output=False,
            embeddings=False,
            reasoning=False,
            web_search=False,
        )
        self._responses = list(responses) if responses is not None else [GenerateResult(text="ok", message=create_text_message("assistant", "ok"), finish_reason="stop")]
        self._stream_events = list(stream_events or [])

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        if not self._responses:
            raise RuntimeError("MockLanguageModel has no responses left.")
        return self._responses.pop(0)

    async def stream(self, input: ModelGenerateInput):
        async def generator():
            if not self._stream_events:
                result = await self.generate(input)
                from .types import StreamFinishEvent, StreamTextDeltaEvent

                yield StreamTextDeltaEvent(text_delta=result.text or "")
                yield StreamFinishEvent(
                    finish_reason=result.finish_reason,
                    provider_finish_reason=result.provider_finish_reason,
                    usage=result.usage,
                )
                return
            for event in self._stream_events.pop(0):
                yield event

        return generator()


def create_mock_language_model(
    *,
    provider: str = "mock",
    model_id: str = "mock-model",
    responses: list[GenerateResult] | None = None,
    stream_events: list[list[StreamEvent]] | None = None,
) -> LanguageModel:
    return MockLanguageModel(provider=provider, model_id=model_id, responses=responses, stream_events=stream_events)


def create_mock_tool(
    name: str,
    *,
    outputs: list[JsonValue] | None = None,
    errors: list[str | Exception] | None = None,
) -> ToolDefinition:
    pending_outputs = list(outputs or ["ok"])
    pending_errors = list(errors or [])

    async def execute(_: Any) -> JsonValue:
        if pending_errors:
            error = pending_errors.pop(0)
            raise error if isinstance(error, Exception) else RuntimeError(error)
        if not pending_outputs:
            raise RuntimeError(f'Mock tool "{name}" has no outputs left.')
        return pending_outputs.pop(0)

    return ToolDefinition(name=name, description=f"Mock tool {name}", schema={"type": "object"}, execute=execute)


def _result_failures(result: AgentRunResult | None, error: Exception | None, expectations: AgentEvaluationExpectations | None) -> list[str]:
    if expectations is None:
        return [] if error is None else [str(error)]
    failures: list[str] = []
    if error is not None:
        if expectations.error_contains is None or expectations.error_contains not in str(error):
            failures.append(str(error))
        return failures
    if result is None:
        return ["Agent did not return a result."]
    status = "completed" if result.finish_reason != "error" else "failed"
    if expectations.status is not None and expectations.status != status:
        failures.append(f"Expected status {expectations.status}, got {status}.")
    if expectations.output_contains is not None and expectations.output_contains not in result.text:
        failures.append(f'Expected output to contain "{expectations.output_contains}".')
    if expectations.output_equals is not None and expectations.output_equals != result.text:
        failures.append("Expected output to equal the configured value.")
    if expectations.finish_reason is not None and expectations.finish_reason != result.finish_reason:
        failures.append(f"Expected finish reason {expectations.finish_reason}, got {result.finish_reason}.")
    tool_names = [tool.tool_name for tool in result.tool_results]
    for expected_tool in expectations.tool_calls:
        if expected_tool not in tool_names:
            failures.append(f'Expected tool call "{expected_tool}".')
    trace = result.trace
    child_agents = trace.orchestration_path[1:] if trace is not None else []
    if expectations.child_run_count is not None and expectations.child_run_count != len(child_agents):
        failures.append(f"Expected {expectations.child_run_count} child runs, got {len(child_agents)}.")
    for child_agent in expectations.child_agents:
        if child_agent not in child_agents:
            failures.append(f'Expected child agent "{child_agent}".')
    state = result.state.metadata.get("state") if result.state is not None else None
    workflow_steps = result.state.metadata.get("workflow_steps") if result.state is not None else None
    failed_steps = result.state.metadata.get("failed_steps") if result.state is not None else None
    if expectations.workflow_steps:
        if not isinstance(workflow_steps, list) or workflow_steps != expectations.workflow_steps:
            failures.append(f"Expected workflow steps {expectations.workflow_steps}, got {workflow_steps}.")
    if expectations.failed_steps:
        if not isinstance(failed_steps, list) or failed_steps != expectations.failed_steps:
            failures.append(f"Expected failed workflow steps {expectations.failed_steps}, got {failed_steps}.")
    if expectations.state_contains:
        if not isinstance(state, dict):
            failures.append("Expected workflow state, got none.")
        else:
            missing = [key for key in expectations.state_contains if key not in state]
            if missing:
                failures.append(f"Expected workflow state keys: {', '.join(missing)}.")
    if expectations.state_equals:
        if not isinstance(state, dict):
            failures.append("Expected workflow state, got none.")
        else:
            for key, expected in expectations.state_equals.items():
                if state.get(key) != expected:
                    failures.append(f'Expected workflow state "{key}" to equal {expected!r}.')
    return failures


async def run_agent_evaluation(
    *,
    agent: Agent | Callable[[AgentEvaluationCase], Agent | Awaitable[Agent]],
    dataset: list[AgentEvaluationCase],
) -> AgentEvaluationResult:
    results: list[AgentEvaluationCaseResult] = []
    for test_case in dataset:
        resolved_agent = agent(test_case) if callable(agent) else agent
        if isinstance(resolved_agent, Awaitable):
            resolved_agent = await resolved_agent
        output: AgentRunResult | None = None
        error: Exception | None = None
        try:
            output = await run_agent(agent=resolved_agent, prompt=test_case.prompt)
        except Exception as exc:
            error = exc
        failures = _result_failures(output, error, test_case.expectations)
        results.append(
            AgentEvaluationCaseResult(
                name=test_case.name,
                ok=not failures,
                output=output,
                failures=failures,
                metadata=test_case.metadata,
            )
        )
    return AgentEvaluationResult(ok=all(item.ok for item in results), cases=results)


def create_agent_evaluation_fixture(
    *,
    name: str,
    dataset: list[AgentEvaluationCase],
    expected_ok: bool = True,
    metadata: dict[str, JsonValue] | None = None,
) -> AgentEvaluationFixture:
    return AgentEvaluationFixture(name=name, dataset=dataset, expected_ok=expected_ok, metadata=metadata or {})


async def run_agent_evaluation_fixture(
    fixture: AgentEvaluationFixture,
    *,
    agent: Agent | Callable[[AgentEvaluationCase], Agent | Awaitable[Agent]],
) -> AgentEvaluationResult:
    result = await run_agent_evaluation(agent=agent, dataset=fixture.dataset)
    if result.ok != fixture.expected_ok:
        return AgentEvaluationResult(
            ok=False,
            cases=[
                *result.cases,
                AgentEvaluationCaseResult(
                    name=f"{fixture.name}:expected_ok",
                    ok=False,
                    failures=[f"Fixture expected ok={fixture.expected_ok}, got ok={result.ok}."],
                ),
            ],
        )
    return result


def create_agent_evaluation_report(result: AgentEvaluationResult, *, metadata: dict[str, JsonValue] | None = None) -> AgentEvaluationReport:
    total = len(result.cases)
    passed = sum(1 for item in result.cases if item.ok)
    cases: list[dict[str, JsonValue]] = [
        {
            "name": item.name,
            "ok": item.ok,
            "failures": list(item.failures),
            "output_preview": (item.output.text[:500] if item.output else ""),
            "metadata": item.metadata,
        }
        for item in result.cases
    ]
    return AgentEvaluationReport(
        ok=result.ok,
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=passed / total if total else 1.0,
        failures=[{"name": item.name, "failures": list(item.failures)} for item in result.cases if item.failures],
        cases=cases,
        metadata=metadata or {},
    )


def _normalize_experiment_variants(
    variants: list[AgentEvaluationVariant] | dict[str, Agent | AgentEvaluationAgentFactory],
) -> list[AgentEvaluationVariant]:
    normalized = (
        list(variants)
        if isinstance(variants, list)
        else [AgentEvaluationVariant(name=name, agent=agent) for name, agent in variants.items()]
    )
    if not normalized:
        raise ValueError("Agent evaluation experiments require at least one variant.")
    names: set[str] = set()
    for variant in normalized:
        if not variant.name:
            raise ValueError("Agent evaluation variant names cannot be empty.")
        if variant.name in names:
            raise ValueError(f'Duplicate agent evaluation variant name "{variant.name}".')
        names.add(variant.name)
    return normalized


def _normalize_experiment_metrics(metrics: list[AgentEvaluationMetric] | None) -> list[AgentEvaluationMetric]:
    normalized = list(metrics or [])
    names = {"pass_rate"}
    for metric in normalized:
        if not metric.name:
            raise ValueError("Agent evaluation metric names cannot be empty.")
        if metric.name in names:
            raise ValueError(f'Duplicate agent evaluation metric name "{metric.name}".')
        names.add(metric.name)
    return normalized


def _finite_metric(value: float, *, metric: str, case: str | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        location = f' for case "{case}"' if case is not None else ""
        raise TypeError(f'Metric "{metric}"{location} must return a number.')
    normalized = float(value)
    if not math.isfinite(normalized):
        location = f' for case "{case}"' if case is not None else ""
        raise ValueError(f'Metric "{metric}"{location} must return a finite number.')
    return normalized


async def _aggregate_experiment_metrics(
    result: AgentEvaluationResult,
    metrics: list[AgentEvaluationMetric],
) -> dict[str, float]:
    report = create_agent_evaluation_report(result)
    aggregated = {"pass_rate": report.pass_rate}
    for metric in metrics:
        values: list[float] = []
        for case in result.cases:
            score = metric.scorer(case)
            if isinstance(score, Awaitable):
                score = await score
            values.append(_finite_metric(score, metric=metric.name, case=case.name))
        try:
            aggregate = math.fsum(values) / len(values) if values else 0.0
        except OverflowError as exc:
            raise ValueError(f'Aggregate metric "{metric.name}" must be finite.') from exc
        aggregated[metric.name] = _finite_metric(aggregate, metric=metric.name)
    return aggregated


def _evaluate_experiment_gates(
    *,
    variants: list[AgentEvaluationVariantResult],
    baseline: AgentEvaluationVariantResult,
    metrics: list[AgentEvaluationMetric],
    gates: list[AgentEvaluationGate],
) -> list[AgentEvaluationGateResult]:
    directions = {"pass_rate": True, **{metric.name: metric.higher_is_better for metric in metrics}}
    results: list[AgentEvaluationGateResult] = []
    for gate in gates:
        if gate.metric not in directions:
            raise ValueError(f'Unknown agent evaluation gate metric "{gate.metric}".')
        if gate.max_regression is not None and gate.max_regression < 0:
            raise ValueError("Agent evaluation max_regression must be non-negative.")
        minimum = None if gate.minimum is None else _finite_metric(gate.minimum, metric=gate.metric)
        maximum = None if gate.maximum is None else _finite_metric(gate.maximum, metric=gate.metric)
        max_regression = (
            None if gate.max_regression is None else _finite_metric(gate.max_regression, metric=gate.metric)
        )
        baseline_value = baseline.metrics[gate.metric]
        higher_is_better = directions[gate.metric]
        for variant in variants:
            value = variant.metrics[gate.metric]
            delta = value - baseline_value
            regression = max(0.0, -delta if higher_is_better else delta)
            failures: list[str] = []
            if minimum is not None and value < minimum:
                failures.append(f"Expected {gate.metric} >= {minimum}, got {value}.")
            if maximum is not None and value > maximum:
                failures.append(f"Expected {gate.metric} <= {maximum}, got {value}.")
            if variant.name != baseline.name and max_regression is not None and regression > max_regression:
                failures.append(
                    f"Regression {regression} for {gate.metric} exceeds allowed {max_regression} "
                    f'against baseline "{baseline.name}".'
                )
            results.append(
                AgentEvaluationGateResult(
                    variant=variant.name,
                    metric=gate.metric,
                    value=value,
                    baseline_value=baseline_value,
                    delta=delta,
                    regression=regression,
                    ok=not failures,
                    minimum=minimum,
                    maximum=maximum,
                    max_regression=max_regression,
                    failures=failures,
                )
            )
    return results


async def run_agent_evaluation_experiment(
    *,
    variants: list[AgentEvaluationVariant] | dict[str, Agent | AgentEvaluationAgentFactory],
    dataset: list[AgentEvaluationCase],
    baseline: str | None = None,
    metrics: list[AgentEvaluationMetric] | None = None,
    gates: list[AgentEvaluationGate] | None = None,
    metadata: dict[str, JsonValue] | None = None,
) -> AgentEvaluationExperimentResult:
    """Evaluate agent variants deterministically and apply baseline-aware CI gates.

    Variants, cases, and custom metrics run in their supplied order. Custom metric
    values are averaged across cases, and every emitted number is required to be
    finite so ``to_json`` always produces strict JSON.
    """

    normalized_variants = _normalize_experiment_variants(variants)
    normalized_metrics = _normalize_experiment_metrics(metrics)
    baseline_name = baseline or normalized_variants[0].name
    if baseline_name not in {variant.name for variant in normalized_variants}:
        raise ValueError(f'Unknown agent evaluation baseline "{baseline_name}".')

    variant_results: list[AgentEvaluationVariantResult] = []
    for variant in normalized_variants:
        result = await run_agent_evaluation(agent=variant.agent, dataset=dataset)
        report = create_agent_evaluation_report(result, metadata=variant.metadata)
        variant_results.append(
            AgentEvaluationVariantResult(
                name=variant.name,
                result=result,
                report=report,
                metrics=await _aggregate_experiment_metrics(result, normalized_metrics),
                metadata=variant.metadata,
            )
        )

    baseline_result = next(variant for variant in variant_results if variant.name == baseline_name)
    normalized_gates = [AgentEvaluationGate()] if gates is None else list(gates)
    gate_results = _evaluate_experiment_gates(
        variants=variant_results,
        baseline=baseline_result,
        metrics=normalized_metrics,
        gates=normalized_gates,
    )
    experiment = AgentEvaluationExperimentResult(
        ok=all(gate.ok for gate in gate_results),
        baseline=baseline_name,
        variants=variant_results,
        gates=gate_results,
        metadata=metadata or {},
    )
    # Validate the full artifact eagerly so invalid metadata never yields a
    # partially usable CI result.
    experiment.to_dict()
    return experiment


async def judge_agent_evaluation(
    result: AgentEvaluationResult,
    judge: Callable[[AgentEvaluationResult], AgentEvaluationJudgeResult | Awaitable[AgentEvaluationJudgeResult]] | None = None,
) -> AgentEvaluationJudgeResult:
    if judge is not None:
        judged = judge(result)
        return await judged if isinstance(judged, Awaitable) else judged
    report = create_agent_evaluation_report(result)
    return AgentEvaluationJudgeResult(score=report.pass_rate, feedback=None if result.ok else "One or more evaluation cases failed.")
