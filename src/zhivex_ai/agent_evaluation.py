from __future__ import annotations

import asyncio
import json
import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent, AgentRunResult, run_agent
from .agent_state import AgentRunState
from .messages import create_text_message
from .types import FinishReason, GenerateResult, JsonValue, LanguageModel, ModelCapabilities, ModelGenerateInput, StreamEvent, TokenUsage, ToolDefinition


AGENT_EVALUATION_ARTIFACT_SCHEMA_VERSION = 1


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
class AgentEvaluationTrajectoryEvent:
    """A deliberately redacted projection of one runtime trace event."""

    type: str
    agent_name: str | None = None
    name: str | None = None
    source_agent: str | None = None
    target_agent: str | None = None
    status: str | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "type": self.type,
            "agent_name": self.agent_name,
            "name": self.name,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "status": self.status,
        }


@dataclass(slots=True)
class AgentEvaluationTrajectory:
    run_id: str
    orchestration_path: list[str]
    events: list[AgentEvaluationTrajectoryEvent]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "run_id": self.run_id,
            "orchestration_path": list(self.orchestration_path),
            "events": [event.to_dict() for event in self.events],
        }


@dataclass(slots=True)
class AgentEvaluationTrialResult:
    repetition: int
    ok: bool
    output: AgentRunResult | None = None
    failures: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    usage: TokenUsage | None = None
    cost: float | None = None
    trajectory: AgentEvaluationTrajectory | None = None


@dataclass(slots=True)
class AgentEvaluationCaseResult:
    name: str
    ok: bool
    output: AgentRunResult | None = None
    failures: list[str] = field(default_factory=list)
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    trials: list[AgentEvaluationTrialResult] = field(default_factory=list)


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
    trial_total: int = 0
    trial_passed: int = 0
    trial_failed: int = 0
    trial_pass_rate: float = 1.0
    metrics: dict[str, float] = field(default_factory=dict)
    gate_failures: list[str] = field(default_factory=list)
    schema_version: int = AGENT_EVALUATION_ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, JsonValue]:
        return _finite_json_object(
            {
                "schema_version": self.schema_version,
                "ok": self.ok,
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": self.pass_rate,
                "failures": self.failures,
                "cases": self.cases,
                "metadata": self.metadata,
                "trial_total": self.trial_total,
                "trial_passed": self.trial_passed,
                "trial_failed": self.trial_failed,
                "trial_pass_rate": self.trial_pass_rate,
                "metrics": self.metrics,
                "gate_failures": self.gate_failures,
            }
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)

    def to_junit_xml(self, *, suite_name: str = "zhivex-agent-evaluation") -> str:
        suite = _evaluation_report_junit_element(self, suite_name=suite_name)
        return ET.tostring(suite, encoding="unicode")


@dataclass(slots=True)
class AgentEvaluationJudgeResult:
    score: float
    feedback: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


AgentEvaluationScorer = Callable[[AgentEvaluationCaseResult], float | Awaitable[float]]
AgentEvaluationAgentFactory = Callable[[AgentEvaluationCase], Agent | Awaitable[Agent]]
AgentEvaluationCostEstimator = Callable[[AgentRunResult], float | None | Awaitable[float | None]]
AgentEvaluationTracePromptExtractor = Callable[[AgentRunState], str | None]
AgentEvaluationTraceExpectationsExtractor = Callable[[AgentRunState], AgentEvaluationExpectations | None]
AgentEvaluationTraceNameExtractor = Callable[[AgentRunState], str]
AgentEvaluationTraceMetadataExtractor = Callable[[AgentRunState], dict[str, JsonValue]]


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
    schema_version: int = AGENT_EVALUATION_ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, JsonValue]:
        return _finite_json_object(
            {
                "schema_version": self.schema_version,
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

    def to_junit_xml(self, *, suite_name: str = "zhivex-agent-evaluation-experiment") -> str:
        suites = ET.Element("testsuites", {"name": suite_name})
        test_count = 0
        failure_count = 0
        total_time = 0.0
        for variant in self.variants:
            suite = _evaluation_report_junit_element(
                variant.report,
                suite_name=f"{suite_name}:{variant.name}",
            )
            suites.append(suite)
            test_count += int(suite.attrib["tests"])
            failure_count += int(suite.attrib["failures"])
            total_time += float(suite.attrib["time"])

        gate_suite = ET.SubElement(suites, "testsuite", {"name": f"{suite_name}:gates"})
        for gate in self.gates:
            test_case = ET.SubElement(
                gate_suite,
                "testcase",
                {
                    "classname": "zhivex.agent-evaluation.gate",
                    "name": f"{gate.variant}:{gate.metric}",
                },
            )
            if not gate.ok:
                failure = ET.SubElement(test_case, "failure", {"message": "; ".join(gate.failures)})
                failure.text = "\n".join(gate.failures)
                failure_count += 1
            test_count += 1
        gate_suite.set("tests", str(len(self.gates)))
        gate_suite.set("failures", str(sum(1 for gate in self.gates if not gate.ok)))
        gate_suite.set("time", "0")
        suites.set("tests", str(test_count))
        suites.set("failures", str(failure_count))
        suites.set("time", _junit_seconds(total_time))
        return ET.tostring(suites, encoding="unicode")


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


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _validate_evaluation_dataset(dataset: list[AgentEvaluationCase]) -> None:
    if not dataset:
        raise ValueError("Agent evaluation datasets cannot be empty.")
    names: set[str] = set()
    for test_case in dataset:
        if not isinstance(test_case.name, str) or not test_case.name:
            raise ValueError("Agent evaluation case names cannot be empty.")
        if test_case.name in names:
            raise ValueError(f'Duplicate agent evaluation case name "{test_case.name}".')
        names.add(test_case.name)
        _finite_json_object(test_case.metadata)


def _event_string(event: Any, field_name: str) -> str | None:
    value = getattr(event, field_name, None)
    return value if isinstance(value, str) and value else None


def create_agent_evaluation_trajectory(result: AgentRunResult) -> AgentEvaluationTrajectory | None:
    """Create a trace projection that excludes messages, tool payloads, and error bodies."""

    if result.trace is None:
        return None
    events: list[AgentEvaluationTrajectoryEvent] = []
    for event in result.trace.events:
        event_type = _event_string(event, "type") or event.__class__.__name__
        name = (
            _event_string(event, "tool_name")
            or _event_string(event, "skill_name")
            or _event_string(event, "guardrail_name")
        )
        tool_call = getattr(event, "tool_call", None)
        if name is None and tool_call is not None:
            name = _event_string(tool_call, "name")
        tool_result = getattr(event, "tool_result", None)
        if name is None and tool_result is not None:
            name = _event_string(tool_result, "tool_name")
        status = _event_string(event, "finish_reason")
        event_ok = getattr(event, "ok", None)
        if status is None and isinstance(event_ok, bool):
            status = "ok" if event_ok else "failed"
        events.append(
            AgentEvaluationTrajectoryEvent(
                type=event_type,
                agent_name=_event_string(event, "agent_name"),
                name=name,
                source_agent=_event_string(event, "source_agent"),
                target_agent=_event_string(event, "target_agent"),
                status=status,
            )
        )
    return AgentEvaluationTrajectory(
        run_id=result.run_id,
        orchestration_path=list(result.trace.orchestration_path),
        events=events,
    )


def create_agent_evaluation_dataset_from_traces(
    traces: list[AgentRunState],
    *,
    prompt_extractor: AgentEvaluationTracePromptExtractor,
    expectations_extractor: AgentEvaluationTraceExpectationsExtractor,
    name_extractor: AgentEvaluationTraceNameExtractor | None = None,
    metadata_extractor: AgentEvaluationTraceMetadataExtractor | None = None,
) -> list[AgentEvaluationCase]:
    """Build evaluation cases without implicitly copying persisted prompt or output data.

    Both content-bearing fields are supplied by application-owned extractors so
    applications can apply their own consent, retention, and redaction policy.
    """

    dataset: list[AgentEvaluationCase] = []
    for state in traces:
        name = name_extractor(state) if name_extractor is not None else state.run_id
        prompt = prompt_extractor(state)
        if prompt is not None and not isinstance(prompt, str):
            raise TypeError("Agent evaluation trace prompt extractors must return str or None.")
        expectations = expectations_extractor(state)
        if expectations is not None and not isinstance(expectations, AgentEvaluationExpectations):
            raise TypeError(
                "Agent evaluation trace expectations extractors must return "
                "AgentEvaluationExpectations or None."
            )
        metadata: dict[str, JsonValue] = {
            "source_run_id": state.run_id,
            "source_provider": state.provider,
            "source_model_id": state.model_id,
        }
        if metadata_extractor is not None:
            extracted_metadata = metadata_extractor(state)
            if not isinstance(extracted_metadata, dict):
                raise TypeError("Agent evaluation trace metadata extractors must return a dict.")
            metadata.update(_finite_json_object(extracted_metadata))
        dataset.append(
            AgentEvaluationCase(
                name=name,
                prompt=prompt,
                expectations=expectations,
                metadata=metadata,
            )
        )
    _validate_evaluation_dataset(dataset)
    return dataset


def _usage_to_json(usage: TokenUsage | None) -> dict[str, JsonValue] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _junit_seconds(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _evaluation_report_junit_element(report: AgentEvaluationReport, *, suite_name: str) -> ET.Element:
    test_total = report.trial_total + len(report.gate_failures)
    failure_total = report.trial_failed + len(report.gate_failures)
    total_seconds = 0.0
    suite = ET.Element(
        "testsuite",
        {
            "name": suite_name,
            "tests": str(test_total),
            "failures": str(failure_total),
        },
    )
    for raw_case in report.cases:
        case_name = raw_case.get("name")
        normalized_name = case_name if isinstance(case_name, str) else "case"
        raw_trials = raw_case.get("trials")
        trials = raw_trials if isinstance(raw_trials, list) else []
        if not trials:
            trials = [
                {
                    "repetition": 1,
                    "ok": raw_case.get("ok") is True,
                    "failures": raw_case.get("failures") if isinstance(raw_case.get("failures"), list) else [],
                    "latency_ms": 0.0,
                }
            ]
        for raw_trial in trials:
            if not isinstance(raw_trial, dict):
                continue
            repetition = raw_trial.get("repetition")
            latency_ms = raw_trial.get("latency_ms")
            seconds = float(latency_ms) / 1000 if isinstance(latency_ms, (int, float)) else 0.0
            total_seconds += seconds
            test_case = ET.SubElement(
                suite,
                "testcase",
                {
                    "classname": "zhivex.agent-evaluation",
                    "name": f"{normalized_name} [repetition {repetition}]",
                    "time": _junit_seconds(seconds),
                },
            )
            if raw_trial.get("ok") is not True:
                raw_failures = raw_trial.get("failures")
                failures = [str(item) for item in raw_failures] if isinstance(raw_failures, list) else []
                message = "; ".join(failures) or "Agent evaluation failed."
                failure = ET.SubElement(test_case, "failure", {"message": message})
                failure.text = "\n".join(failures)
    for index, gate_failure in enumerate(report.gate_failures, start=1):
        test_case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "zhivex.agent-evaluation.gate",
                "name": f"gate-{index}",
                "time": "0",
            },
        )
        failure = ET.SubElement(test_case, "failure", {"message": gate_failure})
        failure.text = gate_failure
    suite.set("time", _junit_seconds(total_seconds))
    return suite


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
    if expectations.error_contains is not None:
        state_error = result.state.error if result.state is not None else None
        if state_error is None or expectations.error_contains not in state_error:
            failures.append(f'Expected an error containing "{expectations.error_contains}".')
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
    child_statuses = [child.status for child in result.state.child_runs] if result.state is not None else []
    for child_status in expectations.child_statuses:
        if child_status not in child_statuses:
            failures.append(f'Expected child status "{child_status}".')
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


async def _run_agent_evaluation_trial(
    *,
    agent: Agent | AgentEvaluationAgentFactory,
    test_case: AgentEvaluationCase,
    repetition: int,
    semaphore: asyncio.Semaphore,
    cost_estimator: AgentEvaluationCostEstimator | None,
) -> AgentEvaluationTrialResult:
    output: AgentRunResult | None = None
    error: Exception | None = None
    async with semaphore:
        started_at = time.perf_counter()
        try:
            resolved_agent = agent(test_case) if callable(agent) else agent
            if isinstance(resolved_agent, Awaitable):
                resolved_agent = await resolved_agent
            output = await run_agent(agent=resolved_agent, prompt=test_case.prompt)
        except Exception as exc:
            error = exc
        latency_ms = (time.perf_counter() - started_at) * 1000
        cost: float | None = None
        if output is not None and cost_estimator is not None:
            estimated_cost = cost_estimator(output)
            if isinstance(estimated_cost, Awaitable):
                estimated_cost = await estimated_cost
            if estimated_cost is not None:
                cost = _finite_metric(estimated_cost, metric="cost", case=test_case.name)

    failures = _result_failures(output, error, test_case.expectations)
    return AgentEvaluationTrialResult(
        repetition=repetition,
        ok=not failures,
        output=output,
        failures=failures,
        latency_ms=_finite_metric(latency_ms, metric="latency_ms", case=test_case.name),
        usage=output.usage if output is not None else None,
        cost=cost,
        trajectory=create_agent_evaluation_trajectory(output) if output is not None else None,
    )


async def run_agent_evaluation(
    *,
    agent: Agent | AgentEvaluationAgentFactory,
    dataset: list[AgentEvaluationCase],
    repetitions: int = 1,
    max_concurrency: int = 1,
    cost_estimator: AgentEvaluationCostEstimator | None = None,
) -> AgentEvaluationResult:
    """Run a dataset with bounded concurrency while preserving input order."""

    _validate_evaluation_dataset(dataset)
    normalized_repetitions = _validate_positive_integer(repetitions, name="repetitions")
    normalized_concurrency = _validate_positive_integer(max_concurrency, name="max_concurrency")
    semaphore = asyncio.Semaphore(normalized_concurrency)
    pending = [
        _run_agent_evaluation_trial(
            agent=agent,
            test_case=test_case,
            repetition=repetition,
            semaphore=semaphore,
            cost_estimator=cost_estimator,
        )
        for test_case in dataset
        for repetition in range(1, normalized_repetitions + 1)
    ]
    ordered_trials = await asyncio.gather(*pending)

    results: list[AgentEvaluationCaseResult] = []
    for case_index, test_case in enumerate(dataset):
        start = case_index * normalized_repetitions
        trials = ordered_trials[start : start + normalized_repetitions]
        failures = (
            list(trials[0].failures)
            if normalized_repetitions == 1
            else [
                f"Repetition {trial.repetition}: {failure}"
                for trial in trials
                for failure in trial.failures
            ]
        )
        results.append(
            AgentEvaluationCaseResult(
                name=test_case.name,
                ok=all(trial.ok for trial in trials),
                output=trials[0].output,
                failures=failures,
                metadata=test_case.metadata,
                trials=list(trials),
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
    repetitions: int = 1,
    max_concurrency: int = 1,
    cost_estimator: AgentEvaluationCostEstimator | None = None,
) -> AgentEvaluationResult:
    result = await run_agent_evaluation(
        agent=agent,
        dataset=fixture.dataset,
        repetitions=repetitions,
        max_concurrency=max_concurrency,
        cost_estimator=cost_estimator,
    )
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


def _effective_case_trials(case: AgentEvaluationCaseResult) -> list[AgentEvaluationTrialResult]:
    if case.trials:
        return case.trials
    return [
        AgentEvaluationTrialResult(
            repetition=1,
            ok=case.ok,
            output=case.output,
            failures=list(case.failures),
            usage=case.output.usage if case.output is not None else None,
            trajectory=create_agent_evaluation_trajectory(case.output) if case.output is not None else None,
        )
    ]


def _usage_total(usage: TokenUsage | None) -> int | None:
    if usage is None:
        return None
    if usage.total_tokens is not None:
        return usage.total_tokens
    if usage.input_tokens is not None and usage.output_tokens is not None:
        return usage.input_tokens + usage.output_tokens
    return None


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _builtin_trial_metrics(result: AgentEvaluationResult) -> dict[str, float]:
    trials = [trial for case in result.cases for trial in _effective_case_trials(case)]
    if not trials:
        return {"trial_pass_rate": 1.0}
    latencies = [trial.latency_ms for trial in trials]
    passed = sum(1 for trial in trials if trial.ok)
    metrics = {
        "trial_pass_rate": passed / len(trials),
        "mean_latency_ms": math.fsum(latencies) / len(latencies),
        "p95_latency_ms": _nearest_rank_percentile(latencies, 0.95),
    }
    costs = [trial.cost for trial in trials if trial.cost is not None]
    if len(costs) == len(trials):
        total_cost = math.fsum(costs)
        metrics["total_cost"] = total_cost
        metrics["mean_cost"] = total_cost / len(costs)
    token_totals = [_usage_total(trial.usage) for trial in trials]
    if all(value is not None for value in token_totals):
        normalized_tokens = [float(value) for value in token_totals if value is not None]
        metrics["mean_total_tokens"] = math.fsum(normalized_tokens) / len(normalized_tokens)
    return {name: _finite_metric(value, metric=name) for name, value in metrics.items()}


def create_agent_evaluation_report(
    result: AgentEvaluationResult,
    *,
    metadata: dict[str, JsonValue] | None = None,
) -> AgentEvaluationReport:
    total = len(result.cases)
    passed = sum(1 for item in result.cases if item.ok)
    cases: list[dict[str, JsonValue]] = [
        {
            "name": item.name,
            "ok": item.ok,
            "failures": list(item.failures),
            "output_preview": (item.output.text[:500] if item.output else ""),
            "metadata": item.metadata,
            "trials": [
                {
                    "repetition": trial.repetition,
                    "ok": trial.ok,
                    "failures": list(trial.failures),
                    "latency_ms": trial.latency_ms,
                    "usage": _usage_to_json(trial.usage),
                    "cost": trial.cost,
                    "trajectory": trial.trajectory.to_dict() if trial.trajectory is not None else None,
                }
                for trial in _effective_case_trials(item)
            ],
        }
        for item in result.cases
    ]
    trials = [trial for case in result.cases for trial in _effective_case_trials(case)]
    trial_passed = sum(1 for trial in trials if trial.ok)
    metrics = _builtin_trial_metrics(result)
    return AgentEvaluationReport(
        ok=result.ok,
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=passed / total if total else 1.0,
        failures=[{"name": item.name, "failures": list(item.failures)} for item in result.cases if item.failures],
        cases=cases,
        metadata=metadata or {},
        trial_total=len(trials),
        trial_passed=trial_passed,
        trial_failed=len(trials) - trial_passed,
        trial_pass_rate=trial_passed / len(trials) if trials else 1.0,
        metrics=metrics,
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
    names = {
        "pass_rate",
        "trial_pass_rate",
        "mean_latency_ms",
        "p95_latency_ms",
        "total_cost",
        "mean_cost",
        "mean_total_tokens",
    }
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
    aggregated = {"pass_rate": report.pass_rate, **report.metrics}
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
    directions = {
        "pass_rate": True,
        "trial_pass_rate": True,
        "mean_latency_ms": False,
        "p95_latency_ms": False,
        "total_cost": False,
        "mean_cost": False,
        "mean_total_tokens": False,
        **{metric.name: metric.higher_is_better for metric in metrics},
    }
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
        missing_variants = [variant.name for variant in variants if gate.metric not in variant.metrics]
        if missing_variants:
            raise ValueError(
                f'Agent evaluation gate metric "{gate.metric}" is unavailable for variants: '
                f"{', '.join(missing_variants)}."
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
    repetitions: int = 1,
    max_concurrency: int = 1,
    cost_estimator: AgentEvaluationCostEstimator | None = None,
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
        result = await run_agent_evaluation(
            agent=variant.agent,
            dataset=dataset,
            repetitions=repetitions,
            max_concurrency=max_concurrency,
            cost_estimator=cost_estimator,
        )
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
