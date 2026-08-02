from __future__ import annotations

import asyncio
import json
import math
import unittest
import xml.etree.ElementTree as ET

from zhivex_ai import Agent, create_mock_language_model
from zhivex_ai.agent_state import AgentRunState
from zhivex_ai.agent_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationExpectations,
    AgentEvaluationGate,
    AgentEvaluationMetric,
    AgentEvaluationVariant,
    create_agent_evaluation_report,
    create_agent_evaluation_dataset_from_traces,
    run_agent_evaluation,
    run_agent_evaluation_experiment,
)
from zhivex_ai.messages import create_text_message
from zhivex_ai.types import GenerateResult, TokenUsage


def _agent_with_text(text: str, *, usage: TokenUsage | None = None) -> Agent:
    return Agent(
        name="assistant",
        model=create_mock_language_model(
            responses=[
                GenerateResult(
                    text=text,
                    message=create_text_message("assistant", text),
                    finish_reason="stop",
                    usage=usage,
                )
            ]
        ),
    )


class AgentEvaluationExperimentTests(unittest.IsolatedAsyncioTestCase):
    async def test_compares_variants_with_async_metrics_and_ci_gates(self) -> None:
        scored_cases: list[str] = []

        async def baseline_factory(_: AgentEvaluationCase) -> Agent:
            return _agent_with_text("ok")

        async def candidate_factory(case: AgentEvaluationCase) -> Agent:
            return _agent_with_text("ok" if case.name == "one" else "wrong answer")

        async def output_length(case_result) -> float:
            scored_cases.append(case_result.name)
            return float(len(case_result.output.text if case_result.output else ""))

        dataset = [
            AgentEvaluationCase(
                name=name,
                prompt="answer",
                expectations=AgentEvaluationExpectations(output_equals="ok"),
            )
            for name in ("one", "two")
        ]

        result = await run_agent_evaluation_experiment(
            variants={"baseline": baseline_factory, "candidate": candidate_factory},
            dataset=dataset,
            baseline="baseline",
            metrics=[AgentEvaluationMetric("output_length", output_length, higher_is_better=False)],
            gates=[
                AgentEvaluationGate("pass_rate", max_regression=0.25),
                AgentEvaluationGate("output_length", maximum=10.0, max_regression=1.0),
            ],
            metadata={"suite": "ci"},
        )

        self.assertFalse(result.ok)
        self.assertEqual([variant.name for variant in result.variants], ["baseline", "candidate"])
        self.assertEqual(result.variants[0].metrics["pass_rate"], 1.0)
        self.assertEqual(result.variants[0].metrics["trial_pass_rate"], 1.0)
        self.assertEqual(result.variants[0].metrics["output_length"], 2.0)
        self.assertEqual(result.variants[1].metrics["pass_rate"], 0.5)
        self.assertEqual(result.variants[1].metrics["trial_pass_rate"], 0.5)
        self.assertEqual(result.variants[1].metrics["output_length"], 7.0)
        self.assertEqual(scored_cases, ["one", "two", "one", "two"])
        candidate_gates = [gate for gate in result.gates if gate.variant == "candidate"]
        self.assertEqual([gate.regression for gate in candidate_gates], [0.5, 5.0])
        self.assertTrue(all(not gate.ok for gate in candidate_gates))

        payload = json.loads(result.to_json(), parse_constant=lambda value: self.fail(value))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["baseline"], "baseline")
        self.assertEqual(payload["metadata"], {"suite": "ci"})
        self.assertEqual(payload["variants"][1]["report"]["failed"], 1)

    async def test_baseline_gate_can_allow_known_failures_without_requiring_perfect_suite(self) -> None:
        cases = [
            AgentEvaluationCase(
                name="known-failure",
                expectations=AgentEvaluationExpectations(output_equals="expected"),
            )
        ]

        result = await run_agent_evaluation_experiment(
            variants=[
                AgentEvaluationVariant("baseline", _agent_with_text("actual")),
                AgentEvaluationVariant("candidate", _agent_with_text("actual")),
            ],
            dataset=cases,
        )

        self.assertTrue(result.ok)
        self.assertFalse(result.variants[0].result.ok)
        self.assertFalse(result.variants[1].result.ok)
        self.assertTrue(all(gate.ok for gate in result.gates))

    async def test_rejects_non_finite_metrics_and_metadata(self) -> None:
        case = AgentEvaluationCase(name="case")

        with self.assertRaisesRegex(ValueError, "finite number"):
            await run_agent_evaluation_experiment(
                variants={"baseline": _agent_with_text("ok")},
                dataset=[case],
                metrics=[AgentEvaluationMetric("invalid", lambda _: math.nan)],
            )

        evaluation = await run_agent_evaluation(agent=_agent_with_text("ok"), dataset=[case])
        report = create_agent_evaluation_report(evaluation, metadata={"invalid": math.inf})
        with self.assertRaisesRegex(ValueError, "finite JSON number"):
            report.to_json()

    async def test_validates_baseline_metric_and_gate_names(self) -> None:
        variant = AgentEvaluationVariant("baseline", _agent_with_text("ok"))

        with self.assertRaisesRegex(ValueError, "Unknown agent evaluation baseline"):
            await run_agent_evaluation_experiment(variants=[variant], dataset=[], baseline="missing")

        with self.assertRaisesRegex(ValueError, "Unknown agent evaluation gate metric"):
            await run_agent_evaluation_experiment(
                variants=[variant],
                dataset=[AgentEvaluationCase(name="case")],
                gates=[AgentEvaluationGate("missing")],
            )

    async def test_repetitions_are_ordered_bounded_and_include_trial_metrics(self) -> None:
        active = 0
        peak = 0

        async def factory(_: AgentEvaluationCase) -> Agent:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return _agent_with_text("ok", usage=TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5))

        result = await run_agent_evaluation(
            agent=factory,
            dataset=[
                AgentEvaluationCase("one", expectations=AgentEvaluationExpectations(output_equals="ok")),
                AgentEvaluationCase("two", expectations=AgentEvaluationExpectations(output_equals="ok")),
            ],
            repetitions=3,
            max_concurrency=2,
            cost_estimator=lambda _: 0.25,
        )

        self.assertTrue(result.ok)
        self.assertLessEqual(peak, 2)
        self.assertEqual([case.name for case in result.cases], ["one", "two"])
        self.assertEqual([[trial.repetition for trial in case.trials] for case in result.cases], [[1, 2, 3], [1, 2, 3]])
        report = create_agent_evaluation_report(result)
        self.assertEqual(report.trial_total, 6)
        self.assertEqual(report.metrics["trial_pass_rate"], 1.0)
        self.assertEqual(report.metrics["total_cost"], 1.5)
        self.assertEqual(report.metrics["mean_total_tokens"], 5.0)
        self.assertGreaterEqual(report.metrics["mean_latency_ms"], 0)
        self.assertGreaterEqual(report.metrics["p95_latency_ms"], report.metrics["mean_latency_ms"])

    async def test_validates_dataset_execution_options_and_previously_ignored_expectations(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            await run_agent_evaluation(agent=_agent_with_text("ok"), dataset=[])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            await run_agent_evaluation(
                agent=_agent_with_text("ok"),
                dataset=[AgentEvaluationCase("same"), AgentEvaluationCase("same")],
            )
        for repetitions, concurrency in ((0, 1), (True, 1), (1, 0), (1, False)):
            with self.subTest(repetitions=repetitions, concurrency=concurrency):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    await run_agent_evaluation(
                        agent=_agent_with_text("ok"),
                        dataset=[AgentEvaluationCase("case")],
                        repetitions=repetitions,
                        max_concurrency=concurrency,
                    )

        async def factory(_: AgentEvaluationCase) -> Agent:
            return _agent_with_text("ok")

        checked = await run_agent_evaluation(
            agent=factory,
            dataset=[
                AgentEvaluationCase(
                    "expected-error",
                    expectations=AgentEvaluationExpectations(error_contains="boom"),
                ),
                AgentEvaluationCase(
                    "expected-child-status",
                    expectations=AgentEvaluationExpectations(child_statuses=["completed"]),
                ),
            ],
        )
        self.assertFalse(checked.ok)
        self.assertIn("Expected an error containing", checked.cases[0].failures[0])
        self.assertIn("Expected child status", checked.cases[1].failures[0])

    async def test_trajectory_is_redacted_and_json_and_junit_are_strict(self) -> None:
        result = await run_agent_evaluation(
            agent=_agent_with_text("secret output"),
            dataset=[AgentEvaluationCase("case<&", prompt="secret prompt")],
        )
        trial = result.cases[0].trials[0]
        self.assertIsNotNone(trial.trajectory)
        trajectory_json = json.dumps(trial.trajectory.to_dict() if trial.trajectory else {})
        self.assertNotIn("secret prompt", trajectory_json)
        self.assertNotIn("secret output", trajectory_json)

        report = create_agent_evaluation_report(result)
        payload = json.loads(report.to_json(), parse_constant=lambda value: self.fail(value))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["trial_total"], 1)
        junit = ET.fromstring(report.to_junit_xml())
        self.assertEqual(junit.tag, "testsuite")
        self.assertEqual(junit.attrib["tests"], "1")

        experiment = await run_agent_evaluation_experiment(
            variants={"baseline": _agent_with_text("ok")},
            dataset=[AgentEvaluationCase("case")],
            gates=[AgentEvaluationGate("pass_rate", minimum=1.1)],
        )
        experiment_junit = ET.fromstring(experiment.to_junit_xml())
        self.assertEqual(experiment_junit.tag, "testsuites")
        self.assertEqual(experiment_junit.attrib["failures"], "1")

    def test_trace_to_dataset_requires_application_owned_extractors_and_provenance(self) -> None:
        state = AgentRunState(
            run_id="run-1",
            agent_name="assistant",
            provider="mock",
            model_id="mock-model",
        )
        dataset = create_agent_evaluation_dataset_from_traces(
            [state],
            prompt_extractor=lambda _: "redacted prompt",
            expectations_extractor=lambda _: AgentEvaluationExpectations(output_contains="approved"),
            metadata_extractor=lambda _: {"source": "reviewed"},
        )
        self.assertEqual(dataset[0].name, "run-1")
        self.assertEqual(dataset[0].metadata["source_run_id"], "run-1")
        self.assertEqual(dataset[0].metadata["source"], "reviewed")

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            create_agent_evaluation_dataset_from_traces(
                [state, state],
                prompt_extractor=lambda _: None,
                expectations_extractor=lambda _: None,
            )
