from __future__ import annotations

import json
import math
import unittest

from zhivex_ai import Agent, create_mock_language_model
from zhivex_ai.agent_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationExpectations,
    AgentEvaluationGate,
    AgentEvaluationMetric,
    AgentEvaluationVariant,
    create_agent_evaluation_report,
    run_agent_evaluation,
    run_agent_evaluation_experiment,
)
from zhivex_ai.messages import create_text_message
from zhivex_ai.types import GenerateResult


def _agent_with_text(text: str) -> Agent:
    return Agent(
        name="assistant",
        model=create_mock_language_model(
            responses=[
                GenerateResult(
                    text=text,
                    message=create_text_message("assistant", text),
                    finish_reason="stop",
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
        self.assertEqual(result.variants[0].metrics, {"pass_rate": 1.0, "output_length": 2.0})
        self.assertEqual(result.variants[1].metrics, {"pass_rate": 0.5, "output_length": 7.0})
        self.assertEqual(scored_cases, ["one", "two", "one", "two"])
        candidate_gates = [gate for gate in result.gates if gate.variant == "candidate"]
        self.assertEqual([gate.regression for gate in candidate_gates], [0.5, 5.0])
        self.assertTrue(all(not gate.ok for gate in candidate_gates))

        payload = json.loads(result.to_json(), parse_constant=lambda value: self.fail(value))
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
                dataset=[],
                gates=[AgentEvaluationGate("missing")],
            )
