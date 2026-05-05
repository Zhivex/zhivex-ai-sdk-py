from __future__ import annotations

import unittest

from zhivex_ai import (
    Agent,
    AgentEvaluationExpectations,
    LoopAgent,
    ParallelAgent,
    SequentialAgent,
    WorkflowStep,
    create_agent_session,
    create_in_memory_agent_run_store,
    create_mock_language_model,
    replay_agent_run,
    run_workflow,
    validate_workflow_expectations,
)
from zhivex_ai.errors import ValidationError
from zhivex_ai.types import GenerateResult


def agent_with_text(name: str, *texts: str) -> Agent:
    return Agent(
        name=name,
        model=create_mock_language_model(
            responses=[GenerateResult(text=text, finish_reason="stop") for text in texts]
        ),
    )


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_state_is_first_class_and_compatible(self) -> None:
        session = create_agent_session(state={"seed": "hello"})

        self.assertEqual(session.state["seed"], "hello")
        self.assertEqual(create_agent_session().state, {})

    async def test_sequential_agent_passes_output_key_to_input_template(self) -> None:
        workflow = SequentialAgent(
            name="pipeline",
            steps=[
                WorkflowStep("extract", agent_with_text("extractor", "application"), prompt="extract", output_key="application"),
                WorkflowStep(
                    "validate",
                    agent_with_text("validator", "valid"),
                    input_template="Validate {application}",
                    output_key="validation",
                    metadata_key="validation_meta",
                ),
            ],
        )

        result = await workflow.run()

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.state["application"], "application")
        self.assertEqual(result.state["validation"], "valid")
        self.assertEqual(result.state["validation_meta"]["run_id"], result.step_results[1].output.run_id)
        self.assertEqual([item.name for item in result.step_results], ["extract", "validate"])

    async def test_sequential_template_missing_state_fails(self) -> None:
        workflow = SequentialAgent(
            name="pipeline",
            steps=[WorkflowStep("validate", agent_with_text("validator", "valid"), input_template="Validate {missing}")],
        )

        result = await workflow.run()

        self.assertEqual(result.status, "failed")
        self.assertIsInstance(result.step_results[0].error, ValidationError)

    async def test_parallel_agent_preserves_order_and_state_outputs(self) -> None:
        workflow = ParallelAgent(
            name="research",
            steps=[
                WorkflowStep("policy", agent_with_text("policy", "policy-result"), output_key="policy"),
                WorkflowStep("risk", agent_with_text("risk", "risk-result"), output_key="risk"),
            ],
        )

        result = await workflow.run(prompt="research")

        self.assertEqual([item.name for item in result.step_results], ["policy", "risk"])
        self.assertEqual(result.state["policy"], "policy-result")
        self.assertEqual(result.state["risk"], "risk-result")

    async def test_parallel_agent_rejects_duplicate_output_keys(self) -> None:
        with self.assertRaisesRegex(ValidationError, "output_key"):
            ParallelAgent(
                name="bad",
                steps=[
                    WorkflowStep("one", agent_with_text("one", "one"), output_key="same"),
                    WorkflowStep("two", agent_with_text("two", "two"), output_key="same"),
                ],
            )

    async def test_loop_agent_stops_by_max_iterations(self) -> None:
        workflow = LoopAgent(
            name="refine",
            steps=[WorkflowStep("draft", agent_with_text("writer", "one", "two"), output_key="draft")],
            max_iterations=2,
        )

        result = await workflow.run(prompt="draft")

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.step_results), 2)
        self.assertEqual(result.state["draft"], "two")

    async def test_loop_agent_stops_by_condition(self) -> None:
        workflow = LoopAgent(
            name="refine",
            steps=[WorkflowStep("draft", agent_with_text("writer", "done", "unused"), output_key="draft")],
            max_iterations=3,
            stop_condition=lambda result: result.state.get("draft") == "done",
        )

        result = await workflow.run(prompt="draft")

        self.assertEqual(len(result.step_results), 1)

    async def test_error_policies_continue_and_capture(self) -> None:
        continue_workflow = SequentialAgent(
            name="continue",
            steps=[
                WorkflowStep("bad", agent_with_text("bad"), error_policy="continue"),
                WorkflowStep("good", agent_with_text("good", "ok"), output_key="good"),
            ],
        )
        capture_workflow = SequentialAgent(
            name="capture",
            steps=[WorkflowStep("bad", agent_with_text("bad"), error_policy="capture", output_key="bad_error")],
        )

        continued = await continue_workflow.run()
        captured = await capture_workflow.run()

        self.assertEqual(continued.status, "failed")
        self.assertEqual(continued.state["good"], "ok")
        self.assertEqual(captured.status, "failed")
        self.assertIn("error", captured.state["bad_error"])

    async def test_workflow_run_state_replay_and_store(self) -> None:
        store = create_in_memory_agent_run_store()
        workflow = SequentialAgent(
            name="pipeline",
            run_store=store,
            steps=[WorkflowStep("step", agent_with_text("agent", "ok"), output_key="out")],
        )

        result = await run_workflow(workflow)
        persisted = await store.load(result.run_id)
        replay = replay_agent_run(result.state_snapshot)

        self.assertIsNotNone(persisted)
        self.assertEqual(result.state_snapshot.metadata["workflow_steps"], ["step"])
        self.assertIn("workflow-step-finish", [event.type for event in replay.timeline])

    async def test_workflow_evaluation_expectations(self) -> None:
        workflow = SequentialAgent(
            name="pipeline",
            steps=[WorkflowStep("step", agent_with_text("agent", "ok"), output_key="out")],
        )
        result = await workflow.run()
        expectations = AgentEvaluationExpectations(
            workflow_steps=["step"],
            state_contains=["out"],
            state_equals={"out": "ok"},
        )

        self.assertEqual(validate_workflow_expectations(result, expectations), [])
