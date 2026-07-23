from __future__ import annotations

import asyncio
import unittest

from zhivex_ai import (
    Agent,
    AgentEvaluationExpectations,
    ApprovalDecision,
    GuardrailResult,
    LoopAgent,
    ParallelAgent,
    SequentialAgent,
    WorkflowStep,
    create_agent_session,
    create_in_memory_agent_run_store,
    create_mock_language_model,
    replay_agent_run,
    run_workflow,
    tool,
    validate_workflow_expectations,
)
from zhivex_ai.errors import ValidationError
from zhivex_ai.types import GenerateResult, ModelMessage, ToolCall, ToolCallPart


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

    async def test_suspended_step_stops_sequential_workflow_without_writing_output(self) -> None:
        continued = False

        async def require_human(_request):
            return ApprovalDecision.require_human("Review required.", approval_id="approval-1")

        async def observe_continuation(_request) -> GuardrailResult:
            nonlocal continued
            continued = True
            return GuardrailResult()

        approval_model = create_mock_language_model(
            responses=[
                GenerateResult(
                    messages=[
                        ModelMessage(
                            role="assistant",
                            parts=[ToolCallPart(tool_call=ToolCall(id="call-1", name="danger", input={}))],
                        )
                    ],
                    finish_reason="tool-calls",
                )
            ]
        )
        approval_agent = Agent(
            name="approval-agent",
            model=approval_model,
            tools={
                "danger": tool(
                    name="danger",
                    schema=dict,
                    execute=lambda _input: "done",
                    requires_approval=True,
                )
            },
            approval_policy=require_human,
            run_store=create_in_memory_agent_run_store(),
        )
        workflow = SequentialAgent(
            name="approval-workflow",
            steps=[
                WorkflowStep("approval", approval_agent, output_key="approved", metadata_key="approval_meta"),
                WorkflowStep(
                    "after",
                    Agent(
                        name="after",
                        model=create_mock_language_model(),
                        input_guardrails=[observe_continuation],
                    ),
                ),
            ],
        )

        result = await workflow.run(prompt="run")

        self.assertEqual(result.status, "suspended")
        self.assertEqual([item.name for item in result.step_results], ["approval"])
        self.assertEqual(result.step_results[0].status, "suspended")
        self.assertFalse(continued)
        self.assertNotIn("approved", result.state)
        self.assertEqual(result.state["approval_meta"]["status"], "suspended")
        self.assertEqual(result.state_snapshot.status, "suspended")
        self.assertEqual(result.state_snapshot.child_runs[0].status, "suspended")
        self.assertEqual(result.state_snapshot.metadata["suspended_steps"], ["approval"])

    async def test_parallel_capture_merges_error_payload_from_isolated_state(self) -> None:
        workflow = ParallelAgent(
            name="parallel-capture",
            steps=[
                WorkflowStep(
                    "bad",
                    agent_with_text("bad"),
                    error_policy="capture",
                    output_key="bad_error",
                    metadata_key="bad_meta",
                )
            ],
        )

        result = await workflow.run()

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.state["bad_error"]["step"], "bad")
        self.assertIn("no responses left", result.state["bad_error"]["error"])
        self.assertEqual(result.state["bad_meta"]["status"], "failed")

    async def test_parallel_fail_fast_cancels_pending_sibling(self) -> None:
        slow_started = asyncio.Event()
        slow_cancelled = asyncio.Event()
        unblock_slow = asyncio.Event()
        side_effects: list[str] = []

        async def slow_tool(_input):
            slow_started.set()
            try:
                await unblock_slow.wait()
            finally:
                slow_cancelled.set()
            side_effects.append("slow-completed")
            return "done"

        async def fail_after_slow_started(_request) -> GuardrailResult:
            await slow_started.wait()
            raise RuntimeError("fail fast")

        slow_model = create_mock_language_model(
            responses=[
                GenerateResult(
                    messages=[
                        ModelMessage(
                            role="assistant",
                            parts=[ToolCallPart(tool_call=ToolCall(id="call-slow", name="slow", input={}))],
                        )
                    ],
                    finish_reason="tool-calls",
                )
            ]
        )
        workflow = ParallelAgent(
            name="parallel-fail-fast",
            steps=[
                WorkflowStep(
                    "bad",
                    Agent(
                        name="bad",
                        model=create_mock_language_model(),
                        input_guardrails=[fail_after_slow_started],
                    ),
                    error_policy="fail_fast",
                ),
                WorkflowStep(
                    "slow",
                    Agent(
                        name="slow",
                        model=slow_model,
                        tools={"slow": tool(name="slow", schema=dict, execute=slow_tool)},
                    ),
                    error_policy="continue",
                ),
            ],
        )

        result = await asyncio.wait_for(workflow.run(), timeout=1)

        self.assertEqual(result.status, "failed")
        self.assertTrue(slow_cancelled.is_set())
        self.assertEqual(side_effects, [])
        self.assertIn("fail fast", str(result.step_results[0].error))
        self.assertIn("Cancelled because another parallel step failed fast", str(result.step_results[1].error))

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
