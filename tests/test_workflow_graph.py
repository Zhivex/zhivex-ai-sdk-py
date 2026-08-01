from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from zhivex_ai import (
    Agent,
    GuardrailResult,
    ToolExecutionContext,
    WorkflowStep,
    create_mock_language_model,
    tool,
)
from zhivex_ai.errors import ProviderHTTPError, ValidationError
from zhivex_ai.types import GenerateResult, ModelMessage, ToolCall, ToolCallPart
from zhivex_ai.workflow import WorkflowFunctionResult, WorkflowRetryPolicy
from zhivex_ai.workflow_adapters import CallbackWorkflowAdapter, WorkflowStepOutcome
from zhivex_ai.workflow_graph import WorkflowBuilder, WorkflowEdge, WorkflowGraph, fork_workflow, resume_workflow
from zhivex_ai.workflow_state import (
    create_in_memory_workflow_checkpoint_store,
    create_sqlite_workflow_checkpoint_store,
)


def agent_with_text(name: str, *texts: str) -> Agent:
    return Agent(
        name=name,
        model=create_mock_language_model(
            responses=[GenerateResult(text=text, finish_reason="stop") for text in texts]
        ),
    )


class WorkflowGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_functional_step_receives_durable_context_and_patches_state(self) -> None:
        seen: list[tuple[str, int, str, object]] = []

        async def calculate(context):
            seen.append((context.step_name, context.attempt, context.idempotency_key, context.deps))
            return WorkflowFunctionResult(
                output={"score": 7},
                state_patch={"decision": "review"},
                metadata={"kind": "deterministic"},
            )

        graph = (
            WorkflowBuilder("functional")
            .add_step(
                WorkflowStep("score", executor=calculate, output_key="score"),
                entrypoint=True,
            )
            .build()
        )
        deps = object()

        result = await graph.run(deps=deps)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.state["score"], {"score": 7})
        self.assertEqual(result.state["decision"], "review")
        self.assertEqual(seen[0][:2], ("score", 1))
        self.assertTrue(seen[0][2].startswith("wfs_"))
        self.assertIs(seen[0][3], deps)

    async def test_branch_decision_is_persisted_and_unselected_path_is_skipped(self) -> None:
        decisions = 0

        def choose_left(context) -> bool:
            nonlocal decisions
            decisions += 1
            return context.source_output == "left"

        graph = (
            WorkflowBuilder("branch", definition_version="1")
            .add_step(WorkflowStep("choose", agent_with_text("chooser", "left"), output_key="choice"), entrypoint=True)
            .add_step(WorkflowStep("left", agent_with_text("left-agent", "accepted"), output_key="left"))
            .add_step(WorkflowStep("right", agent_with_text("right-agent", "rejected"), output_key="right"))
            .add_edge("choose", "left", condition=choose_left, name="choose-left")
            .add_edge("choose", "right", condition=lambda context: context.source_output == "right", name="choose-right")
            .build()
        )

        result = await graph.run()
        history = await graph.checkpoint_store.list_checkpoints(result.run_id)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.state["left"], "accepted")
        self.assertNotIn("right", result.state)
        self.assertEqual(result.checkpoint.nodes["right"].status, "skipped")
        self.assertEqual(decisions, 1)
        self.assertTrue(any(item.transition.type == "workflow-edge-decisions" for item in history))

    async def test_parallel_wave_is_merged_before_outgoing_conditions(self) -> None:
        graph = (
            WorkflowBuilder("fan-in")
            .add_step(WorkflowStep("a", agent_with_text("a-agent", "A"), output_key="a"), entrypoint=True)
            .add_step(WorkflowStep("b", agent_with_text("b-agent", "B"), output_key="b"), entrypoint=True)
            .add_step(WorkflowStep("join", agent_with_text("join-agent", "joined"), output_key="joined"))
            .add_edge("a", "join", condition=lambda context: context.state.get("b") == "B", name="a-ready")
            .add_edge("b", "join", condition=lambda context: context.state.get("a") == "A", name="b-ready")
            .build()
        )

        result = await graph.run()

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.state["joined"], "joined")

    async def test_interrupt_can_resume_after_process_style_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = create_sqlite_workflow_checkpoint_store(str(Path(directory) / "workflows.sqlite3"))
            first = (
                WorkflowBuilder("durable", definition_version="2026-07-31")
                .add_step(WorkflowStep("draft", agent_with_text("writer", "draft"), output_key="draft"), entrypoint=True)
                .add_step(WorkflowStep("publish", agent_with_text("publisher", "published"), output_key="published"))
                .add_edge("draft", "publish")
                .interrupt_after("draft", reason="Human review")
                .build(checkpoint_store=store)
            )
            suspended = await first.run(idempotency_key="durable-1")
            self.assertEqual(suspended.status, "suspended")
            self.assertIsNotNone(suspended.checkpoint.pending_interrupt)

            reconstructed = (
                WorkflowBuilder("durable", definition_version="2026-07-31")
                .add_step(WorkflowStep("draft", agent_with_text("writer", "unused"), output_key="draft"), entrypoint=True)
                .add_step(WorkflowStep("publish", agent_with_text("publisher", "published"), output_key="published"))
                .add_edge("draft", "publish")
                .interrupt_after("draft", reason="Human review")
                .build(checkpoint_store=store)
            )
            resumed = await resume_workflow(
                reconstructed,
                suspended.run_id,
                interrupt_id=suspended.checkpoint.pending_interrupt.interrupt_id,
                resume_value={"approved": True},
            )

            self.assertEqual(resumed.status, "completed")
            self.assertEqual(resumed.state["draft"], "draft")
            self.assertEqual(resumed.state["published"], "published")

    async def test_step_retry_uses_stable_identity_across_attempts(self) -> None:
        calls = 0

        async def fail_once(_request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ProviderHTTPError("retry", 503)

        step = WorkflowStep(
            "retry",
            Agent(
                name="retry-agent",
                model=create_mock_language_model(
                    responses=[GenerateResult(text="ok", finish_reason="stop")]
                ),
                input_guardrails=[fail_once],
            ),
            output_key="answer",
            retry_policy=WorkflowRetryPolicy(max_attempts=2, backoff_ms=0, max_backoff_ms=0),
        )
        graph = WorkflowBuilder("retry").add_step(step, entrypoint=True).build()

        result = await graph.run()
        starts = [
            item
            for item in await graph.checkpoint_store.list_checkpoints(result.run_id)
            if item.transition.type == "workflow-step-start"
        ]

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.step_results[0].attempts, 2)
        self.assertEqual(calls, 2)
        self.assertEqual(starts[0].nodes["retry"].idempotency_key, starts[1].nodes["retry"].idempotency_key)

    async def test_whole_step_retry_keeps_tool_idempotency_context_stable(self) -> None:
        contexts: list[ToolExecutionContext] = []
        output_checks = 0

        async def side_effect(_input, context: ToolExecutionContext) -> str:
            contexts.append(context)
            return "done"

        async def fail_first_output(_request) -> GuardrailResult:
            nonlocal output_checks
            output_checks += 1
            if output_checks == 1:
                raise RuntimeError("retry complete step")
            return GuardrailResult()

        tool_call = GenerateResult(
            messages=[
                ModelMessage(
                    role="assistant",
                    parts=[ToolCallPart(tool_call=ToolCall(id="call-1", name="effect", input={}))],
                )
            ],
            finish_reason="tool-calls",
        )
        final = GenerateResult(text="complete", finish_reason="stop")
        step = WorkflowStep(
            "effect-step",
            Agent(
                name="effect-agent",
                model=create_mock_language_model(responses=[tool_call, final, tool_call, final]),
                tools={"effect": tool(name="effect", schema=dict, execute=side_effect)},
                output_guardrails=[fail_first_output],
            ),
            retry_policy=WorkflowRetryPolicy(
                max_attempts=2,
                backoff_ms=0,
                max_backoff_ms=0,
                retry_if=lambda _error: True,
            ),
        )
        graph = WorkflowBuilder("tool-retry").add_step(step, entrypoint=True).build()

        result = await graph.run()

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(contexts), 2)
        self.assertEqual(contexts[0].idempotency_key, contexts[1].idempotency_key)
        self.assertNotEqual(contexts[0].run_id, contexts[1].run_id)

    async def test_definition_mismatch_fails_closed_on_resume(self) -> None:
        store = create_in_memory_workflow_checkpoint_store()
        original = (
            WorkflowBuilder("versioned", definition_version="1")
            .add_step(WorkflowStep("one", agent_with_text("one", "one")), entrypoint=True)
            .interrupt_before("one")
            .build(checkpoint_store=store)
        )
        suspended = await original.run()
        changed = (
            WorkflowBuilder("versioned", definition_version="2")
            .add_step(WorkflowStep("one", agent_with_text("one", "one")), entrypoint=True)
            .interrupt_before("one")
            .build(checkpoint_store=store)
        )

        with self.assertRaisesRegex(ValidationError, "version changed"):
            await resume_workflow(
                changed,
                suspended.run_id,
                interrupt_id=suspended.checkpoint.pending_interrupt.interrupt_id,
            )

    async def test_fork_replays_from_selected_checkpoint_with_lineage(self) -> None:
        store = create_in_memory_workflow_checkpoint_store()
        graph = (
            WorkflowBuilder("forkable")
            .add_step(WorkflowStep("one", agent_with_text("one", "one"), output_key="one"), entrypoint=True)
            .add_step(WorkflowStep("two", agent_with_text("two", "two"), output_key="two"))
            .add_edge("one", "two")
            .interrupt_before("two")
            .build(checkpoint_store=store)
        )
        source = await graph.run()
        checkpoint_id = source.checkpoint.checkpoint_id

        fork_graph = (
            WorkflowBuilder("forkable")
            .add_step(WorkflowStep("one", agent_with_text("one", "unused"), output_key="one"), entrypoint=True)
            .add_step(WorkflowStep("two", agent_with_text("two", "forked"), output_key="two"))
            .add_edge("one", "two")
            .interrupt_before("two")
            .build(checkpoint_store=store)
        )
        forked = await fork_workflow(
            fork_graph,
            source.run_id,
            checkpoint_id=checkpoint_id,
            state_updates={"branch": "audit"},
        )

        self.assertNotEqual(forked.run_id, source.run_id)
        self.assertEqual(forked.forked_from_run_id, source.run_id)
        self.assertEqual(forked.state["one"], "one")
        self.assertEqual(forked.state["branch"], "audit")
        self.assertEqual(forked.status, "suspended")

    async def test_callback_adapter_receives_versioned_envelope_and_can_resume(self) -> None:
        seen_keys: list[str] = []

        async def callback(request):
            seen_keys.append(request.step_idempotency_key)
            if "step:external" not in request.metadata["workflow_resume_values"]:
                return WorkflowStepOutcome.for_request(
                    request,
                    status="suspended",
                    suspension={"reason": "external signal"},
                )
            return WorkflowStepOutcome.for_request(
                request,
                status="completed",
                output="external-result",
                state_patch={"external_state": "done"},
            )

        graph = WorkflowGraph(
            name="external",
            definition_version="1",
            steps=[
                WorkflowStep(
                    "external",
                    agent_with_text("placeholder", "unused"),
                    output_key="external",
                    executor_ref="tasks.external",
                )
            ],
            edges=[],
            adapter=CallbackWorkflowAdapter(backend="custom", callback=callback),
        )

        suspended = await graph.run()
        resumed = await resume_workflow(
            graph,
            suspended.run_id,
            node_name="external",
            resume_value={"signal": "continue"},
        )

        self.assertEqual(suspended.status, "suspended")
        self.assertEqual(resumed.status, "completed")
        self.assertEqual(resumed.state["external"], "external-result")
        self.assertEqual(resumed.state["external_state"], "done")
        self.assertEqual(seen_keys[0], seen_keys[1])

    async def test_idempotent_reentry_recovers_node_left_running_by_worker_loss(self) -> None:
        calls = 0

        async def callback(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise asyncio.CancelledError()
            return WorkflowStepOutcome.for_request(request, status="completed", output="recovered")

        store = create_in_memory_workflow_checkpoint_store()
        graph = WorkflowGraph(
            name="worker-recovery",
            steps=[
                WorkflowStep(
                    "external",
                    agent_with_text("placeholder", "unused"),
                    executor_ref="tasks.external",
                    output_key="result",
                )
            ],
            edges=[],
            checkpoint_store=store,
            adapter=CallbackWorkflowAdapter(backend="custom", callback=callback),
        )

        with self.assertRaises(asyncio.CancelledError):
            await graph.run(idempotency_key="recover-1")
        running = await store.find_by_idempotency_key("recover-1")
        self.assertEqual(running.nodes["external"].status, "running")

        with self.assertRaisesRegex(ValidationError, "still running"):
            await graph.run(idempotency_key="recover-1")

        recovered = await graph.run(idempotency_key="recover-1", recover_running=True)

        self.assertEqual(recovered.status, "completed")
        self.assertEqual(recovered.state["result"], "recovered")
        self.assertEqual(recovered.checkpoint.nodes["external"].attempt, 2)
        self.assertTrue(
            any(item.transition.type == "workflow-recovered" for item in await store.list_checkpoints(recovered.run_id))
        )

    def test_graph_rejects_cycles_and_unreachable_explicit_entrypoints(self) -> None:
        one = WorkflowStep("one", agent_with_text("one", "one"))
        two = WorkflowStep("two", agent_with_text("two", "two"))
        with self.assertRaisesRegex(ValidationError, "acyclic"):
            WorkflowGraph(
                name="cycle",
                steps=[one, two],
                edges=[
                    WorkflowEdge("one", "two"),
                    WorkflowEdge("two", "one"),
                ],
            )
        with self.assertRaisesRegex(ValidationError, "cannot be reached"):
            WorkflowGraph(name="unreachable", steps=[one, two], edges=[], entrypoints=["one"])

    def test_retry_policy_and_concurrency_require_strict_integers(self) -> None:
        with self.assertRaisesRegex(ValidationError, "max_attempts"):
            WorkflowRetryPolicy(max_attempts=True)
        with self.assertRaisesRegex(ValidationError, "backoff_ms"):
            WorkflowRetryPolicy(backoff_ms=1.5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValidationError, "positive integer"):
            WorkflowGraph(name="invalid", steps=[WorkflowStep("one", agent_with_text("one", "one"))], edges=[], max_concurrency=True)
