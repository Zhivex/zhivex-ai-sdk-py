from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from zhivex_ai import (
    Agent,
    ApprovalDecision,
    GuardrailResult,
    ToolExecutionContext,
    WorkflowStep,
    create_in_memory_agent_run_store,
    create_mock_language_model,
    tool,
)
from zhivex_ai.errors import (
    ProviderHTTPError,
    ValidationError,
    WorkflowConflictError,
    WorkflowDefinitionMismatchError,
    WorkflowInterruptError,
    WorkflowLeaseLostError,
    WorkflowRunNotFoundError,
)
from zhivex_ai.types import GenerateResult, ModelMessage, ToolCall, ToolCallPart
from zhivex_ai.workflow import WorkflowFunctionResult, WorkflowRetryPolicy
from zhivex_ai.workflow_adapters import CallbackWorkflowAdapter, WorkflowStepOutcome
from zhivex_ai.workflow_graph import (
    WorkflowBuilder,
    WorkflowEdge,
    WorkflowGraph,
    cancel_workflow,
    fork_workflow,
    resume_workflow,
)
from zhivex_ai.workflow_state import (
    create_in_memory_workflow_checkpoint_store,
    create_in_memory_workflow_lease_manager,
    create_sqlite_workflow_checkpoint_store,
)


def agent_with_text(name: str, *texts: str) -> Agent:
    return Agent(
        name=name,
        model=create_mock_language_model(
            responses=[GenerateResult(text=text, finish_reason="stop") for text in texts]
        ),
    )


class _RecordingSpan:
    def __init__(self, attributes):
        self.attributes = dict(attributes or {})
        self.error = None

    def end(self, *, attributes=None, error=None):
        self.attributes.update(attributes or {})
        self.error = error


class _RecordingObserver:
    def __init__(self):
        self.spans = []

    def start_span(self, name, attributes=None):
        span = _RecordingSpan(attributes)
        self.spans.append((name, span))
        return span


class WorkflowGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_caller_metadata_cannot_pre_resolve_internal_interrupts(self) -> None:
        graph = (
            WorkflowBuilder("protected-interrupt")
            .add_step(WorkflowStep("review", agent_with_text("reviewer", "approved")), entrypoint=True)
            .interrupt_before("review")
            .build()
        )

        result = await graph.run(
            metadata={
                "resolved_interrupts": ["before:review"],
                "custom": "preserved",
            }
        )

        self.assertEqual(result.status, "suspended")
        self.assertIsNotNone(result.checkpoint.pending_interrupt)
        self.assertEqual(result.checkpoint.metadata["resolved_interrupts"], [])
        self.assertEqual(result.checkpoint.metadata["custom"], "preserved")

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

    async def test_approval_resume_preserves_step_tool_identity_and_metadata_key(self) -> None:
        contexts: list[ToolExecutionContext] = []

        async def require_human(_request):
            return ApprovalDecision.require_human("Review required.", approval_id="approval-1")

        async def side_effect(_input, context: ToolExecutionContext) -> str:
            contexts.append(context)
            return "done"

        tool_call = GenerateResult(
            messages=[
                ModelMessage(
                    role="assistant",
                    parts=[ToolCallPart(tool_call=ToolCall(id="call-approval", name="effect", input={}))],
                )
            ],
            finish_reason="tool-calls",
        )
        graph = (
            WorkflowBuilder("approval-resume")
            .add_step(
                WorkflowStep(
                    "effect-step",
                    Agent(
                        name="effect-agent",
                        model=create_mock_language_model(
                            responses=[tool_call, GenerateResult(text="complete", finish_reason="stop")]
                        ),
                        tools={
                            "effect": tool(
                                name="effect",
                                schema=dict,
                                execute=side_effect,
                                requires_approval=True,
                            )
                        },
                        approval_policy=require_human,
                        run_store=create_in_memory_agent_run_store(),
                    ),
                    output_key="result",
                    metadata_key="result_meta",
                ),
                entrypoint=True,
            )
            .build()
        )

        suspended = await graph.run()
        step_key = suspended.checkpoint.nodes["effect-step"].idempotency_key
        resumed = await resume_workflow(
            graph,
            suspended.run_id,
            approval_id="approval-1",
            approved=True,
        )

        self.assertEqual(resumed.status, "completed")
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].idempotency_key, f"{step_key}:call-approval")
        self.assertEqual(resumed.state["result"], "complete")
        self.assertEqual(
            resumed.state["result_meta"],
            {
                "name": "effect-step",
                "status": "completed",
                "run_id": resumed.checkpoint.nodes["effect-step"].child_run_id,
                "agent_name": "effect-agent",
                "text": "complete",
                "attempts": 1,
                "error": None,
            },
        )

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

    def test_default_definition_revision_preserves_legacy_digest(self) -> None:
        graph = (
            WorkflowBuilder("identity-fixture", definition_version="1")
            .add_step(
                WorkflowStep(
                    "one",
                    agent_with_text("worker", "one"),
                    output_key="out",
                ),
                entrypoint=True,
            )
            .build()
        )

        self.assertEqual(
            graph.definition_digest,
            "68c0597f2c80311bbe51dc08bf2270c419634582261d89718c9bd2cd3b223e30",
        )

    def test_step_definition_revision_captures_agent_configuration(self) -> None:
        def graph_for(instructions: str, revision: str | None = None) -> WorkflowGraph:
            agent = agent_with_text("worker", "done")
            agent.instructions = instructions
            return (
                WorkflowBuilder("agent-identity")
                .add_step(
                    WorkflowStep(
                        "work",
                        agent,
                        definition_revision=revision,
                    ),
                    entrypoint=True,
                )
                .build()
            )

        self.assertEqual(
            graph_for("Use policy A.").definition_digest,
            graph_for("Use policy B.").definition_digest,
        )
        self.assertNotEqual(
            graph_for("Use policy A.", "instructions:v1").definition_digest,
            graph_for("Use policy B.", "instructions:v2").definition_digest,
        )

    def test_step_definition_revision_captures_closure_configuration(self) -> None:
        def configured_executor(multiplier: int):
            async def calculate(context):
                return int(context.input or 0) * multiplier

            return calculate

        unversioned_two = (
            WorkflowBuilder("closure-identity")
            .add_step(
                WorkflowStep("calculate", executor=configured_executor(2)),
                entrypoint=True,
            )
            .build()
        )
        unversioned_three = (
            WorkflowBuilder("closure-identity")
            .add_step(
                WorkflowStep("calculate", executor=configured_executor(3)),
                entrypoint=True,
            )
            .build()
        )
        versioned_two = (
            WorkflowBuilder("closure-identity")
            .add_step(
                WorkflowStep(
                    "calculate",
                    executor=configured_executor(2),
                    definition_revision="multiplier:2",
                ),
                entrypoint=True,
            )
            .build()
        )
        versioned_three = (
            WorkflowBuilder("closure-identity")
            .add_step(
                WorkflowStep(
                    "calculate",
                    executor=configured_executor(3),
                    definition_revision="multiplier:3",
                ),
                entrypoint=True,
            )
            .build()
        )

        self.assertEqual(unversioned_two.definition_digest, unversioned_three.definition_digest)
        self.assertNotEqual(versioned_two.definition_digest, versioned_three.definition_digest)

    def test_edge_definition_revision_captures_condition_configuration(self) -> None:
        def configured_condition(threshold: int):
            return lambda context: int(context.source_output or 0) >= threshold

        def graph_for(threshold: int) -> WorkflowGraph:
            return (
                WorkflowBuilder("edge-identity")
                .add_step(
                    WorkflowStep("source", agent_with_text("source", "10")),
                    entrypoint=True,
                )
                .add_step(WorkflowStep("target", agent_with_text("target", "done")))
                .add_edge(
                    "source",
                    "target",
                    condition=configured_condition(threshold),
                    definition_revision=f"threshold:{threshold}",
                )
                .build()
            )

        self.assertNotEqual(graph_for(5).definition_digest, graph_for(20).definition_digest)

    async def test_step_definition_revision_change_fails_closed_on_resume(self) -> None:
        store = create_in_memory_workflow_checkpoint_store()
        original = (
            WorkflowBuilder("revision-resume", definition_version="1")
            .add_step(
                WorkflowStep(
                    "one",
                    agent_with_text("one", "one"),
                    definition_revision="agent-config:v1",
                ),
                entrypoint=True,
            )
            .interrupt_before("one")
            .build(checkpoint_store=store)
        )
        suspended = await original.run()
        changed = (
            WorkflowBuilder("revision-resume", definition_version="1")
            .add_step(
                WorkflowStep(
                    "one",
                    agent_with_text("one", "one"),
                    definition_revision="agent-config:v2",
                ),
                entrypoint=True,
            )
            .interrupt_before("one")
            .build(checkpoint_store=store)
        )

        with self.assertRaisesRegex(WorkflowDefinitionMismatchError, "definition digest changed"):
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
        self.assertEqual(
            suspended.checkpoint.nodes["external"].suspension,
            {"reason": "external signal"},
        )
        self.assertEqual(resumed.status, "completed")
        self.assertIsNone(resumed.checkpoint.nodes["external"].suspension)
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

    async def test_concurrent_idempotent_start_reuses_the_winning_run(self) -> None:
        inner_store = create_in_memory_workflow_checkpoint_store()

        class RacingStore:
            def __init__(self) -> None:
                self.find_calls = 0
                self.initial_finds_ready = asyncio.Event()

            async def append(self, checkpoint, *, expected_sequence=None):
                return await inner_store.append(checkpoint, expected_sequence=expected_sequence)

            async def load_latest(self, run_id):
                return await inner_store.load_latest(run_id)

            async def load_checkpoint(self, checkpoint_id):
                return await inner_store.load_checkpoint(checkpoint_id)

            async def find_by_idempotency_key(self, idempotency_key):
                self.find_calls += 1
                if self.find_calls <= 2:
                    result = await inner_store.find_by_idempotency_key(idempotency_key)
                    if self.find_calls == 2:
                        self.initial_finds_ready.set()
                    await self.initial_finds_ready.wait()
                    return result
                return await inner_store.find_by_idempotency_key(idempotency_key)

            async def list_checkpoints(self, run_id):
                return await inner_store.list_checkpoints(run_id)

        started = asyncio.Event()
        release = asyncio.Event()

        async def hold_winner(_context):
            started.set()
            await release.wait()
            return "done"

        graph = (
            WorkflowBuilder("concurrent-start")
            .add_step(WorkflowStep("work", executor=hold_winner, output_key="result"), entrypoint=True)
            .build(checkpoint_store=RacingStore())
        )
        runs = [
            asyncio.create_task(graph.run(idempotency_key="same-request")),
            asyncio.create_task(graph.run(idempotency_key="same-request")),
        ]
        await started.wait()
        done, pending = await asyncio.wait(runs, return_when=asyncio.FIRST_COMPLETED)
        reused = next(iter(done)).result()

        self.assertEqual(reused.status, "running")
        release.set()
        winner = await next(iter(pending))

        self.assertEqual(winner.status, "completed")
        self.assertEqual(reused.run_id, winner.run_id)
        self.assertEqual(winner.state["result"], "done")

    async def test_concurrent_idempotent_fork_reuses_the_winning_run(self) -> None:
        inner_store = create_in_memory_workflow_checkpoint_store()

        async def complete(_context):
            return "done"

        step = WorkflowStep("work", executor=complete, output_key="result")
        source_graph = WorkflowBuilder("concurrent-fork").add_step(step, entrypoint=True).build(
            checkpoint_store=inner_store
        )
        source = await source_graph.run()

        class RacingStore:
            def __init__(self) -> None:
                self.find_calls = 0
                self.initial_finds_ready = asyncio.Event()

            async def append(self, checkpoint, *, expected_sequence=None):
                return await inner_store.append(checkpoint, expected_sequence=expected_sequence)

            async def load_latest(self, run_id):
                return await inner_store.load_latest(run_id)

            async def load_checkpoint(self, checkpoint_id):
                return await inner_store.load_checkpoint(checkpoint_id)

            async def find_by_idempotency_key(self, idempotency_key):
                self.find_calls += 1
                if self.find_calls <= 2:
                    result = await inner_store.find_by_idempotency_key(idempotency_key)
                    if self.find_calls == 2:
                        self.initial_finds_ready.set()
                    await self.initial_finds_ready.wait()
                    return result
                return await inner_store.find_by_idempotency_key(idempotency_key)

            async def list_checkpoints(self, run_id):
                return await inner_store.list_checkpoints(run_id)

        graph = WorkflowBuilder("concurrent-fork").add_step(step, entrypoint=True).build(
            checkpoint_store=RacingStore()
        )
        forks = await asyncio.gather(
            fork_workflow(graph, source.run_id, idempotency_key="same-fork"),
            fork_workflow(graph, source.run_id, idempotency_key="same-fork"),
        )

        self.assertEqual(forks[0].run_id, forks[1].run_id)
        self.assertEqual(
            (await inner_store.find_by_idempotency_key("same-fork")).run_id,
            forks[0].run_id,
        )

    async def test_active_lease_blocks_recovery_and_heartbeat_keeps_it_alive(self) -> None:
        lease_manager = create_in_memory_workflow_lease_manager()
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_step(_context):
            started.set()
            await release.wait()
            return "done"

        graph = (
            WorkflowBuilder("leased-heartbeat")
            .add_step(WorkflowStep("work", executor=slow_step, output_key="result"), entrypoint=True)
            .build(
                lease_manager=lease_manager,
                lease_ttl_ms=60,
                lease_heartbeat_ms=10,
            )
        )
        worker = asyncio.create_task(graph.run(idempotency_key="leased-heartbeat"))
        await asyncio.wait_for(started.wait(), timeout=1)
        await asyncio.sleep(0.08)

        with self.assertRaisesRegex(WorkflowConflictError, "active execution lease"):
            await graph.run(idempotency_key="leased-heartbeat", recover_running=True)

        release.set()
        result = await worker
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.state["result"], "done")
        self.assertEqual(result.checkpoint.metadata["execution_lease"]["fencing_token"], 1)
        self.assertIsNone(await lease_manager.get(result.run_id))

    async def test_expired_lease_allows_recovery_and_increments_fence(self) -> None:
        calls = 0

        async def callback(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise asyncio.CancelledError()
            return WorkflowStepOutcome.for_request(request, status="completed", output="recovered")

        store = create_in_memory_workflow_checkpoint_store()
        step = WorkflowStep(
            "external",
            agent_with_text("placeholder", "unused"),
            executor_ref="tasks.external",
            output_key="result",
        )
        unleased = WorkflowGraph(
            name="expired-recovery",
            steps=[step],
            edges=[],
            checkpoint_store=store,
            adapter=CallbackWorkflowAdapter(backend="custom", callback=callback),
        )
        with self.assertRaises(asyncio.CancelledError):
            await unleased.run(idempotency_key="expired-recovery")
        running = await store.find_by_idempotency_key("expired-recovery")
        assert running is not None

        lease_manager = create_in_memory_workflow_lease_manager()
        expired = await lease_manager.acquire(
            running.run_id,
            owner_id="dead-worker",
            ttl_ms=10,
            now_ms=0,
        )
        assert expired is not None
        leased = WorkflowGraph(
            name="expired-recovery",
            steps=[step],
            edges=[],
            checkpoint_store=store,
            adapter=CallbackWorkflowAdapter(backend="custom", callback=callback),
            lease_manager=lease_manager,
        )

        result = await leased.run(idempotency_key="expired-recovery", recover_running=True)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.checkpoint.nodes["external"].attempt, 2)
        self.assertEqual(calls, 2)

    async def test_resume_requires_lease_ownership(self) -> None:
        lease_manager = create_in_memory_workflow_lease_manager()
        graph = (
            WorkflowBuilder("leased-resume")
            .add_step(WorkflowStep("review", agent_with_text("reviewer", "approved")), entrypoint=True)
            .interrupt_before("review")
            .build(lease_manager=lease_manager)
        )
        suspended = await graph.run()
        pending = suspended.checkpoint.pending_interrupt
        assert pending is not None
        other = await lease_manager.acquire(
            suspended.run_id,
            owner_id="other-worker",
            ttl_ms=30_000,
        )
        assert other is not None

        with self.assertRaisesRegex(WorkflowConflictError, "active execution lease"):
            await resume_workflow(
                graph,
                suspended.run_id,
                interrupt_id=pending.interrupt_id,
            )

        self.assertTrue(await lease_manager.release(suspended.run_id, token=other.token))
        resumed = await resume_workflow(
            graph,
            suspended.run_id,
            interrupt_id=pending.interrupt_id,
        )
        self.assertEqual(resumed.status, "completed")

    async def test_stale_lease_owner_cannot_persist_step_result(self) -> None:
        lease_manager = create_in_memory_workflow_lease_manager()
        replacement_tokens: list[str] = []

        async def steal_lease(context):
            current = await lease_manager.get(context.run_id)
            assert current is not None
            self.assertTrue(await lease_manager.release(context.run_id, token=current.token))
            replacement = await lease_manager.acquire(
                context.run_id,
                owner_id="replacement-worker",
                ttl_ms=30_000,
            )
            assert replacement is not None
            replacement_tokens.append(replacement.token)
            return "must-not-commit"

        graph = (
            WorkflowBuilder("lease-fencing")
            .add_step(WorkflowStep("work", executor=steal_lease, output_key="result"), entrypoint=True)
            .build(lease_manager=lease_manager)
        )

        with self.assertRaisesRegex(WorkflowLeaseLostError, "lost"):
            await graph.run(idempotency_key="lease-fencing")

        running = await graph.checkpoint_store.find_by_idempotency_key("lease-fencing")
        assert running is not None
        self.assertEqual(running.nodes["work"].status, "running")
        self.assertNotIn("result", running.state)
        self.assertTrue(await lease_manager.release(running.run_id, token=replacement_tokens[0]))

    async def test_resume_surfaces_typed_not_found_and_interrupt_errors(self) -> None:
        graph = (
            WorkflowBuilder("typed-resume-errors")
            .add_step(WorkflowStep("review", agent_with_text("reviewer", "done")), entrypoint=True)
            .interrupt_before("review")
            .build()
        )
        with self.assertRaises(WorkflowRunNotFoundError):
            await resume_workflow(graph, "missing-run")

        suspended = await graph.run()
        with self.assertRaises(WorkflowInterruptError):
            await resume_workflow(graph, suspended.run_id, interrupt_id="wrong-interrupt")

    async def test_cancel_workflow_stops_running_graph_cooperatively(self) -> None:
        lease_manager = create_in_memory_workflow_lease_manager()
        started = asyncio.Event()
        release = asyncio.Event()

        async def side_effect(_context):
            started.set()
            await release.wait()
            return "late"

        graph = (
            WorkflowBuilder("cancellable")
            .add_step(WorkflowStep("work", executor=side_effect, output_key="result"), entrypoint=True)
            .build(lease_manager=lease_manager)
        )
        worker = asyncio.create_task(graph.run(idempotency_key="cancellable"))
        await asyncio.wait_for(started.wait(), timeout=1)
        running = await graph.checkpoint_store.find_by_idempotency_key("cancellable")
        assert running is not None

        cancelled = await cancel_workflow(graph, running.run_id, reason="operator-stop")
        release.set()
        worker_result = await worker

        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(worker_result.status, "cancelled")
        self.assertEqual(cancelled.checkpoint.nodes["work"].status, "cancelled")
        self.assertNotIn("result", cancelled.state)
        self.assertEqual(cancelled.state_snapshot.cancellation_reason, "operator-stop")
        self.assertIsNone(await lease_manager.get(running.run_id))

    async def test_adapter_cancelled_outcome_cancels_workflow(self) -> None:
        async def callback(request):
            return WorkflowStepOutcome.for_request(request, status="cancelled")

        graph = WorkflowGraph(
            name="adapter-cancel",
            steps=[
                WorkflowStep(
                    "external",
                    agent_with_text("placeholder", "unused"),
                    executor_ref="tasks.external",
                )
            ],
            edges=[],
            adapter=CallbackWorkflowAdapter(backend="custom", callback=callback),
        )

        result = await graph.run()

        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.checkpoint.nodes["external"].status, "cancelled")

    async def test_observer_correlates_workflow_step_and_child_agent_spans(self) -> None:
        observer = _RecordingObserver()
        graph = (
            WorkflowBuilder("observed")
            .add_step(WorkflowStep("work", agent_with_text("worker", "done")), entrypoint=True)
            .build(observer=observer)
        )

        result = await graph.run()
        spans = {name: span for name, span in observer.spans}

        self.assertEqual(spans["zhivex.workflow.run"].attributes["zhivex.workflow.run_id"], result.run_id)
        self.assertEqual(spans["zhivex.workflow.run"].attributes["zhivex.workflow.status"], "completed")
        self.assertEqual(spans["zhivex.workflow.step"].attributes["zhivex.workflow.step_name"], "work")
        self.assertEqual(spans["zhivex.workflow.step"].attributes["zhivex.workflow.step_status"], "completed")
        self.assertIn("zhivex.agent.run", spans)
        self.assertIn("zhivex.agent.model", spans)

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

    def test_definition_revisions_must_be_non_empty_strings(self) -> None:
        with self.assertRaisesRegex(ValidationError, "definition_revision"):
            WorkflowGraph(
                name="invalid-step-revision",
                steps=[
                    WorkflowStep(
                        "one",
                        agent_with_text("one", "one"),
                        definition_revision=" ",
                    )
                ],
                edges=[],
            )
        with self.assertRaisesRegex(ValidationError, "definition_revision"):
            WorkflowGraph(
                name="invalid-edge-revision",
                steps=[
                    WorkflowStep("one", agent_with_text("one", "one")),
                    WorkflowStep("two", agent_with_text("two", "two")),
                ],
                edges=[WorkflowEdge("one", "two", definition_revision="")],
            )
