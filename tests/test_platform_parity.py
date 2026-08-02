from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from zhivex_ai import (
    Agent,
    AgentEvaluationCase,
    AgentEvaluationExpectations,
    AgentGroupMember,
    AgentRunState,
    apply_safety_policy_to_agent,
    cancel_agent_run_tree,
    create_agent_evaluation_report,
    create_agent_run_snapshot,
    create_agent_trace_artifact,
    create_budget_guard,
    create_in_memory_agent_run_store,
    create_mock_language_model,
    create_mock_tool,
    create_redaction_policy,
    create_safety_policy,
    create_sqlite_agent_run_store,
    create_subagent_tool,
    judge_agent_evaluation,
    replay_agent_run,
    run_agent,
    run_agent_evaluation,
    run_agent_group,
    summarize_agent_trace,
)
from zhivex_ai.messages import create_text_message, tool_call_part
from zhivex_ai.types import GenerateResult, ModelMessage, TokenUsage, ToolCall


class PlatformParityTests(unittest.IsolatedAsyncioTestCase):
    async def test_in_memory_run_store_and_cancel_tree(self) -> None:
        store = create_in_memory_agent_run_store()
        parent = AgentRunState(run_id="parent", agent_name="parent", provider="mock", model_id="m")
        child = AgentRunState(
            run_id="child",
            agent_name="child",
            provider="mock",
            model_id="m",
            parent_run_id="parent",
        )
        await store.save(parent)
        await store.save(child)

        result = await cancel_agent_run_tree(store, "parent", reason="stop")

        self.assertEqual(result.root.status if result.root else None, "cancelled")
        self.assertEqual([state.run_id for state in result.cancelled], ["parent", "child"])
        self.assertEqual((await store.load("child")).cancellation_reason, "stop")  # type: ignore[union-attr]

    async def test_sqlite_run_store_idempotency_and_parent_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = create_sqlite_agent_run_store(str(Path(directory) / "runs.sqlite"))
            await store.save(
                AgentRunState(
                    run_id="run_1",
                    agent_name="agent",
                    provider="mock",
                    model_id="m",
                    parent_run_id="parent",
                    idempotency_key="idem",
                )
            )

            self.assertEqual((await store.find_by_idempotency_key("idem")).run_id, "run_1")  # type: ignore[union-attr]
            self.assertEqual([state.run_id for state in await store.find_by_parent_run_id("parent")], ["run_1"])

    async def test_run_agent_persists_state_and_reuses_idempotency_key(self) -> None:
        store = create_in_memory_agent_run_store()
        agent = Agent(name="assistant", model=create_mock_language_model(), run_store=store)

        first = await run_agent(agent=agent, prompt="hello", idempotency_key="idem")
        second = await run_agent(agent=agent, prompt="ignored", idempotency_key="idem")

        self.assertEqual(first.run_id, second.run_id)
        self.assertEqual((await store.load(first.run_id)).status, "completed")  # type: ignore[union-attr]

    async def test_subagent_tool_records_child_run_shape(self) -> None:
        child = Agent(name="researcher", model=create_mock_language_model())
        tool = create_subagent_tool(name="researcher", agent=child, parent_run_id="parent_run")

        output = await tool.execute({"prompt": "research"})

        self.assertEqual(output["text"], "ok")
        self.assertEqual(output["child_run"]["agent_name"], "researcher")
        self.assertEqual(output["child_run"]["parent_run_id"], "parent_run")

    async def test_run_agent_group_preserves_order_and_failures(self) -> None:
        ok = Agent(name="ok", model=create_mock_language_model())
        failing = Agent(name="bad", model=create_mock_language_model(responses=[]))

        result = await run_agent_group(
            [AgentGroupMember("one", ok), AgentGroupMember("two", failing)],
            prompt="go",
            parent_run_id="parent",
        )

        self.assertEqual(result.parent_run_id, "parent")
        self.assertEqual([item.name for item in result.outputs], ["one", "two"])
        self.assertIsNotNone(result.outputs[0].output)
        self.assertIsNotNone(result.outputs[1].error)

    async def test_replay_evaluation_and_report(self) -> None:
        state = AgentRunState(
            run_id="run",
            agent_name="assistant",
            provider="mock",
            model_id="m",
            status="completed",
            current_step=1,
            output_text="hello",
        )
        self.assertEqual(create_agent_run_snapshot(state).output_text, "hello")
        self.assertEqual(replay_agent_run(state).timeline[0].type, "run-start")

        agent = Agent(name="assistant", model=create_mock_language_model())
        result = await run_agent_evaluation(
            agent=agent,
            dataset=[
                AgentEvaluationCase(
                    name="contains",
                    prompt="hello",
                    expectations=AgentEvaluationExpectations(output_contains="ok"),
                )
            ],
        )
        report = create_agent_evaluation_report(result)
        judge = await judge_agent_evaluation(result)

        self.assertTrue(result.ok)
        self.assertEqual(report.pass_rate, 1.0)
        self.assertEqual(judge.score, 1.0)

    async def test_trace_and_cost_summary(self) -> None:
        state = AgentRunState(
            run_id="run",
            agent_name="assistant",
            provider="mock",
            model_id="m",
            status="completed",
            current_step=1,
            output_text="hello",
            usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            started_at_ms=1_000,
            finished_at_ms=1_250,
        )

        artifact = create_agent_trace_artifact(state)
        summary = summarize_agent_trace(state)

        self.assertEqual(artifact.output_preview, "hello")
        self.assertEqual(artifact.duration_ms, 250)
        self.assertEqual(summary.tool_calls, 0)
        self.assertEqual(summary.duration_ms, 250)

    async def test_safety_redaction_budget_and_agent_application(self) -> None:
        redaction = create_redaction_policy(include_emails=True)
        budget = create_budget_guard(max_total_tokens=10)
        policy = create_safety_policy(redaction=redaction, budget=budget)
        agent = apply_safety_policy_to_agent(Agent(name="assistant", model=create_mock_language_model()), policy)

        self.assertIn("[REDACTED]", redaction.redact_text("token=abcdefghij user@example.com"))
        self.assertTrue(budget.evaluate_state(AgentRunState(run_id="r", agent_name="a", provider="p", model_id="m", usage=TokenUsage(total_tokens=11))).tripwire_triggered)
        self.assertEqual(agent.metadata["safety_policy"], "review_sensitive")

    async def test_mock_tool_and_tool_call_response(self) -> None:
        call = ToolCall(id="call_1", name="lookup", input={})
        model = create_mock_language_model(
            responses=[
                GenerateResult(
                    messages=[ModelMessage(role="assistant", parts=[tool_call_part(call)])],
                    finish_reason="tool-calls",
                ),
                GenerateResult(text="done", message=create_text_message("assistant", "done"), finish_reason="stop"),
            ]
        )
        agent = Agent(name="assistant", model=model, tools={"lookup": create_mock_tool("lookup", outputs=["value"])})

        result = await run_agent(agent=agent, prompt="use tool")

        self.assertEqual(result.text, "done")
        self.assertEqual(result.tool_results[0].tool_name, "lookup")
