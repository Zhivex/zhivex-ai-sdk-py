from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase

from zhivex_ai import (
    Agent, AgentRuntime, RunLimits, ValidationError, create_in_memory_agent_run_store,
    run_agent, stream_agent, stream_live_agent,
)
from zhivex_ai.types import ModelGenerateInput, GenerateResult, RealtimeResponseCompletedEvent, RealtimeTextDeltaEvent
from tests.test_agent import EchoAgentModel
from tests.test_realtime import FakeLiveModel, FakeLiveSession


class AgentExecutionRetentionTests(IsolatedAsyncioTestCase):
    async def test_bounded_trace_exports_every_event_and_retains_tail(self) -> None:
        model = EchoAgentModel()
        observed = []

        async def emit(event):
            observed.append(event)

        result = await AgentRuntime().run(
            agent=Agent(name="bounded", model=model, trace_event_limit=2),
            prompt="hello", emit=emit,
        )
        self.assertIsInstance(result.trace.events, list)
        self.assertEqual(result.trace.events, observed[-2:])
        self.assertEqual(result.trace.events_dropped, len(observed) - 2)

        unlimited = await run_agent(agent=Agent(name="unlimited", model=model), prompt="hello")
        self.assertGreater(len(unlimited.trace.events), 2)
        self.assertEqual(unlimited.trace.events_dropped, 0)

    async def test_stream_history_inherits_trace_limit_and_collect_remains_available(self) -> None:
        stream = stream_agent(agent=Agent(name="bounded", model=EchoAgentModel(), trace_event_limit=2), prompt="hello")
        result = await stream.collect()
        self.assertEqual(len(result.trace.events), 2)
        with self.assertRaisesRegex(ValidationError, "retained event history"):
            async for _ in stream.event_stream():
                pass
        self.assertLessEqual(len(stream._broadcast.history), 2)

    async def test_live_trace_and_stream_retention(self) -> None:
        session = FakeLiveSession([
            RealtimeTextDeltaEvent(text_delta="hello"),
            RealtimeResponseCompletedEvent(reason="done"),
        ])
        result_stream = stream_live_agent(agent=Agent(name="live", model=FakeLiveModel(session), trace_event_limit=2))
        result = await result_stream.collect()
        self.assertLessEqual(len(result.trace.events), 2)
        self.assertGreater(result.trace.events_dropped, 0)
        self.assertLessEqual(len(result_stream._broadcast.history), 2)
        self.assertTrue(session.closed)

    async def test_budget_timeout_persists_failure_and_joins_model(self) -> None:
        stopped = asyncio.Event()

        class SlowModel(EchoAgentModel):
            async def generate(self, input: ModelGenerateInput) -> GenerateResult:
                try:
                    await asyncio.sleep(10)
                    return await super().generate(input)
                finally:
                    stopped.set()

        store = create_in_memory_agent_run_store()
        agent = Agent(name="slow", model=SlowModel(), run_store=store)
        with self.assertRaises(TimeoutError):
            await run_agent(agent=agent, prompt="hello", idempotency_key="deadline", total_timeout_ms=20)
        state = await store.find_by_idempotency_key("deadline")
        self.assertEqual(state.status, "failed")
        self.assertTrue(stopped.is_set())

    async def test_budget_includes_middleware_before_model(self) -> None:
        called = False

        async def middleware(request, call_next):
            nonlocal called
            called = True
            await asyncio.sleep(10)
            return await call_next(request)

        with self.assertRaises(TimeoutError):
            await run_agent(agent=Agent(name="middleware", model=EchoAgentModel(), middleware=[middleware]), prompt="hello", total_timeout_ms=20)
        self.assertTrue(called)

    async def test_live_budget_covers_startup_and_persists_failure(self) -> None:
        class SlowConnect(FakeLiveModel):
            async def connect(self, config=None, options=None):
                await asyncio.sleep(10)
                return await super().connect(config, options)

        store = create_in_memory_agent_run_store()
        stream = stream_live_agent(agent=Agent(name="slow-live", model=SlowConnect(FakeLiveSession()), run_store=store), idempotency_key="live-deadline", total_timeout_ms=20)
        with self.assertRaises(TimeoutError):
            await stream.collect()
        state = await store.find_by_idempotency_key("live-deadline")
        self.assertEqual(state.status, "failed")

    async def test_existing_wall_limit_preserves_runtime_error(self) -> None:
        class SlowModel(EchoAgentModel):
            async def generate(self, input: ModelGenerateInput) -> GenerateResult:
                await asyncio.sleep(10)
                return await super().generate(input)

        with self.assertRaisesRegex(RuntimeError, "max wall time"):
            await run_agent(agent=Agent(name="wall", model=SlowModel(), run_limits=RunLimits(max_wall_time_ms=20)), prompt="hello")

    def test_trace_limit_rejects_invalid_values(self) -> None:
        for limit in (True, 0, -1, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValidationError):
                Agent(name="invalid", model=EchoAgentModel(), trace_event_limit=limit)

    async def test_resume_budget_covers_approved_tool_and_reconciles_claim(self) -> None:
        from zhivex_ai import ApprovalDecision, resume_agent_run, tool
        from tests.test_agent import ToolLoopModel

        stopped = asyncio.Event()

        async def suspend(request):
            return ApprovalDecision.require_human(approval_id="approval")

        async def slow(input):
            try:
                await asyncio.sleep(10)
                return "late"
            finally:
                stopped.set()

        store = create_in_memory_agent_run_store()
        agent = Agent(name="approval", model=ToolLoopModel(), run_store=store, approval_policy=suspend, tools={
            "delegate": tool(name="delegate", schema=dict[str, str], execute=slow, requires_approval=True),
        })
        suspended = await run_agent(agent=agent, prompt="hello")
        self.assertEqual(suspended.state.status, "suspended")
        with self.assertRaises(TimeoutError):
            await resume_agent_run(agent=agent, run_id=suspended.run_id, total_timeout_ms=20)
        state = await store.load(suspended.run_id)
        self.assertEqual(state.status, "failed")
        self.assertIn("resume_claim_failure", state.metadata)
        self.assertTrue(stopped.is_set())
