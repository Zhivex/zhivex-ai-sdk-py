from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable
from unittest import IsolatedAsyncioTestCase

from zhivex_ai import (
    Agent,
    ToolExecutionOptions,
    ToolExecutionOutcomeUnknown,
    create_in_memory_agent_run_store,
    create_text_message,
    run_agent,
    tool,
)
from zhivex_ai.types import (
    GenerateResult,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    ToolCall,
    ToolCallPart,
)


CAPABILITIES = ModelCapabilities(
    streaming=False,
    tools=True,
    structured_output=False,
    json_mode=False,
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


class TimeoutToolModel:
    provider = "test"
    model_id = "timeout-tool"
    capabilities = CAPABILITIES

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        self.calls += 1
        if not any(message.role == "tool" for message in input.messages):
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[ToolCallPart(tool_call=ToolCall(id="write-1", name="write", input={}))],
                    )
                ],
                finish_reason="tool-calls",
            )
        return GenerateResult(messages=[create_text_message("assistant", "done")], text="done")

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        del input
        if False:
            yield None


class ToolTimeoutSafetyTests(IsolatedAsyncioTestCase):
    async def test_timed_out_sync_tool_stops_agent_and_reports_unknown_outcome(self) -> None:
        model = TimeoutToolModel()
        store = create_in_memory_agent_run_store()
        side_effects: list[str] = []
        contexts: list[object] = []

        def late_write(_input: dict[str, object], context: object) -> dict[str, bool]:
            contexts.append(context)
            time.sleep(0.1)
            side_effects.append("committed")
            return {"ok": True}

        agent = Agent(
            name="writer",
            model=model,
            run_store=store,
            tools={"write": tool(name="write", schema=dict, execute=late_write)},
        )

        with self.assertRaises(ToolExecutionOutcomeUnknown) as raised:
            await run_agent(
                agent=agent,
                prompt="write",
                idempotency_key="tenant:request-1",
                tool_execution=ToolExecutionOptions(timeout_ms=20),
            )

        self.assertEqual(model.calls, 1)
        self.assertEqual(side_effects, [])
        self.assertEqual(raised.exception.tool_call_id, "write-1")
        self.assertTrue(raised.exception.outcome_unknown)
        self.assertIn("write-1", raised.exception.idempotency_key)
        self.assertEqual(len(contexts), 1)
        context = contexts[0]
        self.assertEqual(getattr(context, "idempotency_key"), raised.exception.idempotency_key)
        self.assertIsNotNone(getattr(context, "deadline_ms"))

        state = await store.find_by_idempotency_key("tenant:request-1")
        self.assertIsNotNone(state)
        self.assertEqual(state.status, "failed")  # type: ignore[union-attr]
        self.assertIn("external outcome is unknown", state.error)  # type: ignore[union-attr]

        # Python cannot terminate a running thread. The late commit demonstrates
        # why the runtime must stop instead of presenting a normal tool error to the model.
        await asyncio.sleep(0.15)
        self.assertEqual(side_effects, ["committed"])

    async def test_tool_timeout_must_be_positive(self) -> None:
        agent = Agent(name="writer", model=TimeoutToolModel(), tools={"write": tool(name="write", schema=dict, execute=lambda _input: {})})

        with self.assertRaisesRegex(Exception, "timeout_ms.*greater than zero"):
            await run_agent(
                agent=agent,
                prompt="write",
                tool_execution=ToolExecutionOptions(timeout_ms=0),
            )
