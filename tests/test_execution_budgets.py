from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase

from zhivex_ai import GatewayConfig, GatewayMessage, GatewayModelTarget, create_gateway, generate_text, stream_text
from zhivex_ai.errors import ProviderHTTPError
from zhivex_ai.providers.base import ProviderAdapter
from zhivex_ai.runtime import current_execution_budget, with_retry
from zhivex_ai.types import GenerateResult, ModelCapabilities, ModelGenerateInput, StreamTextDeltaEvent


class SlowModel:
    provider = "openai"
    model_id = "slow"
    capabilities = ModelCapabilities(streaming=True, tools=True, structured_output=False, json_mode=False, tool_choice=False, parallel_tool_calls=False, vision=False, files=False, audio_input=False, audio_output=False, embeddings=False, reasoning=False, web_search=False)

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False
        self.inherited_budget = False

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        self.calls += 1
        self.inherited_budget = current_execution_budget() is not None
        await asyncio.sleep(0.035)
        raise ProviderHTTPError("busy", 503, retryable=False)

    async def stream(self, input: ModelGenerateInput):
        async def events():
            try:
                yield StreamTextDeltaEvent(text_delta="started")
                await asyncio.sleep(10)
            finally:
                self.closed = True
        return events()


class ExecutionBudgetTests(IsolatedAsyncioTestCase):
    async def test_fallbacks_share_one_total_deadline(self) -> None:
        first, second = SlowModel(), SlowModel()
        gateway = create_gateway(GatewayConfig(
            adapters={
                "openai": ProviderAdapter(name="openai", language_model_factory=lambda _: first),
                "anthropic": ProviderAdapter(name="anthropic", language_model_factory=lambda _: second),
            }, max_retries=0, total_timeout_ms=55, attempt_timeout_ms=1000,
        ))
        with self.assertRaises(TimeoutError):
            await gateway.generate(
                messages=[GatewayMessage(role="user", content="hello")],
                primary=GatewayModelTarget(provider="openai", model_id="slow"),
                fallbacks=[GatewayModelTarget(provider="anthropic", model_id="slow")],
            )
        self.assertEqual((first.calls, second.calls), (1, 1))
        self.assertTrue(first.inherited_budget and second.inherited_budget)

    async def test_foundation_budget_reaches_adapter_retry(self) -> None:
        class RetryingModel(SlowModel):
            async def generate(self, input: ModelGenerateInput) -> GenerateResult:
                async def invoke() -> GenerateResult:
                    self.calls += 1
                    raise ProviderHTTPError("busy", 429, response_headers={"Retry-After": "60"})
                return await with_retry(invoke, max_retries=3)
        model = RetryingModel()
        with self.assertRaises(TimeoutError):
            await generate_text(model=model, prompt="hello", total_timeout_ms=20)
        self.assertEqual(model.calls, 1)

    async def test_stream_deadline_closes_upstream(self) -> None:
        model = SlowModel()
        result = stream_text(model=model, prompt="hello", total_timeout_ms=20)
        with self.assertRaises(TimeoutError):
            await result.collect()
        self.assertTrue(model.closed)
        self.assertIsNone(current_execution_budget())

    async def test_gateway_honors_retry_after_within_total_budget(self) -> None:
        class LimitedModel(SlowModel):
            async def generate(self, input: ModelGenerateInput) -> GenerateResult:
                self.calls += 1
                raise ProviderHTTPError("limited", 429, response_headers={"Retry-After": "60"})
        model = LimitedModel()
        gateway = create_gateway(GatewayConfig(
            adapters={"openai": ProviderAdapter(name="openai", language_model_factory=lambda _: model)},
            max_retries=3, total_timeout_ms=20, retry_backoff_ms=0,
        ))
        with self.assertRaises(TimeoutError):
            await gateway.generate(
                messages=[GatewayMessage(role="user", content="hello")],
                primary=GatewayModelTarget(provider="openai", model_id="slow"),
            )
        self.assertEqual(model.calls, 1)

    async def test_foundation_total_budget_cancels_tool_work(self) -> None:
        from zhivex_ai import tool
        from zhivex_ai.types import ModelMessage, ToolCall, ToolCallPart

        tool_started = asyncio.Event()
        tool_stopped = asyncio.Event()

        async def execute(_: dict[str, str]) -> str:
            tool_started.set()
            try:
                await asyncio.sleep(10)
                return "late"
            finally:
                tool_stopped.set()

        class ToolModel(SlowModel):
            async def generate(self, input: ModelGenerateInput) -> GenerateResult:
                return GenerateResult(messages=[ModelMessage(role="assistant", parts=[
                    ToolCallPart(tool_call=ToolCall(id="call", name="slow", input={}))
                ])])

        with self.assertRaises(TimeoutError):
            await generate_text(
                model=ToolModel(), prompt="hello", max_steps=2, total_timeout_ms=20,
                tools={"slow": tool(name="slow", description="Slow tool", schema=dict[str, str], execute=execute)},
            )
        self.assertTrue(tool_started.is_set())
        self.assertTrue(tool_stopped.is_set())
