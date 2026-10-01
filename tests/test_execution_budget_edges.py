from __future__ import annotations

import asyncio
from unittest import IsolatedAsyncioTestCase

from zhivex_ai import (
    GatewayConfig,
    GatewayMessage,
    GatewayModelTarget,
    create_gateway,
    stream_text,
)
from zhivex_ai.errors import ProviderHTTPError, ValidationError
from zhivex_ai.providers.base import ProviderAdapter
from zhivex_ai.runtime import execution_scope, retry_delay_ms
from zhivex_ai.types import ModelCapabilities, StreamTextDeltaEvent


class SuspendedStreamModel:
    provider = "openai"
    model_id = "slow"
    capabilities = ModelCapabilities(
        streaming=True,
        tools=False,
        structured_output=False,
        json_mode=False,
        tool_choice=False,
        parallel_tool_calls=False,
        vision=False,
        files=False,
        audio_input=False,
        audio_output=False,
        embeddings=False,
        reasoning=False,
        web_search=False,
    )

    def __init__(self):
        self.closed = asyncio.Event()
        self.started = asyncio.Event()

    async def stream(self, request):
        async def iterator():
            try:
                self.started.set()
                yield StreamTextDeltaEvent(text_delta="first")
                await asyncio.sleep(10)
            finally:
                self.closed.set()

        return iterator()


class ExecutionBudgetEdgeTests(IsolatedAsyncioTestCase):
    async def test_total_budget_requires_positive_integer(self) -> None:
        for value in (
            True,
            False,
            0,
            -1,
            1.5,
            float("nan"),
            float("inf"),
            "20",
            10**1000,
        ):
            with self.subTest(value=repr(value)[:40]):
                with self.assertRaises(ValidationError):
                    async with execution_scope(value):
                        self.fail("invalid budget was accepted")
                with self.assertRaises(ValidationError):
                    stream_text(
                        model=SuspendedStreamModel(),
                        prompt="hi",
                        total_timeout_ms=value,
                    )

    async def test_jitter_validation_precedes_retry_after(self) -> None:
        error = ProviderHTTPError("busy", 429, response_headers={"Retry-After": "1"})
        for value in (
            True,
            False,
            -1,
            1.1,
            float("nan"),
            float("inf"),
            "0.5",
            10**1000,
        ):
            with self.subTest(value=repr(value)[:40]):
                with self.assertRaises(ValidationError):
                    retry_delay_ms(
                        error, attempt=0, retry_backoff_ms=0, retry_jitter=value
                    )
                with self.assertRaises(ValidationError):
                    async with execution_scope(20, retry_jitter=value):
                        self.fail("invalid jitter was accepted")

    async def test_timeout_during_callback_closes_suspended_iterator(self) -> None:
        model = SuspendedStreamModel()
        callbacks = 0

        async def on_event(event):
            nonlocal callbacks
            callbacks += 1
            await asyncio.sleep(10)

        result = stream_text(
            model=model, prompt="hi", total_timeout_ms=20, on_event=on_event
        )
        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(result.collect(), timeout=0.5)
        self.assertTrue(model.closed.is_set())
        self.assertEqual(callbacks, 1)

    async def test_external_cancellation_during_callback_closes_iterator(self) -> None:
        model = SuspendedStreamModel()
        in_callback = asyncio.Event()

        async def on_event(event):
            in_callback.set()
            await asyncio.sleep(10)

        result = stream_text(
            model=model, prompt="hi", total_timeout_ms=1000, on_event=on_event
        )
        await in_callback.wait()
        await result.aclose()
        self.assertTrue(model.closed.is_set())
        with self.assertRaises(asyncio.CancelledError):
            await result.collect()

    async def test_gateway_stream_keeps_deadline_after_selection_returns(self) -> None:
        model = SuspendedStreamModel()
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(
                        name="openai", language_model_factory=lambda _: model
                    )
                },
                max_retries=0,
                total_timeout_ms=30,
                attempt_timeout_ms=1000,
            )
        )
        stream = gateway.stream_text(
            messages=[GatewayMessage(role="user", content="hi")],
            primary=GatewayModelTarget(provider="openai", model_id="slow"),
        )
        events = stream.event_stream()
        first = await anext(events)
        self.assertEqual(first.text_delta, "first")
        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(stream.collect(), timeout=0.5)
        self.assertTrue(model.closed.is_set())
        await events.aclose()
