from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import GatewayConfig, GatewayError, GatewayImageAttachment, GatewayMessage, GatewayModelTarget, create_gateway
from zhivex_ai.errors import ProviderHTTPError
from zhivex_ai.messages import create_text_message
from zhivex_ai.providers.base import ProviderAdapter
from zhivex_ai.types import GenerateResult, ModelCapabilities, ModelGenerateInput, StreamFinishEvent, StreamTextDeltaEvent, TokenUsage


class FailingModel:
    provider = "openai"
    model_id = "bad"
    capabilities = ModelCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=True,
        tool_choice=True,
        parallel_tool_calls=False,
        vision=True,
        files=False,
        audio_input=False,
        audio_output=False,
        embeddings=False,
        reasoning=True,
        web_search=False,
    )

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        raise RuntimeError("503 upstream failed")

    async def stream(self, input: ModelGenerateInput):
        raise RuntimeError("not used")


class WorkingModel:
    provider = "anthropic"
    model_id = "good"
    capabilities = FailingModel.capabilities

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        return GenerateResult(messages=[create_text_message("assistant", "fallback ok")], text="fallback ok")

    async def stream(self, input: ModelGenerateInput):
        raise RuntimeError("not used")


class NoVisionModel(WorkingModel):
    capabilities = ModelCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=True,
        tool_choice=True,
        parallel_tool_calls=False,
        vision=False,
        files=False,
        audio_input=False,
        audio_output=False,
        embeddings=False,
        reasoning=True,
        web_search=False,
    )


class TimeoutModel(WorkingModel):
    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        raise asyncio.TimeoutError()


class RateLimitedModel(WorkingModel):
    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        raise ProviderHTTPError("OpenAI request failed with status 429.", 429)


class NonRetryableHTTPModel(WorkingModel):
    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        raise ProviderHTTPError("OpenAI request failed with status 503.", 503, retryable=False)


class StreamingModel(WorkingModel):
    async def stream(self, input: ModelGenerateInput):
        async def generator():
            yield StreamTextDeltaEvent(text_delta='{"status":"ok"}')
            yield StreamFinishEvent(
                finish_reason="stop",
                usage=TokenUsage(input_tokens=2, output_tokens=2, total_tokens=4),
            )

        return generator()


class ObjectModel(WorkingModel):
    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        return GenerateResult(messages=[create_text_message("assistant", '{"status":"ok"}')], text='{"status":"ok"}')


class Payload(BaseModel):
    status: str


class GatewayTests(IsolatedAsyncioTestCase):
    async def test_gateway_falls_back_to_second_provider(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(name="openai", language_model_factory=lambda model_id: FailingModel()),
                    "anthropic": ProviderAdapter(name="anthropic", language_model_factory=lambda model_id: WorkingModel()),
                },
                max_retries=0,
                latency_bias_ms={"anthropic": 1000},
            )
        )
        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            fallbacks=[GatewayModelTarget(provider="anthropic", model_id="claude-3-5-sonnet")],
        )
        self.assertEqual(result.text, "fallback ok")
        self.assertEqual(result.provider_used, "anthropic")
        self.assertEqual(len(result.attempts), 2)

    async def test_gateway_keeps_primary_first_even_if_fallback_scores_higher(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(name="openai", language_model_factory=lambda model_id: WorkingModel()),
                    "anthropic": ProviderAdapter(name="anthropic", language_model_factory=lambda model_id: WorkingModel()),
                },
                latency_bias_ms={"openai": 5000},
            )
        )
        result = await gateway.generate(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            fallbacks=[GatewayModelTarget(provider="anthropic", model_id="claude-3-5-sonnet")],
            routing_mode="quality",
            task_intent="reasoning",
        )
        self.assertEqual(result.provider_used, "openai")
        self.assertEqual(result.attempts[0].provider, "openai")
        self.assertEqual(result.route_decision.ordered_targets[0].provider, "openai")

    async def test_gateway_skips_non_vision_targets_instead_of_dropping_images(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(name="openai", language_model_factory=lambda model_id: NoVisionModel()),
                    "anthropic": ProviderAdapter(name="anthropic", language_model_factory=lambda model_id: WorkingModel()),
                },
                max_retries=0,
            )
        )
        result = await gateway.generate(
            messages=[
                GatewayMessage(
                    role="user",
                    content="describe this",
                    images=[GatewayImageAttachment(data_url="data:image/png;base64,aaaa", mime_type="image/png")],
                )
            ],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            fallbacks=[GatewayModelTarget(provider="anthropic", model_id="claude-3-5-sonnet")],
        )
        self.assertEqual(result.provider_used, "anthropic")
        self.assertEqual(result.attempts[0].provider, "openai")
        self.assertIn("does not support vision input", result.attempts[0].error_message or "")

    async def test_gateway_timeout_error_has_clear_message_and_retryable_flag(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(name="openai", language_model_factory=lambda model_id: TimeoutModel()),
                },
                max_retries=0,
                attempt_timeout_ms=3210,
            )
        )
        with self.assertRaises(GatewayError) as context:
            await gateway.generate(
                messages=[GatewayMessage(role="user", content="hello")],
                primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            )
        self.assertTrue(context.exception.retryable)
        self.assertIn('timed out for "openai/gpt-4o-mini" after 3210 ms', str(context.exception))

    async def test_gateway_provider_http_error_preserves_retryable_status(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(name="openai", language_model_factory=lambda model_id: RateLimitedModel()),
                },
                max_retries=0,
            )
        )
        with self.assertRaises(GatewayError) as context:
            await gateway.generate(
                messages=[GatewayMessage(role="user", content="hello")],
                primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            )
        self.assertTrue(context.exception.retryable)
        self.assertIn("429", str(context.exception))

    async def test_gateway_respects_explicit_non_retryable_http_error(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(name="openai", language_model_factory=lambda model_id: NonRetryableHTTPModel()),
                },
                max_retries=0,
            )
        )
        with self.assertRaises(GatewayError) as context:
            await gateway.generate(
                messages=[GatewayMessage(role="user", content="hello")],
                primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
            )
        self.assertFalse(context.exception.retryable)
        self.assertIn("503", str(context.exception))

    async def test_gateway_stream_text_collects_result(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(name="openai", language_model_factory=lambda model_id: StreamingModel()),
                },
                max_retries=0,
            )
        )
        result = gateway.stream_text(
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
        )
        final = await result.collect()
        self.assertEqual(final.text, '{"status":"ok"}')
        self.assertEqual(final.provider_used, "openai")

    async def test_gateway_generate_object_enriches_gateway_metadata(self) -> None:
        gateway = create_gateway(
            GatewayConfig(
                adapters={
                    "openai": ProviderAdapter(name="openai", language_model_factory=lambda model_id: ObjectModel()),
                },
                max_retries=0,
            )
        )
        result = await gateway.generate_object(
            schema=Payload,
            messages=[GatewayMessage(role="user", content="hello")],
            primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
        )
        self.assertEqual(result.object.status, "ok")
        self.assertEqual(result.provider_used, "openai")
