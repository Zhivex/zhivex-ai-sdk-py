from __future__ import annotations

import sys
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import GatewayConfig, GatewayMessage, GatewayModelTarget, create_gateway
from zhivex_ai.messages import create_text_message
from zhivex_ai.providers.base import ProviderAdapter
from zhivex_ai.types import GenerateResult, ModelCapabilities, ModelGenerateInput


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
