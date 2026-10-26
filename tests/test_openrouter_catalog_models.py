"""OpenRouter ids served by the existing Responses client."""
from __future__ import annotations

import json
from typing import Any
from unittest import IsolatedAsyncioTestCase

from zhivex_ai import ReasoningConfig, create_openrouter, default_model_catalog, generate_text


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.status_code = 200
        self.payload = payload

    async def json(self) -> Any:
        return self.payload

    async def text(self) -> str:
        return json.dumps(self.payload)


class OpenRouterCatalogModelTests(IsolatedAsyncioTestCase):
    async def test_openrouter_catalog_models_post_reasoning_to_responses(self):
        requests: list[dict[str, Any]] = []

        async def fetch(url, **request):
            requests.append({"url": url, "json": request["json_body"]})
            return FakeResponse({
                "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
            })

        provider = create_openrouter(api_key="test", fetch=fetch)
        models = (
            "openai/gpt-6.1-sol",
            "anthropic/claude-sonnet-5.5",
            "anthropic/claude-haiku-5.5",
            "x-ai/grok-4.7",
            "meta/muse-wonder",
            "google/gemini-3.8-flash",
            "meta/muse-spark-open",
        )
        for model_id in models:
            with self.subTest(model_id=model_id):
                result = await generate_text(
                    model=provider.native.language_model(model_id),
                    prompt="hello",
                    reasoning=ReasoningConfig(effort="low"),
                )
                self.assertEqual(result.text, "ok")
                self.assertTrue(requests[-1]["url"].endswith("/responses"))
                self.assertEqual(requests[-1]["json"]["model"], model_id)
                self.assertEqual(requests[-1]["json"]["reasoning"], {"effort": "low"})
                entry = default_model_catalog.find("openrouter", model_id)
                self.assertIsNotNone(entry)
                assert entry is not None and entry.pricing is not None
                self.assertEqual(entry.support_evidence, "offline-contract")
                self.assertIsNotNone(entry.pricing.input_per_1m_tokens)
        self.assertIsNone(default_model_catalog.find("openrouter", "google/gemini-nano-banana-2.1"))
        haiku = default_model_catalog.find("openrouter", "anthropic/claude-haiku-5.5")
        assert haiku is not None and haiku.pricing is not None
        self.assertEqual(haiku.pricing.cache_write_per_1m_tokens, 0.125)
        self.assertEqual(haiku.pricing.long_context_threshold_tokens, 100_000)
        flash = default_model_catalog.find("openrouter", "google/gemini-3.8-flash")
        assert flash is not None and flash.pricing is not None
        self.assertEqual(flash.pricing.input_per_1m_tokens, 0.825)
        self.assertEqual(flash.pricing.output_per_1m_tokens, 4.125)
