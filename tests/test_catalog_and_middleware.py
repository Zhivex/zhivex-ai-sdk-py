from __future__ import annotations

import sys
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    ModelCapabilities,
    create_cached_generate_middleware,
    create_in_memory_generate_cache,
    create_model_catalog,
    create_telemetry_middleware,
    create_text_message,
    default_model_catalog,
    wrap_language_model,
)
from zhivex_ai.types import GenerateResult, ModelGenerateInput


class CountingModel:
    provider = "test"
    model_id = "counting"
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

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        self.calls += 1
        return GenerateResult(messages=[create_text_message("assistant", "cached")], text="cached")

    async def stream(self, input: ModelGenerateInput):
        raise RuntimeError("not used")


class CatalogAndMiddlewareTests(IsolatedAsyncioTestCase):
    async def test_model_catalog_find_supports_lookup(self) -> None:
        openai_entry = default_model_catalog.find("openai", "gpt-5.4-mini")
        self.assertIsNotNone(openai_entry)
        catalog = create_model_catalog([openai_entry])  # type: ignore[list-item]
        entry = catalog.find("openai", "gpt-5.4-mini")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.provider, "openai")

    async def test_cached_middleware_avoids_duplicate_generate_calls(self) -> None:
        model = CountingModel()
        wrapped = wrap_language_model(
            model,
            [
                create_cached_generate_middleware(cache=create_in_memory_generate_cache()),
            ],
        )
        request = ModelGenerateInput(messages=[create_text_message("user", "hello")])
        first = await wrapped.generate(request)
        second = await wrapped.generate(request)
        self.assertEqual(first.text, "cached")
        self.assertEqual(second.text, "cached")
        self.assertEqual(model.calls, 1)

    async def test_telemetry_middleware_emits_start_and_finish_events(self) -> None:
        events: list[dict[str, object]] = []
        model = CountingModel()
        wrapped = wrap_language_model(
            model,
            [
                create_telemetry_middleware(
                    on_event=lambda event: events.append(event),
                )
            ],
        )

        result = await wrapped.generate(ModelGenerateInput(messages=[create_text_message("user", "hello")]))

        self.assertEqual(result.text, "cached")
        self.assertEqual([event["type"] for event in events], ["generate-start", "generate-finish"])
        self.assertEqual(events[0]["model"].provider, "test")
        self.assertEqual(events[1]["model"].model_id, "counting")
        self.assertIn("latencyMs", events[1])
