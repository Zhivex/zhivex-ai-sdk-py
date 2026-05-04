from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    ModelCapabilities,
    create_cached_generate_middleware,
    create_circuit_breaker_middleware,
    create_file_generate_cache,
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

    async def test_default_model_catalog_tracks_reference_models(self) -> None:
        expected = [
            ("openai", "gpt-5.5", {"reasoning", "tools", "vision"}),
            ("openai", "gpt-5.4-mini", {"speed", "tools"}),
            ("anthropic", "claude-opus-4-7", {"reasoning", "tools", "vision"}),
            ("anthropic", "claude-sonnet-4-6", {"reasoning", "tools", "vision"}),
            ("gemini", "gemini-3.1-pro-preview", {"reasoning", "tools", "vision"}),
            ("gemini", "gemini-3-flash-preview", {"speed", "tools", "vision"}),
            ("vertex", "gemini-3.1-pro-preview", {"reasoning", "tools", "vision"}),
            ("bedrock", "anthropic.claude-sonnet-4-6", {"reasoning", "tools", "vision"}),
            ("bedrock", "amazon.nova-premier-v1:0", {"reasoning", "tools", "vision"}),
            ("qwen", "qwen3.6-plus", {"reasoning", "tools", "vision"}),
            ("qwen", "text-embedding-v4", {"embedding", "retrieval"}),
            ("qwen", "qwen3-asr-flash", {"audio", "speed"}),
            ("qwen", "qwen3-tts-instruct-flash", {"audio"}),
            ("kimi", "kimi-k2.6", {"reasoning", "tools", "vision"}),
        ]

        for provider, model_id, recommended in expected:
            entry = default_model_catalog.find(provider, model_id)
            self.assertIsNotNone(entry, f"{provider}/{model_id} missing from default catalog")
            self.assertTrue(recommended.issubset(set(entry.recommended_for)))  # type: ignore[union-attr]

        self.assertEqual(default_model_catalog.find("gemini", "gemini-pro-latest").model_id, "gemini-3.1-pro-preview")  # type: ignore[union-attr]
        self.assertEqual(default_model_catalog.find("anthropic", "claude-haiku-4-5").model_id, "claude-haiku-4-5-20251001")  # type: ignore[union-attr]
        self.assertEqual(default_model_catalog.find("qwen", "qwen3.6-plus-2026-04-02").model_id, "qwen3.6-plus")  # type: ignore[union-attr]
        self.assertEqual(default_model_catalog.find("qwen", "qwen3.5-plus-2026-04-20").model_id, "qwen3.5-plus")  # type: ignore[union-attr]
        self.assertEqual(default_model_catalog.find("qwen", "text-embedding-v3").model_id, "text-embedding-v4")  # type: ignore[union-attr]
        self.assertEqual(default_model_catalog.find("qwen", "qwen3-asr-flash-2026-02-10").model_id, "qwen3-asr-flash")  # type: ignore[union-attr]
        self.assertEqual(default_model_catalog.find("qwen", "qwen3-tts-instruct-flash-2026-01-26").model_id, "qwen3-tts-instruct-flash")  # type: ignore[union-attr]

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

    async def test_telemetry_middleware_emits_error_event(self) -> None:
        class FailingModel(CountingModel):
            async def generate(self, input: ModelGenerateInput) -> GenerateResult:
                raise RuntimeError("boom")

        events: list[dict[str, object]] = []
        wrapped = wrap_language_model(
            FailingModel(),
            [
                create_telemetry_middleware(
                    on_event=lambda event: events.append(event),
                )
            ],
        )

        with self.assertRaises(RuntimeError):
            await wrapped.generate(ModelGenerateInput(messages=[create_text_message("user", "hello")]))

        self.assertEqual([event["type"] for event in events], ["generate-start", "generate-error"])
        self.assertEqual(str(events[1]["error"]), "boom")

    async def test_file_generate_cache_persists_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = create_file_generate_cache(dir=tmpdir)
            value = GenerateResult(messages=[create_text_message("assistant", "cached")], text="cached")

            self.assertIsNone(await cache.get("cache-key"))
            await cache.set("cache-key", value)

            restored = await create_file_generate_cache(dir=tmpdir).get("cache-key")
            self.assertIsNotNone(restored)
            self.assertEqual(restored.text, "cached")

    async def test_circuit_breaker_emits_open_half_open_and_closed(self) -> None:
        class FlakyModel(CountingModel):
            def __init__(self) -> None:
                super().__init__()
                self.fail_next = True

            async def generate(self, input: ModelGenerateInput) -> GenerateResult:
                if self.fail_next:
                    self.fail_next = False
                    raise RuntimeError("temporary failure")
                return await super().generate(input)

        transitions: list[str] = []
        wrapped = wrap_language_model(
            FlakyModel(),
            [
                create_circuit_breaker_middleware(
                    failure_threshold=1,
                    cooldown_ms=0,
                    on_state_change=lambda event: transitions.append(str(event["status"])),
                )
            ],
        )

        with self.assertRaises(RuntimeError):
            await wrapped.generate(ModelGenerateInput(messages=[create_text_message("user", "hello")]))

        result = await wrapped.generate(ModelGenerateInput(messages=[create_text_message("user", "hello")]))

        self.assertEqual(result.text, "cached")
        self.assertEqual(transitions, ["open", "half-open", "closed"])
