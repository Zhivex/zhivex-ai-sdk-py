from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import tempfile
import time
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    ModelCapabilities,
    ModelCatalogEntry,
    ValidationError,
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
from zhivex_ai.catalog import ModelPricing
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
        return GenerateResult(
            messages=[create_text_message("assistant", "cached")], text="cached"
        )

    async def stream(self, input: ModelGenerateInput):
        raise RuntimeError("not used")


class NonStoringGenerateCache:
    async def get(self, key: str) -> GenerateResult | None:
        return None

    async def set(self, key: str, value: GenerateResult) -> None:
        return None


class CatalogAndMiddlewareTests(IsolatedAsyncioTestCase):
    async def test_model_catalog_entry_metadata_defaults_are_compatible(self) -> None:
        entry = ModelCatalogEntry("test", "test-model", ["test-alias"], 1.25, ["chat"])
        other_entry = ModelCatalogEntry("test", "other-model")

        self.assertEqual(entry.api_surface, "language")
        self.assertEqual(entry.availability, "stable")
        self.assertEqual(entry.aliases, ("test-alias",))
        self.assertEqual(entry.recommended_for, ("chat",))
        self.assertEqual(entry.regions, ())
        self.assertEqual(entry.support_evidence, "catalog-only")
        self.assertEqual(entry.source_urls, ())
        self.assertIsNone(entry.max_tool_calls_per_turn)
        self.assertIsNone(entry.parallel_tool_calls)
        self.assertIsNone(entry.structured_output)
        with self.assertRaises(FrozenInstanceError):
            entry.provider = "mutated"  # type: ignore[misc]
        self.assertEqual(other_entry.regions, ())
        self.assertEqual(other_entry.source_urls, ())

    async def test_model_catalog_rejects_identifier_collisions(self) -> None:
        with self.assertRaisesRegex(ValidationError, "identifier collision"):
            create_model_catalog(
                [
                    ModelCatalogEntry("test", "canonical-a", aliases=("shared",)),
                    ModelCatalogEntry("test", "shared"),
                ]
            )

    async def test_model_catalog_returns_defensive_copies(self) -> None:
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
            reasoning=False,
            web_search=False,
        )
        catalog = create_model_catalog(
            [ModelCatalogEntry("test", "safe", capabilities=capabilities)]
        )
        first = catalog.find("test", "safe")
        self.assertIsNotNone(first)
        first.capabilities.tools = False  # type: ignore[union-attr]
        second = catalog.find("test", "safe")
        self.assertTrue(second.capabilities.tools)  # type: ignore[union-attr]

    async def test_model_pricing_preserves_units_and_expires(self) -> None:
        pricing = ModelPricing(
            currency="usd",
            source_url="https://example.test/pricing",
            input_per_1m_tokens=5,
            output_per_1m_tokens=30,
            effective_from="2026-08-01",
            effective_until="2026-08-31",
        )

        self.assertEqual(pricing.currency, "USD")
        self.assertEqual(
            pricing.conservative_cost_per_1k_tokens(as_of=date(2026, 8, 29)), 0.03
        )
        self.assertIsNone(
            pricing.conservative_cost_per_1k_tokens(as_of=date(2026, 9, 1))
        )

    async def test_model_catalog_rejects_ambiguous_or_invalid_prices(self) -> None:
        with self.assertRaises(ValidationError):
            ModelPricing(
                currency="USD",
                source_url="https://example.test/pricing",
                input_per_1m_tokens=-1,
            )
        with self.assertRaisesRegex(ValidationError, "either legacy"):
            ModelCatalogEntry(
                "test",
                "ambiguous",
                cost_per_1k_tokens=1,
                pricing=ModelPricing(
                    currency="USD",
                    source_url="https://example.test/pricing",
                    input_per_1m_tokens=1,
                ),
            )

    async def test_model_catalog_find_supports_lookup(self) -> None:
        openai_entry = default_model_catalog.find("openai", "gpt-5.4-mini")
        self.assertIsNotNone(openai_entry)
        catalog = create_model_catalog([openai_entry])  # type: ignore[list-item]
        entry = catalog.find("openai", "gpt-5.4-mini")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.provider, "openai")

    async def test_default_model_catalog_tracks_reference_models(self) -> None:
        expected = [
            ("openai", "gpt-5.6-sol", {"reasoning", "tools", "vision"}),
            ("openai", "gpt-5.6-terra", {"reasoning", "tools", "vision"}),
            ("openai", "gpt-5.6-luna", {"speed", "tools", "vision"}),
            ("openai", "gpt-5.5", {"reasoning", "tools", "vision"}),
            ("openai", "gpt-5.4-mini", {"speed", "tools"}),
            ("openai", "gpt-image-2", {"vision"}),
            ("openai", "gpt-realtime-2.1", {"audio", "speed"}),
            ("azure-openai", "gpt-5.6-sol", {"reasoning", "tools", "vision"}),
            ("azure-openai", "gpt-5.6-terra", {"reasoning", "tools", "vision"}),
            ("azure-openai", "gpt-5.6-luna", {"speed", "tools", "vision"}),
            ("azure-openai", "gpt-5.5", {"reasoning", "tools", "vision"}),
            ("azure-openai", "gpt-chat-latest", {"chat", "tools"}),
            ("azure-openai", "gpt-image-2", {"vision"}),
            ("azure-openai", "gpt-realtime-2.1", {"audio", "speed"}),
            ("azure-openai", "text-embedding-3-large", {"embedding", "retrieval"}),
            ("anthropic", "claude-fable-5", {"reasoning", "tools", "vision"}),
            ("anthropic", "claude-mythos-5", {"reasoning", "tools", "vision"}),
            ("anthropic", "claude-opus-5", {"reasoning", "tools", "vision"}),
            ("anthropic", "claude-opus-4-8", {"reasoning", "tools", "vision"}),
            ("anthropic", "claude-sonnet-5", {"reasoning", "tools", "vision"}),
            ("anthropic", "claude-sonnet-4-6", {"reasoning", "tools", "vision"}),
            ("gemini", "gemini-3.1-pro-preview", {"reasoning", "tools", "vision"}),
            (
                "gemini",
                "gemini-3.7-flash",
                {"chat", "reasoning", "speed", "tools", "vision"},
            ),
            (
                "gemini",
                "gemini-3.6-flash",
                {"chat", "reasoning", "speed", "tools", "vision"},
            ),
            (
                "gemini",
                "gemini-3.5-flash-lite",
                {"chat", "reasoning", "speed", "tools", "vision"},
            ),
            ("gemini", "gemini-3.5-flash", {"speed", "tools", "vision"}),
            (
                "gemini",
                "gemini-omni-flash-preview",
                {"audio", "speed", "tools", "vision"},
            ),
            ("gemini", "gemini-3.1-flash-lite", {"speed", "tools", "vision"}),
            (
                "gemini",
                "gemini-3.5-live-translate-preview",
                {"audio", "translation", "realtime"},
            ),
            ("gemini", "gemini-3.1-flash-live-preview", {"audio", "speed", "vision"}),
            ("gemini", "gemini-3.1-flash-tts-preview", {"audio"}),
            ("gemini", "gemini-3.1-flash-image", {"vision"}),
            ("gemini", "gemini-3.1-flash-lite-image", {"speed", "vision"}),
            ("gemini", "gemini-3-pro-image", {"reasoning", "vision"}),
            ("gemini", "veo-3.1-fast-generate-preview", {"speed", "vision"}),
            ("gemini", "lyria-3-pro-preview", {"audio"}),
            ("vertex", "gemini-3.1-pro-preview", {"reasoning", "tools", "vision"}),
            (
                "vertex",
                "gemini-3.7-flash",
                {"chat", "reasoning", "speed", "tools", "vision"},
            ),
            (
                "vertex",
                "gemini-3.6-flash",
                {"chat", "reasoning", "speed", "tools", "vision"},
            ),
            (
                "vertex",
                "gemini-3.5-flash-lite",
                {"chat", "reasoning", "speed", "tools", "vision"},
            ),
            ("vertex", "gemini-3.5-flash", {"speed", "tools", "vision"}),
            ("vertex", "gemini-3.1-flash-lite", {"speed", "tools", "vision"}),
            ("vertex", "gemini-3.1-flash-live-preview", {"audio", "speed", "vision"}),
            ("vertex", "gemini-3.1-flash-tts-preview", {"audio"}),
            ("vertex", "gemini-3.1-flash-image", {"vision"}),
            ("vertex", "gemini-3-pro-image", {"reasoning", "vision"}),
            ("vertex", "veo-3.1-fast-generate-preview", {"speed", "vision"}),
            ("vertex", "lyria-3-pro-preview", {"audio"}),
            ("bedrock", "anthropic.claude-opus-4-8", {"reasoning", "tools", "vision"}),
            ("bedrock", "anthropic.claude-sonnet-5", {"reasoning", "tools", "vision"}),
            (
                "bedrock",
                "anthropic.claude-sonnet-4-6",
                {"reasoning", "tools", "vision"},
            ),
            ("bedrock", "amazon.nova-premier-v1:0", {"reasoning", "tools", "vision"}),
            ("qwen", "qwen3.8-max", {"chat", "reasoning", "tools", "vision"}),
            ("qwen", "qwen3.7-max", {"reasoning", "tools"}),
            ("qwen", "qwen3.7-max-2026-06-08", {"reasoning", "tools", "vision"}),
            ("qwen", "qwen3.7-plus", {"reasoning", "tools", "vision"}),
            ("qwen", "qwen3.6-plus", {"reasoning", "tools", "vision"}),
            ("qwen", "text-embedding-v4", {"embedding", "retrieval"}),
            ("qwen", "qwen3-asr-flash", {"audio", "speed"}),
            ("qwen", "qwen3-tts-instruct-flash", {"audio"}),
            ("kimi", "kimi-k3", {"reasoning", "tools", "vision"}),
            ("kimi", "kimi-k2.6", {"reasoning", "tools", "vision"}),
            ("deepseek", "deepseek-v4-pro", {"chat", "reasoning", "tools"}),
            ("deepseek", "deepseek-v4-flash", {"chat", "speed", "reasoning", "tools"}),
            ("meta", "muse-spark-1.2", {"chat", "reasoning", "tools", "vision"}),
            (
                "meta",
                "muse-spark-1.2-contributor",
                {"chat", "reasoning", "tools", "vision"},
            ),
            ("meta", "muse-spark-1.1", {"chat", "reasoning", "tools", "vision"}),
            (
                "openrouter",
                "meta/muse-spark-1.2",
                {"chat", "reasoning", "tools", "vision"},
            ),
            (
                "openrouter",
                "meta/muse-glimmer-30b",
                {"chat", "reasoning", "tools", "vision"},
            ),
            ("ollama", "muse-glimmer:30b", {"chat", "reasoning", "tools", "vision"}),
            (
                "ollama",
                "muse-glimmer:30b-mlx",
                {"chat", "reasoning", "tools", "vision"},
            ),
            (
                "vllm",
                "meta-models/Muse-Glimmer-30B",
                {"chat", "reasoning", "tools", "vision"},
            ),
        ]

        for provider, model_id, recommended in expected:
            entry = default_model_catalog.find(provider, model_id)
            self.assertIsNotNone(
                entry, f"{provider}/{model_id} missing from default catalog"
            )
            self.assertTrue(recommended.issubset(set(entry.recommended_for)))  # type: ignore[union-attr]

        self.assertEqual(
            default_model_catalog.find("gemini", "gemini-pro-latest").model_id,
            "gemini-3.1-pro-preview",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("gemini", "gemini-flash-latest").model_id,
            "gemini-3.7-flash",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find(
                "gemini", "gemini-3.1-flash-lite-preview"
            ).availability,
            "retired",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("vertex", "gemini-flash-latest").model_id,
            "gemini-3.7-flash",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find(
                "vertex", "gemini-3.1-flash-lite-preview"
            ).availability,
            "retired",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("openai", "gpt-realtime").model_id,
            "gpt-realtime",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("openai", "gpt-5.6").model_id, "gpt-5.6-sol"
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("azure-openai", "gpt-5.6").model_id,
            "gpt-5.6-sol",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("openai", "gpt-image-1.5").model_id,
            "gpt-image-1.5",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("azure-openai", "chat-latest").model_id,
            "gpt-chat-latest",
        )  # type: ignore[union-attr]
        self.assertIsNone(default_model_catalog.find("azure-openai", "gpt-realtime"))
        self.assertEqual(
            default_model_catalog.find(
                "gemini", "gemini-3.1-flash-image-preview"
            ).availability,
            "retired",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find(
                "vertex", "gemini-3-pro-image-preview"
            ).availability,
            "retired",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("anthropic", "claude-opus-4-7").model_id,
            "claude-opus-4-7",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("anthropic", "claude-haiku-4-5").model_id,
            "claude-haiku-4-5-20251001",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("qwen", "qwen3.7-max-2026-05-20").model_id,
            "qwen3.7-max-2026-05-20",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("qwen", "qwen3.7-max-2026-06-08").model_id,
            "qwen3.7-max-2026-06-08",
        )  # type: ignore[union-attr]
        self.assertIsNone(default_model_catalog.find("qwen", "qwen3.7-max-2026-05-17"))
        self.assertIsNone(
            default_model_catalog.find("anthropic", "claude-sonnet-4-20250514")
        )
        self.assertIsNone(
            default_model_catalog.find("gemini", "imagen-4.0-generate-001")
        )
        self.assertEqual(
            default_model_catalog.find("qwen", "qwen3.7-plus-2026-05-26").model_id,
            "qwen3.7-plus-2026-05-26",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("qwen", "qwen3.6-plus-2026-04-02").model_id,
            "qwen3.6-plus-2026-04-02",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("qwen", "qwen3.5-plus-2026-04-20").model_id,
            "qwen3.5-plus-2026-04-20",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("qwen", "text-embedding-v3").model_id,
            "text-embedding-v3",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("qwen", "qwen3-asr-flash-2026-02-10").model_id,
            "qwen3-asr-flash-2026-02-10",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find(
                "qwen", "qwen3-tts-instruct-flash-2026-01-26"
            ).model_id,
            "qwen3-tts-instruct-flash-2026-01-26",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("deepseek", "deepseek-chat").availability,
            "retired",
        )  # type: ignore[union-attr]
        self.assertEqual(
            default_model_catalog.find("deepseek", "deepseek-reasoner").availability,
            "retired",
        )  # type: ignore[union-attr]

    async def test_meta_muse_catalog_separates_direct_and_hosted_support(self) -> None:
        direct_model_ids = (
            "muse-spark-1.2",
            "muse-spark-1.2-contributor",
            "muse-spark-1.1",
        )

        for model_id in direct_model_ids:
            entry = default_model_catalog.find("meta", model_id)
            self.assertIsNotNone(entry)
            self.assertTrue(entry.source_urls)  # type: ignore[union-attr]
            self.assertTrue(entry.parallel_tool_calls)  # type: ignore[union-attr]
            self.assertTrue(entry.structured_output)  # type: ignore[union-attr]

        self.assertEqual(
            default_model_catalog.find("meta", "muse-spark-1.2").availability, "stable"
        )  # type: ignore[union-attr]
        for model_id in ("muse-spark-1.2-contributor", "muse-spark-1.1"):
            self.assertEqual(
                default_model_catalog.find("meta", model_id).availability, "preview"
            )  # type: ignore[union-attr]

        self.assertEqual(  # exact-model offline fixture; the other direct IDs remain catalog-only
            default_model_catalog.find("meta", "muse-spark-1.2").support_evidence,  # type: ignore[union-attr]
            "offline-contract",
        )
        self.assertEqual(  # type: ignore[union-attr]
            default_model_catalog.find(
                "meta", "muse-spark-1.2-contributor"
            ).support_evidence,
            "catalog-only",
        )
        self.assertEqual(  # type: ignore[union-attr]
            default_model_catalog.find("meta", "muse-spark-1.1").support_evidence,
            "catalog-only",
        )

        openrouter_spark = default_model_catalog.find(
            "openrouter", "meta/muse-spark-1.2"
        )
        self.assertIsNotNone(openrouter_spark)
        self.assertEqual(openrouter_spark.support_evidence, "catalog-only")  # type: ignore[union-attr]
        self.assertEqual(openrouter_spark.availability, "preview")  # type: ignore[union-attr]
        self.assertIsNone(openrouter_spark.structured_output)  # type: ignore[union-attr]

    async def test_muse_glimmer_host_routes_keep_tool_and_schema_claims_conservative(
        self,
    ) -> None:
        hosted_routes = (
            ("openrouter", "meta/muse-glimmer-30b"),
            ("ollama", "muse-glimmer:30b"),
            ("ollama", "muse-glimmer:30b-mlx"),
            ("vllm", "meta-models/Muse-Glimmer-30B"),
        )

        for provider, model_id in hosted_routes:
            entry = default_model_catalog.find(provider, model_id)
            self.assertIsNotNone(
                entry, f"{provider}/{model_id} missing from default catalog"
            )
            self.assertEqual(entry.support_evidence, "catalog-only")  # type: ignore[union-attr]
            self.assertEqual(entry.max_tool_calls_per_turn, 1)  # type: ignore[union-attr]
            self.assertFalse(entry.parallel_tool_calls)  # type: ignore[union-attr]
            self.assertIsNone(entry.structured_output)  # type: ignore[union-attr]
            self.assertTrue(entry.source_urls)  # type: ignore[union-attr]

    async def test_llama_4_is_cataloged_as_vllm_host_route_not_direct_meta_api(
        self,
    ) -> None:
        model_ids = (
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        )

        for model_id in model_ids:
            hosted = default_model_catalog.find("vllm", model_id)
            self.assertIsNotNone(hosted)
            self.assertEqual(hosted.support_evidence, "catalog-only")  # type: ignore[union-attr]
            self.assertTrue(hosted.source_urls)  # type: ignore[union-attr]
            self.assertIsNone(default_model_catalog.find("meta", model_id))

    async def test_default_model_catalog_tracks_surface_and_availability_metadata(
        self,
    ) -> None:
        expected_surfaces = {
            ("openai", "gpt-image-2"): "image",
            ("openai", "gpt-realtime-2.1"): "realtime",
            ("openai", "gpt-realtime-whisper"): "transcription",
            ("azure-openai", "text-embedding-3-large"): "embedding",
            ("gemini", "gemini-omni-flash-preview"): "interactions",
            ("gemini", "gemini-3.1-flash-tts-preview"): "speech",
            ("gemini", "gemini-3.1-flash-image"): "image",
            ("gemini", "veo-3.1-generate-preview"): "video",
            ("gemini", "lyria-3-pro-preview"): "media",
            ("qwen", "text-embedding-v4"): "embedding",
            ("qwen", "qwen3-rerank"): "rerank",
            ("qwen", "qwen3-asr-flash"): "transcription",
            ("qwen", "qwen3-tts-flash"): "speech",
            ("vertex", "gemini-3.1-flash-live-preview"): "realtime",
        }

        for key, api_surface in expected_surfaces.items():
            entry = default_model_catalog.find(*key)
            self.assertIsNotNone(
                entry, f"{key[0]}/{key[1]} missing from default catalog"
            )
            self.assertEqual(entry.api_surface, api_surface)  # type: ignore[union-attr]

        preview_named_entries = [
            entry
            for entry in default_model_catalog.list()
            if "preview" in entry.model_id
        ]
        self.assertTrue(preview_named_entries)
        for entry in preview_named_entries:
            self.assertIn(
                entry.availability,
                {"preview", "deprecated", "retired"},
                f"{entry.provider}/{entry.model_id}",
            )

        qwen_current = default_model_catalog.find("qwen", "qwen3.8-max")
        self.assertIsNotNone(qwen_current)
        self.assertEqual(qwen_current.api_surface, "language")  # type: ignore[union-attr]
        self.assertEqual(qwen_current.availability, "stable")  # type: ignore[union-attr]
        self.assertEqual(qwen_current.regions, ("cn", "intl", "us"))  # type: ignore[union-attr]
        self.assertEqual(qwen_current.support_evidence, "offline-contract")  # type: ignore[union-attr]
        self.assertTrue(qwen_current.source_urls)  # type: ignore[union-attr]

    async def test_latest_gemini_catalog_entries_include_region_and_evidence_metadata(
        self,
    ) -> None:
        gemini_flash = default_model_catalog.find("gemini", "gemini-3.6-flash")
        gemini_flash_lite = default_model_catalog.find(
            "gemini", "gemini-3.5-flash-lite"
        )
        vertex_flash = default_model_catalog.find("vertex", "gemini-3.6-flash")
        vertex_flash_lite = default_model_catalog.find(
            "vertex", "gemini-3.5-flash-lite"
        )

        for entry in (gemini_flash, gemini_flash_lite, vertex_flash, vertex_flash_lite):
            self.assertIsNotNone(entry)
            self.assertEqual(entry.api_surface, "language")  # type: ignore[union-attr]
            self.assertEqual(entry.availability, "stable")  # type: ignore[union-attr]
            self.assertEqual(entry.support_evidence, "offline-contract")  # type: ignore[union-attr]
            self.assertTrue(entry.source_urls)  # type: ignore[union-attr]

        self.assertEqual(vertex_flash.regions, ("global",))  # type: ignore[union-attr]
        self.assertEqual(vertex_flash_lite.regions, ("global", "us", "eu"))  # type: ignore[union-attr]

    async def test_default_model_catalog_has_operational_evidence_for_every_entry(
        self,
    ) -> None:
        entries = default_model_catalog.list()

        self.assertGreaterEqual(len(entries), 100)
        self.assertTrue(all(entry.source_urls for entry in entries))
        reviewed = {
            ("openai", "gpt-6-astra"), ("azure-openai", "gpt-6-astra"),
            ("anthropic", "claude-fable-5-1"), ("anthropic", "claude-mythos-5-1"),
            ("anthropic", "claude-sonnet-5"), ("gemini", "gemini-3.8-flash"),
            ("vertex", "gemini-3.8-flash"), ("qwen", "qwen3.8-max-0902"),
            ("deepseek", "deepseek-v4-flash-vision-exp"),
        }
        for entry in entries:
            expected_date = "2026-09-05" if (entry.provider, entry.model_id) in reviewed else "2026-08-29"
            self.assertEqual(entry.verified_at, expected_date)
        self.assertTrue(all(entry.capabilities is not None for entry in entries))
        self.assertTrue(all(entry.cost_per_1k_tokens is None for entry in entries))
        self.assertGreater(sum(entry.pricing is not None for entry in entries), 0)

        for entry in entries:
            if entry.availability == "retired":
                self.assertIsNotNone(entry.replacement_model_id)

    async def test_default_model_catalog_keeps_distinct_api_ids_out_of_aliases(
        self,
    ) -> None:
        distinct_ids = (
            ("openai", "gpt-4o-mini", "gpt-5.4-mini"),
            ("openai", "gpt-image-1.5", "gpt-image-2"),
            ("openai", "gpt-realtime-2", "gpt-realtime-2.1"),
            ("anthropic", "claude-opus-4-7", "claude-opus-4-8"),
            ("qwen", "text-embedding-v3", "text-embedding-v4"),
            ("kimi", "kimi-k2.5", "kimi-k2.6"),
        )

        for provider, model_id, other_model_id in distinct_ids:
            entry = default_model_catalog.find(provider, model_id)
            self.assertIsNotNone(entry)
            self.assertEqual(entry.model_id, model_id)  # type: ignore[union-attr]
            self.assertNotEqual(entry.model_id, other_model_id)  # type: ignore[union-attr]

    async def test_cached_middleware_avoids_duplicate_generate_calls(self) -> None:
        model = CountingModel()
        wrapped = wrap_language_model(
            model,
            [
                create_cached_generate_middleware(
                    cache=create_in_memory_generate_cache()
                ),
            ],
        )
        request = ModelGenerateInput(messages=[create_text_message("user", "hello")])
        first = await wrapped.generate(request)
        second = await wrapped.generate(request)
        self.assertEqual(first.text, "cached")
        self.assertEqual(second.text, "cached")
        self.assertEqual(model.calls, 1)

    async def test_cached_middleware_coalesces_100_identical_in_flight_requests(
        self,
    ) -> None:
        class BlockingCountingModel(CountingModel):
            def __init__(self) -> None:
                super().__init__()
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def generate(self, input: ModelGenerateInput) -> GenerateResult:
                self.calls += 1
                self.started.set()
                await self.release.wait()
                return GenerateResult(
                    messages=[create_text_message("assistant", "cached")], text="cached"
                )

        model = BlockingCountingModel()
        wrapped = wrap_language_model(
            model,
            [
                create_cached_generate_middleware(
                    cache=create_in_memory_generate_cache()
                )
            ],
        )
        request = ModelGenerateInput(messages=[create_text_message("user", "hello")])
        tasks = [asyncio.create_task(wrapped.generate(request)) for _ in range(100)]

        try:
            await asyncio.wait_for(model.started.wait(), timeout=1)
            await asyncio.sleep(0.05)
            self.assertEqual(model.calls, 1)
        finally:
            model.release.set()

        results = await asyncio.gather(*tasks)
        self.assertTrue(all(result is results[0] for result in results))
        self.assertEqual(model.calls, 1)

    async def test_cached_middleware_does_not_serialize_independent_keys(self) -> None:
        class ConcurrentKeyModel(CountingModel):
            def __init__(self) -> None:
                super().__init__()
                self.both_started = asyncio.Event()
                self.release = asyncio.Event()

            async def generate(self, input: ModelGenerateInput) -> GenerateResult:
                self.calls += 1
                if self.calls == 2:
                    self.both_started.set()
                await self.release.wait()
                return GenerateResult(
                    messages=[create_text_message("assistant", "cached")], text="cached"
                )

        model = ConcurrentKeyModel()
        request_a = ModelGenerateInput(messages=[create_text_message("user", "a")])
        request_b = ModelGenerateInput(messages=[create_text_message("user", "b")])
        wrapped = wrap_language_model(
            model,
            [
                create_cached_generate_middleware(
                    cache=NonStoringGenerateCache(),
                    get_key=lambda input, _model: "a" if input is request_a else "b",
                )
            ],
        )
        tasks = [
            asyncio.create_task(wrapped.generate(request_a)),
            asyncio.create_task(wrapped.generate(request_b)),
        ]

        try:
            await asyncio.wait_for(model.both_started.wait(), timeout=1)
            self.assertEqual(model.calls, 2)
        finally:
            model.release.set()

        await asyncio.gather(*tasks)

    async def test_cached_middleware_shares_failure_and_retries_after_rejection(
        self,
    ) -> None:
        class FlakyCountingModel(CountingModel):
            async def generate(self, input: ModelGenerateInput) -> GenerateResult:
                self.calls += 1
                attempt = self.calls
                await asyncio.sleep(0.01)
                if attempt == 1:
                    raise RuntimeError("temporary failure")
                return GenerateResult(
                    messages=[create_text_message("assistant", "recovered")],
                    text="recovered",
                )

        model = FlakyCountingModel()
        wrapped = wrap_language_model(
            model,
            [
                create_cached_generate_middleware(
                    cache=create_in_memory_generate_cache()
                )
            ],
        )
        request = ModelGenerateInput(messages=[create_text_message("user", "hello")])

        failures = await asyncio.gather(
            *(wrapped.generate(request) for _ in range(10)),
            return_exceptions=True,
        )

        self.assertEqual(model.calls, 1)
        self.assertTrue(all(isinstance(error, RuntimeError) for error in failures))
        self.assertTrue(all(str(error) == "temporary failure" for error in failures))

        recovered = await wrapped.generate(request)
        self.assertEqual(recovered.text, "recovered")
        self.assertEqual(model.calls, 2)

    async def test_cached_middleware_shields_shared_work_from_waiter_cancellation(
        self,
    ) -> None:
        class BlockingCountingModel(CountingModel):
            def __init__(self) -> None:
                super().__init__()
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def generate(self, input: ModelGenerateInput) -> GenerateResult:
                self.calls += 1
                self.started.set()
                await self.release.wait()
                return GenerateResult(
                    messages=[create_text_message("assistant", "cached")], text="cached"
                )

        model = BlockingCountingModel()
        wrapped = wrap_language_model(
            model,
            [create_cached_generate_middleware(cache=NonStoringGenerateCache())],
        )
        request = ModelGenerateInput(messages=[create_text_message("user", "hello")])
        owner = asyncio.create_task(wrapped.generate(request))
        await asyncio.wait_for(model.started.wait(), timeout=1)
        waiter = asyncio.create_task(wrapped.generate(request))
        await asyncio.sleep(0)

        owner.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await owner

        model.release.set()
        result = await asyncio.wait_for(waiter, timeout=1)
        self.assertEqual(result.text, "cached")
        self.assertEqual(model.calls, 1)

        await asyncio.sleep(0)
        repeated = await wrapped.generate(request)
        self.assertEqual(repeated.text, "cached")
        self.assertEqual(model.calls, 2)

    async def test_cached_middleware_removes_in_flight_entry_after_success(
        self,
    ) -> None:
        model = CountingModel()
        wrapped = wrap_language_model(
            model,
            [create_cached_generate_middleware(cache=NonStoringGenerateCache())],
        )
        request = ModelGenerateInput(messages=[create_text_message("user", "hello")])

        await wrapped.generate(request)
        await wrapped.generate(request)

        self.assertEqual(model.calls, 2)

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

        result = await wrapped.generate(
            ModelGenerateInput(messages=[create_text_message("user", "hello")])
        )

        self.assertEqual(result.text, "cached")
        self.assertEqual(
            [event["type"] for event in events], ["generate-start", "generate-finish"]
        )
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
            await wrapped.generate(
                ModelGenerateInput(messages=[create_text_message("user", "hello")])
            )

        self.assertEqual(
            [event["type"] for event in events], ["generate-start", "generate-error"]
        )
        self.assertEqual(str(events[1]["error"]), "boom")

    async def test_file_generate_cache_persists_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = create_file_generate_cache(dir=tmpdir)
            value = GenerateResult(
                messages=[create_text_message("assistant", "cached")], text="cached"
            )

            self.assertIsNone(await cache.get("cache-key"))
            await cache.set("cache-key", value)

            restored = await create_file_generate_cache(dir=tmpdir).get("cache-key")
            self.assertIsNotNone(restored)
            self.assertEqual(restored.text, "cached")

    async def test_file_generate_cache_replace_failure_preserves_previous_value_and_cleans_temp(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = create_file_generate_cache(dir=tmpdir)
            await cache.set("cache-key", GenerateResult(text="previous"))

            with patch(
                "zhivex_ai.middleware.os.replace",
                side_effect=OSError("simulated crash before replace"),
            ):
                with self.assertRaisesRegex(OSError, "simulated crash"):
                    await cache.set("cache-key", GenerateResult(text="new"))

            restored = await cache.get("cache-key")
            self.assertIsNotNone(restored)
            self.assertEqual(restored.text, "previous")
            self.assertEqual(
                list(Path(tmpdir).glob(".zhivex-generate-cache-*.tmp")), []
            )

    async def test_file_generate_cache_fsync_failure_preserves_previous_value_and_cleans_temp(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = create_file_generate_cache(dir=tmpdir)
            await cache.set("cache-key", GenerateResult(text="previous"))

            with patch(
                "zhivex_ai.middleware.os.fsync",
                side_effect=OSError("simulated flush failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated flush failure"):
                    await cache.set("cache-key", GenerateResult(text="new"))

            restored = await cache.get("cache-key")
            self.assertIsNotNone(restored)
            self.assertEqual(restored.text, "previous")
            self.assertEqual(
                list(Path(tmpdir).glob(".zhivex-generate-cache-*.tmp")), []
            )

    async def test_file_generate_cache_rejects_corrupt_and_non_object_entries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = create_file_generate_cache(dir=tmpdir)
            digest = hashlib.sha256(b"cache-key").hexdigest()
            path = Path(tmpdir) / f"{digest}.json"

            for corrupt_payload in ('{"text":', "[]"):
                path.write_text(corrupt_payload, "utf-8")
                with self.assertRaisesRegex(ValidationError, "corrupt, incompatible"):
                    await cache.get("cache-key")

    async def test_file_generate_cache_never_exposes_partial_json_under_concurrency(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = create_file_generate_cache(dir=tmpdir)
            key = "shared-key"
            values = [f"value-{index}" for index in range(32)]
            expected = {"previous", *values}
            await cache.set(key, GenerateResult(text="previous"))

            async def read_repeatedly() -> list[str | None]:
                observed: list[str | None] = []
                for _ in range(64):
                    value = await cache.get(key)
                    self.assertIsNotNone(value)
                    observed.append(value.text)
                return observed

            readers = [asyncio.create_task(read_repeatedly()) for _ in range(8)]
            writers = [
                asyncio.create_task(cache.set(key, GenerateResult(text=value)))
                for value in values
            ]
            reader_results = await asyncio.gather(*readers)
            await asyncio.gather(*writers)

            observed = {value for result in reader_results for value in result}
            self.assertTrue(observed.issubset(expected))
            self.assertEqual(
                list(Path(tmpdir).glob(".zhivex-generate-cache-*.tmp")), []
            )

    async def test_file_generate_cache_cleanup_is_bounded_and_only_removes_own_stale_temps(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = create_file_generate_cache(dir=tmpdir)
            digest = hashlib.sha256(b"cache-key").hexdigest()
            prefix = f".zhivex-generate-cache-{digest}-"
            stale_paths = [Path(tmpdir) / f"{prefix}{index}.tmp" for index in range(40)]
            stale_time = time.time() - (25 * 60 * 60)
            for path in stale_paths:
                path.write_text("partial", "utf-8")
                os.utime(path, (stale_time, stale_time))

            fresh = Path(tmpdir) / f"{prefix}fresh.tmp"
            unrelated = Path(tmpdir) / "application-owned.tmp"
            fresh.write_text("active", "utf-8")
            unrelated.write_text("keep", "utf-8")

            await cache.set("cache-key", GenerateResult(text="cached"))

            self.assertEqual(sum(path.exists() for path in stale_paths), 8)
            self.assertTrue(fresh.exists())
            self.assertEqual(fresh.read_text("utf-8"), "active")
            self.assertEqual(unrelated.read_text("utf-8"), "keep")

    async def test_file_generate_cache_does_not_follow_entry_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            outside = root / "outside.json"
            outside.write_text('{"text":"outside"}', "utf-8")
            digest = hashlib.sha256(b"cache-key").hexdigest()
            link = cache_dir / f"{digest}.json"
            try:
                link.symlink_to(outside)
            except (NotImplementedError, OSError):
                self.skipTest("Symbolic links are unavailable on this platform.")

            cache = create_file_generate_cache(dir=str(cache_dir))
            with self.assertRaisesRegex(ValidationError, "not a regular file"):
                await cache.get("cache-key")

            await cache.set("cache-key", GenerateResult(text="cached"))
            self.assertFalse(link.is_symlink())
            self.assertEqual(outside.read_text("utf-8"), '{"text":"outside"}')
            restored = await cache.get("cache-key")
            self.assertIsNotNone(restored)
            self.assertEqual(restored.text, "cached")

    async def test_file_generate_cache_permission_failure_does_not_leave_temp_files(
        self,
    ) -> None:
        if os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0):
            self.skipTest("POSIX owner permissions are required for this regression.")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            cache_dir.chmod(0o500)
            try:
                cache = create_file_generate_cache(dir=str(cache_dir))
                with self.assertRaises(PermissionError):
                    await cache.set("cache-key", GenerateResult(text="cached"))
            finally:
                cache_dir.chmod(0o700)

            self.assertEqual(list(cache_dir.glob(".zhivex-generate-cache-*.tmp")), [])

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
                    on_state_change=lambda event: transitions.append(
                        str(event["status"])
                    ),
                )
            ],
        )

        with self.assertRaises(RuntimeError):
            await wrapped.generate(
                ModelGenerateInput(messages=[create_text_message("user", "hello")])
            )

        result = await wrapped.generate(
            ModelGenerateInput(messages=[create_text_message("user", "hello")])
        )

        self.assertEqual(result.text, "cached")
        self.assertEqual(transitions, ["open", "half-open", "closed"])
