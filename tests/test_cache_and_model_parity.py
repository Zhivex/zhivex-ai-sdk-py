"""Offline contracts for cache-write pricing, Gemini 4 Argon, and Anthropic API 2026-10-21."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase

from zhivex_ai import (
    CacheWriteTokensDetails,
    ReasoningConfig,
    TokenUsage,
    UnsupportedFeatureError,
    ValidationError,
    create_anthropic,
    create_gemini,
    create_openai,
    create_vertex,
    generate_text,
    stream_text,
)
from zhivex_ai.cache_usage import read_cache_write_tokens_details
from zhivex_ai.catalog import default_model_catalog
from zhivex_ai.observability import TokenPricing, estimate_token_cost
from zhivex_ai.providers.anthropic import _parse_token_usage
from zhivex_ai.types import ModelMessage, ProviderDataPart, TextPart


@dataclass
class FakeResponse:
    status_code: int
    payload: Any = None
    headers: dict[str, str] = field(default_factory=dict)

    async def json(self) -> Any:
        return self.payload

    async def text(self) -> str:
        return json.dumps(self.payload)

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    async def iter_lines(self) -> AsyncIterable[str]:
        if False:
            yield ""


def _completed_response(usage: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


class CacheWritePricingTests(TestCase):
    def test_reader_rejects_unknown_or_empty_details(self) -> None:
        parsed = read_cache_write_tokens_details(
            {"ttl_30m_tokens": 3, "ttl_12h_tokens": 1}
        )
        self.assertEqual(
            parsed, CacheWriteTokensDetails(ttl_30m_tokens=3, ttl_12h_tokens=1)
        )
        for value in (
            {},
            {"ttl_1h_tokens": 1},
            {"ttl_30m_tokens": True},
            {"ttl_12h_tokens": -1},
        ):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                read_cache_write_tokens_details(value)

    def test_twelve_hour_writes_use_two_and_a_half_times_input(self) -> None:
        usage = TokenUsage(
            input_tokens=4_000,
            output_tokens=500,
            total_tokens=4_500,
            cached_input_tokens=1_000,
            cache_write_tokens=2_000,
            cache_write_tokens_details=CacheWriteTokensDetails(
                ttl_30m_tokens=1_000, ttl_12h_tokens=1_000
            ),
        )
        estimate = estimate_token_cost(
            usage,
            TokenPricing(
                input_cost_per_1k_tokens=0.002,
                output_cost_per_1k_tokens=0.01,
                cached_input_cost_per_1k_tokens=0.0002,
                cache_write_cost_per_1k_tokens=0.0025,
            ),
        )
        self.assertAlmostEqual(estimate.input_cost or 0, 0.002)
        self.assertAlmostEqual(estimate.cached_input_cost or 0, 0.0002)
        self.assertAlmostEqual(estimate.cache_write_cost or 0, 0.0075)
        self.assertAlmostEqual(estimate.output_cost or 0, 0.005)
        self.assertAlmostEqual(estimate.total_cost or 0, 0.0147)

    def test_missing_cache_write_rate_leaves_the_total_unset(self) -> None:
        estimate = estimate_token_cost(
            TokenUsage(input_tokens=100, output_tokens=10, cache_write_tokens=40),
            TokenPricing(
                input_cost_per_1k_tokens=0.002, output_cost_per_1k_tokens=0.01
            ),
        )
        self.assertIsNone(estimate.cache_write_cost)
        self.assertIsNone(estimate.total_cost)

    def test_legacy_estimate_ignores_absent_cache_counters(self) -> None:
        estimate = estimate_token_cost(
            TokenUsage(input_tokens=1_000, output_tokens=1_000, total_tokens=2_000),
            TokenPricing(total_cost_per_1k_tokens=0.003),
        )
        self.assertAlmostEqual(estimate.total_cost or 0, 0.006)
        self.assertIsNone(estimate.cache_write_cost)

    def test_gemini_4_argon_catalog_uses_introductory_rates(self) -> None:
        for provider in ("gemini", "vertex"):
            entry = default_model_catalog.find(provider, "gemini-4-argon")
            self.assertIsNotNone(entry)
            assert entry is not None and entry.pricing is not None
            self.assertEqual(entry.pricing.input_per_1m_tokens, 2)
            self.assertEqual(entry.pricing.cached_input_per_1m_tokens, 0.10)
            self.assertEqual(entry.pricing.output_per_1m_tokens, 10)
            self.assertIsNone(entry.pricing.cache_write_per_1m_tokens)
            self.assertEqual(
                entry.pricing.conservative_cost_per_1k_tokens(as_of=date(2026, 10, 26)),
                0.01,
            )
            self.assertIsNone(
                entry.pricing.conservative_cost_per_1k_tokens(as_of=date(2026, 12, 1))
            )


class OpenAICacheWriteTests(IsolatedAsyncioTestCase):
    async def test_responses_usage_and_twelve_hour_ttl(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            requests.append(kwargs["json_body"])
            return FakeResponse(
                200,
                _completed_response(
                    {
                        "input_tokens": 30,
                        "output_tokens": 4,
                        "total_tokens": 34,
                        "input_tokens_details": {
                            "cached_tokens": 10,
                            "cache_write_tokens": 12,
                            "cache_write_tokens_details": {
                                "ttl_30m_tokens": 8,
                                "ttl_12h_tokens": 4,
                            },
                        },
                    }
                ),
            )

        result = await generate_text(
            model=create_openai(api_key="test", fetch=fetch).native.language_model(
                "gpt-4o"
            ),
            messages=[
                ModelMessage(
                    role="user",
                    parts=[
                        TextPart(
                            text="Remember this prefix.",
                            provider_metadata={
                                "prompt_cache_breakpoint": {
                                    "mode": "explicit",
                                    "ttl": "12h",
                                }
                            },
                        )
                    ],
                )
            ],
            provider_options={"prompt_cache_options": {"ttl": "12h"}},
        )
        self.assertEqual(result.usage.cached_input_tokens, 10)
        self.assertEqual(result.usage.cache_write_tokens, 12)
        self.assertEqual(result.usage.cache_write_tokens_details.ttl_12h_tokens, 4)
        self.assertEqual(requests[0]["prompt_cache_options"], {"ttl": "12h"})
        self.assertEqual(
            requests[0]["input"][0]["content"][0]["prompt_cache_breakpoint"],
            {"mode": "explicit", "ttl": "12h"},
        )

    async def test_invalid_prompt_cache_ttl_fails_before_dispatch(self) -> None:
        async def fetch(*args: Any, **kwargs: Any) -> FakeResponse:
            self.fail("invalid ttl must not reach the provider")

        with self.assertRaises(ValidationError):
            await generate_text(
                model=create_openai(api_key="test", fetch=fetch).native.language_model(
                    "gpt-4o"
                ),
                prompt="hello",
                provider_options={"prompt_cache_options": {"ttl": "1d"}},
            )


class AnthropicCacheAndApiVersionTests(IsolatedAsyncioTestCase):
    def test_cache_counters_are_added_to_input_tokens(self) -> None:
        usage = _parse_token_usage(
            {
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 4,
                    "cache_creation_input_tokens": 6,
                    "output_tokens": 2,
                }
            }
        )
        self.assertEqual(usage.input_tokens, 20)
        self.assertEqual(usage.cached_input_tokens, 4)
        self.assertEqual(usage.cache_write_tokens, 6)
        unchanged = _parse_token_usage(
            {"usage": {"input_tokens": 3, "output_tokens": 1}}
        )
        self.assertEqual(
            unchanged, TokenUsage(input_tokens=3, output_tokens=1, total_tokens=4)
        )

    async def test_default_version_still_sends_temperature(self) -> None:
        requests: list[tuple[dict[str, str], dict[str, Any]]] = []

        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            requests.append((kwargs["headers"], kwargs["json_body"]))
            return FakeResponse(
                200,
                {
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        await generate_text(
            model=create_anthropic(api_key="test", fetch=fetch)("claude-sonnet-4-5"),
            prompt="hello",
            temperature=1,
        )
        self.assertEqual(requests[0][0]["anthropic-version"], "2023-06-01")
        self.assertEqual(requests[0][1]["temperature"], 1)

    async def test_api_version_2026_10_21_rejects_removed_fields_before_dispatch(
        self,
    ) -> None:
        async def fetch(*args: Any, **kwargs: Any) -> FakeResponse:
            self.fail("removed fields must not reach Anthropic")

        provider = create_anthropic(
            api_key="test", anthropic_version="2026-10-21", fetch=fetch
        )
        model = provider.native.language_model("claude-sonnet-4-5")
        document = {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": "notes"},
        }
        cases = (
            {"temperature": 1},
            {"provider_options": {"top_p": 1}},
            {"provider_options": {"top_k": 1}},
            {
                "messages": [
                    ModelMessage(role="user", parts=[TextPart(text="hello")]),
                    ModelMessage(
                        role="assistant",
                        parts=[
                            ProviderDataPart(
                                provider="anthropic",
                                data={"type": "raw_content_block", "block": document},
                            )
                        ],
                    ),
                ]
            },
            {
                "messages": [
                    ModelMessage(
                        role="user",
                        parts=[
                            TextPart(
                                text="quoted",
                                provider_metadata={
                                    "citations": [
                                        {
                                            "type": "char_location",
                                            "cited_text": "quoted",
                                        }
                                    ]
                                },
                            )
                        ],
                    )
                ]
            },
        )
        for kwargs in cases:
            call = dict(kwargs)
            if "messages" not in call:
                call["prompt"] = "hello"
            with (
                self.subTest(keys=sorted(call)),
                self.assertRaises(UnsupportedFeatureError),
            ):
                await generate_text(model=model, **call)
            async with stream_text(model=model, **call) as result:
                with self.assertRaises(UnsupportedFeatureError):
                    await result.collect()

        client = provider.native.messages_client_factory()
        assert client is not None
        with self.assertRaises(UnsupportedFeatureError):
            await client.create(
                {
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 16,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "search_result",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "quoted",
                                            "citations": [{"type": "char_location"}],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            )

    async def test_api_version_2026_10_21_allows_nested_documents_and_assistant_citations(
        self,
    ) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            requests.append(
                {
                    "headers": kwargs.get("headers") or {},
                    "json": kwargs.get("json_body"),
                }
            )
            return FakeResponse(
                200,
                {
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                },
            )

        provider = create_anthropic(
            api_key="test", anthropic_version="2026-10-21", fetch=fetch
        )
        await generate_text(
            model=provider("claude-sonnet-4-5"),
            messages=[
                ModelMessage(role="user", parts=[TextPart(text="hello")]),
                ModelMessage(
                    role="assistant",
                    parts=[
                        TextPart(
                            text="quoted",
                            provider_metadata={
                                "citations": [{"type": "char_location"}]
                            },
                        )
                    ],
                ),
                ModelMessage(role="user", parts=[TextPart(text="continue")]),
            ],
        )
        self.assertEqual(requests[0]["headers"]["anthropic-version"], "2026-10-21")
        self.assertNotIn("temperature", requests[0]["json"])
        client = provider.native.messages_client_factory()
        assert client is not None
        await client.create(
            {
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [
                                    {
                                        "type": "document",
                                        "source": {
                                            "type": "text",
                                            "media_type": "text/plain",
                                            "data": "notes",
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        self.assertEqual(len(requests), 2)


class Gemini4ArgonTests(IsolatedAsyncioTestCase):
    async def test_thinking_level_and_cached_tokens(self) -> None:
        requests: list[tuple[str, dict[str, Any]]] = []

        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            requests.append((url, kwargs["json_body"]))
            return FakeResponse(
                200,
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 120,
                        "candidatesTokenCount": 8,
                        "totalTokenCount": 128,
                        "cachedContentTokenCount": 40,
                    },
                },
            )

        for factory in (
            lambda: create_gemini(api_key="test", fetch=fetch),
            lambda: create_vertex(
                project_id="test", location="global", access_token="test", fetch=fetch
            ),
        ):
            requests.clear()
            provider = factory()
            model = provider.native.language_model("gemini-4-argon")
            result = await generate_text(
                model=model,
                prompt="test",
                reasoning=ReasoningConfig(effort="high"),
            )
            self.assertEqual(result.text, "ok")
            self.assertEqual(result.usage.input_tokens, 120)
            self.assertEqual(result.usage.cached_input_tokens, 40)
            self.assertEqual(
                requests[-1][1]["generationConfig"]["thinkingConfig"],
                {"thinkingLevel": "high"},
            )
            for kwargs in (
                {"reasoning": ReasoningConfig(effort="minimal")},
                {"temperature": 0.2},
                {
                    "provider_options": {
                        "generationConfig": {"thinkingConfig": {"thinkingBudget": 128}}
                    }
                },
                {"provider_options": {"frequencyPenalty": 0.1}},
            ):
                with (
                    self.subTest(provider=provider.name, kwargs=sorted(kwargs)),
                    self.assertRaises(UnsupportedFeatureError),
                ):
                    await generate_text(model=model, prompt="test", **kwargs)
            self.assertEqual(len(requests), 1)
