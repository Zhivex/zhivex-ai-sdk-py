from __future__ import annotations

from datetime import date
import json
from unittest import IsolatedAsyncioTestCase, TestCase

from zhivex_ai import (
    ImagePart,
    ModelMessage,
    ReasoningConfig,
    TextPart,
    ToolChoiceName,
    UnsupportedFeatureError,
    create_anthropic,
    create_azure_openai,
    create_deepseek,
    create_gemini,
    create_openai,
    create_qwen,
    create_vertex,
    generate_text,
    stream_text,
    tool,
)
from zhivex_ai.catalog import ModelPricing, default_model_catalog
from tests.test_anthropic_provider import FakeResponse
from scripts.check_catalog_freshness import pricing_alerts


class ModelRefreshTests(IsolatedAsyncioTestCase):
    async def test_refreshed_models_stream_normalized_text(self):
        responses_events = [
            {"type": "response.output_text.delta", "delta": "ok"},
            {
                "type": "response.completed",
                "response": {"status": "completed", "output": []},
            },
        ]
        cases = [
            (create_openai, {"api_key": "test"}, "gpt-6-astra", responses_events),
            (
                create_azure_openai,
                {"api_key": "test", "endpoint": "https://example.test"},
                "gpt-6-astra",
                responses_events,
            ),
            (
                create_qwen,
                {"api_key": "test", "region": "intl"},
                "qwen3.8-max-0902",
                responses_events,
            ),
            (
                create_anthropic,
                {"api_key": "test"},
                "claude-fable-5-1",
                [
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "ok"},
                    },
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
                    {"type": "message_stop"},
                ],
            ),
            (
                create_gemini,
                {"api_key": "test"},
                "gemini-3.8-flash",
                [
                    {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": "ok"}]},
                                "finishReason": "STOP",
                            }
                        ]
                    }
                ],
            ),
            (
                create_vertex,
                {"access_token": "test", "project_id": "test", "location": "global"},
                "gemini-3.8-flash",
                [
                    {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": "ok"}]},
                                "finishReason": "STOP",
                            }
                        ]
                    }
                ],
            ),
            (
                create_deepseek,
                {"api_key": "test"},
                "deepseek-v4-flash-vision-exp",
                [{"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}],
            ),
        ]
        for factory, options, model_id, events in cases:

            async def fetch(url, **kwargs):
                self.assertTrue(kwargs["stream"])
                return FakeResponse(
                    200,
                    body_text="".join(
                        ("event: " + event["type"] + "\n" if "type" in event else "")
                        + "data: "
                        + json.dumps(event)
                        + "\n\n"
                        for event in events
                    ),
                )

            with self.subTest(model=model_id, factory=factory.__name__):
                async with stream_text(
                    model=factory(fetch=fetch, **options)(model_id),
                    prompt="test",
                    stream_buffer_size=16,
                ) as result:
                    self.assertEqual(
                        "".join([text async for text in result.text_stream()]), "ok"
                    )
                    self.assertEqual((await result.collect()).text, "ok")

    async def test_fable_token_count_rejects_forced_choice(self):
        async def fetch(*args, **kwargs):
            self.fail("Invalid count requests must not reach the provider")

        provider = create_anthropic(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError):
            await provider.tokens().count(
                model_id="claude-fable-5-1",
                prompt="test",
                provider_options={"tool_choice": {"type": "any"}},
            )

    async def test_astra_uses_responses_for_text_and_tools(self):
        requests = []

        async def fetch(url, **kwargs):
            requests.append((url, kwargs["json_body"]))
            return FakeResponse(
                200,
                {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok"}],
                        }
                    ],
                },
            )

        for provider in (
            create_openai(api_key="test", fetch=fetch),
            create_azure_openai(
                api_key="test", endpoint="https://example.test", fetch=fetch
            ),
        ):
            result = await generate_text(
                model=provider("gpt-6-astra"),
                prompt="test",
                reasoning=ReasoningConfig(effort="max"),
                tools={
                    "lookup": tool(
                        name="lookup",
                        schema={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                        execute=lambda value: value,
                    )
                },
            )
            self.assertEqual(result.text, "ok")
            self.assertTrue(requests[-1][0].endswith("/responses"))
            self.assertEqual(requests[-1][1]["reasoning"], {"effort": "max"})
            self.assertEqual(requests[-1][1]["tools"][0]["name"], "lookup")

    async def test_astra_rejects_unsupported_parameters_before_dispatch(self):
        async def fetch(*args, **kwargs):
            self.fail("Invalid requests must not reach the provider")

        model = create_openai(api_key="test", fetch=fetch).native.language_model(
            "gpt-6-astra"
        )
        for kwargs in (
            {"temperature": 1},
            {"reasoning": ReasoningConfig(effort="none")},
            {"reasoning": ReasoningConfig(effort="minimal")},
            {"provider_options": {"top_p": 1}},
            {"provider_options": {"include": ["message.output_text.logprobs"]}},
        ):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaises(UnsupportedFeatureError),
            ):
                await generate_text(model=model, prompt="test", **kwargs)
            async with stream_text(model=model, prompt="test", **kwargs) as result:
                with self.assertRaises(UnsupportedFeatureError):
                    await result.collect()

    async def test_fable_51_forced_tools_and_thinking_fail_before_dispatch(self):
        async def fetch(*args, **kwargs):
            self.fail("Invalid requests must not reach the provider")

        tools = {
            "lookup": tool(
                name="lookup",
                schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                execute=lambda value: value,
            )
        }
        for model_id in ("claude-fable-5-1", "claude-mythos-5-1"):
            model = create_anthropic(api_key="test", fetch=fetch).native.language_model(
                model_id
            )
            for kwargs in (
                {"tool_choice": "required"},
                {"tool_choice": ToolChoiceName(tool_name="lookup")},
                {"reasoning": ReasoningConfig(effort="none")},
                {"reasoning": ReasoningConfig(budget_tokens=1024)},
                {"provider_options": {"tool_choice": {"type": "any"}}},
            ):
                with (
                    self.subTest(model=model_id, kwargs=kwargs),
                    self.assertRaises(UnsupportedFeatureError),
                ):
                    await generate_text(
                        model=model, prompt="test", tools=tools, **kwargs
                    )
                async with stream_text(
                    model=model, prompt="test", tools=tools, **kwargs
                ) as result:
                    with self.assertRaises(UnsupportedFeatureError):
                        await result.collect()

    async def test_fable_51_none_choice_and_adaptive_reasoning(self):
        requests = []

        async def fetch(url, **kwargs):
            requests.append(kwargs["json_body"])
            return FakeResponse(
                200,
                {
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                },
            )

        model = create_anthropic(api_key="test", fetch=fetch)("claude-fable-5-1")
        await generate_text(
            model=model,
            prompt="test",
            tools={
                "lookup": tool(
                    name="lookup",
                    schema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                    execute=lambda value: value,
                )
            },
            tool_choice="none",
            reasoning=ReasoningConfig(effort="high"),
        )
        self.assertEqual(requests[0]["thinking"], {"type": "adaptive"})
        self.assertEqual(requests[0]["tool_choice"], {"type": "none"})

    async def test_gemini_38_and_vertex_reject_minimal_but_accept_medium(self):
        requests = []

        async def fetch(url, **kwargs):
            requests.append((url, kwargs["json_body"]))
            return FakeResponse(
                200,
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}
                    ]
                },
            )

        providers = (
            create_gemini(api_key="test", fetch=fetch),
            create_vertex(
                project_id="test", location="global", access_token="test", fetch=fetch
            ),
        )
        for provider in providers:
            model = provider("gemini-3.8-flash")
            with self.assertRaises(UnsupportedFeatureError):
                await generate_text(
                    model=model,
                    prompt="test",
                    reasoning=ReasoningConfig(effort="minimal"),
                )
            output = await generate_text(
                model=model, prompt="test", reasoning=ReasoningConfig(effort="medium")
            )
            self.assertEqual(output.text, "ok")
            self.assertIn("gemini-3.8-flash", requests[-1][0])
            self.assertEqual(
                requests[-1][1]["generationConfig"]["thinkingConfig"]["thinkingLevel"],
                "medium",
            )

    async def test_qwen_snapshot_keeps_responses_routing_and_model_id(self):
        requests = []

        async def fetch(url, **kwargs):
            requests.append((url, kwargs["json_body"]))
            return FakeResponse(
                200,
                {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok"}],
                        }
                    ],
                },
            )

        provider = create_qwen(api_key="test", region="intl", fetch=fetch)
        for model_id in ("qwen3.8-max-0902", "qwen3.8-max-2026-09-02"):
            await generate_text(model=provider(model_id), prompt="test")
            self.assertTrue(requests[-1][0].endswith("/responses"))
            self.assertEqual(requests[-1][1]["model"], model_id)

    async def test_deepseek_vision_preserves_images_and_text_models_reject_them(self):
        requests = []

        async def fetch(url, **kwargs):
            requests.append(kwargs["json_body"])
            return FakeResponse(
                200,
                {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            )

        provider = create_deepseek(api_key="test", fetch=fetch)
        messages = [
            ModelMessage(
                role="user",
                parts=[
                    TextPart(text="Describe"),
                    ImagePart(image="https://example.test/image.png"),
                ],
            )
        ]
        await generate_text(
            model=provider("deepseek-v4-flash-vision-exp"), messages=messages
        )
        self.assertEqual(
            requests[-1]["messages"][0]["content"][1],
            {
                "type": "image_url",
                "image_url": {"url": "https://example.test/image.png"},
            },
        )
        for model_id in ("deepseek-v4-flash", "deepseek-v4-pro"):
            with self.assertRaises(UnsupportedFeatureError):
                await generate_text(model=provider(model_id), messages=messages)
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider("deepseek-v4-flash-vision-exp"),
                messages=[ModelMessage(role="assistant", parts=messages[0].parts)],
            )
        self.assertEqual(len(requests), 1)


class CatalogFreshnessTests(TestCase):
    def test_pricing_boundaries_are_deterministic(self):
        pricing = ModelPricing(
            currency="USD",
            source_url="https://example.test/pricing",
            input_per_1m_tokens=2,
            output_per_1m_tokens=10,
            effective_from="2026-08-01",
            effective_until="2026-08-31",
        )
        for day, expected in (
            (date(2026, 7, 31), None),
            (date(2026, 8, 1), 0.01),
            (date(2026, 8, 31), 0.01),
            (date(2026, 9, 1), None),
        ):
            self.assertEqual(
                pricing.conservative_cost_per_1k_tokens(as_of=day), expected
            )

    def test_freshness_alerts_anticipate_expiry(self):
        self.assertEqual(
            pricing_alerts(default_model_catalog, as_of=date(2026, 9, 5)), []
        )
        upcoming = pricing_alerts(default_model_catalog, as_of=date(2026, 12, 15))
        self.assertTrue(
            any(
                item["model_id"] == "gemini-3.8-flash" and item["status"] == "expiring"
                for item in upcoming
            )
        )
        expired = pricing_alerts(default_model_catalog, as_of=date(2027, 1, 1))
        self.assertTrue(
            any(
                item["model_id"] == "gemini-3.8-flash" and item["status"] == "expired"
                for item in expired
            )
        )

    def test_restricted_models_and_snapshot_aliases_remain_distinct(self):
        self.assertEqual(
            default_model_catalog.find(
                provider="anthropic", model_id="claude-mythos-5-1"
            ).availability,
            "limited",
        )
        self.assertEqual(
            default_model_catalog.find(
                provider="qwen", model_id="qwen3.8-max-2026-09-02"
            ).model_id,
            "qwen3.8-max-0902",
        )
        self.assertEqual(
            default_model_catalog.find(
                provider="qwen", model_id="qwen3.8-max"
            ).model_id,
            "qwen3.8-max",
        )
