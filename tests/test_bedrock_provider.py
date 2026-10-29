from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import create_bedrock, generate_text, stream_text, tool
from zhivex_ai.catalog import default_model_catalog
from zhivex_ai.errors import UnsupportedFeatureError, ValidationError
from zhivex_ai.types import (
    ImagePart,
    ModelGenerateInput,
    ModelMessage,
    ReasoningConfig,
    TextPart,
    ToolChoiceName,
)


class FakeBedrockClient:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def converse(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        return {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
        }


class FakeBedrockStreamClient(FakeBedrockClient):
    async def converse_stream(self, payload: dict[str, Any]):
        self.payloads.append(payload)

        async def stream():
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hello "}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "world"}}}
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "start": {"toolUse": {"toolUseId": "tooluse_1", "name": "weather"}},
                }
            }
            yield {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"city"'}}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": ':"Madrid"}'}}}}
            yield {"contentBlockStop": {"contentBlockIndex": 1}}
            yield {"messageStop": {"stopReason": "tool_use"}}
            yield {"metadata": {"usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5}}}

        return stream()


class BedrockProviderTests(IsolatedAsyncioTestCase):
    async def test_bedrock_omits_empty_inference_fields(self) -> None:
        client = FakeBedrockClient()
        provider = create_bedrock(client=client)
        model = provider.native.language_model("anthropic.claude-3-5-sonnet")

        result = await model.generate(
            ModelGenerateInput(messages=[ModelMessage(role="user", parts=[TextPart(text="hello")])])
        )

        self.assertEqual(result.text, "ok")
        self.assertEqual(client.payloads[0]["messages"][0]["content"][0]["text"], "hello")
        self.assertNotIn("inferenceConfig", client.payloads[0])
        self.assertNotIn("system", client.payloads[0])

    async def test_bedrock_maps_tool_use_and_tool_results(self) -> None:
        class ToolClient(FakeBedrockClient):
            async def converse(self, payload: dict[str, Any]) -> dict[str, Any]:
                self.payloads.append(payload)
                if len(self.payloads) == 1:
                    return {
                        "output": {
                            "message": {
                                "content": [
                                    {
                                        "toolUse": {
                                            "toolUseId": "tooluse_1",
                                            "name": "weather",
                                            "input": {"city": "Madrid"},
                                        }
                                    }
                                ]
                            }
                        },
                        "stopReason": "tool_use",
                    }
                return {
                    "output": {"message": {"content": [{"text": "sunny"}]}},
                    "stopReason": "end_turn",
                }

        client = ToolClient()
        provider = create_bedrock(client=client)

        result = await generate_text(
            model=provider.native.language_model("anthropic.claude-sonnet-4-6"),
            prompt="weather",
            tools={
                "weather": tool(
                    name="weather",
                    schema={"type": "object", "properties": {"city": {"type": "string"}}},
                    execute=lambda input: {"forecast": "sunny", "city": input["city"]},
                )
            },
            tool_choice="required",
            max_steps=2,
        )

        self.assertEqual(result.text, "sunny")
        self.assertEqual(client.payloads[0]["toolConfig"]["toolChoice"], {"any": {}})
        self.assertEqual(client.payloads[0]["toolConfig"]["tools"][0]["toolSpec"]["name"], "weather")
        self.assertEqual(client.payloads[1]["messages"][1]["content"][0]["toolUse"]["name"], "weather")
        self.assertEqual(client.payloads[1]["messages"][2]["content"][0]["toolResult"]["toolUseId"], "tooluse_1")

    async def test_bedrock_stream_maps_text_and_tool_use_events(self) -> None:
        client = FakeBedrockStreamClient()
        provider = create_bedrock(client=client)

        stream = stream_text(
            model=provider.native.language_model("anthropic.claude-sonnet-4-6"),
            prompt="weather",
            tools={
                "weather": tool(
                    name="weather",
                    schema={"type": "object", "properties": {"city": {"type": "string"}}},
                    execute=lambda input: {"forecast": "sunny"},
                )
            },
        )
        events = [event async for event in stream.event_stream()]

        self.assertEqual([event.text_delta for event in events if event.type == "text-delta"], ["hello ", "world"])
        tool_call = next(event.tool_call for event in events if event.type == "tool-call")
        finish = next(event for event in events if event.type == "finish")
        self.assertEqual(tool_call.name, "weather")
        self.assertEqual(tool_call.input, {"city": "Madrid"})
        self.assertEqual(finish.finish_reason, "tool-calls")
        self.assertEqual(finish.usage.total_tokens, 5)


def _user(text: str = "hello") -> ModelGenerateInput:
    return ModelGenerateInput(messages=[ModelMessage(role="user", parts=[TextPart(text=text)])])


class _RecordingClient:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.response = response or {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
        }

    async def converse(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        return self.response

    async def converse_stream(self, payload: dict[str, Any]):
        self.payloads.append(payload)

        async def stream():
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "think "}}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "more", "signature": "sig-1"}}}}
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"redactedContent": b"hidden"}}}}
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "answer"}}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            yield {
                "metadata": {
                    "usage": {
                        "inputTokens": 10,
                        "outputTokens": 4,
                        "cacheReadInputTokens": 6,
                        "cacheWriteInputTokens": 2,
                    }
                }
            }

        return stream()


class _RejectingClient:
    async def converse(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(payload)

    async def converse_stream(self, payload: dict[str, Any]):
        raise AssertionError(payload)


class BedrockReasoningModelTests(IsolatedAsyncioTestCase):
    def _model(self, model_id: str, client: Any | None = None):
        return create_bedrock(client=client or _RecordingClient()).native.language_model(model_id)

    async def test_other_bedrock_models_still_reject_reasoning(self) -> None:
        client = _RejectingClient()
        model = create_bedrock(client=client).native.language_model("anthropic.claude-sonnet-4-6")
        self.assertFalse(model.capabilities.reasoning)
        with self.assertRaisesRegex(UnsupportedFeatureError, 'Provider "bedrock" does not support "reasoning".'):
            await model.generate(ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(effort="low")))

    async def test_sonnet_55_efforts_and_constraints(self) -> None:
        client = _RecordingClient()
        provider = create_bedrock(client=client)
        model = provider.native.language_model("us.anthropic.claude-sonnet-5-5")
        self.assertTrue(model.capabilities.reasoning)
        for effort in ("low", "medium", "high", "xhigh", "max"):
            await model.generate(ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(effort=effort), temperature=0.2))
            fields = client.payloads[-1]["additionalModelRequestFields"]
            self.assertEqual(fields["thinking"], {"type": "adaptive"})
            self.assertEqual(fields["output_config"], {"effort": effort})
            self.assertEqual(client.payloads[-1]["inferenceConfig"]["temperature"], 0.2)
        await model.generate(_user())
        self.assertNotIn("additionalModelRequestFields", client.payloads[-1])

        arn = "arn:aws:bedrock:us-east-1:123:inference-profile/global.anthropic.claude-sonnet-5-5"
        await provider.native.language_model(arn).generate(
            ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(effort="high"))
        )
        self.assertEqual(client.payloads[-1]["modelId"], arn)
        self.assertEqual(client.payloads[-1]["additionalModelRequestFields"]["output_config"]["effort"], "high")

        for model_id in (
            "anthropic.claude-sonnet-5-5",
            "eu.anthropic.claude-sonnet-5-5",
            "arn:aws:bedrock:us-east-1:123:inference-profile/global.anthropic.claude-sonnet-5-5",
        ):
            closed = create_bedrock(client=_RejectingClient()).native.language_model(model_id)
            with self.assertRaisesRegex(UnsupportedFeatureError, 'reasoning effort "none"'):
                await closed.generate(ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(effort="none")))
        rejecting = create_bedrock(client=_RejectingClient()).native.language_model("anthropic.claude-sonnet-5-5")
        with self.assertRaisesRegex(UnsupportedFeatureError, "budget_tokens"):
            await rejecting.generate(ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(budget_tokens=2048)))
        with self.assertRaisesRegex(UnsupportedFeatureError, "minimal"):
            await rejecting.generate(ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(effort="minimal")))
        with self.assertRaisesRegex(ValidationError, "auto"):
            await rejecting.generate(ModelGenerateInput(messages=_user().messages, tool_choice="required"))
        with self.assertRaisesRegex(ValidationError, "auto"):
            await rejecting.generate(
                ModelGenerateInput(messages=_user().messages, tool_choice=ToolChoiceName(tool_name="weather"))
            )

    async def test_haiku_45_budget_interleaved_thinking_and_stream_limit(self) -> None:
        client = _RecordingClient()
        model = create_bedrock(client=client).native.language_model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        self.assertTrue(model.capabilities.reasoning)
        await model.generate(
            ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(budget_tokens=1024), max_tokens=2048)
        )
        fields = client.payloads[-1]["additionalModelRequestFields"]
        self.assertEqual(fields["thinking"], {"type": "enabled", "budget_tokens": 1024})
        self.assertEqual(fields["anthropic_beta"], ["interleaved-thinking-2025-05-14"])

        await model.generate(ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(effort="none"), temperature=0.4))
        self.assertNotIn("additionalModelRequestFields", client.payloads[-1])
        self.assertEqual(client.payloads[-1]["inferenceConfig"]["temperature"], 0.4)

        weather = tool(name="weather", schema={"type": "object", "properties": {}}, execute=lambda input: {"ok": True})
        await model.generate(
            ModelGenerateInput(
                messages=_user().messages,
                tools={"weather": weather},
                reasoning=ReasoningConfig(budget_tokens=4000),
                max_tokens=2000,
                provider_options={"additionalModelRequestFields": {"anthropic_beta": ["tokens-2025"], "top_k": None}},
            )
        )
        merged = client.payloads[-1]["additionalModelRequestFields"]
        self.assertEqual(merged["anthropic_beta"], ["interleaved-thinking-2025-05-14", "tokens-2025"])
        self.assertEqual(merged["thinking"]["budget_tokens"], 4000)

        rejecting = create_bedrock(client=_RejectingClient()).native.language_model(
            "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        with self.assertRaisesRegex(ValidationError, "budget_tokens"):
            await rejecting.generate(ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(effort="low")))
        with self.assertRaisesRegex(ValidationError, "at least 1024"):
            await rejecting.generate(ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(budget_tokens=1023)))
        with self.assertRaisesRegex(ValidationError, "less than max_tokens"):
            await rejecting.generate(
                ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(budget_tokens=2048), max_tokens=2048)
            )
        with self.assertRaisesRegex(ValidationError, "temperature"):
            await rejecting.generate(
                ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(budget_tokens=2048), temperature=1)
            )
        with self.assertRaisesRegex(ValidationError, "temperature"):
            await rejecting.generate(
                ModelGenerateInput(
                    messages=_user().messages,
                    reasoning=ReasoningConfig(budget_tokens=2048),
                    max_tokens=4000,
                    provider_options={"inferenceConfig": {"topP": 0.9}},
                )
            )
        with self.assertRaisesRegex(ValidationError, "auto"):
            await rejecting.generate(
                ModelGenerateInput(
                    messages=_user().messages,
                    reasoning=ReasoningConfig(budget_tokens=2048),
                    max_tokens=4000,
                    tool_choice="required",
                    tools={"weather": weather},
                )
            )
        with self.assertRaisesRegex(ValidationError, "streaming"):
            await rejecting.generate(
                ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(budget_tokens=2048), max_tokens=21334)
            )
        await model.stream(
            ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(budget_tokens=2048), max_tokens=21334)
        )

    async def test_nova_2_lite_reasoning_effort(self) -> None:
        client = _RecordingClient()
        model = create_bedrock(client=client).native.language_model("amazon.nova-2-lite-v1:0")
        self.assertTrue(model.capabilities.reasoning)
        await model.generate(ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(effort="none"), temperature=0.2, max_tokens=20))
        self.assertEqual(client.payloads[-1]["additionalModelRequestFields"], {"reasoningConfig": {"type": "disabled"}})
        for effort in ("low", "medium"):
            await model.generate(
                ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(effort=effort), temperature=0.2)
            )
            self.assertEqual(
                client.payloads[-1]["additionalModelRequestFields"],
                {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": effort}},
            )
        await model.generate(ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(effort="high")))
        self.assertEqual(client.payloads[-1]["additionalModelRequestFields"]["reasoningConfig"]["maxReasoningEffort"], "high")
        self.assertNotIn("inferenceConfig", client.payloads[-1])

        prefixed = create_bedrock(client=client).native.language_model("global.amazon.nova-2-lite-v1:0")
        await prefixed.generate(ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(effort="low")))
        self.assertEqual(client.payloads[-1]["modelId"], "global.amazon.nova-2-lite-v1:0")

        rejecting = create_bedrock(client=_RejectingClient()).native.language_model("us.amazon.nova-2-lite-v1:0")
        for kwargs, pattern in (
            ({"reasoning": ReasoningConfig(effort="high"), "temperature": 0.1}, "high"),
            ({"reasoning": ReasoningConfig(effort="high"), "max_tokens": 100}, "high"),
            ({"reasoning": ReasoningConfig(effort="high"), "provider_options": {"top_k": 10}}, "high"),
            ({"reasoning": ReasoningConfig(effort="high"), "provider_options": {"inferenceConfig": {"topP": 0.5}}}, "high"),
            ({"reasoning": ReasoningConfig(budget_tokens=1024)}, "budget_tokens"),
            ({"reasoning": ReasoningConfig(effort="minimal")}, "minimal"),
            ({"reasoning": ReasoningConfig(effort="max")}, "max"),
        ):
            with self.assertRaisesRegex((ValidationError, UnsupportedFeatureError), pattern):
                await rejecting.generate(ModelGenerateInput(messages=_user().messages, **kwargs))

    async def test_reasoning_blocks_cache_usage_and_replay(self) -> None:
        response = {
            "output": {
                "message": {
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": "[REDACTED]", "signature": "sig-keep"}}},
                        {"text": "done"},
                    ]
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 4,
                "cacheReadInputTokens": 6,
                "cacheWriteInputTokens": 2,
            },
        }
        client = _RecordingClient(response)
        model = create_bedrock(client=client).native.language_model("anthropic.claude-sonnet-5-5")
        result = await generate_text(model=model, prompt="hello", reasoning=ReasoningConfig(effort="high"))
        self.assertEqual(result.text, "done")
        reasoning = next(part for part in result.messages[-1].parts if part.type == "provider-data")
        self.assertEqual(reasoning.data["reasoningText"], {"text": "[REDACTED]", "signature": "sig-keep"})
        self.assertEqual(result.usage.cached_input_tokens, 6)
        self.assertEqual(result.usage.cache_write_tokens, 2)
        self.assertEqual(result.usage.total_tokens, 14)
        self.assertEqual(result.usage.input_tokens, 10)

        await model.generate(
            ModelGenerateInput(
                messages=[
                    ModelMessage(role="user", parts=[TextPart(text="hello")]),
                    ModelMessage(role="assistant", parts=list(result.messages[-1].parts)),
                    ModelMessage(role="user", parts=[TextPart(text="again")]),
                ],
                reasoning=ReasoningConfig(effort="low"),
            )
        )
        replayed = client.payloads[-1]["messages"][1]["content"][0]
        self.assertEqual(
            replayed,
            {"reasoningContent": {"reasoningText": {"text": "[REDACTED]", "signature": "sig-keep"}}},
        )

        stream_client = _RecordingClient()
        stream_model = create_bedrock(client=stream_client).native.language_model("amazon.nova-2-lite-v1:0")
        stream = await stream_model.stream(ModelGenerateInput(messages=_user().messages, reasoning=ReasoningConfig(effort="medium")))
        events = [event async for event in stream]
        deltas = [event.data["text"] for event in events if event.type == "provider-data" and event.data.get("type") == "reasoning_delta"]
        self.assertEqual(deltas, ["think ", "more"])
        final = next(event for event in events if event.type == "provider-data" and event.data.get("type") == "reasoning_content")
        self.assertTrue(final.data["replay"])
        self.assertEqual(final.data["reasoningText"]["signature"], "sig-1")
        self.assertEqual(final.data["reasoningText"]["text"], "think more")
        self.assertEqual(final.data["redactedContent"], "aGlkZGVu")
        finish = next(event for event in events if event.type == "finish")
        self.assertEqual(finish.usage.cached_input_tokens, 6)
        self.assertEqual(finish.usage.cache_write_tokens, 2)
        self.assertEqual(finish.usage.total_tokens, 14)

    async def test_cache_points_and_vision(self) -> None:
        client = _RecordingClient()
        model = create_bedrock(client=client).native.language_model("anthropic.claude-sonnet-5-5")
        weather = tool(
            name="weather",
            schema={"type": "object", "properties": {}},
            execute=lambda input: {"ok": True},
            cache_control={"type": "ephemeral", "ttl": "5m"},
        )
        await model.generate(
            ModelGenerateInput(
                messages=[
                    ModelMessage(
                        role="system",
                        parts=[TextPart(text="policy", provider_metadata={"cache_control": {"type": "ephemeral", "ttl": "1h"}})],
                    ),
                    ModelMessage(
                        role="user",
                        parts=[
                            TextPart(text="look", provider_metadata={"cache_control": {"type": "ephemeral"}}),
                            ImagePart(image="data:image/png;base64,aGVsbG8=", media_type="image/png"),
                        ],
                    ),
                ],
                tools={"weather": weather},
                reasoning=ReasoningConfig(effort="low"),
            )
        )
        payload = client.payloads[-1]
        self.assertEqual(payload["system"][1], {"cachePoint": {"type": "default", "ttl": "1h"}})
        self.assertEqual(payload["messages"][0]["content"][1], {"cachePoint": {"type": "default"}})
        self.assertEqual(payload["messages"][0]["content"][2]["image"]["format"], "png")
        self.assertEqual(payload["toolConfig"]["tools"][1], {"cachePoint": {"type": "default"}})

        nova = create_bedrock(client=_RejectingClient()).native.language_model("amazon.nova-2-lite-v1:0")
        with self.assertRaisesRegex(ValidationError, "5-minute"):
            await nova.generate(
                ModelGenerateInput(
                    messages=[ModelMessage(role="user", parts=[TextPart(text="x", provider_metadata={"cache_control": {"ttl": "1h"}})])]
                )
            )
        with self.assertRaisesRegex(ValidationError, "tools"):
            await nova.generate(
                ModelGenerateInput(
                    messages=_user().messages,
                    tools={"weather": weather},
                )
            )

    def test_catalog_prices_match_published_bedrock_rates(self) -> None:
        sonnet = default_model_catalog.find("bedrock", "anthropic.claude-sonnet-5-5")
        haiku = default_model_catalog.find("bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        nova = default_model_catalog.find("bedrock", "amazon.nova-2-lite-v1:0")
        unprefixed = default_model_catalog.find("bedrock", "anthropic.claude-haiku-4-5-20251001-v1:0")
        assert sonnet is not None and haiku is not None and nova is not None and unprefixed is not None
        self.assertEqual(sonnet.pricing.input_per_1m_tokens, 2.0)  # type: ignore[union-attr]
        self.assertEqual(sonnet.pricing.output_per_1m_tokens, 10.0)  # type: ignore[union-attr]
        self.assertEqual(sonnet.pricing.cached_input_per_1m_tokens, 0.1)  # type: ignore[union-attr]
        self.assertEqual(sonnet.pricing.cache_write_per_1m_tokens, 2.5)  # type: ignore[union-attr]
        self.assertEqual(haiku.pricing.input_per_1m_tokens, 1.10)  # type: ignore[union-attr]
        self.assertEqual(haiku.pricing.output_per_1m_tokens, 5.50)  # type: ignore[union-attr]
        self.assertEqual(haiku.pricing.cached_input_per_1m_tokens, 0.11)  # type: ignore[union-attr]
        self.assertEqual(haiku.pricing.cache_write_per_1m_tokens, 1.375)  # type: ignore[union-attr]
        self.assertEqual(nova.pricing.input_per_1m_tokens, 0.30)  # type: ignore[union-attr]
        self.assertEqual(nova.pricing.output_per_1m_tokens, 2.50)  # type: ignore[union-attr]
        self.assertIsNone(nova.pricing.cached_input_per_1m_tokens)  # type: ignore[union-attr]
        self.assertIsNone(nova.pricing.cache_write_per_1m_tokens)  # type: ignore[union-attr]
        self.assertIsNone(unprefixed.pricing)
        self.assertEqual(unprefixed.replacement_model_id, "anthropic.claude-haiku-5-5")
        self.assertTrue(sonnet.capabilities.reasoning)  # type: ignore[union-attr]
        self.assertFalse(sonnet.capabilities.structured_output)  # type: ignore[union-attr]
        self.assertEqual(sonnet.pricing.source_url, "https://aws.amazon.com/bedrock/pricing/")  # type: ignore[union-attr]
