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
from zhivex_ai.types import ModelGenerateInput, ModelMessage, TextPart


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
