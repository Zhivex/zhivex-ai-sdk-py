from __future__ import annotations

import json
from collections.abc import AsyncIterable
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import UnsupportedFeatureError, create_anthropic, generate_text, tool
from zhivex_ai.types import ImagePart, ModelGenerateInput, ModelMessage, ReasoningConfig, TextPart, ToolChoiceName


@dataclass
class FakeResponse:
    status_code: int
    payload: Any = None
    body_text: str = ""

    async def json(self) -> Any:
        return self.payload

    async def text(self) -> str:
        return self.body_text or json.dumps(self.payload)

    async def iter_lines(self) -> AsyncIterable[str]:
        for line in self.body_text.splitlines():
            yield line


class AnthropicProviderTests(IsolatedAsyncioTestCase):
    async def test_anthropic_tool_call_roundtrip(self) -> None:
        requests: list[dict[str, Any]] = []
        calls = 0

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            nonlocal calls
            calls += 1
            requests.append(json_body)
            if calls == 1:
                return FakeResponse(
                    status_code=200,
                    payload={
                        "content": [
                            {"type": "thinking", "thinking": "Need math", "signature": "sig-1"},
                            {"type": "tool_use", "id": "tool-1", "name": "math", "input": {"value": 2}},
                        ],
                        "stop_reason": "tool_use",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                )
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": "result is 4"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider("claude-3-5-sonnet"),
            prompt="double 2",
            max_steps=2,
            tools={
                "math": tool(
                    name="math",
                    schema=dict[str, int],
                    execute=lambda input: {"result": input["value"] * 2},
                )
            },
        )
        self.assertEqual(result.text, "result is 4")
        self.assertEqual(result.tool_results[0].tool_name, "math")
        assistant_blocks = requests[1]["messages"][1]["content"]
        self.assertEqual(assistant_blocks[0]["type"], "thinking")
        self.assertEqual(assistant_blocks[1]["type"], "tool_use")

    async def test_anthropic_rejects_forced_tool_choice_with_extended_thinking(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(status_code=200, payload={})

        provider = create_anthropic(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider("claude-sonnet-4-20250514"),
                prompt="double 2",
                tools={
                    "math": tool(
                        name="math",
                        schema=dict[str, int],
                        execute=lambda input: {"result": input["value"] * 2},
                    )
                },
                tool_choice=ToolChoiceName(tool_name="math"),
                reasoning=ReasoningConfig(budget_tokens=1024),
            )
        self.assertEqual(requests, [])

    async def test_anthropic_maps_data_url_images_to_base64_source(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": "looks good"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        model = provider("claude-sonnet-4-20250514")
        result = await model.generate(
            ModelGenerateInput(
                messages=[
                    ModelMessage(
                        role="user",
                        parts=[
                            ImagePart(image="data:image/png;base64,aGVsbG8="),
                            TextPart(text="describe this image"),
                        ],
                    )
                ]
            )
        )
        self.assertEqual(result.text, "looks good")
        image_block = requests[0]["messages"][0]["content"][0]
        self.assertEqual(image_block["source"]["type"], "base64")
        self.assertEqual(image_block["source"]["media_type"], "image/png")
        self.assertEqual(image_block["source"]["data"], "aGVsbG8=")

    async def test_anthropic_stream_includes_thinking_without_null_fields(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                body_text='event: message_stop\ndata: {"stop_reason":"end_turn"}\n',
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        model = provider("claude-sonnet-4-20250514")
        events = []
        async for event in await model.stream(
            ModelGenerateInput(
                messages=[ModelMessage(role="user", parts=[TextPart(text="hello")])],
                reasoning=ReasoningConfig(budget_tokens=2048),
            )
        ):
            events.append(event)

        self.assertEqual(len(events), 1)
        self.assertEqual(requests[0]["thinking"], {"type": "enabled", "budget_tokens": 2048})
        self.assertNotIn("temperature", requests[0])
