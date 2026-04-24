from __future__ import annotations

import json
import os
import sys
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    FilePart,
    ImagePart,
    KIMI_OFFICIAL_TOOL_URIS,
    KimiFormulaClient,
    ReasoningConfig,
    ToolChoiceName,
    UnsupportedFeatureError,
    create_kimi,
    generate_text,
    kimi_formula_toolset,
    stream_text,
    tool,
    user,
)
from zhivex_ai.types import StructuredOutputConfig


@dataclass
class FakeResponse:
    status_code: int
    payload: Any = None
    body_text: str = ""
    body_bytes: bytes = b""
    headers: dict[str, str] | None = None

    async def json(self) -> Any:
        return self.payload

    async def text(self) -> str:
        if self.body_text:
            return self.body_text
        if self.body_bytes:
            return self.body_bytes.decode("utf-8", errors="replace")
        return json.dumps(self.payload)

    async def read(self) -> bytes:
        if self.body_bytes:
            return self.body_bytes
        return (self.body_text or json.dumps(self.payload)).encode("utf-8")

    async def iter_lines(self) -> AsyncIterable[str]:
        for line in self.body_text.splitlines():
            yield line


class KimiProviderConfigTests(TestCase):
    def test_kimi_prefers_moonshot_env_and_falls_back_to_kimi_env(self) -> None:
        with patch.dict(os.environ, {"MOONSHOT_API_KEY": "moon", "KIMI_API_KEY": "kimi", "MOONSHOT_BASE_URL": "https://custom.example/v1"}, clear=True):
            provider = create_kimi()
            model = provider.native.language_model("kimi-k2.6")
            self.assertEqual(model.api_key, "moon")
            self.assertEqual(model.base_url, "https://custom.example/v1")

        with patch.dict(os.environ, {"KIMI_API_KEY": "kimi"}, clear=True):
            provider = create_kimi()
            model = provider.native.language_model("kimi-k2.6")
            self.assertEqual(model.api_key, "kimi")
            self.assertEqual(model.base_url, "https://api.moonshot.ai/v1")


class KimiProviderTests(IsolatedAsyncioTestCase):
    async def test_kimi_uses_chat_completions_and_reasoning_mapping(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            body: Any = None,
            timeout_ms: int | None = None,
            stream: bool = False,
        ):
            requests.append({"url": url, "headers": headers, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={
                    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                },
            )

        provider = create_kimi(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider.native.language_model("kimi-k2.6"),
            prompt="hello",
            max_tokens=32,
            reasoning=ReasoningConfig(effort="none"),
        )

        self.assertEqual(result.text, "ok")
        self.assertEqual(requests[0]["url"], "https://api.moonshot.ai/v1/chat/completions")
        self.assertNotIn("/responses", requests[0]["url"])
        self.assertEqual(requests[0]["headers"]["authorization"], "Bearer test")
        self.assertEqual(requests[0]["json"]["max_completion_tokens"], 32)
        self.assertEqual(requests[0]["json"]["thinking"], {"type": "disabled"})

    async def test_kimi_rejects_k26_sampling_overrides_and_forced_tools_in_thinking_mode(self) -> None:
        provider = create_kimi(api_key="test")
        model = provider.native.language_model("kimi-k2.6")

        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(model=model, prompt="hello", temperature=0.2)

        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=model,
                prompt="hello",
                tools={"weather": tool(name="weather", schema={"type": "object", "properties": {}}, execute=lambda input: {"ok": True})},
                tool_choice=ToolChoiceName("weather"),
            )

    async def test_kimi_maps_structured_output_and_multimodal_inputs(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            body: Any = None,
            timeout_ms: int | None = None,
            stream: bool = False,
        ):
            requests.append({"url": url, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={"choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}, "finish_reason": "stop"}]},
            )

        provider = create_kimi(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("kimi-k2.6"),
            messages=[
                user(
                    [
                        ImagePart(image="data:image/png;base64,aGVsbG8="),
                        FilePart(file_id="file_video", media_type="video/mp4"),
                    ]
                )
            ],
            provider_options={"thinking": {"type": "disabled"}},
            structured_output=StructuredOutputConfig(schema={"type": "object", "properties": {"ok": {"type": "boolean"}}}, mode="native"),
        )

        message = requests[0]["json"]["messages"][0]
        self.assertEqual(requests[0]["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(message["content"][0], {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}})
        self.assertEqual(message["content"][1], {"type": "video_url", "video_url": {"url": "ms://file_video"}})

    async def test_kimi_streaming_emits_chat_completion_deltas(self) -> None:
        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            body: Any = None,
            timeout_ms: int | None = None,
            stream: bool = False,
        ):
            return FakeResponse(
                status_code=200,
                body_text=(
                    'data: {"choices":[{"delta":{"content":"hola"},"finish_reason":null}]}\n\n'
                    'data: {"choices":[{"delta":{"content":" mundo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        provider = create_kimi(api_key="test", fetch=fetch)
        stream = stream_text(model=provider.native.language_model("kimi-k2.6"), prompt="hello", provider_options={"thinking": {"type": "disabled"}})
        events = [event async for event in stream.event_stream()]
        result = await stream.collect()

        self.assertEqual([event.text_delta for event in events if event.type == "text-delta"], ["hola", " mundo"])
        self.assertEqual(result.text, "hola mundo")
        self.assertEqual(result.finish_reason, "stop")

    async def test_kimi_tool_calls_round_trip_through_chat_completions(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            body: Any = None,
            timeout_ms: int | None = None,
            stream: bool = False,
        ):
            requests.append({"url": url, "json": json_body})
            if len(requests) == 1:
                return FakeResponse(
                    status_code=200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "reasoning_content": "need a tool",
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {"name": "weather", "arguments": '{"city":"Madrid"}'},
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                )
            return FakeResponse(
                status_code=200,
                payload={"choices": [{"message": {"role": "assistant", "content": "sunny"}, "finish_reason": "stop"}]},
            )

        provider = create_kimi(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider.native.language_model("kimi-k2.6"),
            prompt="weather",
            tools={
                "weather": tool(
                    name="weather",
                    schema={"type": "object", "properties": {"city": {"type": "string"}}},
                    execute=lambda input: {"city": input["city"], "forecast": "sunny"},
                )
            },
            provider_options={"thinking": {"type": "disabled"}},
            max_steps=2,
        )

        self.assertEqual(result.text, "sunny")
        self.assertEqual(requests[0]["json"]["tools"][0]["function"]["name"], "weather")
        assistant_message = requests[1]["json"]["messages"][1]
        self.assertEqual(assistant_message["reasoning_content"], "need a tool")
        self.assertEqual(requests[1]["json"]["messages"][2]["role"], "tool")

    async def test_kimi_files_batches_tokens_and_formulas_clients(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            body: Any = None,
            timeout_ms: int | None = None,
            stream: bool = False,
        ):
            requests.append({"url": url, "method": method, "json": json_body, "body": body})
            if url.endswith("/files") and method == "POST":
                return FakeResponse(status_code=200, payload={"id": "file_1", "filename": "batch.jsonl", "bytes": 12, "status": "ready"})
            if url.endswith("/batches") and method == "POST":
                return FakeResponse(status_code=200, payload={"id": "batch_1", "status": "validating"})
            if url.endswith("/tokenizers/estimate-token-count"):
                return FakeResponse(status_code=200, payload={"total_tokens": 12})
            if url.endswith("/formulas/moonshot/web-search:latest/tools"):
                return FakeResponse(
                    status_code=200,
                    payload={
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "description": "Search the web",
                                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                                },
                            }
                        ]
                    },
                )
            if url.endswith("/formulas/moonshot/web-search:latest/fibers"):
                return FakeResponse(status_code=200, payload={"status": "succeeded", "context": {"output": {"answer": "found"}}})
            raise AssertionError(url)

        provider = create_kimi(api_key="test", fetch=fetch)
        uploaded = await provider.files().upload(data=b"{}", filename="batch.jsonl", media_type="application/jsonl", purpose="batch")
        batch = await provider.batches().create({"input_file_id": uploaded.id, "endpoint": "/v1/chat/completions", "completion_window": "24h"})
        tokens = await provider.tokens().count(model_id="kimi-k2.6", prompt="hello")
        tools = await provider.formulas().toolset(["moonshot/web-search:latest"])
        output = await tools["search"].execute({"query": "Moonshot"})  # type: ignore[misc, union-attr]
        helper_tools = await kimi_formula_toolset(provider.formulas(), [KIMI_OFFICIAL_TOOL_URIS[1]])

        self.assertEqual(uploaded.id, "file_1")
        self.assertEqual(requests[0]["body"]["data"]["purpose"], "batch")
        self.assertEqual(batch["id"], "batch_1")
        self.assertEqual(tokens.total_tokens, 12)
        self.assertEqual(output, {"answer": "found"})
        self.assertIn("search", helper_tools)
        self.assertIsInstance(provider.formulas(), KimiFormulaClient)
