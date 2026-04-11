from __future__ import annotations

import json
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import ImagePart, MCPServerConfig, MCPToolConfig, ToolChoiceName, create_gemini, generate_text, tool
from zhivex_ai import UnsupportedFeatureError, generate_grounded_text
from zhivex_ai.types import ModelGenerateInput, ModelMessage, StructuredOutputConfig, TextPart
from zhivex_ai.errors import ProviderHTTPError


@dataclass
class FakeResponse:
    status_code: int
    payload: Any = None
    body_text: str = ""

    async def json(self) -> Any:
        return self.payload

    async def text(self) -> str:
        return self.body_text or json.dumps(self.payload)


class GeminiProviderTests(IsolatedAsyncioTestCase):
    async def test_gemini_maps_tool_choice_and_usage(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [{"content": {"parts": [{"text": "sunny"}]}, "finishReason": "STOP"}],
                    "usageMetadata": {
                        "promptTokenCount": 11,
                        "candidatesTokenCount": 7,
                        "totalTokenCount": 18,
                    },
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider("gemini-2.5-flash"),
            prompt="weather",
            tools={"weather": tool(name="weather", schema=dict[str, str], execute=lambda input: {"ok": True})},
            tool_choice=ToolChoiceName(tool_name="weather"),
            structured_output=StructuredOutputConfig(schema=dict[str, str], mode="native"),
        )

        self.assertEqual(result.text, "sunny")
        self.assertEqual(result.usage.total_tokens, 18)
        self.assertEqual(
            requests[0]["toolConfig"]["functionCallingConfig"],
            {"mode": "ANY", "allowedFunctionNames": ["weather"]},
        )
        self.assertEqual(requests[0]["generationConfig"]["responseMimeType"], "application/json")
        self.assertIn("responseJsonSchema", requests[0]["generationConfig"])

    async def test_gemini_omits_null_fields_and_strips_data_url_prefix(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}],
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        model = provider("gemini-2.5-flash")
        await model.generate(
            ModelGenerateInput(
                messages=[
                    ModelMessage(
                        role="user",
                        parts=[
                            ImagePart(image="data:image/png;base64,aGVsbG8="),
                            TextPart(text="describe"),
                        ],
                    )
                ]
            )
        )

        inline_data = requests[0]["contents"][0]["parts"][0]["inlineData"]
        self.assertEqual(inline_data["mimeType"], "image/png")
        self.assertEqual(inline_data["data"], "aGVsbG8=")
        self.assertNotIn("toolConfig", requests[0])
        self.assertNotIn("generationConfig", requests[0])

    async def test_gemini_maps_google_search_provider_option(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [{"content": {"parts": [{"text": "fresh answer"}]}, "finishReason": "STOP"}],
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider("gemini-2.5-flash"),
            prompt="latest news",
            provider_options={"google_search": True},
        )

        self.assertEqual(result.text, "fresh answer")
        self.assertEqual(requests[0]["tools"], [{"googleSearch": {}}])
        self.assertNotIn("google_search", requests[0])

    async def test_gemini_grounded_language_model_returns_sources(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "grounded answer"}]},
                            "finishReason": "STOP",
                            "groundingMetadata": {
                                "groundingChunks": [
                                    {"web": {"uri": "https://example.com/1", "title": "Example 1"}},
                                    {"web": {"uri": "https://example.com/2", "title": "Example 2", "text": "snippet"}},
                                ]
                            },
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 3,
                        "totalTokenCount": 8,
                    },
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        result = await generate_grounded_text(
            model=provider.grounded_language_model("gemini-2.5-flash"),
            prompt="latest news",
        )

        self.assertEqual(result.text, "grounded answer")
        self.assertEqual(result.sources[0].url, "https://example.com/1")
        self.assertEqual(result.sources[1].snippet, "snippet")
        self.assertEqual(requests[0]["tools"], [{"googleSearch": {}}])

    async def test_gemini_reports_builtin_search_tool_without_opt_in(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [
                        {
                            "content": {"parts": [{"functionCall": {"name": "search", "args": {"query": "Apollo"}}}]},
                            "finishReason": "STOP",
                        }
                    ],
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError) as context:
            await generate_text(model=provider("gemini-3-flash-preview"), prompt="Research Apollo.")

        self.assertIn("google_search", str(context.exception))

    async def test_gemini_normalizes_mcp_tool_schema(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}],
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        await generate_text(
            model=provider("gemini-2.5-flash"),
            prompt="hello",
            tools={
                "fs_read_file": tool(
                    name="fs_read_file",
                    schema={
                        "type": "object",
                        "title": "Read File Input",
                        "properties": {
                            "path": {"type": "string", "title": "Path"},
                            "head": {"type": "integer", "default": 20},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                    source="mcp",
                    mcp_config=MCPToolConfig(
                        server=MCPServerConfig(transport="stdio", name="fs", command="npx"),
                        tool_name="read_file",
                    ),
                )
            },
        )

        parameters = requests[0]["tools"][0]["functionDeclarations"][0]["parameters"]
        self.assertEqual(parameters["type"], "object")
        self.assertEqual(parameters["required"], ["path"])
        self.assertNotIn("title", parameters)
        self.assertNotIn("additionalProperties", parameters)
        self.assertNotIn("default", parameters["properties"]["head"])
        self.assertEqual(parameters["properties"]["head"]["type"], "integer")

    async def test_gemini_http_error_includes_response_body(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(status_code=400, body_text='{"error":{"message":"Bad schema: additionalProperties"}}')

        provider = create_gemini(api_key="test", fetch=fetch)
        with self.assertRaises(ProviderHTTPError) as context:
            await generate_text(model=provider("gemini-2.5-flash"), prompt="hello")

        self.assertIn("Response body:", str(context.exception))
        self.assertIn("Bad schema", str(context.exception))

    async def test_gemini_preserves_thought_signature_across_tool_loop(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            if len(requests) == 1:
                return FakeResponse(
                    status_code=200,
                    payload={
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {"name": "weather", "args": {"city": "Madrid"}},
                                            "thoughtSignature": "sig-123",
                                        }
                                    ]
                                },
                                "finishReason": "STOP",
                            }
                        ]
                    },
                )
            return FakeResponse(
                status_code=200,
                payload={
                    "candidates": [{"content": {"parts": [{"text": "sunny"}]}, "finishReason": "STOP"}],
                },
            )

        provider = create_gemini(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider("gemini-2.5-flash"),
            prompt="weather",
            max_steps=2,
            tools={
                "weather": tool(
                    name="weather",
                    schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                    execute=lambda input: {"forecast": "sunny"},
                )
            },
        )

        self.assertEqual(result.text, "sunny")
        second_request_parts = requests[1]["contents"][1]["parts"]
        function_call_part = next(part for part in second_request_parts if "functionCall" in part)
        self.assertEqual(function_call_part["thought_signature"], "sig-123")
