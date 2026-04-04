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

from zhivex_ai import ImagePart, ToolChoiceName, create_gemini, generate_text, tool
from zhivex_ai.types import ModelGenerateInput, ModelMessage, StructuredOutputConfig, TextPart


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
