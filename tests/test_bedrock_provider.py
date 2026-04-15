from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import create_bedrock
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
