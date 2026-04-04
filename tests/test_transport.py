from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    create_text_message,
    create_ui_message_json_response,
    create_ui_message_lines_response,
    from_ui_messages,
    parse_ui_message_request,
    serialize_ui_message,
    stream_text,
    to_ui_message_stream,
    to_ui_messages,
)
from zhivex_ai.types import (  # noqa: E402
    GenerateResult,
    ModelCapabilities,
    ModelGenerateInput,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    TokenUsage,
)


class FakeLanguageModel:
    provider = "test"
    model_id = "model"
    capabilities = ModelCapabilities(
        streaming=True,
        tools=False,
        structured_output=False,
        json_mode=False,
        tool_choice=False,
        parallel_tool_calls=False,
        vision=False,
        files=False,
        audio_input=False,
        audio_output=False,
        embeddings=False,
        reasoning=False,
        web_search=False,
    )

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        return GenerateResult(text="ignored")

    async def stream(self, input: ModelGenerateInput):
        async def generator():
            yield StreamTextDeltaEvent(text_delta="hello")
            yield StreamFinishEvent(
                finish_reason="stop",
                usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        return generator()


class _FakeRequest:
    def __init__(self, body: str, content_type: str) -> None:
        self._body = body
        self.headers = {"content-type": content_type}

    async def text(self) -> str:
        return self._body


class TransportTests(IsolatedAsyncioTestCase):
    async def test_ui_message_roundtrip(self) -> None:
        source = [create_text_message("user", "hello")]
        ui_messages = to_ui_messages(source)
        restored = from_ui_messages(ui_messages)
        self.assertEqual(restored[0].parts[0].text, "hello")

    async def test_parse_ui_message_request_accepts_ndjson(self) -> None:
        message = to_ui_messages([create_text_message("user", "hello")])[0]
        request = _FakeRequest(serialize_ui_message(message), "application/x-ndjson")
        parsed = await parse_ui_message_request(request)
        self.assertEqual(parsed[0].id, message.id)

    async def test_create_ui_responses_set_content_types(self) -> None:
        messages = to_ui_messages([create_text_message("user", "hello")])
        json_response = create_ui_message_json_response(messages)
        lines_response = create_ui_message_lines_response(messages)
        self.assertIn("application/json", json_response.headers["content-type"])
        self.assertIn("application/x-ndjson", lines_response.headers["content-type"])

    async def test_to_ui_message_stream_emits_finish_chunk(self) -> None:
        result = stream_text(model=FakeLanguageModel(), prompt="hello")
        chunks = []
        async for chunk in to_ui_message_stream(result, "assistant-1"):
            chunks.append(chunk)
        self.assertEqual(chunks[0].type, "text-delta")
        self.assertEqual(chunks[-1].type, "finish")
