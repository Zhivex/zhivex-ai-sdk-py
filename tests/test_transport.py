from __future__ import annotations

import json
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
    stream_sse,
    serialize_ui_message,
    stream_text,
    to_sse_response,
    to_text_stream_response,
    to_ui_message_stream_response,
    to_ui_message_stream,
    to_ui_messages,
)
from zhivex_ai.types import (  # noqa: E402
    CodeExecutionResultPart,
    GenerateResult,
    GeneratedCodePart,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    TokenUsage,
)


async def _collect_async_bytes(body):
    chunks: list[bytes] = []
    async for chunk in body:
        chunks.append(chunk)
    return b"".join(chunks)


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


class _FakeJSONRequest(_FakeRequest):
    async def json(self):
        return json.loads(self._body)


class _FakeReadRequest:
    def __init__(self, body: bytes, content_type: str) -> None:
        self._body = body
        self.headers = {"content-type": content_type}

    async def read(self) -> bytes:
        return self._body


class _FakeResponse:
    def __init__(self, *, status_code: int, lines: list[str], text: str = "") -> None:
        self.status_code = status_code
        self._lines = lines
        self._text = text

    async def text(self) -> str:
        return self._text

    async def iter_lines(self):
        for line in self._lines:
            yield line


class TransportTests(IsolatedAsyncioTestCase):
    async def test_ui_message_roundtrip(self) -> None:
        source = [
            ModelMessage(
                role="assistant",
                parts=[
                    GeneratedCodePart(code="print('hello')", language="python"),
                    CodeExecutionResultPart(output="hello", outcome="ok"),
                ],
            )
        ]
        ui_messages = to_ui_messages(source)
        restored = from_ui_messages(ui_messages)
        self.assertEqual(restored[0].parts[0].type, "generated-code")
        self.assertEqual(restored[0].parts[1].type, "code-result")

    async def test_parse_ui_message_request_accepts_ndjson(self) -> None:
        message = to_ui_messages([create_text_message("user", "hello")])[0]
        request = _FakeRequest(serialize_ui_message(message), "application/x-ndjson")
        parsed = await parse_ui_message_request(request)
        self.assertEqual(parsed[0].id, message.id)

    async def test_parse_ui_message_request_accepts_json_requests_and_bytes(self) -> None:
        message = to_ui_messages([create_text_message("user", "hello")])[0]
        payload = json.dumps([json.loads(serialize_ui_message(message))])

        parsed_json = await parse_ui_message_request(_FakeJSONRequest(payload, "application/json"))
        parsed_bytes = await parse_ui_message_request(payload.encode("utf-8"))

        self.assertEqual(parsed_json[0].id, message.id)
        self.assertEqual(parsed_bytes[0].id, message.id)

    async def test_parse_ui_message_request_accepts_readable_requests(self) -> None:
        message = to_ui_messages([create_text_message("user", "hello")])[0]
        request = _FakeReadRequest(f"{serialize_ui_message(message)}\n".encode("utf-8"), "application/x-ndjson")

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

    async def test_stream_sse_parses_events_and_raises_for_errors(self) -> None:
        response = _FakeResponse(
            status_code=200,
            lines=["event: message", 'data: {"ok": true}', "", "data: plain", ""],
        )

        events = [event async for event in stream_sse(response)]

        self.assertEqual(events, [{"event": "message", "data": '{"ok": true}'}, {"event": None, "data": "plain"}])

        with self.assertRaisesRegex(Exception, "Streaming request failed with status 503"):
            failing = _FakeResponse(status_code=503, lines=[], text="unavailable")
            [event async for event in stream_sse(failing)]

    async def test_sse_and_text_responses_encode_streams(self) -> None:
        async def source():
            yield {"step": 1}
            yield {"step": 2}

        sse_response = to_sse_response(source(), event=lambda payload: f"step-{payload['step']}")
        self.assertIn("text/event-stream", sse_response.headers["content-type"])
        sse_body = await _collect_async_bytes(sse_response.body)
        self.assertIn(b"event: step-1", sse_body)
        self.assertIn(b'data: {"step": 2}', sse_body)

        text_response = to_text_stream_response(stream_text(model=FakeLanguageModel(), prompt="hello"))
        self.assertIn("text/plain", text_response.headers["content-type"])
        text_body = await _collect_async_bytes(text_response.body)
        self.assertEqual(text_body, b"hello")

    async def test_ui_message_stream_response_wraps_sse(self) -> None:
        response = to_ui_message_stream_response(stream_text(model=FakeLanguageModel(), prompt="hello"), message_id="assistant-1")

        self.assertIn("text/event-stream", response.headers["content-type"])
        body = await _collect_async_bytes(response.body)
        self.assertIn(b"event: text-delta", body)
        self.assertIn(b"event: finish", body)
