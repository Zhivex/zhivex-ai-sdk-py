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

from zhivex_ai import (
    AudioInput,
    ToolChoiceName,
    create_openai,
    generate_grounded_text,
    generate_speech,
    generate_text,
    stream_text,
    tool,
    transcribe_audio,
)


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


class OpenAIProviderTests(IsolatedAsyncioTestCase):
    async def test_openai_maps_chat_completion(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            body: Any = None,
            timeout_ms: int | None,
            stream: bool = False,
        ):
            requests.append({"url": url, "headers": headers, "json": json_body, "body": body, "stream": stream})
            return FakeResponse(
                status_code=200,
                payload={
                    "choices": [{"finish_reason": "stop", "message": {"content": "hello from openai"}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = await generate_text(model=provider("gpt-4o-mini"), prompt="hello")
        self.assertEqual(result.text, "hello from openai")
        self.assertEqual(result.usage.total_tokens, 7)
        self.assertEqual(requests[0]["json"]["model"], "gpt-4o-mini")

    async def test_openai_stream_collects_text(self) -> None:
        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            body: Any = None,
            timeout_ms: int | None,
            stream: bool = False,
        ):
            return FakeResponse(
                status_code=200,
                body_text=(
                    'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                    'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = stream_text(model=provider("gpt-4o-mini"), prompt="hello")
        final = await result.collect()
        self.assertEqual(final.text, "hello world")

    async def test_openai_maps_tool_choice(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            body: Any = None,
            timeout_ms: int | None,
            stream: bool = False,
        ):
            requests.append({"json": json_body})
            return FakeResponse(
                status_code=200,
                payload={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]},
            )

        provider = create_openai(api_key="test", fetch=fetch)
        await generate_text(
            model=provider("gpt-4o-mini"),
            prompt="hello",
            tools={
                "weather": tool(name="weather", schema=dict[str, str], execute=lambda input: {"ok": True})
            },
            tool_choice=ToolChoiceName(tool_name="weather"),
        )
        self.assertEqual(requests[0]["json"]["tool_choice"]["function"]["name"], "weather")

    async def test_openai_transcribes_audio(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            body: Any = None,
            timeout_ms: int | None,
            stream: bool = False,
        ):
            requests.append({"url": url, "body": body})
            return FakeResponse(status_code=200, payload={"text": "transcribed"})

        provider = create_openai(api_key="test", fetch=fetch)
        result = await transcribe_audio(
            model=provider.transcription_model("gpt-4o-mini-transcribe"),
            audio=AudioInput(data=b"abc", media_type="audio/wav", filename="clip.wav"),
        )
        self.assertEqual(result.text, "transcribed")
        self.assertIn("file", requests[0]["body"]["files"])

    async def test_openai_generates_speech(self) -> None:
        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            body: Any = None,
            timeout_ms: int | None,
            stream: bool = False,
        ):
            return FakeResponse(
                status_code=200,
                body_bytes=b"voice-bytes",
                headers={"content-type": "audio/mpeg"},
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = await generate_speech(
            model=provider.speech_model("gpt-4o-mini-tts"),
            input="hello",
        )
        self.assertEqual(result.audio, b"voice-bytes")
        self.assertEqual(result.media_type, "audio/mpeg")

    async def test_openai_generates_grounded_text(self) -> None:
        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            body: Any = None,
            timeout_ms: int | None,
            stream: bool = False,
        ):
            return FakeResponse(
                status_code=200,
                payload={
                    "status": "completed",
                    "output_text": "fresh answer",
                    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                    "citations": [{"url": "https://example.com", "title": "Example"}],
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = await generate_grounded_text(
            model=provider.grounded_language_model("gpt-4o-search-preview"),
            prompt="latest news",
        )
        self.assertEqual(result.text, "fresh answer")
        self.assertEqual(result.sources[0].url, "https://example.com")
