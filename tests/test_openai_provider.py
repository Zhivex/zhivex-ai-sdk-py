from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterable
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    AudioInput,
    MCPServerConfig,
    MCPToolConfig,
    UnsupportedFeatureError,
    ToolChoiceName,
    ValidationError,
    create_openai,
    create_openrouter,
    create_qwen,
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


class WeatherToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str


class OpenAIProviderTests(IsolatedAsyncioTestCase):
    async def test_openai_maps_responses_request(self) -> None:
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
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hello from openai"}],
                        }
                    ],
                    "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = await generate_text(model=provider("gpt-4o-mini"), prompt="hello")
        self.assertEqual(result.text, "hello from openai")
        self.assertEqual(result.usage.total_tokens, 7)
        self.assertEqual(requests[0]["json"]["model"], "gpt-4o-mini")
        self.assertEqual(requests[0]["url"], "https://api.openai.com/v1/responses")
        self.assertEqual(requests[0]["json"]["input"][0]["content"][0]["text"], "hello")
        self.assertNotIn("max_tokens", requests[0]["json"])
        self.assertNotIn("max_completion_tokens", requests[0]["json"])

    async def test_openai_uses_max_output_tokens(self) -> None:
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
            requests.append({"url": url, "headers": headers, "json": json_body, "body": body, "stream": stream})
            return FakeResponse(
                status_code=200,
                payload={
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hello from gpt-5"}],
                        }
                    ],
                    "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = await generate_text(model=provider("gpt-5-nano"), prompt="hello", max_tokens=123)
        self.assertEqual(result.text, "hello from gpt-5")
        self.assertEqual(requests[0]["json"]["max_output_tokens"], 123)
        self.assertNotIn("max_tokens", requests[0]["json"])
        self.assertNotIn("max_completion_tokens", requests[0]["json"])

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
                    'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
                    'data: {"type":"response.output_text.delta","delta":" world"}\n\n'
                    'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}}\n\n'
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
                payload={
                    "status": "completed",
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        await generate_text(
            model=provider("gpt-4o-mini"),
            prompt="hello",
            tools={
                "weather": tool(name="weather", schema=WeatherToolInput, execute=lambda input: {"ok": True})
            },
            tool_choice=ToolChoiceName(tool_name="weather"),
        )
        self.assertEqual(requests[0]["json"]["tool_choice"]["name"], "weather")
        self.assertEqual(requests[0]["json"]["tools"][0]["name"], "weather")
        self.assertFalse(requests[0]["json"]["tools"][0]["parameters"]["additionalProperties"])

    async def test_openai_rejects_non_strict_tool_schema_before_request(self) -> None:
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
            requests.append({"url": url, "json": json_body})
            return FakeResponse(status_code=200, payload={})

        provider = create_openai(api_key="test", fetch=fetch)
        with self.assertRaises(ValidationError) as context:
            await generate_text(
                model=provider("gpt-4o-mini"),
                prompt="hello",
                tools={
                    "weather": tool(name="weather", schema=dict[str, str], execute=lambda input: {"ok": True})
                },
                tool_choice=ToolChoiceName(tool_name="weather"),
            )

        self.assertIn("strict mode", str(context.exception))
        self.assertIn("additionalProperties", str(context.exception))
        self.assertEqual(requests, [])

    async def test_openai_normalizes_mcp_tool_schema_for_strict_mode(self) -> None:
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
            requests.append({"url": url, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={
                    "status": "completed",
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        await generate_text(
            model=provider("gpt-4o-mini"),
            prompt="hello",
            tools={
                "fs_read_file": tool(
                    name="fs_read_file",
                    schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "head": {"type": "integer"},
                            "tail": {"type": "integer"},
                        },
                        "required": ["path"],
                    },
                    source="mcp",
                    mcp_config=MCPToolConfig(
                        server=MCPServerConfig(transport="stdio", name="fs", command="npx"),
                        tool_name="read_file",
                    ),
                )
            },
        )
        parameters = requests[0]["json"]["tools"][0]["parameters"]
        self.assertFalse(parameters["additionalProperties"])
        self.assertEqual(parameters["required"], ["path", "head", "tail"])
        self.assertEqual(parameters["properties"]["path"]["type"], "string")
        self.assertEqual(
            parameters["properties"]["head"]["anyOf"],
            [{"type": "integer"}, {"type": "null"}],
        )
        self.assertEqual(
            parameters["properties"]["tail"]["anyOf"],
            [{"type": "integer"}, {"type": "null"}],
        )

    async def test_openai_serializes_failed_tool_results_without_slots_error(self) -> None:
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
            requests.append({"url": url, "json": json_body})
            if len(requests) == 1:
                return FakeResponse(
                    status_code=200,
                    payload={
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "weather",
                                "arguments": '{"city":"Madrid"}',
                            }
                        ],
                    },
                )
            return FakeResponse(
                status_code=200,
                payload={
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "tool failed"}],
                        }
                    ],
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider("gpt-4o-mini"),
            prompt="hello",
            max_steps=2,
            tools={
                "weather": tool(
                    name="weather",
                    schema=WeatherToolInput,
                    execute=lambda input: (_ for _ in ()).throw(RuntimeError("boom")),
                )
            },
        )

        self.assertEqual(result.text, "tool failed")
        self.assertEqual(requests[1]["json"]["input"][-1]["output"], '{"message": "boom"}')

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

    async def test_openrouter_rejects_required_tool_choice(self) -> None:
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
            requests.append({"url": url, "json": json_body})
            return FakeResponse(status_code=200, payload={})

        provider = create_openrouter(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider("openai/o4-mini"),
                prompt="hello",
                tools={"weather": tool(name="weather", schema=WeatherToolInput, execute=lambda input: {"ok": True})},
                tool_choice="required",
            )

        self.assertEqual(requests, [])

    async def test_openrouter_generates_speech_via_chat_audio_stream(self) -> None:
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
            requests.append({"url": url, "json": json_body, "stream": stream})
            return FakeResponse(
                status_code=200,
                body_text=(
                    'data: {"choices":[{"delta":{"audio":{"data":"'
                    + base64.b64encode(b"hello ").decode("ascii")
                    + '"}}}]}\n\n'
                    'data: {"choices":[{"delta":{"audio":{"data":"'
                    + base64.b64encode(b"world").decode("ascii")
                    + '"}}}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        provider = create_openrouter(api_key="test", fetch=fetch)
        result = await generate_speech(
            model=provider.speech_model("openai/gpt-4o-mini-tts"),
            input="hello",
            voice="alloy",
        )

        self.assertEqual(result.audio, b"hello world")
        self.assertEqual(result.media_type, "audio/wav")
        self.assertEqual(requests[0]["url"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertTrue(requests[0]["stream"])
        self.assertEqual(requests[0]["json"]["modalities"], ["text", "audio"])
        self.assertEqual(requests[0]["json"]["audio"], {"voice": "alloy", "format": "wav"})

    async def test_qwen_reports_tools_as_unsupported_for_this_adapter(self) -> None:
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
            requests.append({"url": url, "json": json_body})
            return FakeResponse(status_code=200, payload={})

        provider = create_qwen(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider("qwen-plus"),
                prompt="hello",
                tools={"weather": tool(name="weather", schema=WeatherToolInput, execute=lambda input: {"ok": True})},
            )

        self.assertEqual(requests, [])

    async def test_qwen_generates_speech(self) -> None:
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
            requests.append({"url": url, "method": method, "json": json_body})
            if method == "GET":
                return FakeResponse(
                    status_code=200,
                    body_bytes=b"qwen-voice",
                    headers={"content-type": "audio/wav"},
                )
            return FakeResponse(
                status_code=200,
                payload={
                    "output": {
                        "finish_reason": "stop",
                        "audio": {
                            "url": "https://files.example.com/audio.wav",
                        },
                    }
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        result = await generate_speech(
            model=provider.speech_model("qwen3-tts-flash"),
            input="hello",
        )

        self.assertEqual(result.audio, b"qwen-voice")
        self.assertEqual(result.media_type, "audio/wav")
        self.assertIn("/api/v1/services/aigc/multimodal-generation/generation", requests[0]["url"])
        self.assertEqual(requests[0]["json"]["input"]["voice"], "Cherry")
        self.assertEqual(requests[1]["method"], "GET")
        self.assertEqual(requests[1]["url"], "https://files.example.com/audio.wav")
