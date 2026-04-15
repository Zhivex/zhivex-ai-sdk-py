from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import UnsupportedFeatureError, create_ollama, embed, generate_object, generate_text, stream_text, tool
from zhivex_ai.types import StreamFinishEvent, StreamTextDeltaEvent


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


class Forecast(BaseModel):
    city: str
    summary: str


class OllamaProviderTests(IsolatedAsyncioTestCase):
    def test_create_ollama_uses_local_compat_defaults(self) -> None:
        provider = create_ollama()
        language_model = provider.native.language_model("llama3.2")
        embedding_model = provider.native.embedding_model("nomic-embed-text")

        self.assertEqual(provider.name, "ollama")
        self.assertEqual(provider.tier, "compatibility")
        self.assertFalse(provider.portable_support.portable_badge)
        self.assertTrue(provider.portable_support.text_generation)
        self.assertTrue(provider.portable_support.streaming)
        self.assertTrue(provider.portable_support.structured_output)
        self.assertTrue(provider.portable_support.tools)
        self.assertTrue(provider.portable_support.embeddings)
        self.assertTrue(provider.portable_support.retrieval)
        self.assertFalse(provider.portable_support.grounding)
        self.assertFalse(provider.portable_support.transcription)
        self.assertFalse(provider.portable_support.speech)
        self.assertEqual(language_model.base_url, "http://localhost:11434/v1")
        self.assertEqual(language_model.api_key, "ollama")
        self.assertEqual(embedding_model.base_url, "http://localhost:11434/v1")
        self.assertEqual(embedding_model.api_key, "ollama")

    def test_ollama_portable_model_construction_is_rejected(self) -> None:
        provider = create_ollama()

        with self.assertRaises(UnsupportedFeatureError):
            provider("llama3.2")

    async def test_ollama_native_text_generation_uses_responses_api(self) -> None:
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
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body, "stream": stream})
            return FakeResponse(
                status_code=200,
                payload={
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "local ok"}],
                        }
                    ],
                    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                },
            )

        provider = create_ollama(fetch=fetch)
        result = await generate_text(
            model=provider.native.language_model("llama3.2"),
            prompt="hello",
        )

        self.assertEqual(result.text, "local ok")
        self.assertEqual(result.usage.total_tokens, 5)
        self.assertEqual(requests[0]["url"], "http://localhost:11434/v1/responses")
        self.assertEqual(requests[0]["json"]["model"], "llama3.2")
        self.assertEqual(requests[0]["json"]["input"][0]["content"][0]["text"], "hello")
        self.assertEqual(requests[0]["headers"]["authorization"], "Bearer ollama")
        self.assertFalse(requests[0]["stream"])

    async def test_ollama_streaming_emits_text_deltas_and_finish_event(self) -> None:
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
            requests.append({"url": url, "json": json_body, "stream": stream})
            return FakeResponse(
                status_code=200,
                body_text=(
                    'data: {"type":"response.output_text.delta","delta":"hola"}\n\n'
                    'data: {"type":"response.output_text.delta","delta":" mundo"}\n\n'
                    'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        provider = create_ollama(fetch=fetch)
        result = stream_text(model=provider.native.language_model("llama3.2"), prompt="hola")
        events = [event async for event in result.event_stream()]
        final = await result.collect()

        self.assertTrue(requests[0]["stream"])
        self.assertEqual(final.text, "hola mundo")
        self.assertEqual(
            [event.text_delta for event in events if isinstance(event, StreamTextDeltaEvent)],
            ["hola", " mundo"],
        )
        finish_event = next(event for event in events if isinstance(event, StreamFinishEvent))
        self.assertEqual(finish_event.finish_reason, "stop")
        self.assertEqual(finish_event.usage.total_tokens, 6)

    async def test_ollama_native_structured_output_uses_json_schema(self) -> None:
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
                payload={
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": '{"city":"Buenos Aires","summary":"sunny"}'}],
                        }
                    ],
                    "usage": {"input_tokens": 8, "output_tokens": 7, "total_tokens": 15},
                },
            )

        provider = create_ollama(fetch=fetch)
        result = await generate_object(
            model=provider.native.language_model("llama3.2"),
            prompt="return weather",
            schema=Forecast,
        )

        self.assertEqual(result.object.city, "Buenos Aires")
        self.assertEqual(result.object.summary, "sunny")
        self.assertEqual(result.object_mode, "native")
        self.assertEqual(requests[0]["json"]["text"]["format"]["type"], "json_schema")
        self.assertEqual(requests[0]["json"]["text"]["format"]["name"], "response")

    async def test_ollama_native_tool_calls_round_trip_function_outputs(self) -> None:
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
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "weather",
                                "arguments": '{"city":"Buenos Aires"}',
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
                            "content": [{"type": "output_text", "text": "sunny and warm"}],
                        }
                    ],
                    "usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
                },
            )

        provider = create_ollama(fetch=fetch)
        result = await generate_text(
            model=provider.native.language_model("llama3.2"),
            prompt="weather?",
            max_steps=2,
            tools={
                "weather": tool(
                    name="weather",
                    schema=WeatherToolInput,
                    execute=lambda input: {"forecast": f"sunny in {input.city}"},
                )
            },
        )

        self.assertEqual(result.text, "sunny and warm")
        self.assertEqual(len(result.tool_results), 1)
        self.assertEqual(result.tool_results[0].tool_name, "weather")
        self.assertEqual(requests[0]["json"]["tools"][0]["name"], "weather")
        self.assertEqual(requests[1]["json"]["input"][-1]["type"], "function_call_output")
        self.assertEqual(requests[1]["json"]["input"][-1]["call_id"], "call_1")
        self.assertEqual(
            json.loads(requests[1]["json"]["input"][-1]["output"]),
            {"forecast": "sunny in Buenos Aires"},
        )

    async def test_ollama_native_embeddings_use_openai_compatible_endpoint(self) -> None:
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
                payload={
                    "data": [{"embedding": [0.1, 0.2, 0.3]}],
                    "usage": {"prompt_tokens": 4, "total_tokens": 4},
                },
            )

        provider = create_ollama(fetch=fetch)
        result = await embed(
            model=provider.native.embedding_model("nomic-embed-text"),
            value="hello",
        )

        self.assertEqual(result.embeddings, [[0.1, 0.2, 0.3]])
        self.assertEqual(result.usage.total_tokens, 4)
        self.assertEqual(requests[0]["url"], "http://localhost:11434/v1/embeddings")
        self.assertEqual(requests[0]["json"], {"model": "nomic-embed-text", "input": ["hello"]})

    def test_ollama_does_not_expose_unimplemented_native_extras(self) -> None:
        provider = create_ollama()

        with self.assertRaises(AttributeError):
            provider.native.speech_model("tts")
        with self.assertRaises(AttributeError):
            provider.native.transcription_model("whisper")
        with self.assertRaises(AttributeError):
            provider.native.realtime_model("realtime")
        with self.assertRaises(AttributeError):
            provider.files()
        with self.assertRaises(AttributeError):
            provider.images()
        with self.assertRaises(AttributeError):
            provider.uploads()
        with self.assertRaises(AttributeError):
            provider.responses()
        with self.assertRaises(AttributeError):
            provider.conversations()
