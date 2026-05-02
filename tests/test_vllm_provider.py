from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    AudioFrame,
    AudioInput,
    RealtimeConnectOptions,
    RealtimeResponseCompletedEvent,
    RealtimeSessionConfig,
    RealtimeTranscriptEvent,
    create_vllm,
    embed,
    generate_object,
    generate_text,
    stream_text,
    tool,
    transcribe_audio,
)
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


class FakeRealtimeConnection:
    def __init__(self, incoming: list[dict[str, Any]]) -> None:
        self._incoming = list(incoming)
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def recv_json(self) -> Any:
        if self.closed or not self._incoming:
            return None
        return self._incoming.pop(0)

    async def close(self) -> None:
        self.closed = True


class WeatherToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str


class Forecast(BaseModel):
    city: str
    summary: str


class VllmProviderTests(IsolatedAsyncioTestCase):
    def test_create_vllm_uses_local_defaults(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            provider = create_vllm()
        language_model = provider("NousResearch/Meta-Llama-3-8B-Instruct")
        embedding_model = provider.embedding_model("BAAI/bge-small-en-v1.5")

        self.assertEqual(provider.name, "vllm")
        self.assertEqual(provider.tier, "portable")
        self.assertTrue(provider.portable_support.portable_badge)
        self.assertTrue(provider.portable_support.text_generation)
        self.assertTrue(provider.portable_support.streaming)
        self.assertTrue(provider.portable_support.structured_output)
        self.assertTrue(provider.portable_support.tools)
        self.assertTrue(provider.portable_support.embeddings)
        self.assertTrue(provider.portable_support.transcription)
        self.assertFalse(provider.portable_support.grounding)
        self.assertFalse(provider.portable_support.speech)
        self.assertEqual(language_model.native_model.base_url, "http://localhost:8000/v1")
        self.assertEqual(language_model.native_model.api_key, "vllm")
        self.assertEqual(embedding_model.native_model.base_url, "http://localhost:8000/v1")
        self.assertEqual(embedding_model.native_model.api_key, "vllm")

    def test_create_vllm_uses_explicit_and_env_configuration(self) -> None:
        with patch.dict("os.environ", {"VLLM_API_KEY": "env-key", "VLLM_BASE_URL": "https://vllm.example/v1"}):
            env_provider = create_vllm()
            explicit_provider = create_vllm(api_key="explicit-key", base_url="http://localhost:9000/v1/")

        self.assertEqual(env_provider.native.language_model("model").api_key, "env-key")
        self.assertEqual(env_provider.native.language_model("model").base_url, "https://vllm.example/v1")
        self.assertEqual(explicit_provider.native.language_model("model").api_key, "explicit-key")
        self.assertEqual(explicit_provider.native.language_model("model").base_url, "http://localhost:9000/v1")

    async def test_vllm_portable_text_generation_uses_responses_api(self) -> None:
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
            requests.append({"url": url, "headers": headers, "json": json_body, "stream": stream})
            return FakeResponse(
                status_code=200,
                payload={
                    "status": "completed",
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "vllm ok"}]}],
                    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                },
            )

        provider = create_vllm(fetch=fetch)
        result = await generate_text(model=provider("meta-llama/Llama-3.1-8B-Instruct"), prompt="hello")

        self.assertEqual(result.text, "vllm ok")
        self.assertEqual(result.usage.total_tokens, 5)
        self.assertEqual(requests[0]["url"], "http://localhost:8000/v1/responses")
        self.assertEqual(requests[0]["json"]["model"], "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(requests[0]["headers"]["authorization"], "Bearer vllm")
        self.assertFalse(requests[0]["stream"])

    async def test_vllm_streaming_emits_text_deltas_and_finish_event(self) -> None:
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
            requests.append({"url": url, "stream": stream})
            return FakeResponse(
                status_code=200,
                body_text=(
                    'data: {"type":"response.output_text.delta","delta":"hola"}\n\n'
                    'data: {"type":"response.output_text.delta","delta":" mundo"}\n\n'
                    'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        provider = create_vllm(fetch=fetch)
        result = stream_text(model=provider("llama"), prompt="hola")
        events = [event async for event in result.event_stream()]
        final = await result.collect()

        self.assertTrue(requests[0]["stream"])
        self.assertEqual(final.text, "hola mundo")
        self.assertEqual([event.text_delta for event in events if isinstance(event, StreamTextDeltaEvent)], ["hola", " mundo"])
        self.assertEqual(next(event for event in events if isinstance(event, StreamFinishEvent)).usage.total_tokens, 6)

    async def test_vllm_structured_output_uses_json_schema(self) -> None:
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
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": '{"city":"Buenos Aires","summary":"sunny"}'}]}],
                },
            )

        provider = create_vllm(fetch=fetch)
        result = await generate_object(model=provider("llama"), prompt="return weather", schema=Forecast)

        self.assertEqual(result.object.city, "Buenos Aires")
        self.assertEqual(requests[0]["json"]["text"]["format"]["type"], "json_schema")

    async def test_vllm_tool_calls_round_trip_function_outputs(self) -> None:
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
                return FakeResponse(status_code=200, payload={"status": "completed", "output": [{"type": "function_call", "call_id": "call_1", "name": "weather", "arguments": '{"city":"Buenos Aires"}'}]})
            return FakeResponse(status_code=200, payload={"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "sunny"}]}]})

        provider = create_vllm(fetch=fetch)
        result = await generate_text(
            model=provider("llama"),
            prompt="weather?",
            max_steps=2,
            tools={"weather": tool(name="weather", schema=WeatherToolInput, execute=lambda input: {"forecast": f"sunny in {input.city}"})},
        )

        self.assertEqual(result.text, "sunny")
        self.assertEqual(requests[0]["json"]["tools"][0]["name"], "weather")
        self.assertEqual(requests[1]["json"]["input"][-1]["type"], "function_call_output")

    async def test_vllm_embeddings_use_openai_compatible_endpoint(self) -> None:
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
            return FakeResponse(status_code=200, payload={"data": [{"embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 4, "total_tokens": 4}})

        provider = create_vllm(fetch=fetch)
        result = await embed(model=provider.embedding_model("bge"), value="hello")

        self.assertEqual(result.embeddings, [[0.1, 0.2]])
        self.assertEqual(requests[0]["url"], "http://localhost:8000/v1/embeddings")

    async def test_vllm_transcribes_audio(self) -> None:
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
            requests.append({"url": url, "body": body})
            return FakeResponse(status_code=200, payload={"text": "transcribed"})

        provider = create_vllm(fetch=fetch)
        result = await transcribe_audio(
            model=provider.transcription_model("openai/whisper-large-v3-turbo"),
            audio=AudioInput(data=b"abc", media_type="audio/wav", filename="clip.wav"),
        )

        self.assertEqual(result.text, "transcribed")
        self.assertEqual(requests[0]["url"], "http://localhost:8000/v1/audio/transcriptions")
        self.assertIn("file", requests[0]["body"]["files"])

    async def test_vllm_realtime_derives_websocket_url_and_parses_asr_events(self) -> None:
        connections: list[FakeRealtimeConnection] = []
        connection_meta: list[dict[str, Any]] = []

        async def connection_factory(url: str, headers: dict[str, str], options: RealtimeConnectOptions | None):
            connection_meta.append({"url": url, "headers": headers, "options": options})
            connection = FakeRealtimeConnection(
                [
                    {"type": "transcription.delta", "delta": "hola"},
                    {"type": "transcription.done", "text": "hola mundo"},
                    {"type": "response.done"},
                ]
            )
            connections.append(connection)
            return connection

        provider = create_vllm(realtime_connection_factory=connection_factory)
        session = await provider.native.realtime_model("openai/whisper-large-v3-turbo").connect(
            RealtimeSessionConfig(input_audio_media_type="audio/pcm", input_sample_rate_hz=16000),
            RealtimeConnectOptions(timeout_ms=500),
        )

        await session.send_audio(AudioFrame(data=b"\x00\x01", media_type="audio/pcm", sample_rate_hz=16000, is_final=True))
        events = []
        async for event in session.event_stream():
            events.append(event)
            if isinstance(event, RealtimeResponseCompletedEvent):
                break
        await session.aclose()

        self.assertIn("/v1/realtime?model=openai%2Fwhisper-large-v3-turbo", connection_meta[0]["url"])
        self.assertEqual(connection_meta[0]["headers"]["authorization"], "Bearer vllm")
        self.assertEqual(connections[0].sent[0]["type"], "session.update")
        self.assertEqual(connections[0].sent[1]["type"], "input_audio_buffer.append")
        self.assertEqual(connections[0].sent[2]["type"], "input_audio_buffer.commit")
        self.assertTrue(any(isinstance(event, RealtimeTranscriptEvent) and event.text == "hola" and not event.is_final for event in events))
        self.assertTrue(any(isinstance(event, RealtimeTranscriptEvent) and event.text == "hola mundo" and event.is_final for event in events))
