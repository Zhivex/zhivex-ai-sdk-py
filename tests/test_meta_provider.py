from __future__ import annotations

import json
import os
import sys
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    ConfigurationError,
    FilePart,
    ImagePart,
    ProviderHTTPError,
    ReasoningConfig,
    ToolChoiceName,
    UnsupportedFeatureError,
    ValidationError,
    generate_object,
    generate_text,
    tool,
    user,
)
from zhivex_ai.providers.meta import (  # noqa: E402
    META_DEFAULT_BASE_URL,
    create_meta,
    meta_hosted_tool,
    meta_tool_search_tool,
    meta_web_search_tool,
)
from zhivex_ai._http import Fetcher  # noqa: E402
from zhivex_ai.types import (  # noqa: E402
    ModelGenerateInput,
    StreamFinishEvent,
    StreamProviderDataEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
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


class WeatherInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str


class Forecast(BaseModel):
    city: str
    summary: str


def chat_text(text: str = "ok") -> FakeResponse:
    return FakeResponse(
        status_code=200,
        payload={
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    )


def responses_text(text: str = "ok") -> FakeResponse:
    return FakeResponse(
        status_code=200,
        payload={
            "id": "resp_1",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        },
    )


def as_fetcher(value: Any) -> Fetcher:
    return cast(Fetcher, value)


class MetaProviderConfigTests(TestCase):
    def test_meta_requires_model_api_key_and_uses_explicit_then_environment_configuration(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ConfigurationError):
            create_meta()

        with patch.dict(os.environ, {"MODEL_API_KEY": "env-key"}, clear=True):
            environment = create_meta()
            explicit = create_meta(api_key="explicit-key", base_url="https://proxy.example/meta/v1/")

        environment_model = environment.native.language_model("muse-spark-1.2")
        explicit_model = explicit.native.language_model("muse-spark-1.2")
        self.assertEqual(environment_model.api_key, "env-key")
        self.assertEqual(environment_model.base_url, META_DEFAULT_BASE_URL)
        self.assertEqual(explicit_model.api_key, "explicit-key")
        self.assertEqual(explicit_model.base_url, "https://proxy.example/meta/v1")

    def test_meta_bundle_is_portable_beta_surface_with_tier_c_agent_capabilities(self) -> None:
        provider = create_meta(api_key="test")
        model = provider("muse-spark-1.2")

        self.assertEqual(provider.name, "meta")
        self.assertEqual(provider.tier, "portable")
        self.assertTrue(provider.portable_support.portable_badge)
        self.assertTrue(provider.portable_support.text_generation)
        self.assertTrue(provider.portable_support.streaming)
        self.assertTrue(provider.portable_support.structured_output)
        self.assertTrue(provider.portable_support.tools)
        self.assertFalse(provider.portable_support.embeddings)
        self.assertTrue(provider.native_support.files)
        self.assertTrue(provider.native_support.responses)
        self.assertEqual(provider.agent_capabilities.support_tier, "tier-c")
        self.assertTrue(provider.agent_capabilities.hosted_web_search)
        self.assertTrue(provider.agent_capabilities.toolsets)
        self.assertTrue(model.capabilities.vision)
        self.assertTrue(model.capabilities.audio_input)
        self.assertTrue(model.capabilities.parallel_tool_calls)

    def test_meta_rejects_unsafe_base_urls(self) -> None:
        unsafe = (
            "http://api.meta.ai/v1",
            "https://localhost/v1",
            "https://127.0.0.1/v1",
            "https://api.meta.ai/v1?target=other",
            "https://api.meta.ai/v1#fragment",
            "https://api.meta.ai/v1/../admin",
            "https://user:pass@api.meta.ai/v1",
        )
        for base_url in unsafe:
            with self.subTest(base_url=base_url), self.assertRaises(ValidationError):
                create_meta(api_key="test", base_url=base_url)

    def test_meta_hosted_tool_helpers_are_provider_scoped(self) -> None:
        web = meta_web_search_tool(search_context_size="medium")
        search = meta_tool_search_tool(namespace="crm")
        custom = meta_hosted_tool("custom_hosted", name="custom", enabled=True)

        self.assertEqual((web.provider, web.type, web.tool_class), ("meta", "web_search", "web-search"))
        self.assertEqual(web.config, {"search_context_size": "medium"})
        self.assertEqual((search.provider, search.type, search.tool_class), ("meta", "tool_search", "toolset"))
        self.assertEqual(custom.name, "custom")
        self.assertEqual(custom.config, {"enabled": True})


class MetaProviderTests(IsolatedAsyncioTestCase):
    async def test_meta_defaults_to_chat_completions_for_portable_text(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            requests.append({"url": url, **kwargs})
            return chat_text("Meta chat ok")

        provider = create_meta(api_key="test-key", fetch=as_fetcher(fetch))
        result = await generate_text(
            model=provider("muse-spark-1.2"),
            prompt="hello",
            max_tokens=64,
            reasoning=ReasoningConfig(effort="low"),
            timeout_ms=321,
        )

        self.assertEqual(result.text, "Meta chat ok")
        self.assertEqual(result.finish_reason, "stop")
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.total_tokens if result.usage else None, 5)
        self.assertEqual(requests[0]["url"], "https://api.meta.ai/v1/chat/completions")
        self.assertEqual(requests[0]["headers"]["authorization"], "Bearer test-key")
        self.assertEqual(requests[0]["json_body"]["model"], "muse-spark-1.2")
        self.assertEqual(requests[0]["json_body"]["max_tokens"], 64)
        self.assertEqual(requests[0]["json_body"]["reasoning_effort"], "low")
        self.assertEqual(requests[0]["timeout_ms"], 321)
        self.assertFalse(requests[0]["stream"])

    async def test_meta_routes_hosted_tools_and_previous_response_id_to_responses(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            requests.append({"url": url, **kwargs})
            return responses_text("Meta responses ok")

        provider = create_meta(api_key="test", fetch=as_fetcher(fetch))
        native = provider.native.language_model("muse-spark-1.2")
        hosted_result = await generate_text(
            model=native,
            prompt="search",
            tools={
                "web": meta_web_search_tool(search_context_size="low", type="tool_search"),
                "catalog": meta_tool_search_tool(namespace="sales"),
            },
        )
        chained_result = await generate_text(
            model=native,
            prompt="continue",
            provider_options={"previous_response_id": "resp_0"},
        )

        self.assertEqual(hosted_result.text, "Meta responses ok")
        self.assertEqual(hosted_result.finish_reason, "stop")
        self.assertEqual(chained_result.text, "Meta responses ok")
        self.assertEqual([item["url"] for item in requests], [f"{META_DEFAULT_BASE_URL}/responses"] * 2)
        self.assertEqual(requests[0]["json_body"]["tools"][0], {"type": "web_search", "search_context_size": "low"})
        self.assertEqual(requests[0]["json_body"]["tools"][1], {"type": "tool_search", "namespace": "sales"})
        self.assertEqual(requests[1]["json_body"]["previous_response_id"], "resp_0")

    async def test_meta_api_mode_controls_chat_and_responses_and_rejects_incompatible_chat(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            requests.append({"url": url, **kwargs})
            return responses_text()

        provider = create_meta(api_key="test", fetch=as_fetcher(fetch))
        native = provider.native.language_model("muse-spark-1.2")
        await generate_text(model=native, prompt="hello", provider_options={"api_mode": "responses"})

        with self.assertRaisesRegex(UnsupportedFeatureError, "requires api_mode"):
            await generate_text(
                model=native,
                prompt="search",
                tools={"web": meta_web_search_tool()},
                provider_options={"api_mode": "chat"},
            )
        with self.assertRaises(ValidationError):
            await generate_text(model=native, prompt="hello", provider_options={"api_mode": "invalid"})
        with self.assertRaisesRegex(ValidationError, "reserved field"):
            await generate_text(model=native, prompt="hello", provider_options={"model": "override"})

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["url"], f"{META_DEFAULT_BASE_URL}/responses")

    async def test_meta_maps_native_json_schema_for_chat_and_responses(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            requests.append({"url": url, **kwargs})
            content = '{"city":"Buenos Aires","summary":"sunny"}'
            return responses_text(content) if url.endswith("/responses") else chat_text(content)

        provider = create_meta(api_key="test", fetch=as_fetcher(fetch))
        chat_result = await generate_object(
            model=provider("muse-spark-1.2"),
            prompt="weather",
            schema=Forecast,
            schema_name="forecast",
        )
        responses_result = await generate_object(
            model=provider.native.language_model("muse-spark-1.2"),
            prompt="weather",
            schema=Forecast,
            schema_name="forecast",
            provider_options={"api_mode": "responses"},
        )

        chat_format = requests[0]["json_body"]["response_format"]
        responses_format = requests[1]["json_body"]["text"]["format"]
        self.assertEqual(chat_result.object.city, "Buenos Aires")
        self.assertEqual(responses_result.object.summary, "sunny")
        self.assertEqual(chat_format["type"], "json_schema")
        self.assertEqual(chat_format["json_schema"]["name"], "forecast")
        self.assertTrue(chat_format["json_schema"]["strict"])
        self.assertEqual(responses_format["type"], "json_schema")
        self.assertEqual(responses_format["name"], "forecast")
        self.assertTrue(responses_format["strict"])

    async def test_meta_maps_function_tools_and_chat_tool_round_trip(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            requests.append({"url": url, **kwargs})
            if len(requests) == 1:
                return FakeResponse(
                    status_code=200,
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {"name": "weather", "arguments": '{"city":"Buenos Aires"}'},
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                )
            return chat_text("sunny")

        weather = tool(
            name="weather",
            schema=WeatherInput,
            execute=lambda value: {"city": value.city, "summary": "sunny"},
        )
        provider = create_meta(api_key="test", fetch=as_fetcher(fetch))
        result = await generate_text(
            model=provider("muse-spark-1.2"),
            prompt="weather?",
            tools={"weather": weather},
            tool_choice="auto",
            max_steps=2,
        )

        first_tool = requests[0]["json_body"]["tools"][0]
        replay = requests[1]["json_body"]["messages"]
        self.assertEqual(result.text, "sunny")
        self.assertEqual(first_tool["type"], "function")
        self.assertEqual(first_tool["function"]["name"], "weather")
        self.assertEqual(requests[0]["json_body"]["tool_choice"], "auto")
        self.assertEqual(replay[-2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(replay[-1]["tool_call_id"], "call_1")

    async def test_meta_maps_vision_and_mp3_wav_audio_to_responses(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            requests.append({"url": url, **kwargs})
            return responses_text("heard and seen")

        provider = create_meta(api_key="test", fetch=as_fetcher(fetch))
        result = await generate_text(
            model=provider("muse-spark-1.2"),
            messages=[
                user(
                    [
                        ImagePart(image="https://images.example/cat.png"),
                        FilePart(data="UklGRg==", media_type="audio/wav", filename="clip.wav"),
                    ]
                )
            ],
        )

        content = requests[0]["json_body"]["input"][0]["content"]
        self.assertEqual(result.text, "heard and seen")
        self.assertEqual(requests[0]["url"], f"{META_DEFAULT_BASE_URL}/responses")
        self.assertEqual(content[0], {"type": "input_image", "image_url": "https://images.example/cat.png"})
        self.assertEqual(
            content[1],
            {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
        )

        await generate_text(
            model=provider("muse-spark-1.2"),
            messages=[user([FilePart(file_id="file_audio", media_type="audio/mpeg")])],
        )
        self.assertEqual(requests[1]["json_body"]["input"][0]["content"][0], {"type": "input_file", "file_id": "file_audio"})

    async def test_meta_rejects_unsupported_tool_choice_reasoning_and_audio_before_fetch(self) -> None:
        calls = 0

        async def fetch(*args: Any, **kwargs: Any) -> FakeResponse:
            nonlocal calls
            calls += 1
            raise AssertionError("fetch should not run")

        model = create_meta(api_key="test", fetch=as_fetcher(fetch)).native.language_model("muse-spark-1.2")
        weather = tool(name="weather", schema=WeatherInput, execute=lambda value: value.city)
        invalid_inputs = (
            {"prompt": "hello", "tools": {"weather": weather}, "tool_choice": "required"},
            {"prompt": "hello", "tools": {"weather": weather}, "tool_choice": "none"},
            {"prompt": "hello", "tools": {"weather": weather}, "tool_choice": ToolChoiceName("weather")},
            {"prompt": "hello", "reasoning": ReasoningConfig(effort="none")},
            {"prompt": "hello", "reasoning": ReasoningConfig(effort="high", budget_tokens=100)},
        )
        for kwargs in invalid_inputs:
            with self.subTest(kwargs=kwargs), self.assertRaises(UnsupportedFeatureError):
                await generate_text(model=model, **kwargs)

        with self.assertRaisesRegex(UnsupportedFeatureError, "only MP3 and WAV"):
            await generate_text(
                model=model,
                messages=[user([FilePart(data="AAAA", media_type="audio/flac")])],
            )
        with self.assertRaisesRegex(UnsupportedFeatureError, "not a remote URL"):
            await generate_text(
                model=model,
                messages=[user([FilePart(url="https://audio.example/clip.mp3", media_type="audio/mpeg")])],
            )
        self.assertEqual(calls, 0)

    async def test_meta_chat_stream_buffers_fragmented_function_arguments(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            requests.append({"url": url, **kwargs})
            return FakeResponse(
                status_code=200,
                body_text=(
                    'data: {"choices":[{"delta":{"content":"checking "}}]}\n\n'
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"weather","arguments":"{\\"city\\""}}]}}]}\n\n'
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"Buenos Aires\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        weather = tool(name="weather", schema=WeatherInput, execute=lambda value: value.city)
        model = create_meta(api_key="test", fetch=as_fetcher(fetch)).native.language_model("muse-spark-1.2")
        stream = await model.stream(
            ModelGenerateInput(messages=[user("weather?")], tools={"weather": weather}, tool_choice="auto")
        )
        events = [event async for event in stream]

        text_event = next(event for event in events if isinstance(event, StreamTextDeltaEvent))
        tool_event = next(event for event in events if isinstance(event, StreamToolCallEvent))
        finish_event = next(event for event in events if isinstance(event, StreamFinishEvent))
        self.assertEqual(text_event.text_delta, "checking ")
        self.assertEqual(tool_event.tool_call.id, "call_1")
        self.assertEqual(tool_event.tool_call.input, {"city": "Buenos Aires"})
        self.assertEqual(finish_event.finish_reason, "tool-calls")
        self.assertEqual(requests[0]["url"], f"{META_DEFAULT_BASE_URL}/chat/completions")
        self.assertTrue(requests[0]["stream"])

    async def test_meta_responses_stream_buffers_fragmented_arguments_and_hosted_events(self) -> None:
        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            return FakeResponse(
                status_code=200,
                body_text=(
                    'event: response.output_item.added\n'
                    'data: {"type":"response.output_item.added","item":{"id":"item_1","type":"web_search_call","status":"in_progress"}}\n\n'
                    'data: {"type":"response.output_text.delta","delta":"working "}\n\n'
                    'data: {"type":"response.output_item.added","item":{"id":"item_2","type":"function_call","call_id":"call_2","name":"weather","arguments":""}}\n\n'
                    'data: {"type":"response.function_call_arguments.delta","item_id":"item_2","delta":"{\\"city\\""}\n\n'
                    'data: {"type":"response.function_call_arguments.delta","item_id":"item_2","delta":":\\"Córdoba\\"}"}\n\n'
                    'data: {"type":"response.function_call_arguments.done","item_id":"item_2","arguments":"{\\"city\\":\\"Córdoba\\"}"}\n\n'
                    'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":4,"output_tokens":3,"total_tokens":7}}}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        weather = tool(name="weather", schema=WeatherInput, execute=lambda value: value.city)
        model = create_meta(api_key="test", fetch=as_fetcher(fetch)).native.language_model("muse-spark-1.2")
        stream = await model.stream(
            ModelGenerateInput(
                messages=[user("weather?")],
                tools={"weather": weather},
                provider_options={"api_mode": "responses"},
            )
        )
        events = [event async for event in stream]

        provider_event = next(event for event in events if isinstance(event, StreamProviderDataEvent))
        tool_event = next(event for event in events if isinstance(event, StreamToolCallEvent))
        finish_event = next(event for event in events if isinstance(event, StreamFinishEvent))
        self.assertEqual(provider_event.data["type"], "web_search_call")
        self.assertEqual(tool_event.tool_call.input, {"city": "Córdoba"})
        self.assertEqual(finish_event.finish_reason, "tool-calls")
        self.assertIsNotNone(finish_event.usage)
        self.assertEqual(finish_event.usage.total_tokens if finish_event.usage else None, 7)

    async def test_meta_files_and_raw_responses_clients_use_official_endpoints(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            requests.append({"url": url, **kwargs})
            if url.endswith("/files"):
                return FakeResponse(
                    status_code=200,
                    payload={"id": "file_1", "filename": "clip.wav", "bytes": 3, "status": "processed"},
                )
            return FakeResponse(status_code=200, payload={"id": "resp_1", "status": "completed"})

        provider = create_meta(api_key="test", fetch=as_fetcher(fetch))
        uploaded = await provider.files().upload(data=b"wav", filename="clip.wav", media_type="audio/wav")
        response = await provider.responses().create({"model": "muse-spark-1.2", "input": "hello"})

        self.assertEqual(uploaded.id, "file_1")
        self.assertEqual(response["id"], "resp_1")
        self.assertEqual(requests[0]["url"], f"{META_DEFAULT_BASE_URL}/files")
        self.assertEqual(requests[0]["body"]["data"]["purpose"], "user_data")
        self.assertEqual(requests[1]["url"], f"{META_DEFAULT_BASE_URL}/responses")
        self.assertEqual(requests[1]["headers"]["authorization"], "Bearer test")

    async def test_meta_retries_retryable_http_errors_and_preserves_timeout(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            requests.append({"url": url, **kwargs})
            if len(requests) == 1:
                return FakeResponse(
                    status_code=429,
                    body_text='{"error":{"message":"slow down"}}',
                    headers={"retry-after": "0"},
                )
            return chat_text("retried")

        model = create_meta(api_key="test", fetch=as_fetcher(fetch))("muse-spark-1.2")
        result = await generate_text(
            model=model,
            prompt="hello",
            timeout_ms=456,
            max_retries=1,
            retry_backoff_ms=0,
        )

        self.assertEqual(result.text, "retried")
        self.assertEqual(len(requests), 2)
        self.assertEqual([request["timeout_ms"] for request in requests], [456, 456])

    async def test_meta_http_error_retains_status_body_and_headers(self) -> None:
        async def fetch(url: str, **kwargs: Any) -> FakeResponse:
            return FakeResponse(
                status_code=503,
                body_text='{"error":{"message":"unavailable"}}',
                headers={"x-request-id": "req_1"},
            )

        model = create_meta(api_key="test", fetch=as_fetcher(fetch))("muse-spark-1.2")
        with self.assertRaises(ProviderHTTPError) as raised:
            await generate_text(model=model, prompt="hello")

        self.assertEqual(raised.exception.status, 503)
        self.assertIn("unavailable", raised.exception.response_body or "")
        self.assertEqual(raised.exception.response_headers["x-request-id"], "req_1")
