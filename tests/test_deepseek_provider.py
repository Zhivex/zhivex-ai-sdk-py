from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    ConfigurationError,
    ImagePart,
    ProviderHTTPError,
    ReasoningConfig,
    TextPart,
    ToolChoiceName,
    UnsupportedFeatureError,
    ValidationError,
    assistant,
    create_deepseek,
    generate_object,
    generate_text,
    provider_data_part,
    stream_text,
    tool,
    user,
)


@dataclass
class FakeResponse:
    status_code: int
    payload: Any = None
    body_text: str = ""
    headers: dict[str, str] | None = None

    async def json(self) -> Any:
        return self.payload

    async def text(self) -> str:
        return self.body_text or json.dumps(self.payload)

    async def read(self) -> bytes:
        return (self.body_text or json.dumps(self.payload)).encode("utf-8")

    async def iter_lines(self) -> AsyncIterable[str]:
        for line in self.body_text.splitlines():
            yield line


class WeatherInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str


class Forecast(BaseModel):
    city: str
    forecast: str


class DeepSeekProviderConfigTests(TestCase):
    def test_deepseek_uses_explicit_then_environment_configuration(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "env-key",
                "DEEPSEEK_BASE_URL": "https://proxy.example/deepseek/",
            },
            clear=True,
        ):
            environment = create_deepseek()
            explicit = create_deepseek(
                api_key="explicit-key",
                base_url="https://explicit.example/api/",
            )

        environment_model = environment.native.language_model("deepseek-v4-flash")
        explicit_model = explicit.native.language_model("deepseek-v4-pro")
        self.assertEqual(environment_model.api_key, "env-key")
        self.assertEqual(environment_model.base_url, "https://proxy.example/deepseek")
        self.assertEqual(explicit_model.api_key, "explicit-key")
        self.assertEqual(explicit_model.base_url, "https://explicit.example/api")

    def test_deepseek_requires_an_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ConfigurationError):
            create_deepseek()

    def test_deepseek_capabilities_match_the_official_text_only_surface(self) -> None:
        provider = create_deepseek(api_key="test")
        model = provider("deepseek-v4-pro")

        self.assertEqual(provider.name, "deepseek")
        self.assertEqual(provider.tier, "portable")
        self.assertTrue(provider.portable_support.portable_badge)
        self.assertTrue(model.capabilities.streaming)
        self.assertTrue(model.capabilities.structured_output)
        self.assertTrue(model.capabilities.tools)
        self.assertTrue(model.capabilities.reasoning)
        self.assertFalse(model.capabilities.vision)
        self.assertFalse(model.capabilities.embeddings)
        self.assertFalse(provider.native_support.embeddings)
        with self.assertRaisesRegex(UnsupportedFeatureError, "portable embeddings"):
            provider.embedding_model("text-embedding")


class DeepSeekProviderTests(IsolatedAsyncioTestCase):
    async def test_deepseek_uses_chat_completions_and_preserves_usage_details(self) -> None:
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
        ) -> FakeResponse:
            requests.append(
                {
                    "url": url,
                    "method": method,
                    "headers": headers,
                    "json": json_body,
                    "stream": stream,
                }
            )
            return FakeResponse(
                status_code=200,
                headers={},
                payload={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "brief reasoning",
                                "content": "DeepSeek ok",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 8,
                        "prompt_cache_hit_tokens": 5,
                        "prompt_cache_miss_tokens": 3,
                        "completion_tokens": 4,
                        "completion_tokens_details": {"reasoning_tokens": 2},
                        "total_tokens": 12,
                    },
                },
            )

        provider = create_deepseek(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider("deepseek-v4-flash"),
            prompt="hello",
            max_tokens=64,
        )

        self.assertEqual(result.text, "DeepSeek ok")
        self.assertEqual(result.usage.total_tokens, 12)
        self.assertEqual(requests[0]["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(requests[0]["headers"]["authorization"], "Bearer test")
        self.assertEqual(requests[0]["json"]["max_tokens"], 64)
        self.assertNotIn("thinking", requests[0]["json"])
        self.assertNotIn("/responses", requests[0]["url"])
        provider_data = [
            part.data
            for part in result.messages[-1].parts
            if part.type == "provider-data"
        ]
        self.assertIn({"reasoning_content": "brief reasoning"}, provider_data)
        self.assertTrue(
            any(
                data.get("usage", {}).get("prompt_cache_hit_tokens") == 5
                for data in provider_data
            )
        )

    async def test_deepseek_maps_all_supported_reasoning_efforts(self) -> None:
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
        ) -> FakeResponse:
            requests.append(json_body or {})
            return FakeResponse(
                status_code=200,
                headers={},
                payload={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

        model = create_deepseek(api_key="test", fetch=fetch).native.language_model(
            "deepseek-v4-pro"
        )
        for effort in ("none", "low", "medium", "high", "xhigh", "max"):
            await generate_text(
                model=model,
                prompt="hello",
                reasoning=ReasoningConfig(effort=effort),  # type: ignore[arg-type]
            )

        self.assertEqual(
            [(item["thinking"], item.get("reasoning_effort")) for item in requests],
            [
                ({"type": "disabled"}, None),
                ({"type": "enabled"}, "high"),
                ({"type": "enabled"}, "high"),
                ({"type": "enabled"}, "high"),
                ({"type": "enabled"}, "max"),
                ({"type": "enabled"}, "max"),
            ],
        )

    async def test_deepseek_preserves_portable_sampling_and_tool_choice(self) -> None:
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
        ) -> FakeResponse:
            requests.append(json_body or {})
            return FakeResponse(
                status_code=200,
                headers={},
                payload={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

        model = create_deepseek(api_key="test", fetch=fetch)(
            "deepseek-v4-flash"
        )
        weather = tool(
            name="weather",
            schema=WeatherInput,
            execute=lambda input: {"city": input.city, "forecast": "sunny"},
        )
        await generate_text(
            model=model,
            prompt="hello",
            temperature=0.2,
        )
        await generate_text(
            model=model,
            prompt="weather",
            tools={"weather": weather},
            tool_choice=ToolChoiceName("weather"),
        )

        self.assertEqual(requests[0]["thinking"], {"type": "disabled"})
        self.assertEqual(requests[0]["temperature"], 0.2)
        self.assertEqual(requests[1]["thinking"], {"type": "disabled"})
        self.assertEqual(
            requests[1]["tool_choice"],
            {"type": "function", "function": {"name": "weather"}},
        )

    async def test_deepseek_rejects_incompatible_reasoning_and_options_before_fetch(
        self,
    ) -> None:
        calls = 0

        async def fetch(*args: Any, **kwargs: Any) -> FakeResponse:
            nonlocal calls
            calls += 1
            raise AssertionError("fetch should not run")

        model = create_deepseek(api_key="test", fetch=fetch).native.language_model(
            "deepseek-v4-pro"
        )
        weather = tool(
            name="weather",
            schema=WeatherInput,
            execute=lambda input: {"ok": True},
        )

        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=model,
                prompt="hello",
                reasoning=ReasoningConfig(effort="minimal"),
            )
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=model,
                prompt="hello",
                reasoning=ReasoningConfig(effort="high", budget_tokens=100),
            )
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=model,
                prompt="hello",
                temperature=0.2,
                reasoning=ReasoningConfig(effort="high"),
            )
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=model,
                prompt="hello",
                tools={"weather": weather},
                tool_choice="required",
                reasoning=ReasoningConfig(effort="high"),
            )
        with self.assertRaises(ValidationError):
            await generate_text(
                model=model,
                prompt="hello",
                reasoning=ReasoningConfig(effort="high"),
                provider_options={"thinking": {"type": "disabled"}},
            )
        with self.assertRaises(ValidationError):
            await generate_text(
                model=model,
                prompt="hello",
                provider_options={
                    "thinking": {"type": "disabled"},
                    "reasoning_effort": "high",
                },
            )
        self.assertEqual(calls, 0)

    async def test_deepseek_structured_output_injects_json_schema_instruction(
        self,
    ) -> None:
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
        ) -> FakeResponse:
            requests.append(json_body or {})
            return FakeResponse(
                status_code=200,
                headers={},
                payload={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '{"city":"Buenos Aires","forecast":"sunny"}',
                            },
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

        provider = create_deepseek(api_key="test", fetch=fetch)
        result = await generate_object(
            model=provider("deepseek-v4-flash"),
            schema=Forecast,
            prompt="Forecast Buenos Aires",
        )

        self.assertEqual(result.object.city, "Buenos Aires")
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})
        self.assertEqual(requests[0]["messages"][0]["role"], "system")
        self.assertIn("valid JSON", requests[0]["messages"][0]["content"])
        self.assertIn('"city"', requests[0]["messages"][0]["content"])

    async def test_deepseek_tool_loop_replays_reasoning_and_non_null_content(
        self,
    ) -> None:
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
        ) -> FakeResponse:
            requests.append(json_body or {})
            if len(requests) == 1:
                return FakeResponse(
                    status_code=200,
                    headers={},
                    payload={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "reasoning_content": "I need the weather tool.",
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "weather",
                                                "arguments": '{"city":"Madrid"}',
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    },
                )
            return FakeResponse(
                status_code=200,
                headers={},
                payload={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "The tool says sunny.",
                                "content": "sunny",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

        model = create_deepseek(api_key="test", fetch=fetch)(
            "deepseek-v4-pro"
        )
        result = await generate_text(
            model=model,
            prompt="Weather in Madrid?",
            tools={
                "weather": tool(
                    name="weather",
                    schema=WeatherInput,
                    execute=lambda input: {
                        "city": input.city,
                        "forecast": "sunny",
                    },
                )
            },
            max_steps=2,
        )

        self.assertEqual(result.text, "sunny")
        replayed = requests[1]["messages"][1]
        self.assertEqual(replayed["role"], "assistant")
        self.assertEqual(replayed["content"], "")
        self.assertEqual(
            replayed["reasoning_content"],
            "I need the weather tool.",
        )
        self.assertEqual(replayed["tool_calls"][0]["id"], "call_1")
        self.assertEqual(requests[1]["messages"][2]["role"], "tool")

    async def test_deepseek_streaming_accumulates_reasoning_tools_and_final_usage(
        self,
    ) -> None:
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
        ) -> FakeResponse:
            requests.append(json_body or {})
            if len(requests) == 1:
                return FakeResponse(
                    status_code=200,
                    headers={},
                    body_text=(
                        ": keep-alive\n\n"
                        'data: {"choices":[{"delta":{"reasoning_content":"need "},"finish_reason":null}],"usage":null}\n\n'
                        'data: {"choices":[{"delta":{"reasoning_content":"weather","tool_calls":[{"index":0,"id":"call_1","function":{"name":"weather","arguments":"{\\"city\\":"}}]},"finish_reason":null}],"usage":null}\n\n'
                        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"Madrid\\"}"}}]},"finish_reason":"tool_calls"}],"usage":null}\n\n'
                        'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8,"prompt_cache_hit_tokens":2}}\n\n'
                        "data: [DONE]\n\n"
                    ),
                )
            return FakeResponse(
                status_code=200,
                headers={},
                body_text=(
                    'data: {"choices":[{"delta":{"reasoning_content":"answer "},"finish_reason":null}],"usage":null}\n\n'
                    'data: {"choices":[{"delta":{"content":"sunny"},"finish_reason":"stop"}],"usage":null}\n\n'
                    'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11}}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        model = create_deepseek(api_key="test", fetch=fetch)(
            "deepseek-v4-pro"
        )
        stream = stream_text(
            model=model,
            prompt="Weather?",
            tools={
                "weather": tool(
                    name="weather",
                    schema=WeatherInput,
                    execute=lambda input: {
                        "city": input.city,
                        "forecast": "sunny",
                    },
                )
            },
            max_steps=2,
        )
        result = await stream.collect()

        self.assertEqual(result.text, "sunny")
        self.assertEqual(result.usage.total_tokens, 19)
        self.assertEqual(requests[0]["stream_options"], {"include_usage": True})
        self.assertEqual(
            requests[1]["messages"][1]["reasoning_content"],
            "need weather",
        )
        self.assertEqual(
            requests[1]["messages"][1]["tool_calls"][0]["function"]["arguments"],
            '{"city": "Madrid"}',
        )

    async def test_deepseek_strict_tools_and_chat_prefix_select_beta_route(self) -> None:
        closed_schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        }
        strict_tool = tool(
            name="weather",
            schema=closed_schema,
            strict=True,
            execute=lambda input: {"ok": True},
        )

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
        ) -> FakeResponse:
            requests.append({"url": url, "json": json_body})
            return FakeResponse(
                status_code=200,
                headers={},
                payload={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

        provider = create_deepseek(
            api_key="test",
            fetch=fetch,
        )
        await generate_text(
            model=provider.native.language_model("deepseek-v4-flash"),
            messages=[
                user("Complete this sentence"),
                assistant(
                    [
                        TextPart(text="DeepSeek is"),
                        provider_data_part("deepseek", {"prefix": True}),
                    ]
                ),
            ],
            tools={"weather": strict_tool},
        )

        self.assertEqual(
            requests[0]["url"],
            "https://api.deepseek.com/beta/chat/completions",
        )
        self.assertTrue(requests[0]["json"]["tools"][0]["function"]["strict"])
        self.assertTrue(requests[0]["json"]["messages"][-1]["prefix"])

    async def test_deepseek_rejects_retired_models_and_non_text_inputs(self) -> None:
        provider = create_deepseek(api_key="test")
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider("deepseek-chat"),
                prompt="hello",
            )
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider("deepseek-reasoner"),
                prompt="hello",
            )
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider("deepseek-v4-pro"),
                messages=[user([ImagePart(image="https://example.com/image.png")])],
            )

    async def test_deepseek_retries_retryable_http_and_resource_failures(
        self,
    ) -> None:
        responses = [
            FakeResponse(
                status_code=429,
                headers={"retry-after": "0"},
                payload={"error": {"message": "rate limited"}},
            ),
            FakeResponse(
                status_code=200,
                headers={},
                payload={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "insufficient_system_resource",
                        }
                    ]
                },
            ),
            FakeResponse(
                status_code=200,
                headers={},
                payload={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            ),
        ]
        calls = 0

        async def fetch(*args: Any, **kwargs: Any) -> FakeResponse:
            nonlocal calls
            response = responses[calls]
            calls += 1
            return response

        result = await generate_text(
            model=create_deepseek(api_key="test", fetch=fetch)(
                "deepseek-v4-flash"
            ),
            prompt="hello",
            max_retries=2,
            retry_backoff_ms=0,
        )

        self.assertEqual(result.text, "ok")
        self.assertEqual(calls, 3)

    async def test_deepseek_does_not_retry_non_retryable_http_errors(self) -> None:
        calls = 0

        async def fetch(*args: Any, **kwargs: Any) -> FakeResponse:
            nonlocal calls
            calls += 1
            return FakeResponse(
                status_code=401,
                headers={},
                payload={
                    "error": {
                        "message": "invalid api_key=super-secret",
                    }
                },
            )

        with self.assertRaises(ProviderHTTPError) as captured:
            await generate_text(
                model=create_deepseek(api_key="test", fetch=fetch)(
                    "deepseek-v4-flash"
                ),
                prompt="hello",
                max_retries=2,
                retry_backoff_ms=0,
            )

        self.assertEqual(calls, 1)
        self.assertEqual(captured.exception.status, 401)
        self.assertNotIn("super-secret", str(captured.exception))
