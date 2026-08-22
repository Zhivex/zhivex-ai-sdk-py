from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    ToolChoiceName,
    create_anthropic,
    create_azure_openai,
    create_deepseek,
    create_gemini,
    create_kimi,
    create_meta,
    create_openai,
    create_qwen,
    create_vertex,
    create_vllm,
    generate_object,
    generate_text,
    stream_text,
    tool,
)
from zhivex_ai.provider_support import TIER_1_PROVIDERS, build_provider_support_rows  # noqa: E402
from zhivex_ai.errors import ValidationError  # noqa: E402
from zhivex_ai.types import StreamFinishEvent, StreamTextDeltaEvent  # noqa: E402


@dataclass(slots=True)
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


class WeatherToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str


class Forecast(BaseModel):
    city: str
    summary: str


@dataclass(slots=True)
class ProviderContractCase:
    provider_name: str
    model_id: str
    create_provider: Callable[[Callable[..., Any]], Any]
    response_family: str
    expected_request_marker: str
    supports_named_tool_choice: bool = True


def _openai_compatible_response(stream: bool) -> FakeResponse:
    if stream:
        return FakeResponse(
            status_code=200,
            body_text=(
                'data: {"type":"response.output_text.delta","delta":"contract"}\n\n'
                'data: {"type":"response.output_text.delta","delta":" ok"}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":2,"output_tokens":2,"total_tokens":4}}}\n\n'
                "data: [DONE]\n\n"
            ),
        )
    return FakeResponse(
        status_code=200,
        payload={
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": '{"city":"Buenos Aires","summary":"contract ok"}'}],
                }
            ],
            "usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        },
    )


def _anthropic_response(stream: bool) -> FakeResponse:
    if stream:
        return FakeResponse(
            status_code=200,
            body_text=(
                'event: content_block_delta\n'
                'data: {"delta":{"type":"text_delta","text":"contract "}}\n\n'
                'event: content_block_delta\n'
                'data: {"delta":{"type":"text_delta","text":"ok"}}\n\n'
                'event: message_delta\n'
                'data: {"delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":2,"output_tokens":2}}\n\n'
                'event: message_stop\n'
                'data: {"stop_reason":"end_turn"}\n'
            ),
        )
    return FakeResponse(
        status_code=200,
        payload={
            "content": [{"type": "text", "text": '{"city":"Buenos Aires","summary":"contract ok"}'}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 2},
        },
    )


def _gemini_response(stream: bool) -> FakeResponse:
    if stream:
        return FakeResponse(
            status_code=200,
            body_text=(
                'data: {"candidates":[{"content":{"parts":[{"text":"contract "}]}}],"usageMetadata":{"totalTokenCount":3}}\n\n'
                'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":4}}\n\n'
            ),
        )
    return FakeResponse(
        status_code=200,
        payload={
            "candidates": [
                {
                    "content": {"parts": [{"text": '{"city":"Buenos Aires","summary":"contract ok"}'}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 2, "totalTokenCount": 4},
        },
    )


def _chat_completions_response(stream: bool) -> FakeResponse:
    if stream:
        return FakeResponse(
            status_code=200,
            body_text=(
                'data: {"choices":[{"delta":{"content":"contract "},"finish_reason":null}]}\n\n'
                'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}\n\n'
                "data: [DONE]\n\n"
            ),
        )
    return FakeResponse(
        status_code=200,
        payload={
            "choices": [
                {
                    "message": {"role": "assistant", "content": '{"city":"Buenos Aires","summary":"contract ok"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
        },
    )


def _contract_cases() -> list[ProviderContractCase]:
    return [
        ProviderContractCase(
            provider_name="openai",
            model_id="gpt-5.6-terra",
            create_provider=lambda fetch: create_openai(api_key="test", fetch=fetch),
            response_family="openai",
            expected_request_marker="/responses",
        ),
        ProviderContractCase(
            provider_name="anthropic",
            model_id="claude-opus-5",
            create_provider=lambda fetch: create_anthropic(api_key="test", fetch=fetch),
            response_family="anthropic",
            expected_request_marker="/messages",
        ),
        ProviderContractCase(
            provider_name="azure-openai",
            model_id="gpt-5.6-terra",
            create_provider=lambda fetch: create_azure_openai(
                api_key="test",
                endpoint="https://example.openai.azure.com",
                fetch=fetch,
            ),
            response_family="openai",
            expected_request_marker="/responses",
        ),
        ProviderContractCase(
            provider_name="gemini",
            model_id="gemini-3.5-flash",
            create_provider=lambda fetch: create_gemini(api_key="test", fetch=fetch),
            response_family="gemini",
            expected_request_marker=":generateContent",
        ),
        ProviderContractCase(
            provider_name="vertex",
            model_id="gemini-3.5-flash",
            create_provider=lambda fetch: create_vertex(access_token="test", project_id="project", fetch=fetch),
            response_family="gemini",
            expected_request_marker=":generateContent",
        ),
        ProviderContractCase(
            provider_name="qwen",
            model_id="qwen3.8-max",
            create_provider=lambda fetch: create_qwen(api_key="test", fetch=fetch),
            response_family="openai",
            expected_request_marker="/responses",
        ),
        ProviderContractCase(
            provider_name="kimi",
            # K3 is the current catalog reference, but its always-on reasoning
            # contract cannot force a named tool. Keep the compatibility K2
            # fixture for the portable named-tool-choice contract and test K3
            # separately in the provider suite.
            model_id="kimi-k2",
            create_provider=lambda fetch: create_kimi(api_key="test", fetch=fetch),
            response_family="chat",
            expected_request_marker="/chat/completions",
        ),
        ProviderContractCase(
            provider_name="deepseek",
            model_id="deepseek-v4-flash",
            create_provider=lambda fetch: create_deepseek(api_key="test", fetch=fetch),
            response_family="chat",
            expected_request_marker="/chat/completions",
        ),
        ProviderContractCase(
            provider_name="meta",
            model_id="muse-spark-1.2",
            create_provider=lambda fetch: create_meta(api_key="test", fetch=fetch),
            response_family="chat",
            expected_request_marker="/chat/completions",
            supports_named_tool_choice=False,
        ),
        ProviderContractCase(
            provider_name="vllm",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            create_provider=lambda fetch: create_vllm(api_key="test", fetch=fetch),
            response_family="openai",
            expected_request_marker="/responses",
        ),
    ]


def _fake_fetch(case: ProviderContractCase, requests: list[dict[str, Any]]) -> Callable[..., Any]:
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
        requests.append({"url": url, "method": method, "headers": headers, "json": json_body, "body": body, "stream": stream})
        if case.provider_name == "qwen" and url.endswith("/chat/completions"):
            return _chat_completions_response(stream)
        if case.response_family == "anthropic":
            return _anthropic_response(stream)
        if case.response_family == "gemini":
            return _gemini_response(stream)
        if case.response_family == "chat":
            return _chat_completions_response(stream)
        return _openai_compatible_response(stream)

    return fetch


@pytest.mark.parametrize("case", _contract_cases(), ids=lambda case: case.provider_name)
def test_tier_1_provider_metadata_has_shared_contract_row(case: ProviderContractCase) -> None:
    provider = case.create_provider(lambda **_: None)
    rows = build_provider_support_rows([provider])

    assert case.provider_name in TIER_1_PROVIDERS
    assert rows[0].provider == case.provider_name
    assert rows[0].tier == "portable"
    assert rows[0].portable_badge
    assert rows[0].portable_support.text_generation
    assert rows[0].portable_support.streaming
    assert rows[0].portable_support.structured_output
    assert rows[0].portable_support.tools


@pytest.mark.parametrize("case", _contract_cases(), ids=lambda case: case.provider_name)
def test_tier_1_portable_and_native_model_boundaries(case: ProviderContractCase) -> None:
    provider = case.create_provider(lambda **_: None)
    portable_model = provider(case.model_id)
    native_model = provider.native.language_model(case.model_id)

    assert provider.name == case.provider_name
    assert portable_model.provider == case.provider_name
    assert portable_model.model_id == case.model_id
    assert portable_model.native_model is native_model
    assert native_model.provider == case.provider_name
    assert native_model.model_id == case.model_id


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _contract_cases(), ids=lambda case: case.provider_name)
async def test_tier_1_portable_text_stream_and_structured_output(case: ProviderContractCase) -> None:
    requests: list[dict[str, Any]] = []
    provider = case.create_provider(_fake_fetch(case, requests))
    model = provider(case.model_id)

    text_result = await generate_text(model=model, prompt="return contract json")
    stream_result = stream_text(model=model, prompt="stream contract")
    stream_events = [event async for event in stream_result.event_stream()]
    object_result = await generate_object(model=model, prompt="return weather json", schema=Forecast)

    assert text_result.text == '{"city":"Buenos Aires","summary":"contract ok"}'
    assert object_result.object.city == "Buenos Aires"
    assert object_result.object.summary == "contract ok"
    assert "".join(event.text_delta for event in stream_events if isinstance(event, StreamTextDeltaEvent)) == "contract ok"
    assert any(isinstance(event, StreamFinishEvent) for event in stream_events)
    assert any(case.expected_request_marker in request["url"] for request in requests)
    assert any(request["stream"] for request in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _contract_cases(), ids=lambda case: case.provider_name)
async def test_tier_1_tool_choice_contract_is_mapped(case: ProviderContractCase) -> None:
    requests: list[dict[str, Any]] = []
    provider = case.create_provider(_fake_fetch(case, requests))

    await generate_text(
        model=provider(case.model_id),
        prompt="weather",
        tools={"weather": tool(name="weather", schema=WeatherToolInput, execute=lambda input: {"forecast": f"sunny in {input.city}"})},
        tool_choice=(
            ToolChoiceName(tool_name="weather")
            if case.supports_named_tool_choice
            else "auto"
        ),
    )

    request = requests[0]["json"]
    assert isinstance(request, dict)
    if case.response_family == "gemini":
        assert request["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"] == ["weather"]
        assert request["tools"][0]["functionDeclarations"][0]["name"] == "weather"
    elif case.response_family == "anthropic":
        assert request["tool_choice"] == {"type": "tool", "name": "weather"}
        assert request["tools"][0]["name"] == "weather"
    elif case.provider_name == "meta":
        assert request["tool_choice"] == "auto"
        assert request["tools"][0]["function"]["name"] == "weather"
    elif case.response_family == "chat":
        assert request["tool_choice"] == {"type": "function", "function": {"name": "weather"}}
        assert request["tools"][0]["function"]["name"] == "weather"
    elif case.provider_name == "qwen":
        assert request["tool_choice"] == {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "weather"}]}
        assert request["tools"][0]["name"] == "weather"
    else:
        assert request["tool_choice"]["name"] == "weather"
        assert request["tools"][0]["name"] == "weather"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _contract_cases(), ids=lambda case: case.provider_name)
async def test_tier_1_portable_models_reject_provider_options(case: ProviderContractCase) -> None:
    provider = case.create_provider(_fake_fetch(case, []))

    with pytest.raises(ValidationError):
        await generate_text(model=provider(case.model_id), prompt="hello", provider_options={"native": True})
