from __future__ import annotations

import json
import os
import sys
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    AudioInput,
    ConfigurationError,
    FilePart,
    ImagePart,
    ModelMessage,
    ProviderDataPart,
    ReasoningConfig,
    TextPart,
    ToolChoiceName,
    UnsupportedFeatureError,
    ValidationError,
    create_qwen,
    embed,
    generate_object,
    generate_speech,
    generate_text,
    qwen_code_interpreter_tool,
    qwen_file_search_tool,
    qwen_image_search_tool,
    qwen_mcp_tool,
    qwen_web_extractor_tool,
    qwen_web_search_tool,
    qwen_web_search_image_tool,
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


class Forecast(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str
    summary: str


class QwenProviderTests(IsolatedAsyncioTestCase):
    def test_create_qwen_resolves_official_regions_and_overrides(self) -> None:
        default_provider = create_qwen(api_key="test")
        us_provider = create_qwen(api_key="test", region="us")
        cn_provider = create_qwen(api_key="test", region="cn")
        override_provider = create_qwen(api_key="test", region="cn", base_url="https://proxy.example.com/compatible-mode/v1/")

        self.assertEqual(default_provider.native.language_model("qwen3.5-plus").base_url, "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(default_provider.native.embedding_model("text-embedding-v4").base_url, "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(us_provider.native.language_model("qwen3.5-plus").base_url, "https://dashscope-us.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(cn_provider.native.language_model("qwen3.5-plus").base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(override_provider.native.embedding_model("text-embedding-v4").base_url, "https://proxy.example.com/compatible-mode/v1")
        self.assertEqual(override_provider.native.language_model("qwen3.5-plus").base_url, "https://proxy.example.com/compatible-mode/v1")

    def test_create_qwen_allows_explicit_responses_base_url_and_validates_region(self) -> None:
        provider = create_qwen(
            api_key="test",
            base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            responses_base_url="https://responses.example.com/v1",
        )

        self.assertEqual(provider.native.language_model("qwen3.5-plus").base_url, "https://responses.example.com/v1")
        self.assertEqual(provider.native.embedding_model("text-embedding-v4").base_url, "https://dashscope-us.aliyuncs.com/compatible-mode/v1")

        with self.assertRaises(ConfigurationError):
            create_qwen(api_key="test", region="eu")  # type: ignore[arg-type]

    async def test_qwen_exposes_raw_responses_client_without_file_search_lifecycle(self) -> None:
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
            return FakeResponse(status_code=200, payload={"id": "resp_qwen", "status": "completed"})

        provider = create_qwen(api_key="test", fetch=fetch)
        payload = await provider.responses().create({"model": "qwen3.5-plus", "input": "hello"})

        self.assertEqual(payload["id"], "resp_qwen")
        self.assertEqual(requests[0]["url"], "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/responses")
        self.assertEqual(requests[0]["json"], {"model": "qwen3.5-plus", "input": "hello"})
        self.assertTrue(provider.native_support.responses)
        self.assertTrue(provider.native_support.files)
        self.assertTrue(provider.native_support.batches)
        self.assertFalse(provider.native_support.file_search)
        with self.assertRaises(AttributeError):
            provider.file_search_stores()

    async def test_qwen_exposes_openai_compatible_files_and_batches_without_vector_store_lifecycle(self) -> None:
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
            requests.append({"url": url, "method": method, "json": json_body, "body": body})
            if url.endswith("/files") and method == "POST":
                return FakeResponse(status_code=200, payload={"id": "file_qwen", "filename": "batch.jsonl", "bytes": 42, "status": "processed"})
            if url.endswith("/batches") and method == "POST":
                return FakeResponse(status_code=200, payload={"id": "batch_qwen", "status": "validating"})
            raise AssertionError(url)

        provider = create_qwen(api_key="test", fetch=fetch)
        uploaded = await provider.files().upload(data=b"{}", filename="batch.jsonl", media_type="application/jsonl")
        batch = await provider.batches().create(
            {"input_file_id": uploaded.id, "endpoint": "/v1/chat/completions", "completion_window": "24h"}
        )

        self.assertEqual(uploaded.id, "file_qwen")
        self.assertEqual(batch["id"], "batch_qwen")
        self.assertEqual(requests[0]["url"], "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/files")
        self.assertEqual(requests[0]["body"]["data"]["purpose"], "batch")
        self.assertEqual(requests[1]["url"], "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/batches")

    async def test_qwen_uses_responses_endpoint_and_supports_tools(self) -> None:
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
                    "id": "resp_qwen",
                    "status": "completed",
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider.native.language_model("qwen3.5-plus"),
            prompt="hello",
            tools={"weather": tool(name="weather", schema=WeatherToolInput, execute=lambda input: {"ok": True})},
            provider_options={"previous_response_id": "resp_previous", "conversation": "conv_123"},
        )

        self.assertEqual(result.text, "ok")
        self.assertEqual(requests[0]["url"], "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/responses")
        self.assertEqual(requests[0]["json"]["input"], [{"role": "user", "content": "hello"}])
        self.assertEqual(requests[0]["json"]["tools"][0]["name"], "weather")
        self.assertEqual(requests[0]["json"]["previous_response_id"], "resp_previous")
        self.assertEqual(requests[0]["json"]["conversation"], "conv_123")

    async def test_qwen_maps_mixed_multimodal_content_to_current_responses_input_types(self) -> None:
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
                    "output": [
                        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "blue"}]}
                    ],
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        for model_id in ("qwen3.7-max-2026-06-08", "qwen3.8-max"):
            with self.subTest(model_id=model_id):
                result = await generate_text(
                    model=provider.native.language_model(model_id),
                    messages=[
                        ModelMessage(
                            role="user",
                            parts=[
                                TextPart(text="What color is this?"),
                                ImagePart(image="data:image/png;base64,iVBORw0KGgo="),
                            ],
                        )
                    ],
                )

                self.assertEqual(result.text, "blue")
                self.assertEqual(
                    requests[-1]["json"]["input"][0]["content"],
                    [
                        {"type": "text", "text": "What color is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                )
                self.assertEqual(
                    requests[-1]["url"],
                    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/responses",
                )

    async def test_qwen38_routes_json_schema_output_to_non_thinking_chat(self) -> None:
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
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '{"city":"Buenos Aires","summary":"sunny"}',
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12},
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        result = await generate_object(
            model=provider.native.language_model("qwen3.8-max"),
            prompt="Return the weather.",
            schema=Forecast,
        )

        self.assertEqual(result.object.summary, "sunny")
        self.assertEqual(
            requests[0]["url"],
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        self.assertFalse(requests[0]["json"]["enable_thinking"])
        self.assertEqual(requests[0]["json"]["response_format"]["type"], "json_schema")
        self.assertEqual(
            requests[0]["json"]["response_format"]["json_schema"]["schema"]["properties"].keys(),
            {"city", "summary"},
        )

    async def test_qwen38_rejects_thinking_structured_output_before_fetch(self) -> None:
        fetch = AsyncMock()
        provider = create_qwen(api_key="test", fetch=fetch)

        with self.assertRaisesRegex(UnsupportedFeatureError, "structured output requires"):
            await generate_object(
                model=provider.native.language_model("qwen3.8-max"),
                prompt="Return the weather.",
                schema=Forecast,
                reasoning=ReasoningConfig(effort="high"),
            )

        fetch.assert_not_called()

    async def test_qwen38_routes_video_file_parts_to_chat(self) -> None:
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
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "A short demo."},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        model = provider.native.language_model("qwen3.8-max")
        self.assertTrue(model.capabilities.files)
        result = await generate_text(
            model=model,
            messages=[
                ModelMessage(
                    role="user",
                    parts=[
                        TextPart(text="Describe this video."),
                        FilePart(url="https://example.com/demo.mp4", media_type="video/mp4"),
                    ],
                )
            ],
            reasoning=ReasoningConfig(effort="medium"),
        )

        self.assertEqual(result.text, "A short demo.")
        self.assertTrue(requests[0]["url"].endswith("/chat/completions"))
        self.assertEqual(
            requests[0]["json"]["messages"][0]["content"],
            [
                {"type": "text", "text": "Describe this video."},
                {
                    "type": "video_url",
                    "video_url": {"url": "https://example.com/demo.mp4"},
                },
            ],
        )
        self.assertEqual(requests[0]["json"]["reasoning_effort"], "medium")
        self.assertTrue(requests[0]["json"]["enable_thinking"])

    async def test_qwen38_routes_reasoning_budgets_to_chat_and_preserves_history(self) -> None:
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
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "new reasoning",
                                "content": "done",
                            },
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider.native.language_model("qwen3.8-max"),
            messages=[
                ModelMessage(
                    role="assistant",
                    parts=[
                        TextPart(text="previous answer"),
                        ProviderDataPart(
                            provider="qwen",
                            data={"reasoning_content": "previous reasoning"},
                        ),
                    ],
                ),
                ModelMessage(role="user", parts=[TextPart(text="continue")]),
            ],
            reasoning=ReasoningConfig(budget_tokens=16_384),
        )

        self.assertEqual(result.text, "done")
        self.assertEqual(requests[0]["json"]["thinking_budget"], 16_384)
        self.assertTrue(requests[0]["json"]["enable_thinking"])
        self.assertEqual(requests[0]["json"]["messages"][0]["reasoning_content"], "previous reasoning")
        reasoning_parts = [
            part
            for part in result.steps[0].response.messages[0].parts
            if isinstance(part, ProviderDataPart)
        ]
        self.assertEqual(reasoning_parts[0].data, {"reasoning_content": "new reasoning"})

    async def test_qwen_streams_responses_with_tools_and_provider_data(self) -> None:
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
                    'data: {"type":"response.reasoning_summary_text.delta","delta":"thinking"}\n\n'
                    'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
                    'data: {"type":"response.output_text.delta","delta":" qwen"}\n\n'
                    'data: {"type":"response.output_item.done","item":{"type":"web_search_call","id":"search_1","action":{"query":"Alibaba Cloud"}}}\n\n'
                    'data: {"type":"response.completed","response":{"id":"resp_stream","status":"completed","usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        stream = stream_text(
            model=provider.native.language_model("qwen3.5-plus"),
            prompt="search",
            tools={"search": qwen_web_search_tool()},
            provider_options={"previous_response_id": "resp_previous"},
        )
        events = [event async for event in stream.event_stream()]
        result = await stream.collect()

        self.assertEqual(result.text, "hello qwen")
        self.assertEqual(requests[0]["url"], "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/responses")
        self.assertTrue(requests[0]["stream"])
        self.assertTrue(requests[0]["json"]["stream"])
        self.assertEqual(requests[0]["json"]["previous_response_id"], "resp_previous")
        self.assertEqual(requests[0]["json"]["tools"][0]["type"], "web_search")
        provider_events = [event for event in events if event.type == "provider-data"]
        self.assertTrue(provider_events)
        self.assertTrue(
            any(
                isinstance(event.data, dict)
                and event.data.get("type") == "response.reasoning_summary_text.delta"
                for event in provider_events
            )
        )

    async def test_qwen_keeps_embeddings_on_openai_compatible_endpoint(self) -> None:
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
                payload={"data": [{"embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 2, "total_tokens": 2}},
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        result = await embed(model=provider.native.embedding_model("text-embedding-v4"), value="hello")

        self.assertEqual(result.embedding, [0.1, 0.2])
        self.assertEqual(requests[0]["url"], "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/embeddings")

    async def test_qwen_maps_hosted_tools_with_required_reasoning(self) -> None:
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
                    "output": [
                        {"type": "reasoning", "summary": [{"text": "thinking"}]},
                        {"type": "web_search_call", "id": "search_1", "action": {"query": "Alibaba Cloud"}},
                        {"type": "web_search_image_call", "id": "t2i_1", "action": {"query": "Qwen logo"}},
                        {"type": "image_search_call", "id": "i2i_1", "action": {"image_url": "https://example.com/logo.png"}},
                        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
                    ],
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider.native.language_model("qwen3.5-plus"),
            prompt="research",
            tools={
                "search": qwen_web_search_tool(),
                "extract": qwen_web_extractor_tool(),
                "code": qwen_code_interpreter_tool(),
                "t2i": qwen_web_search_image_tool(),
                "i2i": qwen_image_search_tool(),
                "files": qwen_file_search_tool(vector_store_ids=["vs_123"]),
                "weather": tool(name="weather", schema=WeatherToolInput, execute=lambda input: {"ok": True}),
            },
            reasoning=ReasoningConfig(effort="high"),
        )

        self.assertEqual(result.text, "done")
        self.assertEqual(
            [item["type"] for item in requests[0]["json"]["tools"]],
            [
                "web_search",
                "web_extractor",
                "code_interpreter",
                "web_search_image",
                "image_search",
                "file_search",
                "function",
            ],
        )
        self.assertEqual(requests[0]["json"]["tools"][5]["vector_store_ids"], ["vs_123"])
        self.assertNotIn("tool_choice", requests[0]["json"])
        self.assertEqual(requests[0]["json"]["reasoning"], {"effort": "high"})
        self.assertNotIn("enable_thinking", requests[0]["json"])
        provider_parts = [part for part in result.steps[0].response.messages[0].parts if getattr(part, "type", None) == "provider-data"]
        self.assertEqual(provider_parts[0].provider, "qwen")
        managed_calls = [
            part.tool_call.name
            for part in result.steps[0].response.messages[0].parts
            if getattr(part, "type", None) == "tool-call"
        ]
        self.assertIn("web_search_image", managed_calls)
        self.assertIn("image_search", managed_calls)

    async def test_qwen_rejects_invalid_hosted_tool_combinations_before_fetch(self) -> None:
        fetch = AsyncMock()
        provider = create_qwen(api_key="test", fetch=fetch)
        model = provider.native.language_model("qwen3.7-plus")

        with self.assertRaisesRegex(ValidationError, "requires the .*web_search.* tool"):
            await generate_text(
                model=model,
                prompt="extract",
                tools={"extract": qwen_web_extractor_tool()},
                reasoning=ReasoningConfig(effort="high"),
            )
        with self.assertRaisesRegex(ValidationError, 'does not support reasoning effort "none"'):
            await generate_text(
                model=model,
                prompt="calculate",
                tools={"code": qwen_code_interpreter_tool()},
                reasoning=ReasoningConfig(effort="none"),
            )
        with self.assertRaisesRegex(UnsupportedFeatureError, "required or named tool choice"):
            await generate_text(
                model=model,
                prompt="calculate",
                tools={
                    "code": qwen_code_interpreter_tool(),
                    "weather": tool(name="weather", schema=WeatherToolInput, execute=lambda input: {"ok": True}),
                },
                tool_choice=ToolChoiceName(tool_name="weather"),
                reasoning=ReasoningConfig(effort="high"),
            )

        fetch.assert_not_called()

    async def test_qwen_maps_mcp_tool_and_accepts_dashscope_env_key(self) -> None:
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
            requests.append({"headers": headers, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={
                    "status": "completed",
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
                },
            )

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "dashscope-key"}, clear=False):
            provider = create_qwen(fetch=fetch)
        await generate_text(
            model=provider.native.language_model("qwen3.5-plus"),
            prompt="use mcp",
            tools={
                "maps": qwen_mcp_tool(
                    server_label="maps",
                    server_url="https://mcp.example.com/sse",
                    server_description="Map tools",
                    headers={"Authorization": "Bearer token"},
                )
            },
        )

        self.assertEqual(requests[0]["headers"]["authorization"], "Bearer dashscope-key")
        self.assertEqual(
            requests[0]["json"]["tools"][0],
            {
                "type": "mcp",
                "server_label": "maps",
                "server_url": "https://mcp.example.com/sse",
                "server_protocol": "sse",
                "server_description": "Map tools",
                "headers": {"Authorization": "Bearer token"},
            },
        )

    async def test_qwen_transcribes_audio_with_openai_compatible_chat_api(self) -> None:
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
                    "choices": [
                        {
                            "message": {
                                "content": "Welcome to Alibaba Cloud.",
                                "annotations": [{"type": "audio_info", "language": "en", "emotion": "neutral"}],
                            }
                        }
                    ]
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        result = await transcribe_audio(
            model=provider.native.transcription_model("qwen3-asr-flash"),
            audio=AudioInput(data=b"audio", media_type="audio/wav", filename="sample.wav"),
            prompt="Product names: Zhivex",
            language="en",
            provider_options={"asr_options": {"enable_itn": True}},
        )

        self.assertEqual(result.text, "Welcome to Alibaba Cloud.")
        self.assertEqual(requests[0]["url"], "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.assertEqual(requests[0]["json"]["messages"][0], {"role": "system", "content": [{"text": "Product names: Zhivex"}]})
        self.assertEqual(requests[0]["json"]["messages"][1]["content"][0]["type"], "input_audio")
        self.assertTrue(requests[0]["json"]["messages"][1]["content"][0]["input_audio"]["data"].startswith("data:audio/wav;base64,"))
        self.assertEqual(requests[0]["json"]["asr_options"], {"enable_itn": True, "language": "en"})

    async def test_qwen_rejects_reasoning_budget_for_responses_api(self) -> None:
        provider = create_qwen(api_key="test")

        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider.native.language_model("qwen3.5-plus"),
                prompt="hello",
                reasoning=ReasoningConfig(budget_tokens=50),
            )

    async def test_qwen_preserves_all_responses_reasoning_effort_levels(self) -> None:
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
            requests.append(json_body or {})
            return FakeResponse(
                status_code=200,
                payload={
                    "status": "completed",
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        efforts = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
        for effort in efforts:
            await generate_text(
                model=provider.native.language_model("qwen3.8-max"),
                prompt="hello",
                reasoning=ReasoningConfig(effort=effort),  # type: ignore[arg-type]
            )

        self.assertEqual([request["reasoning"]["effort"] for request in requests], efforts)
        self.assertTrue(all("enable_thinking" not in request for request in requests))

    async def test_qwen38_disables_default_thinking_for_forced_tool_choice(self) -> None:
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
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok"}],
                        }
                    ],
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("qwen3.8-max"),
            prompt="weather",
            tools={
                "weather": tool(
                    name="weather",
                    schema=WeatherToolInput,
                    execute=lambda input: {"ok": True},
                )
            },
            tool_choice=ToolChoiceName(tool_name="weather"),
        )

        self.assertTrue(requests[0]["url"].endswith("/responses"))
        self.assertEqual(requests[0]["json"]["reasoning"], {"effort": "none"})

    async def test_qwen38_chat_disables_default_thinking_for_forced_tool_choices(self) -> None:
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
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        model = provider.native.language_model("qwen3.8-max")
        weather = tool(
            name="weather",
            schema=WeatherToolInput,
            execute=lambda input: {"ok": True},
        )
        for tool_choice in (ToolChoiceName(tool_name="weather"), "required"):
            with self.subTest(tool_choice=tool_choice):
                await generate_text(
                    model=model,
                    messages=[
                        ModelMessage(
                            role="user",
                            parts=[
                                TextPart(text="Describe the weather in this video."),
                                FilePart(url="https://example.com/weather.mp4", media_type="video/mp4"),
                            ],
                        )
                    ],
                    tools={"weather": weather},
                    tool_choice=tool_choice,  # type: ignore[arg-type]
                )

        self.assertEqual(len(requests), 2)
        self.assertTrue(all(request["url"].endswith("/chat/completions") for request in requests))
        self.assertTrue(all(request["json"]["enable_thinking"] is False for request in requests))

    async def test_qwen38_chat_rejects_explicit_thinking_for_forced_tool_choices(self) -> None:
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
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        model = provider.native.language_model("qwen3.8-max")
        weather = tool(
            name="weather",
            schema=WeatherToolInput,
            execute=lambda input: {"ok": True},
        )
        thinking_configs: tuple[tuple[str, dict[str, Any]], ...] = (
            ("reasoning", {"reasoning": ReasoningConfig(effort="high")}),
            ("enable_thinking", {"provider_options": {"enable_thinking": True}}),
            ("reasoning_effort", {"provider_options": {"reasoning_effort": "high"}}),
            ("thinking_budget", {"provider_options": {"thinking_budget": 1024}}),
        )
        for tool_choice in (ToolChoiceName(tool_name="weather"), "required"):
            for config_name, thinking_options in thinking_configs:
                with self.subTest(tool_choice=tool_choice, thinking_config=config_name):
                    with self.assertRaisesRegex(UnsupportedFeatureError, "while thinking is enabled"):
                        await generate_text(
                            model=model,
                            messages=[
                                ModelMessage(
                                    role="user",
                                    parts=[
                                        TextPart(text="Describe the weather in this video."),
                                        FilePart(url="https://example.com/weather.mp4", media_type="video/mp4"),
                                    ],
                                )
                            ],
                            tools={"weather": weather},
                            tool_choice=tool_choice,  # type: ignore[arg-type]
                            **thinking_options,
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
                            "url": "https://dashscope-result.oss-cn-beijing.aliyuncs.com/audio.wav",
                        },
                    }
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        result = await generate_speech(
            model=provider.native.speech_model("qwen3-tts-flash"),
            input="hello",
            provider_options={"language_type": "English", "instructions": "Speak clearly."},
        )

        self.assertEqual(result.audio, b"qwen-voice")
        self.assertEqual(result.media_type, "audio/wav")
        self.assertIn("/api/v1/services/aigc/multimodal-generation/generation", requests[0]["url"])
        self.assertEqual(requests[0]["json"]["input"]["voice"], "Cherry")
        self.assertEqual(requests[0]["json"]["input"]["language_type"], "English")
        self.assertEqual(requests[0]["json"]["input"]["instructions"], "Speak clearly.")
        self.assertEqual(requests[1]["method"], "GET")
        self.assertEqual(
            requests[1]["url"],
            "https://dashscope-result.oss-cn-beijing.aliyuncs.com/audio.wav",
        )

    async def test_qwen_upgrades_official_signed_audio_url_to_https(self) -> None:
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
            requests.append({"url": url, "method": method})
            if method == "GET":
                return FakeResponse(
                    status_code=200,
                    body_bytes=b"qwen-voice",
                    headers={"content-type": "audio/mpeg"},
                )
            return FakeResponse(
                status_code=200,
                payload={
                    "output": {
                        "audio": {
                            "url": (
                                "http://dashscope-result-sgp.oss-ap-southeast-1.aliyuncs.com/"
                                "audio.mp3?Expires=123&Signature=signed"
                            )
                        }
                    }
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        result = await generate_speech(
            model=provider.native.speech_model("qwen3-tts-flash"),
            input="hello",
        )

        self.assertEqual(result.audio, b"qwen-voice")
        self.assertEqual(
            requests[1]["url"],
            (
                "https://dashscope-result-sgp.oss-ap-southeast-1.aliyuncs.com/"
                "audio.mp3?Expires=123&Signature=signed"
            ),
        )

    async def test_qwen_speech_rejects_untrusted_audio_url(self) -> None:
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
            if method == "GET":
                raise AssertionError("Unsafe audio URL should not be fetched.")
            return FakeResponse(
                status_code=200,
                payload={
                    "output": {
                        "finish_reason": "stop",
                        "audio": {"url": "http://aliyuncs.com.evil.example/admin"},
                    }
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        with self.assertRaises(ValidationError):
            await generate_speech(model=provider.native.speech_model("qwen-tts"), input="hello")

    async def test_qwen_speech_rejects_noncanonical_loopback_audio_url(self) -> None:
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
            if method == "GET":
                raise AssertionError("Unsafe audio URL should not be fetched.")
            return FakeResponse(
                status_code=200,
                payload={
                    "output": {
                        "finish_reason": "stop",
                        "audio": {"url": "https://2130706433/admin"},
                    }
                },
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        with self.assertRaises(ValidationError):
            await generate_speech(model=provider.native.speech_model("qwen-tts"), input="hello")
