from __future__ import annotations

import json
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
import sys
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    FilePart,
    UnsupportedFeatureError,
    ValidationError,
    anthropic_code_execution_tool,
    anthropic_mcp_server,
    anthropic_web_search_tool,
    create_anthropic,
    generate_grounded_text,
    generate_object,
    generate_text,
    stream_text,
    tool,
)
from zhivex_ai.types import ImagePart, ModelGenerateInput, ModelMessage, ReasoningConfig, StructuredOutputConfig, TextPart, ToolChoiceName


@dataclass
class FakeResponse:
    status_code: int
    payload: Any = None
    body_text: str = ""
    body_bytes: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)

    async def json(self) -> Any:
        return self.payload

    async def text(self) -> str:
        return self.body_text or json.dumps(self.payload)

    async def read(self) -> bytes:
        if self.body_bytes is not None:
            return self.body_bytes
        return (self.body_text or json.dumps(self.payload)).encode("utf-8")

    async def iter_lines(self) -> AsyncIterable[str]:
        for line in self.body_text.splitlines():
            yield line


class AnthropicProviderTests(IsolatedAsyncioTestCase):
    async def test_anthropic_portable_generate_text_supports_tier_one_path(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": "portable ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 2, "output_tokens": 2},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        result = await generate_text(model=provider("claude-sonnet-4-20250514"), prompt="hello")

        self.assertEqual(result.text, "portable ok")

    async def test_anthropic_portable_streaming_and_structured_output_work(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            if stream:
                return FakeResponse(
                    status_code=200,
                    body_text=(
                        'event: content_block_delta\n'
                        'data: {"delta":{"type":"text_delta","text":"hello "}}\n\n'
                        'event: content_block_delta\n'
                        'data: {"delta":{"type":"text_delta","text":"world"}}\n\n'
                        'event: message_delta\n'
                        'data: {"delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":2,"output_tokens":2}}\n\n'
                        'event: message_stop\n'
                        'data: {"stop_reason":"end_turn"}\n'
                    ),
                )
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": '{"answer":"portable json"}'}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)

        streamed = stream_text(model=provider("claude-sonnet-4-20250514"), prompt="hello")
        streamed_result = await streamed.collect()
        structured = await generate_object(
            model=provider("claude-sonnet-4-20250514"),
            prompt="return json",
            schema={"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
        )

        self.assertEqual(streamed_result.text, "hello world")
        self.assertEqual(structured.object["answer"], "portable json")
        self.assertEqual(requests[1]["output_config"]["format"]["type"], "json_schema")

    async def test_anthropic_portable_grounded_generation_is_supported(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srv_1",
                            "content": [
                                {
                                    "type": "web_search_result",
                                    "url": "https://example.com/portable",
                                    "title": "Portable Source",
                                }
                            ],
                        },
                        {"type": "text", "text": "grounded portable"},
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 2, "output_tokens": 2},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        result = await generate_grounded_text(
            model=provider.grounded_language_model("claude-sonnet-4-20250514"),
            prompt="find one fact",
        )

        self.assertEqual(result.text, "grounded portable")
        self.assertEqual(result.sources[0].url, "https://example.com/portable")

    async def test_anthropic_portable_models_reject_provider_options(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(status_code=200, payload={})

        provider = create_anthropic(api_key="test", fetch=fetch)

        with self.assertRaises(ValidationError):
            await generate_text(
                model=provider("claude-sonnet-4-20250514"),
                prompt="hello",
                provider_options={"tools": [anthropic_web_search_tool()]},
            )

    async def test_anthropic_tool_call_roundtrip(self) -> None:
        requests: list[dict[str, Any]] = []
        calls = 0

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            nonlocal calls
            calls += 1
            requests.append(json_body)
            if calls == 1:
                return FakeResponse(
                    status_code=200,
                    payload={
                        "content": [
                            {"type": "thinking", "thinking": "Need math", "signature": "sig-1"},
                            {"type": "tool_use", "id": "tool-1", "name": "math", "input": {"value": 2}},
                        ],
                        "stop_reason": "tool_use",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                )
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": "result is 4"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider.native.language_model("claude-3-5-sonnet"),
            prompt="double 2",
            max_steps=2,
            tools={
                "math": tool(
                    name="math",
                    schema=dict[str, int],
                    execute=lambda input: {"result": input["value"] * 2},
                )
            },
        )
        self.assertEqual(result.text, "result is 4")
        self.assertEqual(result.tool_results[0].tool_name, "math")
        assistant_blocks = requests[1]["messages"][1]["content"]
        self.assertEqual(assistant_blocks[0]["type"], "thinking")
        self.assertEqual(assistant_blocks[1]["type"], "tool_use")

    async def test_anthropic_maps_hosted_tools_from_tools_set(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
                headers={},
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("claude-sonnet-4-20250514"),
            prompt="hello",
            tools={
                "lookup": tool(name="lookup", schema=dict[str, str], execute=lambda input: {"ok": True}),
                "search": anthropic_web_search_tool(max_uses=2),
                "mcp": anthropic_mcp_server(url="https://mcp.example.com", name="example-mcp", allowed_tools=["echo"]),
                "code": anthropic_code_execution_tool(),
            },
        )

        self.assertEqual(requests[0]["tools"][0]["name"], "lookup")
        self.assertEqual(requests[0]["tools"][1]["type"], "web_search_20250305")
        self.assertEqual(requests[0]["tools"][2]["type"], "mcp_toolset")
        self.assertEqual(requests[0]["tools"][2]["mcp_server_name"], "example-mcp")
        self.assertEqual(requests[0]["tools"][3]["type"], "code_execution_20250825")
        self.assertEqual(requests[0]["mcp_servers"][0]["name"], "example-mcp")
        self.assertEqual(requests[0]["mcp_servers"][0]["url"], "https://mcp.example.com")

    async def test_anthropic_rejects_duplicate_mcp_toolsets_for_same_server(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            raise AssertionError("request should not be dispatched")

        provider = create_anthropic(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError) as context:
            await generate_text(
                model=provider.native.language_model("claude-sonnet-4-20250514"),
                prompt="hello",
                tools={
                    "mcp_a": anthropic_mcp_server(url="https://mcp.example.com", name="example-mcp", allowed_tools=["echo"]),
                    "mcp_b": anthropic_mcp_server(url="https://mcp.example.com", name="example-mcp", allowed_tools=["sum"]),
                },
            )

        self.assertIn('multiple "mcp_toolset" entries', str(context.exception))

    async def test_anthropic_rejects_duplicate_mcp_server_declarations_across_tools_and_provider_options(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            raise AssertionError("request should not be dispatched")

        provider = create_anthropic(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError) as context:
            await generate_text(
                model=provider.native.language_model("claude-sonnet-4-20250514"),
                prompt="hello",
                tools={"mcp": anthropic_mcp_server(url="https://mcp.example.com", name="example-mcp")},
                provider_options={"mcp_servers": [{"name": "example-mcp", "url": "https://mcp.example.com"}]},
            )

        self.assertIn('declaring MCP server "example-mcp" in both hosted toolsets', str(context.exception))

    async def test_anthropic_rejects_mixed_first_class_and_raw_mcp_toolsets(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            raise AssertionError("request should not be dispatched")

        provider = create_anthropic(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError) as context:
            await generate_text(
                model=provider.native.language_model("claude-sonnet-4-20250514"),
                prompt="hello",
                tools={"mcp": anthropic_mcp_server(url="https://mcp.example.com", name="example-mcp")},
                provider_options={"tools": [{"type": "mcp_toolset", "mcp_server_name": "backup-mcp"}]},
            )

        self.assertIn('mixing first-class "mcp_toolset" tools', str(context.exception))

    async def test_anthropic_rejects_forced_tool_choice_with_extended_thinking(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(status_code=200, payload={})

        provider = create_anthropic(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider.native.language_model("claude-sonnet-4-20250514"),
                prompt="double 2",
                tools={
                    "math": tool(
                        name="math",
                        schema=dict[str, int],
                        execute=lambda input: {"result": input["value"] * 2},
                    )
                },
                tool_choice=ToolChoiceName(tool_name="math"),
                reasoning=ReasoningConfig(budget_tokens=1024),
            )
        self.assertEqual(requests, [])

    async def test_anthropic_maps_data_url_images_to_base64_source(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": "looks good"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        model = provider.native.language_model("claude-sonnet-4-20250514")
        result = await model.generate(
            ModelGenerateInput(
                messages=[
                    ModelMessage(
                        role="user",
                        parts=[
                            ImagePart(image="data:image/png;base64,aGVsbG8="),
                            TextPart(text="describe this image"),
                        ],
                    )
                ]
            )
        )
        self.assertEqual(result.text, "looks good")
        image_block = requests[0]["messages"][0]["content"][0]
        self.assertEqual(image_block["source"]["type"], "base64")
        self.assertEqual(image_block["source"]["media_type"], "image/png")
        self.assertEqual(image_block["source"]["data"], "aGVsbG8=")

    async def test_anthropic_maps_inline_pdf_file_input(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": "done"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("claude-sonnet-4-20250514"),
            messages=[ModelMessage(role="user", parts=[FilePart(data="JVBERi0xLjQK", media_type="application/pdf", filename="stub.pdf")])],
        )

        document_block = requests[0]["messages"][0]["content"][0]
        self.assertEqual(document_block["type"], "document")
        self.assertEqual(document_block["source"]["type"], "base64")
        self.assertEqual(document_block["source"]["data"], "JVBERi0xLjQK")

    async def test_anthropic_maps_pdf_urls_to_document_source(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": "done"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("claude-sonnet-4-20250514"),
            messages=[ModelMessage(role="user", parts=[FilePart(url="https://example.com/doc.pdf", title="Doc")])],
        )

        document_block = requests[0]["messages"][0]["content"][0]
        self.assertEqual(document_block["type"], "document")
        self.assertEqual(document_block["source"]["type"], "url")
        self.assertEqual(document_block["source"]["url"], "https://example.com/doc.pdf")
        self.assertEqual(document_block["title"], "Doc")

    async def test_anthropic_maps_file_id_pdf_input(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": "done"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("claude-sonnet-4-20250514"),
            messages=[ModelMessage(role="user", parts=[FilePart(file_id="file_123", filename="stub.pdf")])],
        )

        document_block = requests[0]["messages"][0]["content"][0]
        self.assertEqual(document_block["source"]["type"], "file")
        self.assertEqual(document_block["source"]["file_id"], "file_123")

    async def test_anthropic_maps_text_documents_with_citations(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": "done"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("claude-sonnet-4-20250514"),
            messages=[
                ModelMessage(
                    role="user",
                    parts=[
                        FilePart(
                            text="Quarterly revenue grew 20% year over year.",
                            title="Q1 Update",
                            context="company=Acme",
                            citations_enabled=True,
                        )
                    ],
                )
            ],
        )

        document_block = requests[0]["messages"][0]["content"][0]
        self.assertEqual(document_block["source"]["type"], "text")
        self.assertEqual(document_block["source"]["data"], "Quarterly revenue grew 20% year over year.")
        self.assertEqual(document_block["context"], "company=Acme")
        self.assertEqual(document_block["citations"], {"enabled": True})

    async def test_anthropic_stream_includes_thinking_without_null_fields(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                body_text='event: message_stop\ndata: {"stop_reason":"end_turn"}\n',
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        model = provider.native.language_model("claude-sonnet-4-20250514")
        events = []
        async for event in await model.stream(
            ModelGenerateInput(
                messages=[ModelMessage(role="user", parts=[TextPart(text="hello")])],
                reasoning=ReasoningConfig(budget_tokens=2048),
            )
        ):
            events.append(event)

        self.assertEqual(len(events), 1)
        self.assertEqual(requests[0]["thinking"], {"type": "enabled", "budget_tokens": 2048})
        self.assertNotIn("temperature", requests[0])

    async def test_anthropic_stream_handles_server_tool_events(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(
                status_code=200,
                body_text=(
                    'event: content_block_start\n'
                    'data: {"index":1,"content_block":{"type":"server_tool_use","id":"srv_1","name":"web_search"}}\n\n'
                    'event: content_block_delta\n'
                    'data: {"index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"query\\":\\"latest mars news\\"}"}}\n\n'
                    'event: content_block_stop\n'
                    'data: {"index":1}\n\n'
                    'event: message_delta\n'
                    'data: {"delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":7,"output_tokens":3}}\n\n'
                    'event: message_stop\n'
                    'data: {"stop_reason":"end_turn"}\n'
                ),
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        model = provider.native.language_model("claude-sonnet-4-20250514")
        events = []
        async for event in await model.stream(ModelGenerateInput(messages=[ModelMessage(role="user", parts=[TextPart(text="search")])])):
            events.append(event)

        self.assertEqual(events[0].tool_call.name, "web_search")
        self.assertTrue(events[0].tool_call.provider_metadata["provider_managed"])
        self.assertEqual(events[0].tool_call.input, {"query": "latest mars news"})
        self.assertEqual(events[-1].usage.total_tokens, 10)

    async def test_anthropic_stream_handles_mcp_tool_events(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(
                status_code=200,
                body_text=(
                    'event: content_block_start\n'
                    'data: {"index":1,"content_block":{"type":"mcp_tool_use","id":"mcp_1","name":"echo","server_name":"example-mcp"}}\n\n'
                    'event: content_block_delta\n'
                    'data: {"index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"value\\":\\"hi\\"}"}}\n\n'
                    'event: content_block_stop\n'
                    'data: {"index":1}\n\n'
                    'event: message_stop\n'
                    'data: {"stop_reason":"end_turn"}\n'
                ),
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        model = provider.native.language_model("claude-sonnet-4-20250514")
        events = []
        async for event in await model.stream(
            ModelGenerateInput(messages=[ModelMessage(role="user", parts=[TextPart(text="search")])], provider_options={"mcp_servers": [anthropic_mcp_server(url="https://mcp.example.com", name="example-mcp")]})
        ):
            events.append(event)

        self.assertEqual(events[0].tool_call.name, "echo")
        self.assertTrue(events[0].tool_call.provider_metadata["provider_managed"])
        self.assertEqual(events[0].tool_call.provider_metadata["server_name"], "example-mcp")
        self.assertEqual(events[0].tool_call.input, {"value": "hi"})

    async def test_anthropic_adds_current_mcp_beta_header(self) -> None:
        headers_seen: list[dict[str, str]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            headers_seen.append(headers)
            return FakeResponse(
                status_code=200,
                payload={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}},
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("claude-sonnet-4-20250514"),
            prompt="use MCP",
            provider_options={"mcp_servers": [anthropic_mcp_server(url="https://mcp.example.com", name="example-mcp")]},
        )

        self.assertEqual(headers_seen[0]["anthropic-beta"], "mcp-client-2025-04-04")

    async def test_anthropic_adds_code_execution_beta_header(self) -> None:
        headers_seen: list[dict[str, str]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            headers_seen.append(headers)
            return FakeResponse(
                status_code=200,
                payload={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}},
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("claude-sonnet-4-20250514"),
            prompt="run code",
            provider_options={"tools": [anthropic_code_execution_tool()]},
        )

        self.assertIn("code-execution-2025-08-25", headers_seen[0]["anthropic-beta"])

    async def test_anthropic_files_client_crud(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any] | None = None,
            body: Any = None,
            timeout_ms: int | None,
            stream: bool = False,
            method: str = "POST",
        ):
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body, "body": body})
            if method == "GET" and url.endswith("/files"):
                return FakeResponse(status_code=200, payload={"data": [{"id": "file_1", "filename": "stub.pdf", "size_bytes": 12, "status": "processed", "downloadable": True}]})
            if method == "GET" and url.endswith("/content"):
                return FakeResponse(status_code=200, body_bytes=b"file-bytes")
            if method == "GET":
                return FakeResponse(status_code=200, payload={"id": "file_1", "filename": "stub.pdf", "size_bytes": 12, "status": "processed", "downloadable": True})
            if method == "DELETE":
                return FakeResponse(status_code=200, payload={"id": "file_1", "type": "file_deleted"})
            return FakeResponse(status_code=200, payload={"id": "file_1", "filename": "stub.pdf", "size_bytes": 12, "status": "processed", "downloadable": True})

        provider = create_anthropic(api_key="test", fetch=fetch)
        files = provider.files()
        created = await files.upload(data=b"hello", filename="notes.txt", media_type="text/plain")
        listed = await files.list()
        fetched = await files.get("file_1")
        downloaded = await files.download("file_1")
        deleted = await files.delete("file_1")

        self.assertEqual(created.id, "file_1")
        self.assertEqual(listed[0].size_bytes, 12)
        self.assertEqual(fetched.filename, "stub.pdf")
        self.assertEqual(downloaded, b"file-bytes")
        self.assertTrue(deleted)
        self.assertEqual(requests[0]["headers"]["anthropic-beta"], "files-api-2025-04-14")
        self.assertEqual(requests[0]["body"]["files"]["file"][2], "text/plain")

    async def test_anthropic_maps_structured_output_and_tool_metadata(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": '{"value":4}'}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        await provider.native.language_model("claude-sonnet-4-20250514").generate(
            ModelGenerateInput(
                messages=[ModelMessage(role="user", parts=[TextPart(text="return json")])],
                structured_output=StructuredOutputConfig(
                    schema={"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]},
                    mode="native",
                    name="calc",
                ),
            )
        )

        tool_def = tool(
            name="lookup",
            description="Look up data.",
            schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
            execute=lambda input: input,
            strict=True,
            eager_input_streaming=True,
            input_examples=[{"q": "weather in NYC"}],
        )
        await generate_text(
            model=provider.native.language_model("claude-sonnet-4-20250514"),
            prompt="lookup",
            tools={"lookup": tool_def},
        )

        self.assertEqual(requests[0]["output_config"]["format"]["type"], "json_schema")
        self.assertTrue(requests[1]["tools"][0]["strict"])
        self.assertTrue(requests[1]["tools"][0]["eager_input_streaming"])
        self.assertEqual(requests[1]["tools"][0]["input_examples"][0]["q"], "weather in NYC")

    async def test_anthropic_merges_local_and_provider_managed_tools(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": "done"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("claude-sonnet-4-20250514"),
            prompt="search and lookup",
            tools={
                "lookup": tool(
                    name="lookup",
                    description="Look up data.",
                    schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
                    execute=lambda input: input,
                )
            },
            provider_options={"tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}]},
        )

        self.assertEqual(len(requests[0]["tools"]), 2)
        self.assertEqual(requests[0]["tools"][0]["name"], "lookup")
        self.assertEqual(requests[0]["tools"][1]["name"], "web_search")

    async def test_anthropic_grounded_text_uses_web_search_and_extracts_sources(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [
                        {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "mars rover"}},
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srv_1",
                            "content": [
                                {
                                    "type": "web_search_result",
                                    "url": "https://example.com/mars",
                                    "title": "Mars Update",
                                    "encrypted_content": "enc",
                                }
                            ],
                        },
                        {
                            "type": "text",
                            "text": "Latest rover update.",
                            "citations": [
                                {
                                    "type": "web_search_result_location",
                                    "url": "https://example.com/mars",
                                    "title": "Mars Update",
                                    "cited_text": "Rover update snippet",
                                    "encrypted_index": "idx",
                                }
                            ],
                        },
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 7, "server_tool_use": {"web_search_requests": 1}},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        result = await generate_grounded_text(
            model=provider.native.grounded_language_model("claude-sonnet-4-20250514"),
            prompt="What is the latest Mars rover update?",
        )

        self.assertEqual(requests[0]["tools"][0]["type"], "web_search_20250305")
        self.assertEqual(result.text, "Latest rover update.")
        self.assertEqual(result.sources[0].url, "https://example.com/mars")
        self.assertTrue(any(source.snippet == "Rover update snippet" for source in result.sources))

    async def test_anthropic_counts_tokens(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str,
            *,
            headers: dict[str, str],
            json_body: dict[str, Any],
            timeout_ms: int | None,
            stream: bool = False,
        ):
            requests.append({"url": url, "headers": headers, "json": json_body})
            return FakeResponse(status_code=200, payload={"input_tokens": 88})

        provider = create_anthropic(api_key="test", fetch=fetch)
        result = await provider.tokens().count(
            model_id="claude-opus-4-20250514",
            prompt="Can you write a formal proof?",
        )

        self.assertEqual(result.total_tokens, 88)
        self.assertEqual(requests[0]["url"], "https://api.anthropic.com/v1/messages/count_tokens")
        self.assertEqual(requests[0]["json"]["model"], "claude-opus-4-20250514")
        self.assertEqual(requests[0]["json"]["messages"][0]["content"][0]["text"], "Can you write a formal proof?")

    def test_anthropic_hosted_tool_builders(self) -> None:
        web_search = anthropic_web_search_tool(max_uses=2, allowed_domains=["example.com"])
        mcp_server = anthropic_mcp_server(url="https://mcp.example.com", name="example-mcp", allowed_tools=["echo"])
        code_execution = anthropic_code_execution_tool()

        self.assertEqual(web_search.type, "web_search_20250305")
        self.assertEqual(web_search.config["max_uses"], 2)
        self.assertEqual(mcp_server.type, "mcp_toolset")
        self.assertEqual(mcp_server.config["server"]["name"], "example-mcp")
        self.assertEqual(mcp_server.config["default_config"]["allowed_tools"], ["echo"])
        self.assertEqual(code_execution.type, "code_execution_20250825")
