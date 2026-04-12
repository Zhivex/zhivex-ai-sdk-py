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
    FilePart,
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
    openai_apply_patch_tool,
    openai_code_interpreter_container,
    openai_code_interpreter_tool,
    openai_computer_use_tool,
    openai_custom_tool,
    openai_custom_tool_format_grammar,
    openai_custom_tool_format_text,
    openai_domain_secret,
    openai_file_search_filter,
    openai_file_search_filter_group,
    openai_file_search_tool,
    openai_image_generation_tool,
    openai_inline_skill,
    openai_inline_skill_source,
    openai_local_shell_tool,
    openai_local_skill,
    openai_mcp_tool,
    openai_namespace_tool,
    openai_network_policy_allowlist,
    openai_response_options,
    openai_shell_environment,
    openai_shell_tool,
    openai_skill_reference,
    openai_tool_search_tool,
    openai_user_location,
    openai_web_search_tool,
    stream_text,
    tool,
    transcribe_audio,
)
from zhivex_ai.types import ModelMessage, TextPart


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

    async def test_openai_maps_inline_pdf_file_input(self) -> None:
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
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        await generate_text(
            model=provider("gpt-4o-mini"),
            messages=[
                ModelMessage(
                    role="user",
                    parts=[
                        FilePart(data="JVBERi0xLjQK", media_type="application/pdf", filename="stub.pdf"),
                        TextPart(text="summarize"),
                    ],
                )
            ],
        )

        content = requests[0]["json"]["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_file")
        self.assertEqual(content[0]["file_data"], "JVBERi0xLjQK")
        self.assertEqual(content[0]["filename"], "stub.pdf")

    async def test_openai_maps_file_id_and_url_pdf_inputs(self) -> None:
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
            messages=[
                ModelMessage(role="user", parts=[FilePart(file_id="file_123"), FilePart(url="https://example.com/doc.pdf")])
            ],
        )

        content = requests[0]["json"]["input"][0]["content"]
        self.assertEqual(content[0]["file_id"], "file_123")
        self.assertEqual(content[1]["file_url"], "https://example.com/doc.pdf")

    async def test_openai_stream_includes_pdf_inputs(self) -> None:
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
            requests.append({"json": json_body, "stream": stream})
            return FakeResponse(
                status_code=200,
                body_text=(
                    'data: {"type":"response.output_text.delta","delta":"done"}\n\n'
                    'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = stream_text(
            model=provider("gpt-4o-mini"),
            messages=[ModelMessage(role="user", parts=[FilePart(file_id="file_123"), TextPart(text="summarize")])],
        )
        await result.collect()

        self.assertTrue(requests[0]["stream"])
        self.assertEqual(requests[0]["json"]["input"][0]["content"][0]["type"], "input_file")

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

    async def test_openai_files_client_crud(self) -> None:
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
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body, "body": body})
            if method == "GET" and url.endswith("/files"):
                return FakeResponse(status_code=200, payload={"data": [{"id": "file_1", "filename": "stub.pdf", "bytes": 12, "status": "processed"}]})
            if method == "GET":
                return FakeResponse(status_code=200, payload={"id": "file_1", "filename": "stub.pdf", "bytes": 12, "status": "processed"})
            if method == "DELETE":
                return FakeResponse(status_code=200, payload={"id": "file_1", "deleted": True})
            return FakeResponse(status_code=200, payload={"id": "file_1", "filename": "stub.pdf", "bytes": 12, "status": "processed"})

        provider = create_openai(api_key="test", fetch=fetch)
        files = provider.files()
        created = await files.upload(data=b"%PDF-1.4", filename="stub.pdf")
        listed = await files.list()
        fetched = await files.get("file_1")
        deleted = await files.delete("file_1")

        self.assertEqual(created.id, "file_1")
        self.assertEqual(listed[0].filename, "stub.pdf")
        self.assertEqual(fetched.size_bytes, 12)
        self.assertTrue(deleted)
        self.assertEqual(requests[0]["body"]["data"]["purpose"], "assistants")
        self.assertIn("file", requests[0]["body"]["files"])

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
        self.assertEqual(requests[0]["json"]["tools"][0]["type"], "web_search")

    async def test_openai_merges_hosted_and_local_tools(self) -> None:
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
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        await generate_text(
            model=provider("gpt-5.4-mini"),
            prompt="hello",
            tools={"weather": tool(name="weather", schema=WeatherToolInput, execute=lambda input: {"ok": True})},
            provider_options={"tools": [{"type": "web_search"}]},
        )

        self.assertEqual([item["type"] for item in requests[0]["json"]["tools"]], ["function", "web_search"])

    async def test_openai_ignores_provider_managed_tool_calls_in_local_loop(self) -> None:
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
            return FakeResponse(
                status_code=200,
                payload={
                    "status": "completed",
                    "output": [
                        {
                            "type": "web_search_call",
                            "id": "ws_1",
                            "status": "completed",
                            "action": {"query": "weather in BA"},
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "sunny"}],
                        },
                    ],
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider("gpt-5.4-mini"),
            prompt="weather?",
            provider_options={"tools": [{"type": "web_search"}]},
        )

        self.assertEqual(result.text, "sunny")
        assistant_messages = [message for message in result.messages if message.role == "assistant"]
        tool_calls = [part.tool_call for message in assistant_messages for part in message.parts if getattr(part, "type", None) == "tool-call"]
        self.assertTrue(any(call.provider_metadata.get("provider_managed") for call in tool_calls))

    async def test_openai_maps_image_generation_outputs_to_image_parts(self) -> None:
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
            return FakeResponse(
                status_code=200,
                payload={
                    "status": "completed",
                    "output": [
                        {
                            "type": "image_generation_call",
                            "id": "img_1",
                            "result": "QUJD",
                            "output_format": "png",
                            "status": "completed",
                        }
                    ],
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider("gpt-5.4-mini"),
            prompt="draw a square",
            provider_options={"tools": [{"type": "image_generation"}]},
        )

        assistant_parts = [part for message in result.messages if message.role == "assistant" for part in message.parts]
        image_parts = [part for part in assistant_parts if getattr(part, "type", None) == "image"]
        self.assertEqual(len(image_parts), 1)
        self.assertTrue(image_parts[0].image.startswith("data:image/png;base64,"))

    async def test_openai_maps_code_interpreter_image_outputs_to_image_parts(self) -> None:
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
            return FakeResponse(
                status_code=200,
                payload={
                    "status": "completed",
                    "output": [
                        {
                            "type": "code_interpreter_call",
                            "id": "ci_1",
                            "container_id": "cont_1",
                            "code": "print('ok')",
                            "outputs": [{"type": "image", "url": "https://example.com/chart.png"}],
                            "status": "completed",
                        }
                    ],
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider("gpt-5.4-mini"),
            prompt="plot a chart",
            provider_options={"tools": [openai_code_interpreter_tool()]},
        )

        assistant_parts = [part for message in result.messages if message.role == "assistant" for part in message.parts]
        image_parts = [part for part in assistant_parts if getattr(part, "type", None) == "image"]
        self.assertEqual(len(image_parts), 1)
        self.assertEqual(image_parts[0].image, "https://example.com/chart.png")

    async def test_openai_maps_code_interpreter_logs_and_hosted_outputs(self) -> None:
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
            return FakeResponse(
                status_code=200,
                payload={
                    "status": "completed",
                    "output": [
                        {
                            "type": "code_interpreter_call",
                            "id": "ci_1",
                            "container_id": "cont_1",
                            "code": "print('hola')",
                            "outputs": [{"type": "logs", "logs": "hola\n"}],
                            "status": "completed",
                        },
                        {
                            "type": "computer_call_output",
                            "id": "co_1",
                            "call_id": "cc_1",
                            "output": {"type": "computer_screenshot", "image_url": "https://example.com/screen.png"},
                        },
                        {
                            "type": "file_search_call",
                            "id": "fs_1",
                            "queries": ["sdk"],
                            "results": [{"file_id": "file_1", "filename": "notes.txt", "text": "matching chunk", "score": 0.9}],
                            "status": "completed",
                        },
                    ],
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider("gpt-5.4-mini"),
            prompt="inspect tools",
            provider_options={"tools": [openai_code_interpreter_tool(), openai_computer_use_tool(), openai_file_search_tool(vector_store_ids=["vs_1"])]},
        )

        assistant_parts = [part for message in result.messages if message.role == "assistant" for part in message.parts]
        self.assertTrue(any(getattr(part, "type", None) == "generated-code" for part in assistant_parts))
        self.assertTrue(any(getattr(part, "type", None) == "code-result" for part in assistant_parts))
        self.assertTrue(any(getattr(part, "type", None) == "image" and getattr(part, "image", None) == "https://example.com/screen.png" for part in assistant_parts))
        self.assertTrue(any(getattr(part, "type", None) == "file" and getattr(part, "text", None) == "matching chunk" for part in assistant_parts))

    async def test_openai_exposes_responses_client(self) -> None:
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
            requests.append({"url": url, "method": method, "json": json_body})
            return FakeResponse(status_code=200, payload={"ok": True, "url": url, "method": method})

        provider = create_openai(api_key="test", fetch=fetch)
        client = provider.responses()

        await client.create({"model": "gpt-5.4-mini", "input": "hello"})
        await client.create_background({"model": "gpt-5.4-mini", "input": "hello again"})
        await client.retrieve("resp_1", include=["output"], stream=True, starting_after=42, include_obfuscation=True)
        await client.list_input_items("resp_1", limit=20, order="asc")
        await client.count_input_tokens({"model": "gpt-5.4-mini", "input": "hello"})
        await client.cancel("resp_1")
        await client.compact({"model": "gpt-5.4-mini", "conversation": "conv_1"})
        await client.delete("resp_1")

        self.assertEqual(requests[0]["url"], "https://api.openai.com/v1/responses")
        self.assertTrue(requests[1]["json"]["background"])
        self.assertIn("/responses/resp_1?include=output&stream=True&starting_after=42&include_obfuscation=True", requests[2]["url"])
        self.assertIn("/responses/resp_1/input_items?limit=20&order=asc", requests[3]["url"])
        self.assertEqual(requests[4]["url"], "https://api.openai.com/v1/responses/input_tokens")
        self.assertEqual(requests[5]["url"], "https://api.openai.com/v1/responses/resp_1/cancel")
        self.assertEqual(requests[6]["url"], "https://api.openai.com/v1/responses/compact")
        self.assertEqual(requests[7]["method"], "DELETE")

    async def test_openai_responses_wait_polls_until_terminal_status(self) -> None:
        requests: list[str] = []
        payloads = [
            {"id": "resp_1", "status": "queued"},
            {"id": "resp_1", "status": "in_progress"},
            {"id": "resp_1", "status": "completed", "output_text": "done"},
        ]

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
            requests.append(url)
            return FakeResponse(status_code=200, payload=payloads.pop(0))

        provider = create_openai(api_key="test", fetch=fetch)
        result = await provider.responses().wait("resp_1", poll_interval_ms=0, include=["output"], starting_after=7)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(requests), 3)
        self.assertTrue(all("/responses/resp_1?include=output&starting_after=7" in url for url in requests))

    async def test_openai_hosted_tool_builders(self) -> None:
        location = openai_user_location(city="Buenos Aires", country="AR", timezone="America/Argentina/Buenos_Aires")
        search_tool = openai_web_search_tool(search_context_size="high", user_location=location)
        file_tool = openai_file_search_tool(
            vector_store_ids=["vs_123"],
            filters=openai_file_search_filter_group(
                "and",
                [openai_file_search_filter(key="lang", operator="eq", value="es")],
            ),
            max_num_results=5,
        )
        image_tool = openai_image_generation_tool(model="gpt-image-1.5", output_format="webp", quality="high")
        shell_tool = openai_shell_tool(
            environment=openai_shell_environment(
                use_local=True,
                local_skills=[openai_local_skill(name="checks", path="/skills/checks", description="Run checks")],
            )
        )
        code_tool = openai_code_interpreter_tool(
            container=openai_code_interpreter_container(
                file_ids=["file_123"],
                memory_limit="4g",
                network_policy=openai_network_policy_allowlist(
                    allowed_domains=["example.com"],
                    domain_secrets=[openai_domain_secret(domain="example.com", name="TOKEN", value="secret")],
                ),
            )
        )
        custom_tool = openai_custom_tool(
            name="raw_query",
            description="Run a raw query",
            format=openai_custom_tool_format_grammar(syntax="regex", definition=".*"),
        )
        namespace_tool = openai_namespace_tool(name="workspace", description="Workspace tools", tools=[custom_tool])
        tool_search_tool = openai_tool_search_tool(description="Find deferred tools", execution="client", parameters={"type": "object"})
        inline_skill = openai_inline_skill(
            name="zip-skill",
            description="Inline zip skill",
            source=openai_inline_skill_source(data="UEsDBAoAAAAA"),
        )
        skill_ref = openai_skill_reference(skill_id="skill_123", version="latest")
        options = openai_response_options(
            tools=[
                search_tool,
                file_tool,
                image_tool,
                shell_tool,
                code_tool,
                namespace_tool,
                tool_search_tool,
                openai_local_shell_tool(),
                openai_apply_patch_tool(),
                openai_computer_use_tool(environment="browser"),
                openai_mcp_tool(server_url="https://mcp.example.com", server_label="Example MCP"),
            ],
            background=True,
            conversation="conv_123",
            previous_response_id="resp_prev",
            include=["reasoning.encrypted_content"],
            metadata={"feature": "hosted-tools"},
            store=True,
        )

        self.assertEqual(search_tool["type"], "web_search")
        self.assertEqual(search_tool["user_location"]["city"], "Buenos Aires")
        self.assertEqual(file_tool["filters"]["filters"][0]["value"], "es")
        self.assertEqual(image_tool["output_format"], "webp")
        self.assertEqual(shell_tool["environment"]["type"], "local")
        self.assertEqual(code_tool["container"]["type"], "auto")
        self.assertEqual(code_tool["container"]["network_policy"]["domain_secrets"][0]["name"], "TOKEN")
        self.assertEqual(namespace_tool["tools"][0]["format"]["type"], "grammar")
        self.assertEqual(tool_search_tool["type"], "tool_search")
        self.assertEqual(inline_skill["source"]["type"], "base64")
        self.assertEqual(skill_ref["type"], "skill_reference")
        self.assertEqual(openai_custom_tool_format_text()["type"], "text")
        self.assertEqual(options["tools"][-1]["type"], "mcp")
        self.assertTrue(options["background"])
        self.assertEqual(options["conversation"], "conv_123")

    async def test_openai_response_options_builder_integrates_with_generate_text(self) -> None:
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
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        await generate_text(
            model=provider("gpt-5.4-mini"),
            prompt="hello",
            provider_options=openai_response_options(
                tools=[openai_file_search_tool(vector_store_ids=["vs_123"])],
                background=True,
                conversation="conv_123",
                previous_response_id="resp_prev",
            ),
        )

        self.assertEqual(requests[0]["json"]["tools"][0]["type"], "file_search")
        self.assertTrue(requests[0]["json"]["background"])
        self.assertEqual(requests[0]["json"]["conversation"], "conv_123")
        self.assertEqual(requests[0]["json"]["previous_response_id"], "resp_prev")

    async def test_openai_exposes_conversations_client(self) -> None:
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
            requests.append({"url": url, "method": method, "json": json_body})
            return FakeResponse(status_code=200, payload={"ok": True, "url": url, "method": method})

        provider = create_openai(api_key="test", fetch=fetch)
        client = provider.conversations()

        await client.create({"metadata": {"topic": "demo"}})
        await client.retrieve("conv_1")
        await client.update("conv_1", {"metadata": {"topic": "updated"}})
        await client.create_item(
            "conv_1",
            {"items": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}]},
            include=["output"],
        )
        await client.retrieve_item("conv_1", "msg_1", include=["output"])
        await client.list_items("conv_1", limit=10, order="desc")
        await client.delete_item("conv_1", "msg_1")
        await client.delete("conv_1")

        self.assertEqual(requests[0]["url"], "https://api.openai.com/v1/conversations")
        self.assertEqual(requests[1]["method"], "GET")
        self.assertEqual(requests[2]["method"], "POST")
        self.assertEqual(requests[3]["url"], "https://api.openai.com/v1/conversations/conv_1/items?include=output")
        self.assertEqual(requests[4]["url"], "https://api.openai.com/v1/conversations/conv_1/items/msg_1?include=output")
        self.assertIn("/conversations/conv_1/items?limit=10&order=desc", requests[5]["url"])
        self.assertEqual(requests[6]["method"], "DELETE")
        self.assertEqual(requests[7]["method"], "DELETE")

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
