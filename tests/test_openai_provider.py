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
    assistant,
    create_openai,
    create_openrouter,
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
    openai_mcp_approval_response,
    openai_mcp_tool,
    openai_namespace_tool,
    openai_network_policy_allowlist,
    openai_response_options,
    openai_response_reference,
    openai_shell_environment,
    openai_shell_tool,
    openai_skill_reference,
    openai_tool_search_tool,
    openai_user_location,
    openai_web_search_tool,
    get_openai_response_id,
    get_openai_response_reference,
    provider_data_part,
    parse_openai_provider_data_part,
    stream_text,
    tool,
    transcribe_audio,
    user,
)
from zhivex_ai.providers.openai import (  # noqa: E402
    get_last_openai_mcp_call,
    get_last_openai_mcp_list_tools_event,
    get_openai_mcp_calls,
    get_openai_mcp_list_tools_events,
    openai_provider_data_tool_call,
)
from zhivex_ai.types import ModelMessage, OpenAIMcpApprovalResponse, TextPart


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

    async def test_openai_file_search_stores_crud(self) -> None:
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
            requests.append({"url": url, "method": method, "json": json_body, "body": body})
            if method == "POST" and url.endswith("/vector_stores"):
                return FakeResponse(status_code=200, payload={"id": "vs_1", "name": "Docs", "created_at": 123})
            if method == "GET" and url.endswith("/vector_stores"):
                return FakeResponse(status_code=200, payload={"data": [{"id": "vs_1", "name": "Docs", "created_at": 123}], "has_more": False})
            if method == "GET" and url.endswith("/vector_stores/vs_1"):
                return FakeResponse(status_code=200, payload={"id": "vs_1", "name": "Docs", "created_at": 123})
            if method == "GET" and url.endswith("/vector_stores/vs_1/files"):
                return FakeResponse(
                    status_code=200,
                    payload={
                        "data": [
                            {
                                "id": "file_1",
                                "vector_store_id": "vs_1",
                                "status": "completed",
                                "filename": "manual.pdf",
                                "attributes": {"lang": "es"},
                                "usage_bytes": 12,
                                "created_at": 456,
                            }
                        ],
                        "has_more": False,
                    },
                )
            if method == "GET" and url.endswith("/vector_stores/vs_1/files/file_1"):
                return FakeResponse(
                    status_code=200,
                    payload={
                        "id": "file_1",
                        "vector_store_id": "vs_1",
                        "status": "completed",
                        "filename": "manual.pdf",
                        "attributes": {"lang": "es"},
                        "usage_bytes": 12,
                        "created_at": 456,
                    },
                )
            if method == "GET" and url.endswith("/vector_stores/vs_1/files/file_1/content"):
                return FakeResponse(status_code=200, body_bytes=b"manual-content")
            if method == "DELETE" and url.endswith("/vector_stores/vs_1/files/file_1"):
                return FakeResponse(status_code=200, payload={"id": "file_1", "deleted": True})
            if method == "DELETE" and url.endswith("/vector_stores/vs_1"):
                return FakeResponse(status_code=200, payload={"id": "vs_1", "deleted": True})
            return FakeResponse(status_code=404, payload={"error": "unexpected"})

        provider = create_openai(api_key="test", fetch=fetch)
        stores = provider.file_search_stores()
        created = await stores.create(display_name="Docs")
        listed = await stores.list()
        fetched = await stores.get("vs_1")
        documents = await stores.list_documents(file_search_store_name="vs_1")
        document = await stores.get_document("vector_stores/vs_1/files/file_1")
        content = await stores.download_document("vector_stores/vs_1/files/file_1")
        deleted_document = await stores.delete_document("vector_stores/vs_1/files/file_1")
        deleted_store = await stores.delete("vs_1")

        self.assertEqual(created.name, "vs_1")
        self.assertEqual(listed.stores[0].display_name, "Docs")
        self.assertEqual(fetched.display_name, "Docs")
        self.assertEqual(documents.documents[0].name, "vector_stores/vs_1/files/file_1")
        self.assertEqual(document.custom_metadata[0]["lang"], "es")
        self.assertEqual(content, b"manual-content")
        self.assertTrue(deleted_document)
        self.assertTrue(deleted_store)

    async def test_openai_file_search_store_upload_waits_on_document_status(self) -> None:
        requests: list[dict[str, Any]] = []
        poll_count = 0

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
            nonlocal poll_count
            requests.append({"url": url, "method": method, "json": json_body, "body": body})
            if method == "POST" and url.endswith("/vector_stores/vs_1/files"):
                return FakeResponse(
                    status_code=200,
                    payload={
                        "id": "file_1",
                        "vector_store_id": "vs_1",
                        "status": "in_progress",
                        "attributes": {"lang": "es"},
                    },
                )
            if method == "POST" and url.endswith("/files"):
                return FakeResponse(status_code=200, payload={"id": "upload_1", "filename": "manual.pdf", "status": "processed"})
            if method == "GET" and url.endswith("/vector_stores/vs_1/files/file_1"):
                poll_count += 1
                return FakeResponse(
                    status_code=200,
                    payload={
                        "id": "file_1",
                        "vector_store_id": "vs_1",
                        "status": "completed" if poll_count >= 2 else "in_progress",
                        "filename": "manual.pdf",
                        "attributes": {"lang": "es"},
                    },
                )
            return FakeResponse(status_code=404, payload={"error": "unexpected"})

        provider = create_openai(api_key="test", fetch=fetch)
        stores = provider.file_search_stores()
        operation = await stores.upload(
            file_search_store_name="vs_1",
            data=b"%PDF-1.4",
            filename="manual.pdf",
            media_type="application/pdf",
            custom_metadata=[{"key": "lang", "value": "es"}],
        )
        waited = await stores.wait_operation(operation.name, poll_interval_ms=0, timeout_ms=50)

        self.assertEqual(operation.name, "vector_stores/vs_1/files/file_1")
        self.assertFalse(operation.done)
        self.assertTrue(waited.done)
        self.assertEqual(requests[1]["json"]["attributes"]["lang"], "es")

    async def test_openai_file_search_stores_update_and_search(self) -> None:
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
            requests.append({"url": url, "method": method, "json": json_body, "body": body})
            if url.endswith("/vector_stores/vs_1") and method == "POST":
                return FakeResponse(status_code=200, payload={"id": "vs_1", "name": "Docs v2", "metadata": {"env": "prod"}})
            if url.endswith("/vector_stores/vs_1/files/file_1") and method == "POST":
                return FakeResponse(
                    status_code=200,
                    payload={"id": "file_1", "vector_store_id": "vs_1", "attributes": {"topic": "billing"}},
                )
            return FakeResponse(
                status_code=200,
                payload={
                    "data": [
                        {
                            "file_id": "file_1",
                            "filename": "guide.md",
                            "score": 0.98,
                            "content": [{"type": "text", "text": "return policy"}],
                        }
                    ]
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        stores = provider.file_search_stores()

        updated = await stores.update("vs_1", display_name="Docs v2", metadata={"env": "prod"})
        document = await stores.update_document("vs_1/file_1", custom_metadata=[{"topic": "billing"}])
        search = await stores.search(file_search_store_name="vs_1", query="return policy", max_num_results=3, rewrite_query=True)

        self.assertEqual(updated.display_name, "Docs v2")
        self.assertEqual(document.custom_metadata, [{"topic": "billing"}])
        self.assertEqual(search.results[0]["filename"], "guide.md")
        self.assertEqual(requests[0]["json"], {"name": "Docs v2", "metadata": {"env": "prod"}})
        self.assertEqual(requests[1]["json"], {"attributes": {"topic": "billing"}})
        self.assertEqual(requests[2]["json"], {"query": "return policy", "max_num_results": 3, "rewrite_query": True})

    async def test_openai_file_search_batches(self) -> None:
        requests: list[dict[str, Any]] = []
        polls = 0

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
            nonlocal polls
            requests.append({"url": url, "method": method, "json": json_body})
            if url.endswith("/vector_stores/vs_1/file_batches") and method == "POST":
                return FakeResponse(status_code=200, payload={"id": "batch_1", "vector_store_id": "vs_1", "status": "in_progress"})
            if url.endswith("/vector_stores/vs_1/file_batches/batch_1") and method == "GET":
                polls += 1
                return FakeResponse(
                    status_code=200,
                    payload={"id": "batch_1", "vector_store_id": "vs_1", "status": "completed" if polls > 1 else "in_progress"},
                )
            if url.endswith("/vector_stores/vs_1/file_batches/batch_1/cancel"):
                return FakeResponse(status_code=200, payload={"id": "batch_1", "vector_store_id": "vs_1", "status": "cancelled"})
            return FakeResponse(
                status_code=200,
                payload={"data": [{"id": "file_1", "vector_store_id": "vs_1", "status": "completed", "filename": "doc.pdf"}], "has_more": False},
            )

        provider = create_openai(api_key="test", fetch=fetch)
        stores = provider.file_search_stores()
        batch = await stores.create_batch(file_search_store_name="vs_1", file_names=["file_1", "file_2"], custom_metadata=[{"team": "docs"}])
        files = await stores.list_batch_documents(name=batch.name)
        waited = await stores.wait_batch(batch.name, poll_interval_ms=0, timeout_ms=50)
        cancelled = await stores.cancel_batch(batch.name)

        self.assertEqual(batch.name, "vector_stores/vs_1/file_batches/batch_1")
        self.assertEqual(files.documents[0].display_name, "doc.pdf")
        self.assertEqual(waited.state, "completed")
        self.assertEqual(cancelled.state, "cancelled")
        self.assertEqual(requests[0]["json"], {"file_ids": ["file_1", "file_2"], "attributes": {"team": "docs"}})

    async def test_openai_images_client_generate_edit_and_variation(self) -> None:
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
            return FakeResponse(
                status_code=200,
                payload={
                    "created": 1719184911,
                    "data": [{"b64_json": base64.b64encode(b"png-bytes").decode("ascii"), "revised_prompt": "a cleaner skyline"}],
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        generated = await provider.images().generate(prompt="city skyline", model="gpt-image-2")
        edited = await provider.images().edit(
            prompt="remove the clouds",
            image=b"png",
            image_filenames="skyline.png",
            model="gpt-image-2",
            mask=b"mask",
        )
        varied = await provider.images().variation(
            image=b"png",
            image_filename="skyline.png",
            model="gpt-image-2",
        )

        self.assertEqual(generated.images[0].revised_prompt, "a cleaner skyline")
        self.assertEqual(edited.images[0].b64_json, base64.b64encode(b"png-bytes").decode("ascii"))
        self.assertEqual(varied.images[0].b64_json, base64.b64encode(b"png-bytes").decode("ascii"))
        self.assertEqual(requests[0]["url"], "https://api.openai.com/v1/images/generations")
        self.assertEqual(requests[0]["json"]["prompt"], "city skyline")
        self.assertEqual(requests[0]["json"]["model"], "gpt-image-2")
        self.assertEqual(requests[1]["url"], "https://api.openai.com/v1/images/edits")
        files_payload = requests[1]["body"]["files"]
        self.assertEqual(files_payload[0][0], "image[]")
        self.assertEqual(files_payload[1][0], "mask")
        self.assertEqual(requests[2]["url"], "https://api.openai.com/v1/images/variations")
        self.assertEqual(requests[2]["body"]["files"]["image"][0], "skyline.png")

    async def test_openai_uploads_client_uploads_bytes_in_parts(self) -> None:
        requests: list[dict[str, Any]] = []
        part_counter = 0

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
            nonlocal part_counter
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body, "body": body})
            if url.endswith("/uploads"):
                return FakeResponse(
                    status_code=200,
                    payload={"id": "upload_1", "filename": "training.jsonl", "purpose": "fine-tune", "bytes": 10, "status": "pending"},
                )
            if url.endswith("/parts"):
                part_counter += 1
                return FakeResponse(
                    status_code=200,
                    payload={"id": f"part_{part_counter}", "upload_id": "upload_1", "created_at": 1719185911},
                )
            if url.endswith("/complete"):
                return FakeResponse(
                    status_code=200,
                    payload={
                        "id": "upload_1",
                        "status": "completed",
                        "file": {"id": "file_1", "filename": "training.jsonl", "bytes": 10, "purpose": "fine-tune"},
                    },
                )
            return FakeResponse(status_code=200, payload={"id": "upload_1", "status": "cancelled"})

        provider = create_openai(api_key="test", fetch=fetch)
        created_file = await provider.uploads().upload_bytes(
            data=b"0123456789",
            filename="training.jsonl",
            mime_type="text/jsonl",
            purpose="fine-tune",
            part_size_bytes=4,
        )

        self.assertEqual(created_file.id, "file_1")
        self.assertEqual(requests[0]["json"]["bytes"], 10)
        self.assertEqual(requests[1]["body"]["files"]["data"][0], "training.jsonl.part-1")
        self.assertEqual(requests[4]["json"]["part_ids"], ["part_1", "part_2", "part_3"])

    async def test_openai_moderations_and_batches_clients(self) -> None:
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
            if url.endswith("/moderations"):
                return FakeResponse(status_code=200, payload={"id": "mod_1", "results": [{"flagged": False}]})
            if url.endswith("/batches") and method == "POST":
                return FakeResponse(status_code=200, payload={"id": "batch_1", "status": "validating"})
            if url.endswith("/batches/batch_1") and method == "GET":
                return FakeResponse(status_code=200, payload={"id": "batch_1", "status": "completed"})
            if "/batches?" in url:
                return FakeResponse(status_code=200, payload={"data": [{"id": "batch_1", "status": "completed"}], "has_more": False})
            return FakeResponse(status_code=200, payload={"id": "batch_1", "status": "cancelling"})

        provider = create_openai(api_key="test", fetch=fetch)
        moderation = await provider.moderations().create({"model": "omni-moderation-latest", "input": "hello"})
        created = await provider.batches().create({"input_file_id": "file_1", "endpoint": "/v1/responses", "completion_window": "24h"})
        retrieved = await provider.batches().retrieve("batch_1")
        listed = await provider.batches().list(limit=10)
        cancelled = await provider.batches().cancel("batch_1")

        self.assertFalse(moderation["results"][0]["flagged"])
        self.assertEqual(created["id"], "batch_1")
        self.assertEqual(retrieved["status"], "completed")
        self.assertEqual(listed["data"][0]["id"], "batch_1")
        self.assertEqual(cancelled["status"], "cancelling")

    async def test_openai_containers_and_skills_clients(self) -> None:
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
            requests.append({"url": url, "method": method, "json": json_body, "body": body})
            if "/containers/" in url and url.endswith("/content"):
                return FakeResponse(status_code=200, body_bytes=b"print('hi')")
            if url.endswith("/containers/ctr_1/files/file_1"):
                return FakeResponse(status_code=200, payload={"id": "file_1", "filename": "main.py", "bytes": 11})
            if url.endswith("/containers/ctr_1/files"):
                if method == "GET":
                    return FakeResponse(status_code=200, payload={"data": [{"id": "file_1", "filename": "main.py", "bytes": 11}]})
                return FakeResponse(status_code=200, payload={"id": "file_1", "filename": "main.py", "bytes": 11})
            if url.endswith("/containers/ctr_1"):
                if method == "DELETE":
                    return FakeResponse(status_code=200, payload={"id": "ctr_1", "deleted": True})
                return FakeResponse(status_code=200, payload={"id": "ctr_1", "status": "ready"})
            if "/containers?" in url:
                return FakeResponse(status_code=200, payload={"data": [{"id": "ctr_1", "status": "ready"}]})
            if url.endswith("/containers"):
                return FakeResponse(status_code=200, payload={"id": "ctr_1", "status": "ready"})

            if "/skills/skill_1/versions/v1/content" in url:
                return FakeResponse(status_code=200, body_bytes=b"zip-bytes")
            if "/skills/skill_1/versions/v1" in url:
                if method == "DELETE":
                    return FakeResponse(status_code=200, payload={"id": "v1", "deleted": True})
                return FakeResponse(status_code=200, payload={"id": "v1", "status": "ready"})
            if "/skills/skill_1/versions" in url:
                if method == "GET":
                    return FakeResponse(status_code=200, payload={"data": [{"id": "v1", "status": "ready"}]})
                return FakeResponse(status_code=200, payload={"id": "v1", "status": "ready"})
            if url.endswith("/skills/skill_1/content"):
                return FakeResponse(status_code=200, body_bytes=b"skill-zip")
            if url.endswith("/skills/skill_1"):
                if method == "DELETE":
                    return FakeResponse(status_code=200, payload={"id": "skill_1", "deleted": True})
                return FakeResponse(status_code=200, payload={"id": "skill_1", "name": "demo-skill"})
            if "/skills?" in url:
                return FakeResponse(status_code=200, payload={"data": [{"id": "skill_1", "name": "demo-skill"}]})
            if url.endswith("/skills"):
                return FakeResponse(status_code=200, payload={"id": "skill_1", "name": "demo-skill"})

            return FakeResponse(status_code=200, payload={"deleted": True})

        provider = create_openai(api_key="test", fetch=fetch)

        container = await provider.containers().create({"name": "sandbox"})
        container_list = await provider.containers().list(limit=10)
        container_file = await provider.containers().create_file(container_id="ctr_1", data=b"print('hi')", filename="main.py")
        container_files = await provider.containers().list_files("ctr_1")
        container_file_content = await provider.containers().retrieve_file_content("ctr_1", "file_1")
        skill = await provider.skills().create({"name": "demo-skill"})
        skill_content = await provider.skills().retrieve_content("skill_1")
        skill_version = await provider.skills().create_version("skill_1", {"source": {"type": "inline"}})
        skill_versions = await provider.skills().list_versions("skill_1")
        skill_version_content = await provider.skills().retrieve_version_content("skill_1", "v1")

        self.assertEqual(container["id"], "ctr_1")
        self.assertEqual(container_list["data"][0]["id"], "ctr_1")
        self.assertEqual(container_file.filename, "main.py")
        self.assertEqual(container_files[0].id, "file_1")
        self.assertEqual(container_file_content, b"print('hi')")
        self.assertEqual(skill["id"], "skill_1")
        self.assertEqual(skill_content, b"skill-zip")
        self.assertEqual(skill_version["id"], "v1")
        self.assertEqual(skill_versions["data"][0]["id"], "v1")
        self.assertEqual(skill_version_content, b"zip-bytes")

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
            model=provider.native.language_model("gpt-5.4-mini"),
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
            model=provider.native.language_model("gpt-5.4-mini"),
            prompt="weather?",
            provider_options={"tools": [{"type": "web_search"}]},
        )

        self.assertEqual(result.text, "sunny")
        assistant_messages = [message for message in result.messages if message.role == "assistant"]
        tool_calls = [part.tool_call for message in assistant_messages for part in message.parts if getattr(part, "type", None) == "tool-call"]
        self.assertTrue(any(call.provider_metadata.get("provider_managed") for call in tool_calls))

    async def test_openai_normalizes_tool_search_provider_managed_calls(self) -> None:
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
                            "type": "tool_search_call",
                            "id": "ts_1",
                            "status": "completed",
                            "action": {"query": "filesystem"},
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "found tools"}],
                        },
                    ],
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider.native.language_model("gpt-5.4-mini"),
            prompt="find a tool",
            provider_options={"tools": [openai_tool_search_tool(description="Find a tool")]},
        )

        assistant_messages = [message for message in result.messages if message.role == "assistant"]
        tool_calls = [part.tool_call for message in assistant_messages for part in message.parts if getattr(part, "type", None) == "tool-call"]
        self.assertTrue(any(call.name == "tool_search" for call in tool_calls))

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
            model=provider.native.language_model("gpt-5.4-mini"),
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
            model=provider.native.language_model("gpt-5.4-mini"),
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
            model=provider.native.language_model("gpt-5.4-mini"),
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
        image_tool = openai_image_generation_tool(model="gpt-image-2", output_format="webp", quality="high")
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

        self.assertEqual(search_tool.type, "web_search")
        self.assertEqual(search_tool.tool_class, "web-search")
        self.assertEqual(search_tool.config["user_location"]["city"], "Buenos Aires")
        self.assertEqual(file_tool.config["filters"]["filters"][0]["value"], "es")
        self.assertEqual(image_tool.config["output_format"], "webp")
        self.assertEqual(shell_tool.config["environment"]["type"], "local")
        self.assertEqual(code_tool.config["container"]["type"], "auto")
        self.assertEqual(code_tool.config["container"]["network_policy"]["domain_secrets"][0]["name"], "TOKEN")
        self.assertEqual(namespace_tool.config["tools"][0]["format"]["type"], "grammar")
        self.assertEqual(tool_search_tool.type, "tool_search")
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
            model=provider.native.language_model("gpt-5.4-mini"),
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

    async def test_openai_response_reference_helpers_extract_from_parts_messages_and_results(self) -> None:
        response_part = openai_response_reference(response_id="resp_from_part")
        response_message = assistant([response_part])
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
                    "id": "resp_from_result",
                    "status": "completed",
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
                },
            )
        response_result = await generate_text(
            model=create_openai(
                api_key="test",
                fetch=fetch,
            ).native.language_model("gpt-4o-mini"),
            prompt="hello",
        )

        self.assertEqual(get_openai_response_reference(response_part).response_id, "resp_from_part")  # type: ignore[union-attr]
        self.assertEqual(get_openai_response_id(response_message), "resp_from_part")
        self.assertEqual(get_openai_response_id(response_result), "resp_from_result")

    async def test_openai_response_options_accepts_previous_response_objects(self) -> None:
        option_from_part = openai_response_options(previous_response=openai_response_reference(response_id="resp_part"))
        option_from_message = openai_response_options(
            previous_response=assistant([provider_data_part("openai", {"response_id": "resp_message"})])
        )

        self.assertEqual(option_from_part["previous_response_id"], "resp_part")
        self.assertEqual(option_from_message["previous_response_id"], "resp_message")

    async def test_openai_maps_hosted_tools_from_tools_set(self) -> None:
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
            model=provider.native.language_model("gpt-4o-mini"),
            prompt="hello",
            tools={
                "weather": tool(name="weather", schema=WeatherToolInput, execute=lambda input: {"ok": True}),
                "search": openai_web_search_tool(search_context_size="high"),
                "mcp": openai_mcp_tool(server_url="https://mcp.example.com", server_label="Example MCP"),
            },
        )

        mapped_tools = requests[0]["json"]["tools"]
        self.assertEqual(mapped_tools[0]["type"], "function")
        self.assertEqual(mapped_tools[1]["type"], "web_search")
        self.assertEqual(mapped_tools[1]["search_context_size"], "high")
        self.assertEqual(mapped_tools[2]["type"], "mcp")
        self.assertEqual(mapped_tools[2]["server_label"], "Example MCP")

    async def test_openai_maps_mcp_approval_response_provider_data_parts(self) -> None:
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
            model=provider.native.language_model("gpt-4o-mini"),
            messages=[
                assistant([openai_mcp_approval_response(approval_request_id="apr_123", approve=True)]),
                user("continue"),
            ],
        )

        self.assertEqual(requests[0]["json"]["input"][0]["type"], "mcp_approval_response")
        self.assertEqual(requests[0]["json"]["input"][0]["approval_request_id"], "apr_123")
        self.assertTrue(requests[0]["json"]["input"][0]["approve"])

    async def test_openai_parse_helper_recognizes_typed_and_legacy_provider_data(self) -> None:
        typed_part = openai_mcp_approval_response(approval_request_id="apr_typed", approve=True)
        legacy_payload = provider_data_part(
            "openai",
            {"type": "mcp_approval_response", "approval_request_id": "apr_legacy", "approve": False},
        )

        typed = parse_openai_provider_data_part(typed_part)
        legacy = parse_openai_provider_data_part(legacy_payload)

        self.assertIsInstance(typed, OpenAIMcpApprovalResponse)
        self.assertEqual(typed.approval_request_id, "apr_typed")
        self.assertIsInstance(legacy, OpenAIMcpApprovalResponse)
        self.assertEqual(legacy.approval_request_id, "apr_legacy")
        self.assertFalse(legacy.approve)

    async def test_openai_parses_provider_data_parts_from_responses_payload(self) -> None:
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
                payload={
                    "id": "resp_123",
                    "status": "completed",
                    "output": [
                        {
                            "type": "mcp_approval_request",
                            "id": "apr_123",
                            "arguments": '{"query":"apollo"}',
                            "name": "docs_search",
                            "server_label": "Docs",
                        },
                        {
                            "type": "mcp_call",
                            "id": "call_123",
                            "arguments": '{"query":"apollo"}',
                            "name": "docs_search",
                            "server_label": "Docs",
                            "status": "completed",
                        },
                        {
                            "type": "mcp_list_tools",
                            "id": "list_123",
                            "server_label": "Docs",
                            "tools": [{"name": "docs_search"}],
                        },
                    ],
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider.native.language_model("gpt-4o-mini"),
            prompt="hello",
        )

        provider_parts = [part for part in result.steps[0].response.messages[0].parts if getattr(part, "type", None) == "provider-data"]
        self.assertEqual(len(provider_parts), 4)
        self.assertEqual(parse_openai_provider_data_part(provider_parts[0]).response_id, "resp_123")  # type: ignore[union-attr]
        self.assertEqual(getattr(parse_openai_provider_data_part(provider_parts[1]), "type", None), "mcp_approval_request")
        self.assertEqual(getattr(parse_openai_provider_data_part(provider_parts[2]), "type", None), "mcp_call")
        self.assertEqual(getattr(parse_openai_provider_data_part(provider_parts[3]), "type", None), "mcp_list_tools")

    async def test_openai_provider_data_helpers_extract_mcp_events(self) -> None:
        message = assistant(
            [
                openai_response_reference(response_id="resp_1"),
                provider_data_part(
                    "openai",
                    {
                        "type": "mcp_call",
                        "id": "call_1",
                        "arguments": '{"query":"apollo"}',
                        "name": "docs_search",
                        "server_label": "Docs",
                        "status": "completed",
                    },
                ),
                provider_data_part(
                    "openai",
                    {
                        "type": "mcp_list_tools",
                        "id": "list_1",
                        "server_label": "Docs",
                        "tools": [{"name": "docs_search"}],
                    },
                ),
            ]
        )

        mcp_calls = get_openai_mcp_calls(message)
        mcp_list_tools_events = get_openai_mcp_list_tools_events(message)

        self.assertEqual(len(mcp_calls), 1)
        self.assertEqual(mcp_calls[0].id, "call_1")
        self.assertEqual(get_last_openai_mcp_call(message).server_label, "Docs")  # type: ignore[union-attr]
        self.assertEqual(len(mcp_list_tools_events), 1)
        self.assertEqual(mcp_list_tools_events[0].id, "list_1")
        self.assertEqual(get_last_openai_mcp_list_tools_event(message).server_label, "Docs")  # type: ignore[union-attr]

    async def test_openai_provider_data_tool_call_helper_normalizes_mcp_events(self) -> None:
        mcp_call = openai_provider_data_tool_call(
            provider_data_part(
                "openai",
                {
                    "type": "mcp_call",
                    "id": "call_1",
                    "arguments": '{"query":"apollo"}',
                    "name": "docs_search",
                    "server_label": "Docs",
                    "status": "completed",
                },
            )
        )
        list_tools_call = openai_provider_data_tool_call(
            provider_data_part(
                "openai",
                {
                    "type": "mcp_list_tools",
                    "id": "list_1",
                    "server_label": "Docs",
                    "tools": [{"name": "docs_search"}],
                },
            )
        )

        self.assertEqual(mcp_call.name, "docs_search")  # type: ignore[union-attr]
        self.assertEqual(mcp_call.input["query"], "apollo")  # type: ignore[union-attr]
        self.assertTrue(mcp_call.provider_metadata["provider_managed"])  # type: ignore[union-attr]
        self.assertEqual(mcp_call.provider_metadata["provider_event_type"], "mcp_call")  # type: ignore[union-attr]
        self.assertEqual(list_tools_call.name, "mcp_list_tools")  # type: ignore[union-attr]
        self.assertEqual(list_tools_call.input["tools"][0]["name"], "docs_search")  # type: ignore[union-attr]

    async def test_openai_stream_emits_provider_data_events_for_mcp_items(self) -> None:
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
                payload={},
                body_text=(
                    'data: {"type":"response.output_item.done","item":{"type":"mcp_approval_request","id":"apr_1","arguments":"{\\"query\\":\\"apollo\\"}","name":"docs_search","server_label":"Docs"}}\n\n'
                    'data: {"type":"response.completed","response":{"id":"resp_456","status":"completed"}}\n\n'
                ),
            )

        provider = create_openai(api_key="test", fetch=fetch)
        stream = stream_text(
            model=provider.native.language_model("gpt-4o-mini"),
            prompt="hello",
        )
        events = [event async for event in stream.event_stream()]
        result = await stream.collect()

        provider_events = [event for event in events if event.type == "provider-data"]
        self.assertEqual(len(provider_events), 2)
        self.assertEqual(getattr(provider_events[0].data, "type", None), "mcp_approval_request")
        self.assertEqual(getattr(provider_events[1].data, "response_id", None), "resp_456")
        provider_parts = [part for part in result.steps[0].response.messages[0].parts if getattr(part, "type", None) == "provider-data"]
        self.assertEqual(len(provider_parts), 2)

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
                model=provider.native.language_model("openai/o4-mini"),
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
            model=provider.native.speech_model("openai/gpt-4o-mini-tts"),
            input="hello",
            voice="alloy",
        )

        self.assertEqual(result.audio, b"hello world")
        self.assertEqual(result.media_type, "audio/wav")
        self.assertEqual(requests[0]["url"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertTrue(requests[0]["stream"])
        self.assertEqual(requests[0]["json"]["modalities"], ["text", "audio"])
        self.assertEqual(requests[0]["json"]["audio"], {"voice": "alloy", "format": "wav"})
