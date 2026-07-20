from __future__ import annotations

import sys
import json
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    assistant,
    azure_openai_mcp_approval_response,
    azure_openai_mcp_tool,
    azure_openai_response_reference,
    azure_openai_web_search_tool,
    create_azure_openai,
    generate_text,
    get_azure_openai_response_id,
    get_azure_openai_response_reference,
    hosted_tool,
    parse_azure_openai_provider_data_part,
    tool,
    user,
)
from zhivex_ai.errors import ConfigurationError
from zhivex_ai.providers.azure_openai import (  # noqa: E402
    azure_openai_provider_data_tool_call,
    get_azure_openai_mcp_calls,
    get_azure_openai_mcp_list_tools_events,
    get_last_azure_openai_mcp_call,
    get_last_azure_openai_mcp_list_tools_event,
)
from zhivex_ai.types import AzureOpenAIMcpApprovalResponse, ProviderDataPart


class WeatherToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str


class FakeResponse:
    def __init__(self, *, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self.payload = payload

    async def json(self) -> object:
        return self.payload

    async def text(self) -> str:
        return json.dumps(self.payload)


class FakeRealtimeConnection:
    async def send_json(self, payload: dict[str, object]) -> None:
        return None

    async def recv_json(self) -> object:
        return {"type": "session.closed"}

    async def close(self) -> None:
        return None


class AzureOpenAIProviderTests(TestCase):
    def test_azure_openai_uses_versionless_v1_base_url(self) -> None:
        provider = create_azure_openai(
            api_key="test",
            endpoint="https://example.openai.azure.com",
            api_version="2024-10-21",
        )

        model = provider.native.language_model("gpt-4o-mini")
        self.assertEqual(model.base_url, "https://example.openai.azure.com/openai/v1")

    def test_azure_openai_rejects_api_key_and_entra_auth_together(self) -> None:
        with self.assertRaises(ConfigurationError):
            create_azure_openai(
                api_key="test",
                endpoint="https://example.openai.azure.com",
                entra_token="token",
            )

    def test_azure_openai_requires_api_key_or_entra_auth(self) -> None:
        with patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "", "AZURE_OPENAI_ENDPOINT": ""}, clear=False):
            with self.assertRaisesRegex(ConfigurationError, "API key or Entra ID"):
                create_azure_openai(endpoint="https://example.openai.azure.com")


class AzureOpenAIHostedToolTests(IsolatedAsyncioTestCase):
    async def test_azure_openai_entra_token_uses_bearer_auth(self) -> None:
        requests: list[dict[str, object]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, object] | None = None,
            body: object = None,
            timeout_ms: int | None = None,
            stream: bool = False,
        ):
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body})
            return FakeResponse(
                status_code=200,
                payload={
                    "status": "completed",
                    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
                },
            )

        provider = create_azure_openai(
            endpoint="https://example.openai.azure.com",
            entra_token="entra-token",
            fetch=fetch,
        )

        await generate_text(model=provider.native.language_model("gpt-4o-mini"), prompt="hello")

        self.assertEqual(requests[0]["headers"]["authorization"], "Bearer entra-token")  # type: ignore[index]
        self.assertNotIn("api-key", requests[0]["headers"])  # type: ignore[operator]

    async def test_azure_openai_entra_token_provider_can_be_sync_or_async(self) -> None:
        requests: list[dict[str, object]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, object] | None = None,
            body: object = None,
            timeout_ms: int | None = None,
            stream: bool = False,
        ):
            requests.append({"url": url, "headers": headers})
            return FakeResponse(status_code=200, payload={"id": "resp_123", "status": "completed"})

        sync_provider = create_azure_openai(
            endpoint="https://example.openai.azure.com",
            entra_token_provider=lambda: "sync-token",
            fetch=fetch,
        )
        await sync_provider.responses().create({"model": "gpt-4o-mini", "input": "hello"})

        async def async_token_provider() -> str:
            return "async-token"

        async_provider = create_azure_openai(
            endpoint="https://example.openai.azure.com",
            entra_token_provider=async_token_provider,
            fetch=fetch,
        )
        await async_provider.responses().create({"model": "gpt-4o-mini", "input": "hello"})

        self.assertEqual(requests[0]["headers"]["authorization"], "Bearer sync-token")  # type: ignore[index]
        self.assertEqual(requests[1]["headers"]["authorization"], "Bearer async-token")  # type: ignore[index]

    async def test_azure_openai_entra_auth_reaches_lifecycle_clients_and_realtime(self) -> None:
        requests: list[dict[str, object]] = []
        connection_meta: list[dict[str, object]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, object] | None = None,
            body: object = None,
            timeout_ms: int | None = None,
            stream: bool = False,
        ):
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body})
            if "conversations" in url:
                return FakeResponse(status_code=200, payload={"id": "conv_123", "object": "conversation"})
            if "vector_stores" in url:
                return FakeResponse(status_code=200, payload={"id": "vs_123", "name": "Docs"})
            return FakeResponse(status_code=200, payload={"id": "resp_123", "status": "completed", "client_secret": {"value": "browser-token"}})

        async def connection_factory(url, headers, options):
            connection_meta.append({"url": url, "headers": headers, "options": options})
            return FakeRealtimeConnection()

        provider = create_azure_openai(
            endpoint="https://example.openai.azure.com",
            entra_token="entra-token",
            fetch=fetch,
            realtime_connection_factory=connection_factory,
        )

        await provider.responses().create({"model": "gpt-4o-mini", "input": "hello"})
        await provider.conversations().create({"metadata": {"team": "sdk"}})
        await provider.file_search_stores().create(display_name="Docs")
        await provider.native.realtime_model("gpt-realtime-2.1").create_browser_token()
        session = await provider.native.realtime_model("gpt-realtime-2.1").connect()
        await session.aclose()

        self.assertTrue(all(request["headers"]["authorization"] == "Bearer entra-token" for request in requests))  # type: ignore[index]
        self.assertTrue(all("api-key" not in request["headers"] for request in requests))  # type: ignore[operator]
        self.assertEqual(connection_meta[0]["headers"]["authorization"], "Bearer entra-token")  # type: ignore[index]
        self.assertNotIn("api-key", connection_meta[0]["headers"])  # type: ignore[operator]

    async def test_azure_openai_exposes_responses_lifecycle_client(self) -> None:
        requests: list[dict[str, object]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, object] | None = None,
            body: object = None,
            timeout_ms: int | None = None,
            stream: bool = False,
        ):
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body})
            return FakeResponse(status_code=200, payload={"id": "resp_123", "status": "completed"})

        provider = create_azure_openai(
            api_key="test",
            endpoint="https://example.openai.azure.com",
            fetch=fetch,
        )

        payload = await provider.responses().create({"model": "gpt-4o-mini", "input": "hello"})

        self.assertEqual(payload["id"], "resp_123")
        self.assertEqual(requests[0]["url"], "https://example.openai.azure.com/openai/v1/responses")
        self.assertEqual(requests[0]["headers"]["api-key"], "test")  # type: ignore[index]
        self.assertNotIn("authorization", requests[0]["headers"])  # type: ignore[operator]
        self.assertEqual(requests[0]["json"], {"model": "gpt-4o-mini", "input": "hello"})

    async def test_azure_openai_exposes_conversations_lifecycle_client(self) -> None:
        requests: list[dict[str, object]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, object] | None = None,
            body: object = None,
            timeout_ms: int | None = None,
            stream: bool = False,
        ):
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body})
            return FakeResponse(status_code=200, payload={"id": "conv_123", "object": "conversation"})

        provider = create_azure_openai(
            api_key="test",
            endpoint="https://example.openai.azure.com",
            fetch=fetch,
        )

        payload = await provider.conversations().create({"metadata": {"team": "sdk"}})

        self.assertEqual(payload["id"], "conv_123")
        self.assertEqual(requests[0]["url"], "https://example.openai.azure.com/openai/v1/conversations")
        self.assertEqual(requests[0]["headers"]["api-key"], "test")  # type: ignore[index]
        self.assertEqual(requests[0]["json"], {"metadata": {"team": "sdk"}})

    async def test_azure_openai_exposes_file_search_stores_lifecycle_client(self) -> None:
        requests: list[dict[str, object]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, object] | None = None,
            body: object = None,
            timeout_ms: int | None = None,
            stream: bool = False,
        ):
            requests.append({"url": url, "method": method, "headers": headers, "json": json_body})
            if method == "DELETE":
                return FakeResponse(status_code=200, payload={"id": "vs_123", "deleted": True})
            if method == "GET":
                return FakeResponse(
                    status_code=200,
                    payload={"object": "list", "data": [{"id": "vs_123", "name": "Docs"}], "has_more": False},
                )
            return FakeResponse(status_code=200, payload={"id": "vs_123", "name": "Docs"})

        provider = create_azure_openai(
            api_key="test",
            endpoint="https://example.openai.azure.com",
            fetch=fetch,
        )

        created = await provider.file_search_stores().create(display_name="Docs")
        listed = await provider.file_search_stores().list()
        deleted = await provider.file_search_stores().delete("vs_123")

        self.assertEqual(created.name, "vs_123")
        self.assertEqual(listed.stores[0].display_name, "Docs")
        self.assertTrue(deleted)
        self.assertEqual(requests[0]["url"], "https://example.openai.azure.com/openai/v1/vector_stores")
        self.assertEqual(requests[0]["headers"]["api-key"], "test")  # type: ignore[index]
        self.assertEqual(requests[0]["json"], {"name": "Docs"})
        self.assertEqual(requests[1]["method"], "GET")
        self.assertEqual(requests[2]["url"], "https://example.openai.azure.com/openai/v1/vector_stores/vs_123")

    async def test_azure_openai_maps_hosted_tools_from_tools_set(self) -> None:
        requests: list[dict[str, object]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, object] | None = None,
            body: object = None,
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

        provider = create_azure_openai(
            api_key="test",
            endpoint="https://example.openai.azure.com",
            fetch=fetch,
        )
        await generate_text(
            model=provider.native.language_model("gpt-4o-mini"),
            prompt="hello",
            tools={
                "weather": tool(name="weather", schema=WeatherToolInput, execute=lambda input: {"ok": True}),
                "search": azure_openai_web_search_tool(search_context_size="high"),
                "mcp": azure_openai_mcp_tool(server_url="https://mcp.example.com", server_label="Example MCP"),
            },
        )

        mapped_tools = requests[0]["json"]["tools"]  # type: ignore[index]
        assert isinstance(mapped_tools, list)
        self.assertEqual(mapped_tools[0]["type"], "function")
        self.assertEqual(mapped_tools[1]["type"], "web_search_preview")
        self.assertEqual(mapped_tools[2]["type"], "mcp")

    async def test_azure_openai_maps_mcp_approval_response_provider_data_parts(self) -> None:
        requests: list[dict[str, object]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, object] | None = None,
            body: object = None,
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

        provider = create_azure_openai(
            api_key="test",
            endpoint="https://example.openai.azure.com",
            fetch=fetch,
        )
        await generate_text(
            model=provider.native.language_model("gpt-4o-mini"),
            messages=[
                assistant([azure_openai_mcp_approval_response(approval_request_id="apr_456", approve=False, reason="deny")]),
                user("continue"),
            ],
        )

        mapped_input = requests[0]["json"]["input"]  # type: ignore[index]
        assert isinstance(mapped_input, list)
        self.assertEqual(mapped_input[0]["type"], "mcp_approval_response")
        self.assertEqual(mapped_input[0]["approval_request_id"], "apr_456")
        self.assertFalse(mapped_input[0]["approve"])

    async def test_azure_openai_parse_helper_recognizes_typed_and_legacy_provider_data(self) -> None:
        typed = parse_azure_openai_provider_data_part(
            azure_openai_mcp_approval_response(approval_request_id="apr_1", approve=True)
        )
        legacy = parse_azure_openai_provider_data_part(
            ProviderDataPart(
                provider="azure-openai",
                data={"type": "mcp_approval_response", "approval_request_id": "apr_2", "approve": False},
            )
        )

        self.assertIsInstance(typed, AzureOpenAIMcpApprovalResponse)
        self.assertEqual(typed.approval_request_id, "apr_1")
        self.assertIsInstance(legacy, AzureOpenAIMcpApprovalResponse)
        self.assertEqual(legacy.approval_request_id, "apr_2")
        self.assertFalse(legacy.approve)

    async def test_azure_openai_response_reference_helpers_extract_from_parts_messages_and_results(self) -> None:
        response_part = azure_openai_response_reference(response_id="resp_from_part")
        response_message = assistant([response_part])

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, object] | None = None,
            body: object = None,
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

        provider = create_azure_openai(
            api_key="test",
            endpoint="https://example.openai.azure.com",
            fetch=fetch,
        )
        response_result = await generate_text(
            model=provider.native.language_model("gpt-4o-mini"),
            prompt="hello",
        )

        self.assertEqual(get_azure_openai_response_reference(response_part).response_id, "resp_from_part")  # type: ignore[union-attr]
        self.assertEqual(get_azure_openai_response_id(response_message), "resp_from_part")
        self.assertEqual(get_azure_openai_response_id(response_result), "resp_from_result")

    async def test_azure_openai_provider_data_helpers_extract_mcp_events(self) -> None:
        message = assistant(
            [
                azure_openai_response_reference(response_id="resp_1"),
                ProviderDataPart(
                    provider="azure-openai",
                    data={
                        "type": "mcp_call",
                        "id": "call_1",
                        "arguments": '{"query":"apollo"}',
                        "name": "docs_search",
                        "server_label": "Docs",
                        "status": "completed",
                    },
                ),
                ProviderDataPart(
                    provider="azure-openai",
                    data={
                        "type": "mcp_list_tools",
                        "id": "list_1",
                        "server_label": "Docs",
                        "tools": [{"name": "docs_search"}],
                    },
                ),
            ]
        )

        mcp_calls = get_azure_openai_mcp_calls(message)
        mcp_list_tools_events = get_azure_openai_mcp_list_tools_events(message)

        self.assertEqual(len(mcp_calls), 1)
        self.assertEqual(mcp_calls[0].id, "call_1")
        self.assertEqual(get_last_azure_openai_mcp_call(message).server_label, "Docs")  # type: ignore[union-attr]
        self.assertEqual(len(mcp_list_tools_events), 1)
        self.assertEqual(mcp_list_tools_events[0].id, "list_1")
        self.assertEqual(get_last_azure_openai_mcp_list_tools_event(message).server_label, "Docs")  # type: ignore[union-attr]

    async def test_azure_openai_provider_data_tool_call_helper_normalizes_mcp_events(self) -> None:
        mcp_call = azure_openai_provider_data_tool_call(
            ProviderDataPart(
                provider="azure-openai",
                data={
                    "type": "mcp_call",
                    "id": "call_1",
                    "arguments": '{"query":"apollo"}',
                    "name": "docs_search",
                    "server_label": "Docs",
                    "status": "completed",
                },
            )
        )
        list_tools_call = azure_openai_provider_data_tool_call(
            ProviderDataPart(
                provider="azure-openai",
                data={
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

    async def test_azure_openai_accepts_openai_targeted_hosted_tools(self) -> None:
        requests: list[dict[str, object]] = []

        async def fetch(
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, object] | None = None,
            body: object = None,
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

        provider = create_azure_openai(
            api_key="test",
            endpoint="https://example.openai.azure.com",
            fetch=fetch,
        )
        await generate_text(
            model=provider.native.language_model("gpt-4o-mini"),
            prompt="hello",
            tools={
                "computer": hosted_tool(
                    name="computer",
                    provider="openai",
                    type="computer_use_preview",
                    tool_class="computer-use",
                )
            },
        )

        mapped_tools = requests[0]["json"]["tools"]  # type: ignore[index]
        assert isinstance(mapped_tools, list)
        self.assertEqual(mapped_tools[0]["type"], "computer_use_preview")
