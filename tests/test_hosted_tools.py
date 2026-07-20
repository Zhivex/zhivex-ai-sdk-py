from __future__ import annotations

import sys
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    AgentCapabilities,
    HostedToolDefinition,
    ModelCapabilities,
    ToolChoiceName,
    ValidationError,
    assistant,
    generate_text,
    get_hosted_tool_class,
    get_last_provider_data_part,
    get_provider_data_parts,
    hosted_tool,
    is_callable_tool_definition,
    is_hosted_tool_class,
    is_hosted_tool_definition,
    provider_data_part,
    tool,
)
from zhivex_ai._serde import deserialize_content_part, serialize_content_part  # noqa: E402
from zhivex_ai.generate_text import _execute_tool  # noqa: E402
from zhivex_ai.messages import get_last_provider_data_entry, get_provider_data_entries  # noqa: E402
from zhivex_ai.providers.openai import parse_openai_provider_data_part  # noqa: E402
from zhivex_ai.ui import deserialize_ui_message, deserialize_ui_message_chunk, serialize_ui_message, serialize_ui_message_chunk, to_ui_message, to_ui_message_stream  # noqa: E402
from zhivex_ai.types import GenerateResult, ModelGenerateInput, OpenAIMcpCall, StreamProviderDataEvent, ToolCall  # noqa: E402


class _HostedValidationModel:
    portable = False

    def __init__(
        self,
        *,
        provider: str,
        agent_capabilities: AgentCapabilities,
    ) -> None:
        self.provider = provider
        self.model_id = "demo"
        self.capabilities = ModelCapabilities(
            streaming=False,
            tools=True,
            structured_output=True,
            json_mode=True,
            tool_choice=True,
            parallel_tool_calls=False,
            vision=False,
            files=False,
            audio_input=False,
            audio_output=False,
            embeddings=False,
            reasoning=False,
            web_search=False,
            agent_capabilities=agent_capabilities,
        )

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        return GenerateResult(messages=[assistant("ok")], text="ok")


class _PortableHostedValidationModel(_HostedValidationModel):
    portable = True


class HostedToolHelperTests(IsolatedAsyncioTestCase):
    def test_hosted_tool_defaults_and_inference(self) -> None:
        search = hosted_tool(name="search", provider="openai", type="web_search")
        mcp = hosted_tool(name="mcp", provider="openai", type="mcp")
        toolset = hosted_tool(name="workspace", provider="anthropic", type="mcp_toolset", config={"server": {"name": "docs"}})
        file_search = hosted_tool(name="files", type="file_search")
        computer = hosted_tool(name="computer", type="computer_use_preview")
        code = hosted_tool(name="code", type="code_execution")
        custom = hosted_tool(name="custom", type="image_generation")

        self.assertEqual(search.kind, "hosted")
        self.assertEqual(get_hosted_tool_class(search), "web-search")
        self.assertEqual(get_hosted_tool_class(mcp), "remote-mcp")
        self.assertEqual(get_hosted_tool_class(toolset), "toolset")
        self.assertEqual(get_hosted_tool_class(file_search), "file-search")
        self.assertEqual(get_hosted_tool_class(computer), "computer-use")
        self.assertEqual(get_hosted_tool_class(code), "code-execution")
        self.assertEqual(get_hosted_tool_class(custom), "custom")

    def test_hosted_tool_inspectors_and_exports(self) -> None:
        hosted = hosted_tool(name="search", provider="openai", type="web_search")
        callable_tool = tool(name="weather", schema=dict[str, str], execute=lambda input: {"ok": True})

        self.assertIsInstance(hosted, HostedToolDefinition)
        self.assertTrue(is_hosted_tool_definition(hosted))
        self.assertFalse(is_callable_tool_definition(hosted))
        self.assertTrue(is_callable_tool_definition(callable_tool))
        self.assertFalse(is_hosted_tool_definition(callable_tool))
        self.assertTrue(is_hosted_tool_class(hosted, "web-search"))

    def test_provider_data_part_round_trips_through_serde_and_ui(self) -> None:
        message = assistant([provider_data_part("openai", {"type": "mcp_approval_response", "approve": True})])
        serialized_part = serialize_content_part(message.parts[0])
        round_tripped_part = deserialize_content_part(serialized_part)
        ui_payload = serialize_ui_message(to_ui_message(message, id="msg_123"))
        ui_message = deserialize_ui_message(ui_payload)

        self.assertEqual(serialized_part["type"], "provider-data")
        self.assertEqual(round_tripped_part.type, "provider-data")
        self.assertEqual(round_tripped_part.provider, "openai")
        self.assertEqual(ui_message.parts[0].type, "provider-data")
        self.assertTrue(ui_message.parts[0].data["approve"])

    def test_provider_data_helpers_extract_parts_from_messages(self) -> None:
        message = assistant(
            [
                provider_data_part("openai", {"response_id": "resp_1"}),
                provider_data_part("azure-openai", {"response_id": "resp_2"}),
            ]
        )

        all_parts = get_provider_data_parts(message)
        openai_part = get_last_provider_data_part(message, provider="openai")

        self.assertEqual(len(all_parts), 2)
        self.assertEqual(openai_part.provider, "openai")
        self.assertEqual(openai_part.data["response_id"], "resp_1")

    def test_provider_data_entry_helpers_can_parse_and_filter_typed_payloads(self) -> None:
        message = assistant(
            [
                provider_data_part("openai", {"response_id": "resp_1"}),
                provider_data_part(
                    "openai",
                    OpenAIMcpCall(
                        id="call_1",
                        arguments='{"query":"apollo"}',
                        name="docs_search",
                        server_label="Docs",
                        status="completed",
                    ),
                ),
            ]
        )

        entries = get_provider_data_entries(
            message,
            provider="openai",
            parser=parse_openai_provider_data_part,
            data_type="mcp_call",
        )
        last_entry = get_last_provider_data_entry(
            message,
            provider="openai",
            parser=parse_openai_provider_data_part,
            data_type="mcp_call",
        )

        self.assertEqual(len(entries), 1)
        self.assertIsInstance(entries[0], OpenAIMcpCall)
        self.assertEqual(entries[0].id, "call_1")
        self.assertIsInstance(last_entry, OpenAIMcpCall)
        self.assertEqual(last_entry.server_label, "Docs")

    async def test_to_ui_message_stream_preserves_provider_data_events(self) -> None:
        async def source():
            yield StreamProviderDataEvent(provider="openai", data={"response_id": "resp_123"})

        chunks = [chunk async for chunk in to_ui_message_stream(source(), message_id="msg_1")]
        payload = serialize_ui_message_chunk(chunks[0])
        round_tripped = deserialize_ui_message_chunk(payload)

        self.assertEqual(chunks[0].type, "provider-data")
        self.assertEqual(chunks[0].provider, "openai")
        self.assertEqual(chunks[0].data["response_id"], "resp_123")
        self.assertEqual(round_tripped.type, "provider-data")


class HostedToolRuntimeTests(IsolatedAsyncioTestCase):
    async def test_mixed_tool_set_executes_callable_and_rejects_hosted_local_execution(self) -> None:
        tools = {
            "weather": tool(name="weather", schema=dict[str, str], execute=lambda input: {"city": input["city"]}),
            "search": hosted_tool(name="search", provider="openai", type="web_search"),
        }

        callable_result = await _execute_tool(
            ToolCall(id="call_1", name="weather", input={"city": "Buenos Aires"}),
            tools,
        )
        self.assertFalse(callable_result.is_error)
        self.assertEqual(callable_result.output["city"], "Buenos Aires")

        with self.assertRaises(ValidationError):
            await _execute_tool(
                ToolCall(id="call_2", name="search", input={}),
                tools,
            )

    async def test_portable_models_reject_hosted_tools(self) -> None:
        model = _PortableHostedValidationModel(
            provider="openai",
            agent_capabilities=AgentCapabilities(hosted_web_search=True),
        )

        with self.assertRaisesRegex(ValidationError, "provider.native"):
            await generate_text(
                model=model,
                prompt="hello",
                tools={"search": hosted_tool(name="search", provider="openai", type="web_search")},
            )

    async def test_unsupported_hosted_tool_class_fails_early(self) -> None:
        model = _HostedValidationModel(
            provider="anthropic",
            agent_capabilities=AgentCapabilities(hosted_web_search=True),
        )

        with self.assertRaisesRegex(Exception, 'hosted tool class "remote-mcp"'):
            await generate_text(
                model=model,
                prompt="hello",
                tools={"mcp": hosted_tool(name="mcp", provider="anthropic", type="mcp")},
            )

    async def test_tool_choice_none_requires_capability(self) -> None:
        model = _HostedValidationModel(
            provider="ollama",
            agent_capabilities=AgentCapabilities(hosted_web_search=True, tool_choice_none=False),
        )

        with self.assertRaisesRegex(Exception, 'tool_choice="none"'):
            await generate_text(
                model=model,
                prompt="hello",
                tools={"search": hosted_tool(name="search", provider="ollama", type="web_search")},
                tool_choice="none",
            )

    async def test_named_tool_choice_rejects_hosted_tools(self) -> None:
        model = _HostedValidationModel(
            provider="openai",
            agent_capabilities=AgentCapabilities(hosted_web_search=True),
        )

        with self.assertRaisesRegex(ValidationError, "only supports hosted tools for anthropic"):
            await generate_text(
                model=model,
                prompt="hello",
                tools={"search": hosted_tool(name="search", provider="openai", type="web_search")},
                tool_choice=ToolChoiceName("search"),
            )

    async def test_named_tool_choice_allows_hosted_tools_for_anthropic(self) -> None:
        model = _HostedValidationModel(
            provider="anthropic",
            agent_capabilities=AgentCapabilities(hosted_web_search=True, toolsets=True),
        )

        result = await generate_text(
            model=model,
            prompt="hello",
            tools={"search": hosted_tool(name="search", provider="anthropic", type="web_search_20260318")},
            tool_choice=ToolChoiceName("search"),
        )

        self.assertEqual(result.text, "ok")

    async def test_required_tool_choice_rejects_hosted_only_gemini_runs(self) -> None:
        model = _HostedValidationModel(
            provider="gemini",
            agent_capabilities=AgentCapabilities(hosted_web_search=True, tool_choice_none=True),
        )

        with self.assertRaisesRegex(Exception, 'tool_choice="required"'):
            await generate_text(
                model=model,
                prompt="hello",
                tools={"search": hosted_tool(name="search", provider="gemini", type="google_search")},
                tool_choice="required",
            )

    async def test_azure_accepts_openai_targeted_hosted_tools_but_other_providers_reject_them(self) -> None:
        azure_model = _HostedValidationModel(
            provider="azure-openai",
            agent_capabilities=AgentCapabilities(hosted_web_search=True),
        )
        anthropic_model = _HostedValidationModel(
            provider="anthropic",
            agent_capabilities=AgentCapabilities(hosted_web_search=True),
        )
        tool_definition = hosted_tool(name="search", provider="openai", type="web_search")

        azure_result = await generate_text(
            model=azure_model,
            prompt="hello",
            tools={"search": tool_definition},
        )

        self.assertEqual(azure_result.text, "ok")
        with self.assertRaisesRegex(ValidationError, 'targets provider "openai"'):
            await generate_text(
                model=anthropic_model,
                prompt="hello",
                tools={"search": tool_definition},
            )
