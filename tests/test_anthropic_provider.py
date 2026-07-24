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
    anthropic_web_fetch_tool,
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

    async def test_anthropic_refusal_stop_reason_is_normalized(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": "I cannot help with that."}],
                    "stop_reason": "refusal",
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        model = provider.native.language_model("claude-fable-5")
        result = await generate_text(model=model, prompt="hello")
        provider_result = await model.generate(
            ModelGenerateInput(messages=[ModelMessage(role="user", parts=[TextPart(text="hello")])])
        )

        self.assertEqual(result.finish_reason, "refusal")
        self.assertEqual(result.provider_finish_reason, "refusal")
        self.assertEqual(provider_result.raw_response["stop_reason"], "refusal")

    async def test_anthropic_streaming_refusal_stop_reason_is_normalized(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(
                status_code=200,
                body_text=(
                    'event: content_block_delta\n'
                    'data: {"delta":{"type":"text_delta","text":"I cannot help."}}\n\n'
                    'event: message_delta\n'
                    'data: {"delta":{"stop_reason":"refusal"},"usage":{"input_tokens":2,"output_tokens":3}}\n\n'
                    'event: message_stop\n'
                    'data: {"stop_reason":"refusal"}\n'
                ),
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        result = await stream_text(model=provider.native.language_model("claude-fable-5"), prompt="hello").collect()

        self.assertEqual(result.text, "I cannot help.")
        self.assertEqual(result.finish_reason, "refusal")
        self.assertEqual(result.provider_finish_reason, "refusal")

    async def test_anthropic_grounded_refusal_stop_reason_is_normalized(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": "I cannot help with that."}],
                    "stop_reason": "refusal",
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        result = await generate_grounded_text(model=provider.native.grounded_language_model("claude-fable-5"), prompt="hello")

        self.assertEqual(result.finish_reason, "refusal")
        self.assertEqual(result.provider_finish_reason, "refusal")

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
                "fetch": anthropic_web_fetch_tool(max_uses=3, citations_enabled=True),
                "mcp": anthropic_mcp_server(url="https://mcp.example.com", name="example-mcp", allowed_tools=["echo"]),
                "code": anthropic_code_execution_tool(),
            },
        )

        self.assertEqual(requests[0]["tools"][0]["name"], "lookup")
        self.assertEqual(requests[0]["tools"][1]["type"], "web_search_20260318")
        self.assertEqual(requests[0]["tools"][2]["type"], "web_fetch_20260318")
        self.assertEqual(requests[0]["tools"][2]["citations"], {"enabled": True})
        self.assertEqual(requests[0]["tools"][3]["type"], "mcp_toolset")
        self.assertEqual(requests[0]["tools"][3]["mcp_server_name"], "example-mcp")
        self.assertEqual(requests[0]["tools"][4]["type"], "code_execution_20260521")
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

    async def test_anthropic_opus_4_8_maps_effort_to_adaptive_thinking(self) -> None:
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
            model=provider.native.language_model("claude-opus-4-8"),
            prompt="migrate this codebase",
            reasoning=ReasoningConfig(effort="xhigh"),
            provider_options={"speed": "fast"},
        )

        self.assertEqual(requests[0]["model"], "claude-opus-4-8")
        self.assertEqual(requests[0]["speed"], "fast")
        self.assertEqual(requests[0]["thinking"], {"type": "adaptive"})
        self.assertEqual(requests[0]["output_config"], {"effort": "xhigh"})

    async def test_anthropic_opus_4_8_accepts_max_effort(self) -> None:
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
            model=provider.native.language_model("claude-opus-4-8"),
            prompt="solve the hard problem",
            reasoning=ReasoningConfig(effort="max"),
        )

        self.assertEqual(requests[0]["output_config"]["effort"], "max")

    async def test_anthropic_fable_5_maps_effort_to_adaptive_thinking(self) -> None:
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
            model=provider.native.language_model("claude-fable-5"),
            prompt="plan the migration",
            reasoning=ReasoningConfig(effort="high"),
        )

        self.assertEqual(requests[0]["model"], "claude-fable-5")
        self.assertEqual(requests[0]["thinking"], {"type": "adaptive"})
        self.assertEqual(requests[0]["output_config"], {"effort": "high"})

    async def test_anthropic_mythos_5_maps_effort_to_adaptive_thinking(self) -> None:
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
            model=provider.native.language_model("claude-mythos-5"),
            prompt="plan the migration",
            reasoning=ReasoningConfig(effort="max"),
        )

        self.assertEqual(requests[0]["model"], "claude-mythos-5")
        self.assertEqual(requests[0]["thinking"], {"type": "adaptive"})
        self.assertEqual(requests[0]["output_config"], {"effort": "max"})

    async def test_anthropic_sonnet_5_maps_effort_to_adaptive_thinking(self) -> None:
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
            model=provider.native.language_model("claude-sonnet-5"),
            prompt="plan the implementation",
            reasoning=ReasoningConfig(effort="high"),
        )

        self.assertEqual(requests[0]["model"], "claude-sonnet-5")
        self.assertEqual(requests[0]["thinking"], {"type": "adaptive"})
        self.assertEqual(requests[0]["output_config"], {"effort": "high"})

    async def test_anthropic_fable_5_rejects_non_adaptive_thinking_config(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            raise AssertionError("request should not be dispatched")

        provider = create_anthropic(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider.native.language_model("claude-fable-5"),
                prompt="hello",
                reasoning=ReasoningConfig(budget_tokens=1024),
            )
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider.native.language_model("claude-fable-5"),
                prompt="hello",
                provider_options={"thinking": {"type": "enabled", "budget_tokens": 1024}},
            )
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider.native.language_model("claude-fable-5"),
                prompt="hello",
                provider_options={"thinking": {"type": "disabled"}},
            )

    async def test_anthropic_fable_5_rejects_non_default_sampling_parameters(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            raise AssertionError("request should not be dispatched")

        provider = create_anthropic(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider.native.language_model("claude-fable-5"),
                prompt="hello",
                temperature=0.2,
            )
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider.native.language_model("claude-fable-5"),
                prompt="hello",
                provider_options={"top_p": 0.9},
            )
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider.native.language_model("claude-fable-5"),
                prompt="hello",
                provider_options={"top_k": 40},
            )

    async def test_anthropic_merges_structured_output_with_effort(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": '{"answer":"ok"}'}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        result = await generate_object(
            model=provider.native.language_model("claude-opus-4-8"),
            prompt="return json",
            schema={"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
            reasoning=ReasoningConfig(effort="medium"),
        )

        self.assertEqual(result.object["answer"], "ok")
        self.assertEqual(requests[0]["output_config"]["effort"], "medium")
        self.assertEqual(requests[0]["output_config"]["format"]["type"], "json_schema")

    async def test_anthropic_rejects_conflicting_output_config_effort(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            raise AssertionError("request should not be dispatched")

        provider = create_anthropic(api_key="test", fetch=fetch)
        with self.assertRaises(ValidationError):
            await generate_text(
                model=provider.native.language_model("claude-opus-4-8"),
                prompt="hello",
                reasoning=ReasoningConfig(effort="high"),
                provider_options={"output_config": {"effort": "low"}},
            )

    async def test_anthropic_opus_4_8_rejects_manual_thinking_budget(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            raise AssertionError("request should not be dispatched")

        provider = create_anthropic(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError) as context:
            await generate_text(
                model=provider.native.language_model("claude-opus-4-8"),
                prompt="hello",
                reasoning=ReasoningConfig(budget_tokens=1024),
            )

        self.assertIn("does not support reasoning.budget_tokens", str(context.exception))

    async def test_anthropic_opus_4_8_rejects_non_default_sampling_parameters(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            raise AssertionError("request should not be dispatched")

        provider = create_anthropic(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider.native.language_model("claude-opus-4-8"),
                prompt="hello",
                temperature=0.2,
            )
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(
                model=provider.native.language_model("claude-opus-4-8"),
                prompt="hello",
                provider_options={"top_p": 0.9},
            )

    async def test_anthropic_opus_4_8_preserves_mid_conversation_system_messages(self) -> None:
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
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        model = provider.native.language_model("claude-opus-4-8")
        await model.generate(
            ModelGenerateInput(
                messages=[
                    ModelMessage(role="system", parts=[TextPart(text="base instructions")]),
                    ModelMessage(role="user", parts=[TextPart(text="start task")]),
                    ModelMessage(role="system", parts=[TextPart(text="narrow the scope")]),
                ]
            )
        )

        self.assertEqual(requests[0]["system"], "base instructions")
        self.assertEqual([message["role"] for message in requests[0]["messages"]], ["user", "system"])
        self.assertEqual(requests[0]["messages"][1]["content"][0]["text"], "narrow the scope")

    async def test_anthropic_current_families_preserve_mid_conversation_system_messages(self) -> None:
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
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        for model_id in ("claude-fable-5", "claude-mythos-5", "claude-opus-4-8"):
            with self.subTest(model_id=model_id):
                await provider.native.language_model(model_id).generate(
                    ModelGenerateInput(
                        messages=[
                            ModelMessage(role="system", parts=[TextPart(text="base")]),
                            ModelMessage(role="user", parts=[TextPart(text="start")]),
                            ModelMessage(role="system", parts=[TextPart(text="update")]),
                        ]
                    )
                )

        for request in requests:
            self.assertEqual(request["system"], "base")
            self.assertEqual([message["role"] for message in request["messages"]], ["user", "system"])

    async def test_anthropic_sonnet_5_keeps_system_messages_top_level(self) -> None:
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
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        await provider.native.language_model("claude-sonnet-5").generate(
            ModelGenerateInput(
                messages=[
                    ModelMessage(role="system", parts=[TextPart(text="base")]),
                    ModelMessage(role="user", parts=[TextPart(text="start")]),
                    ModelMessage(role="system", parts=[TextPart(text="update")]),
                ]
            )
        )

        self.assertEqual(requests[0]["system"], "base\nupdate")
        self.assertEqual([message["role"] for message in requests[0]["messages"]], ["user"])

    async def test_anthropic_mid_conversation_system_messages_validate_placement(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            raise AssertionError("request should not be dispatched")

        provider = create_anthropic(api_key="test", fetch=fetch)
        model = provider.native.language_model("claude-opus-4-8")
        with self.assertRaises(ValidationError):
            await model.generate(
                ModelGenerateInput(
                    messages=[
                        ModelMessage(role="user", parts=[TextPart(text="start task")]),
                        ModelMessage(role="system", parts=[TextPart(text="narrow the scope")]),
                        ModelMessage(role="user", parts=[TextPart(text="continue")]),
                    ]
                )
            )

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

    async def test_anthropic_stream_handles_current_web_fetch_and_code_results(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(
                status_code=200,
                body_text=(
                    'event: content_block_start\n'
                    'data: {"index":1,"content_block":{"type":"web_fetch_tool_result","tool_use_id":"fetch_1","content":{"type":"web_fetch_result","url":"https://example.com"}}}\n\n'
                    'event: content_block_start\n'
                    'data: {"index":2,"content_block":{"type":"bash_code_execution_tool_result","tool_use_id":"code_1","content":{"type":"bash_code_execution_result","stdout":"42\\n","return_code":0,"content":[]}}}\n\n'
                    'event: message_stop\n'
                    'data: {"stop_reason":"end_turn"}\n'
                ),
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        model = provider.native.language_model("claude-opus-4-8")
        events = []
        async for event in await model.stream(
            ModelGenerateInput(messages=[ModelMessage(role="user", parts=[TextPart(text="fetch and compute")])])
        ):
            events.append(event)

        self.assertEqual(events[0].tool_result.tool_call_id, "fetch_1")
        self.assertEqual(events[1].tool_result.tool_call_id, "code_1")
        self.assertEqual(events[1].tool_result.output["stdout"], "42\n")

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
                    'event: content_block_start\n'
                    'data: {"index":2,"content_block":{"type":"mcp_tool_result","tool_use_id":"mcp_1","is_error":false,"content":[{"type":"text","text":"hi"}]}}\n\n'
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
        self.assertEqual(events[1].tool_result.tool_call_id, "mcp_1")
        self.assertFalse(events[1].tool_result.is_error)

    async def test_anthropic_adds_legacy_mcp_beta_header_by_default(self) -> None:
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

    async def test_anthropic_adds_current_mcp_beta_header_when_helper_opts_in(self) -> None:
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
            provider_options={
                "mcp_servers": [anthropic_mcp_server(url="https://mcp.example.com", name="example-mcp", version="current")]
            },
        )

        self.assertEqual(headers_seen[0]["anthropic-beta"], "mcp-client-2025-11-20")

    async def test_anthropic_accepts_raw_current_mcp_beta_header_override(self) -> None:
        headers_seen: list[dict[str, str]] = []
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            headers_seen.append(headers)
            requests.append(json_body)
            return FakeResponse(
                status_code=200,
                payload={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}},
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        await generate_text(
            model=provider.native.language_model("claude-sonnet-4-20250514"),
            prompt="use MCP",
            provider_options={
                "anthropic_mcp_beta": "mcp-client-2025-11-20",
                "mcp_servers": [{"type": "url", "url": "https://mcp.example.com", "name": "example-mcp"}],
            },
        )

        self.assertEqual(headers_seen[0]["anthropic-beta"], "mcp-client-2025-11-20")
        self.assertNotIn("anthropic_mcp_beta", requests[0])

    async def test_anthropic_current_code_execution_does_not_add_legacy_beta_header(self) -> None:
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

        self.assertNotIn("anthropic-beta", headers_seen[0])

    async def test_anthropic_parses_current_server_tool_results(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [
                        {
                            "type": "web_fetch_tool_result",
                            "tool_use_id": "srv_fetch",
                            "content": {
                                "type": "web_fetch_result",
                                "url": "https://example.com/article",
                            },
                        },
                        {
                            "type": "bash_code_execution_tool_result",
                            "tool_use_id": "srv_code",
                            "content": {
                                "type": "bash_code_execution_result",
                                "stdout": "42\n",
                                "stderr": "",
                                "return_code": 0,
                                "content": [],
                            },
                        },
                        {"type": "text", "text": "done"},
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        result = await provider.native.language_model("claude-opus-4-8").generate(
            ModelGenerateInput(messages=[ModelMessage(role="user", parts=[TextPart(text="fetch and compute")])])
        )

        assert result.messages is not None
        self.assertEqual([part.type for part in result.messages[0].parts], ["tool-result", "code-result", "text"])
        self.assertEqual(result.messages[0].parts[0].tool_result.tool_call_id, "srv_fetch")
        self.assertEqual(result.messages[0].parts[1].output, "42\n")
        self.assertEqual(result.messages[0].parts[1].outcome, "bash_code_execution_result")

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

        self.assertEqual(requests[0]["tools"][0]["type"], "web_search_20260318")
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
        legacy_web_search = anthropic_web_search_tool(tool_type="web_search_20250305")
        web_fetch = anthropic_web_fetch_tool(
            max_uses=3,
            citations_enabled=True,
            use_cache=False,
            response_inclusion="excluded",
        )
        mcp_server = anthropic_mcp_server(url="https://mcp.example.com", name="example-mcp", allowed_tools=["echo"])
        current_mcp_server = anthropic_mcp_server(url="https://mcp.example.com", name="example-mcp", version="current")
        code_execution = anthropic_code_execution_tool()

        self.assertEqual(web_search.type, "web_search_20260318")
        self.assertEqual(legacy_web_search.type, "web_search_20250305")
        self.assertEqual(web_search.config["max_uses"], 2)
        self.assertEqual(web_fetch.type, "web_fetch_20260318")
        self.assertEqual(web_fetch.config["citations"], {"enabled": True})
        self.assertFalse(web_fetch.config["use_cache"])
        self.assertEqual(web_fetch.config["response_inclusion"], "excluded")
        self.assertEqual(mcp_server.type, "mcp_toolset")
        self.assertEqual(mcp_server.config["server"]["name"], "example-mcp")
        self.assertEqual(mcp_server.config["default_config"]["allowed_tools"], ["echo"])
        self.assertEqual(current_mcp_server.metadata["anthropic_mcp_beta"], "mcp-client-2025-11-20")
        self.assertEqual(code_execution.type, "code_execution_20260521")
