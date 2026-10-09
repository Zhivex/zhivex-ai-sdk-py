from __future__ import annotations

import json
from typing import Any
from unittest import IsolatedAsyncioTestCase

from zhivex_ai import UnsupportedFeatureError, create_anthropic, hosted_tool
from zhivex_ai.types import ModelGenerateInput, ModelMessage, ReasoningConfig, TextPart, ToolExecutionResult, ToolResultPart


class Response:
    status_code = 200

    def __init__(self, payload: dict[str, Any], body: str = "") -> None:
        self.payload = payload
        self.body = body

    async def json(self) -> dict[str, Any]:
        return self.payload

    async def iter_lines(self):
        for line in self.body.splitlines():
            yield line


class Anthropic55Tests(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.response = Response({"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"})

        async def fetch(url, *, headers, json_body, timeout_ms, stream=False, method="POST"):
            self.requests.append(json_body)
            return self.response

        self.provider = create_anthropic(api_key="test", fetch=fetch)
        self.messages = [ModelMessage(role="user", parts=[TextPart(text="hello")])]

    async def test_sonnet_none_maps_to_between_tools_without_changing_older_models(self) -> None:
        for model_id, expected in [("claude-sonnet-5-5", "between_tools"), ("claude-sonnet-5", "disabled"), ("claude-opus-5", "disabled")]:
            with self.subTest(model_id=model_id):
                await self.provider.native.language_model(model_id).generate(ModelGenerateInput(messages=self.messages, reasoning=ReasoningConfig(effort="none")))
                self.assertEqual(self.requests[-1]["thinking"], {"type": expected})

    async def test_opus_always_on_rejects_none_and_both_models_reject_manual_budget(self) -> None:
        for model_id, reasoning in [("claude-opus-5-5", ReasoningConfig(effort="none")), ("claude-opus-5-5", ReasoningConfig(budget_tokens=1024)), ("claude-sonnet-5-5", ReasoningConfig(budget_tokens=1024)), ("claude-haiku-5-5", ReasoningConfig(effort="none")), ("claude-haiku-5-5", ReasoningConfig(budget_tokens=1024))]:
            with self.subTest(model_id=model_id, reasoning=reasoning), self.assertRaises(UnsupportedFeatureError):
                await self.provider.native.language_model(model_id).generate(ModelGenerateInput(messages=self.messages, reasoning=reasoning))
        self.assertEqual(self.requests, [])

    async def test_haiku55_adaptive_effort_and_forced_tools_and_toolset(self) -> None:
        model = self.provider.native.language_model("claude-haiku-5-5")
        await model.generate(ModelGenerateInput(messages=self.messages, reasoning=ReasoningConfig(effort="low")))
        self.assertEqual(self.requests[-1]["thinking"], {"type": "adaptive"})
        self.assertEqual(self.requests[-1]["output_config"]["effort"], "low")
        await model.generate(ModelGenerateInput(messages=self.messages, tool_choice="required"))
        self.assertEqual(self.requests[-1]["tool_choice"], {"type": "any"})
        with self.assertRaises(UnsupportedFeatureError):
            await model.generate(ModelGenerateInput(messages=self.messages, provider_options={"thinking": {"type": "between_tools"}}))
        with self.assertRaises(UnsupportedFeatureError):
            await model.generate(ModelGenerateInput(messages=self.messages, provider_options={"tools": [{"type": "computer_20251124", "name": "computer"}]}))
        await model.generate(ModelGenerateInput(messages=self.messages, tools={"computer": hosted_tool(name="computer", provider="anthropic", type="computer_toolset_20260801", tool_class="toolset")}))
        self.assertEqual(self.requests[-1]["tools"], [{"type": "computer_toolset_20260801"}])

    async def test_sonnet_between_tools_effort_cap_applies_after_merging(self) -> None:
        for effort in ["high", "xhigh", "max"]:
            input = ModelGenerateInput(messages=self.messages, reasoning=ReasoningConfig(effort="none"), provider_options={"output_config": {"effort": effort}})
            if effort == "high":
                await self.provider.native.language_model("claude-sonnet-5-5").generate(input)
                self.assertEqual(self.requests[-1]["output_config"]["effort"], "high")
            else:
                with self.assertRaises(UnsupportedFeatureError):
                    await self.provider.native.language_model("claude-sonnet-5-5").generate(input)
        self.assertEqual(len(self.requests), 1)

    async def test_forced_choices_rejected_in_generate_stream_native_and_count(self) -> None:
        for model_id in ["claude-opus-5-5", "claude-sonnet-5-5"]:
            model = self.provider.native.language_model(model_id)
            for kind in ["any", "tool"]:
                options = {"tool_choice": {"type": kind, "name": "lookup"}}
                input = ModelGenerateInput(messages=self.messages, provider_options=options)
                for operation in [lambda: model.generate(input), lambda: model.stream(input), lambda: self.provider.native.messages().create({"model": model_id, **options}), lambda: self.provider.native.tokens().count(model_id=model_id, messages=self.messages, provider_options=options)]:
                    with self.subTest(model=model_id, kind=kind), self.assertRaises(UnsupportedFeatureError):
                        await operation()
            with self.assertRaises(UnsupportedFeatureError):
                await model.generate(ModelGenerateInput(messages=self.messages, tool_choice="required"))
        self.assertEqual(self.requests, [])

    async def test_raw_thinking_rejects_disabled_and_accepts_sonnet_between_tools(self) -> None:
        for model_id in ["claude-opus-5-5", "claude-sonnet-5-5"]:
            with self.assertRaises(UnsupportedFeatureError):
                await self.provider.native.messages().create({"model": model_id, "thinking": {"type": "disabled"}})
        await self.provider.native.messages().create({"model": "claude-sonnet-5-5", "thinking": {"type": "between_tools"}})
        self.assertEqual(self.requests[-1]["thinking"]["type"], "between_tools")

    async def test_new_computer_toolset_has_no_legacy_name_and_member_roundtrips(self) -> None:
        thinking = {"type": "thinking", "thinking": "", "signature": "signed", "summary": "progress"}
        member = {"type": "tool_use", "toolset_name": "computer", "id": "tool-1", "name": "screenshot", "input": {}}
        self.response = Response({"content": [thinking, member], "stop_reason": "tool_use", "input_transformations": [{"reason": "organization_binding_mismatch"}]})
        model = self.provider.native.language_model("claude-opus-5-5")
        result = await model.generate(ModelGenerateInput(messages=self.messages, tools={"computer": hosted_tool(name="computer", provider="anthropic", type="computer_toolset_20260801", tool_class="toolset")}))
        self.assertEqual(self.requests[-1]["tools"], [{"type": "computer_toolset_20260801"}])
        call = result.messages[0].parts[0].tool_call
        self.assertEqual(call.provider_metadata["toolset_name"], "computer")
        output = ToolExecutionResult(tool_call_id=call.id, tool_name=call.name, output="image", provider_metadata={"toolset_name": "computer"})
        await model.generate(ModelGenerateInput(messages=[*self.messages, *result.messages, ModelMessage(role="tool", parts=[ToolResultPart(tool_result=output)])]))
        self.assertEqual(self.requests[-1]["messages"][1]["content"], [thinking, member])
        self.assertEqual(self.requests[-1]["messages"][2]["content"][0]["toolset_name"], "computer")
        self.assertEqual(result.raw_response["input_transformations"][0]["reason"], "organization_binding_mismatch")

    async def test_legacy_computer_rejected_but_web_fetch_allowed_on_55(self) -> None:
        for model_id in ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]:
            model = self.provider.native.language_model(model_id)
            with self.assertRaises(UnsupportedFeatureError):
                await model.generate(ModelGenerateInput(messages=self.messages, provider_options={"tools": [{"type": "computer_20251124", "name": "computer"}]}))
            await model.generate(ModelGenerateInput(messages=self.messages, provider_options={"tools": [{"type": "web_fetch_20260318", "name": "web_fetch"}]}))
        self.assertEqual(len(self.requests), 3)

    async def test_stream_preserves_member_identity_and_signed_thinking(self) -> None:
        events = [("content_block_start", {"index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": "signed"}}), ("content_block_stop", {"index": 0}), ("content_block_start", {"index": 1, "content_block": {"type": "tool_use", "id": "tool-1", "name": "navigate", "toolset_name": "browser", "input": {}}}), ("content_block_delta", {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"url":"https://example.com"}'}}), ("content_block_stop", {"index": 1}), ("message_stop", {"stop_reason": "tool_use"})]
        self.response = Response({}, "".join(f"event: {event}\ndata: {json.dumps(payload)}\n\n" for event, payload in events))
        model = self.provider.native.language_model("claude-sonnet-5-5")
        stream = await model.stream(ModelGenerateInput(messages=self.messages))
        emitted = [event async for event in stream]
        call = next(event.tool_call for event in emitted if event.type == "tool-call")
        self.assertEqual(call.provider_metadata["toolset_name"], "browser")
        self.assertEqual(call.provider_metadata["anthropic_thinking_blocks"][0]["signature"], "signed")
        self.assertEqual(call.input, {"url": "https://example.com"})

    async def test_sonnet55_mid_conversation_system_stays_in_history(self) -> None:
        messages = [*self.messages, ModelMessage(role="system", parts=[TextPart(text="new instruction")])]
        await self.provider.native.language_model("claude-sonnet-5-5").generate(ModelGenerateInput(messages=messages))
        self.assertEqual(self.requests[-1]["messages"][-1], {"role": "system", "content": [{"type": "text", "text": "new instruction"}]})

    async def test_advisor_incompatible_pair_rejected_and_encrypted_result_stream_preserved(self) -> None:
        model = self.provider.native.language_model("claude-sonnet-5-5")
        with self.assertRaises(UnsupportedFeatureError):
            await model.generate(ModelGenerateInput(messages=self.messages, provider_options={"tools": [{"type": "advisor_20260301", "name": "advisor", "model": "claude-opus-4-8"}]}))
        block = {"type": "advisor_tool_result", "tool_use_id": "srv-1", "content": {"type": "advisor_redacted_result", "encrypted_content": "opaque"}}
        self.response = Response({}, f'event: content_block_start\ndata: {json.dumps({"index": 0, "content_block": block})}\n\nevent: message_stop\ndata: {{"stop_reason": "end_turn"}}\n\n')
        stream = await model.stream(ModelGenerateInput(messages=self.messages))
        events = [event async for event in stream]
        event = next(event for event in events if event.type == "provider-data")
        self.assertEqual(event.data["block"], block)

    async def test_stream_surfaces_service_thinking_binding_transformations(self) -> None:
        transformations = [{"reason": "organization_binding_mismatch", "message_index": 1, "block_index": 0}]
        self.response = Response({}, f'event: message_start\ndata: {json.dumps({"message": {"input_transformations": transformations}})}\n\nevent: message_stop\ndata: {{"stop_reason": "end_turn"}}\n\n')
        stream = await self.provider.native.language_model("claude-sonnet-5-5").stream(ModelGenerateInput(messages=self.messages))
        events = [event async for event in stream]
        self.assertEqual(events[0].data["input_transformations"], transformations)
