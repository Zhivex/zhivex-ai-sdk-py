from __future__ import annotations

import json
from unittest import IsolatedAsyncioTestCase

from zhivex_ai import (
    FilePart, ImagePart, ModelMessage, ReasoningConfig, TextPart,
    UnsupportedFeatureError, ValidationError, create_qwen, generate_text,
    qwen_code_interpreter_tool, qwen_web_search_tool, stream_text, tool,
)
from tests.test_qwen_provider import FakeResponse


class QwenOmniTests(IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.payload = {"id": "resp_omni", "status": "completed", "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "OK"}]},
        ], "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}}

        async def fetch(url, **kwargs):
            self.requests.append((url, kwargs["json_body"]))
            if url.endswith("chat/completions") and kwargs.get("stream"):
                events = [
                    {"choices": [{"delta": {"reasoning_content": "check"}}]},
                    {"choices": [{"delta": {"content": "OK"}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}},
                ]
                return FakeResponse(200, body_text="\n".join("data: " + json.dumps(event) + "\n" for event in events) + "\ndata: [DONE]\n\n")
            if url.endswith("chat/completions"):
                return FakeResponse(200, {"choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}]})
            if kwargs.get("stream"):
                events = [
                    {"type": "response.output_text.delta", "delta": "OK"},
                    {"type": "response.completed", "response": self.payload},
                ]
                return FakeResponse(200, body_text="\n".join("data: " + json.dumps(event) + "\n" for event in events))
            return FakeResponse(200, self.payload)
        self.provider = create_qwen(api_key="test", fetch=fetch)
        self.model = self.provider.native.language_model("qwen3.8-omni-flash")

    async def test_mixed_media_responses_and_stream_usage(self):
        messages = [ModelMessage(role="user", parts=[
            TextPart(text="Describe"), ImagePart(image="https://example.com/image.png"),
            FilePart(url="https://example.com/audio.wav", media_type="audio/wav", provider_metadata={"qwen": {"use_multichannel": True}}),
            FilePart(data="AAAA", media_type="video/mp4"),
        ])]
        result = await stream_text(model=self.model, messages=messages).collect()
        self.assertEqual(result.text, "OK")
        self.assertEqual(result.usage.total_tokens, 15)
        url, body = self.requests[0]
        self.assertTrue(url.endswith("/responses"))
        content = body["input"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["input_text", "input_image", "input_audio", "input_video"])
        self.assertEqual(content[2], {"type": "input_audio", "audio_url": "https://example.com/audio.wav", "format": "wav", "use_multichannel": True})
        self.assertEqual(content[3]["video_url"], "data:video/mp4;base64,AAAA")

    async def test_inline_audio_and_image(self):
        await generate_text(model=self.model, messages=[ModelMessage(role="user", parts=[
            FilePart(data="AAAA", media_type="audio/mpeg"), FilePart(data="BBBB", media_type="image/png"),
        ])])
        content = self.requests[0][1]["input"][0]["content"]
        self.assertEqual(content[0]["format"], "mp3")
        self.assertEqual(content[0]["audio_url"], "data:audio/mpeg;base64,AAAA")
        self.assertEqual(content[1]["image_url"], "data:image/png;base64,BBBB")

    async def test_rejects_non_user_media_and_documents(self):
        for role in ["assistant", "system", "tool"]:
            with self.subTest(role=role), self.assertRaises(ValidationError):
                await generate_text(model=self.model, messages=[ModelMessage(role=role, parts=[FilePart(data="AAAA", media_type="audio/wav")])])
        for part in [FilePart(data="AAAA", media_type="application/pdf"), FilePart(file_id="file-x", media_type="audio/wav")]:
            with self.assertRaises(ValidationError):
                await generate_text(model=self.model, messages=[ModelMessage(role="user", parts=[part])])
        self.assertEqual(self.requests, [])

    async def test_hosted_tool_scope_and_override_guards(self):
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(model=self.model, prompt="test", tools={"code": qwen_code_interpreter_tool()})
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(model=self.model, prompt="test", provider_options={"tools": [{"type": "mcp"}]})
        with self.assertRaises(ValidationError):
            await generate_text(model=self.model, prompt="test", provider_options={"input": "override"})
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(model=self.model, prompt="test", provider_options={"modalities": ["audio"]})
        self.assertEqual(self.requests, [])
        await generate_text(model=self.model, prompt="search", tools={"search": qwen_web_search_tool()})
        self.assertEqual(self.requests[0][1]["tools"][0]["type"], "web_search")

    async def test_reasoning_and_default_forced_tool_choice(self):
        def lookup(value: str) -> str:
            return value
        await generate_text(model=self.model, prompt="call lookup", tools={"lookup": tool(lookup)}, tool_choice="required")
        self.assertEqual(self.requests[0][1]["reasoning"], {"effort": "none"})
        for effort in ["none", "minimal", "low", "medium", "high", "xhigh", "max"]:
            await generate_text(model=self.model, prompt="test", reasoning=ReasoningConfig(effort=effort))
            self.assertEqual(self.requests[-1][1]["reasoning"]["effort"], effort)
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(model=self.model, prompt="test", tools={"lookup": tool(lookup)}, tool_choice="required", reasoning=ReasoningConfig(effort="low"))

    async def test_budget_routes_to_chat_with_audio(self):
        await generate_text(model=self.model, messages=[ModelMessage(role="user", parts=[FilePart(data="AAAA", media_type="audio/wav")])], reasoning=ReasoningConfig(budget_tokens=1024), max_tokens=2048)
        url, body = self.requests[0]
        self.assertTrue(url.endswith("/chat/completions"))
        self.assertEqual(body["thinking_budget"], 1024)
        self.assertEqual(body["max_tokens"], 2048)
        self.assertEqual(body["messages"][0]["content"][0]["input_audio"]["data"], "data:audio/wav;base64,AAAA")
        self.assertNotIn("max_completion_tokens", body)

    def test_capabilities_are_model_specific(self):
        self.assertTrue(self.model.capabilities.audio_input)
        self.assertTrue(self.model.capabilities.files)
        self.assertFalse(self.model.capabilities.audio_output)
        self.assertFalse(self.model.capabilities.structured_output)
        self.assertFalse(self.model.capabilities.agent_capabilities.remote_mcp)
        self.assertFalse(self.provider.native.language_model("qwen3.8-flash").capabilities.files)

    async def test_native_reasoning_options_are_normalized_and_conflicts_rejected(self):
        await generate_text(model=self.model, prompt="test", provider_options={"reasoning_effort": "low"})
        self.assertEqual(self.requests[-1][1]["reasoning"], {"effort": "low"})
        self.assertNotIn("reasoning_effort", self.requests[-1][1])
        for options in [
            {"reasoning_effort": "low", "thinking_budget": 1024},
            {"thinking_budget": -1}, {"thinking_budget": True},
            {"reasoning_effort": "invalid"}, {"enable_thinking": False},
        ]:
            with self.subTest(options=options), self.assertRaises(ValidationError):
                await generate_text(model=self.model, prompt="test", provider_options=options)
        with self.assertRaises(ValidationError):
            await generate_text(model=self.model, prompt="test", reasoning=ReasoningConfig(effort="low", budget_tokens=1024))

    async def test_budget_search_uses_chat_search(self):
        result = await generate_text(model=self.model, prompt="search", reasoning=ReasoningConfig(budget_tokens=1024), tools={"search": qwen_web_search_tool()})
        self.assertEqual(result.text, "OK")
        self.assertEqual(result.usage.total_tokens, 15)
        self.assertTrue(self.requests[-1][1]["stream"])
        self.assertTrue(self.requests[-1][0].endswith("/chat/completions"))
        self.assertTrue(self.requests[-1][1]["enable_search"])
        self.assertEqual(self.requests[-1][1]["search_options"], {"search_strategy": "agent"})
        self.assertNotIn("tools", self.requests[-1][1])

    async def test_invalid_spatial_audio_metadata(self):
        for metadata in [{"use_multichannel": "true"}, {"arbitrary": True}]:
            with self.assertRaises(ValidationError):
                await generate_text(model=self.model, messages=[ModelMessage(role="user", parts=[FilePart(data="AAAA", media_type="audio/wav", provider_metadata={"qwen": metadata})])])

    def test_catalog_matches_omni_modalities(self):
        from zhivex_ai import default_model_catalog
        entry = default_model_catalog.find("qwen", "qwen3.8-omni-flash")
        self.assertIsNotNone(entry)
        self.assertTrue(entry.capabilities.audio_input)
        self.assertTrue(entry.capabilities.files)
        self.assertFalse(entry.capabilities.audio_output)
        self.assertFalse(entry.capabilities.structured_output)

    async def test_budget_cannot_force_a_tool(self):
        def lookup(value: str) -> str:
            return value
        with self.assertRaises(UnsupportedFeatureError):
            await generate_text(model=self.model, prompt="test", tools={"lookup": tool(lookup)}, tool_choice="required", reasoning=ReasoningConfig(budget_tokens=1024))
        self.assertEqual(self.requests, [])

    async def test_collected_search_rejects_unfinished_stream(self):
        async def fetch(url, **kwargs):
            return FakeResponse(200, body_text='data: {"choices":[{"delta":{"content":"partial"}}]}\n\ndata: [DONE]\n\n')
        provider = create_qwen(api_key="test", fetch=fetch)
        with self.assertRaisesRegex(ValidationError, "terminal finish"):
            await generate_text(model=provider.native.language_model("qwen3.8-omni-flash"), prompt="search", reasoning=ReasoningConfig(budget_tokens=1024), tools={"search": qwen_web_search_tool()})
