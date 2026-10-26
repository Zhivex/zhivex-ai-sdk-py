"""Vertex partner chat, E5 embeddings, and routes that must not hit Gemini."""
from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

from pydantic import BaseModel

from zhivex_ai import create_vertex, default_model_catalog, tool
from zhivex_ai.errors import ConfigurationError, UnsupportedFeatureError, ValidationError
from zhivex_ai.types import ImagePart, ModelGenerateInput, ModelMessage, ReasoningConfig, TextPart, ToolChoiceName
from tests.test_gemini_provider import FakeResponse


class Answer(BaseModel):
    value: str


def _ok(**kwargs):
    async def fetch(url, **request):
        fetch.seen.append((url, request.get("json_body")))
        return FakeResponse(200, {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]})

    fetch.seen = []
    return fetch


class VertexPartnerModelTests(IsolatedAsyncioTestCase):
    async def test_partner_chat_routes_and_reasoning_controls(self):
        fetch = _ok()
        provider = create_vertex(access_token="t", project_id="p", location="global", fetch=fetch)
        model = provider.native.language_model("gpt-oss-20b-maas")
        self.assertEqual(model.model_id, "openai/gpt-oss-20b-maas")
        self.assertTrue(model.capabilities.reasoning)
        self.assertTrue(fetch.seen == [])
        await model.generate(ModelGenerateInput(
            messages=[ModelMessage(role="user", parts=[TextPart(text="think")])],
            reasoning=ReasoningConfig(effort="high"),
            tools={"lookup": tool(name="lookup", schema=Answer, execute=lambda item: item)},
        ))
        url, body = fetch.seen[-1]
        self.assertTrue(url.endswith("/endpoints/openapi/chat/completions"))
        self.assertEqual(body["model"], "openai/gpt-oss-20b-maas")
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(body["tools"][0]["function"]["description"], "")

        for effort in ("xhigh", "none"):
            with self.assertRaises(UnsupportedFeatureError):
                await model.generate(ModelGenerateInput(
                    messages=[ModelMessage(role="user", parts=[TextPart(text="think")])],
                    reasoning=ReasoningConfig(effort=effort),
                ))
        with self.assertRaises(UnsupportedFeatureError):
            await model.generate(ModelGenerateInput(
                messages=[ModelMessage(role="user", parts=[TextPart(text="think")])],
                reasoning=ReasoningConfig(budget_tokens=128),
            ))
        with self.assertRaises(UnsupportedFeatureError):
            await model.generate(ModelGenerateInput(
                messages=[ModelMessage(role="user", parts=[TextPart(text="think")])],
                tools={"lookup": tool(name="lookup", schema=Answer, execute=lambda item: item)},
                tool_choice=ToolChoiceName(tool_name="lookup"),
            ))

        hybrid = provider.native.language_model("deepseek-ai/deepseek-v3.1-maas")
        await hybrid.generate(ModelGenerateInput(
            messages=[ModelMessage(role="user", parts=[TextPart(text="think")])],
            reasoning=ReasoningConfig(effort="low"),
            provider_options={"chat_template_kwargs": {"keep": True}},
        ))
        self.assertEqual(fetch.seen[-1][1]["chat_template_kwargs"], {"keep": True, "thinking": True})
        self.assertNotIn("reasoning_effort", fetch.seen[-1][1])
        await hybrid.generate(ModelGenerateInput(
            messages=[ModelMessage(role="user", parts=[TextPart(text="think")])],
            provider_options={"reasoning_effort": "none"},
        ))
        self.assertIs(fetch.seen[-1][1]["chat_template_kwargs"]["thinking"], False)

        glm = provider.native.language_model("zai-org/glm-4.7-maas")
        await glm.generate(ModelGenerateInput(
            messages=[ModelMessage(role="user", parts=[TextPart(text="think")])],
            reasoning=ReasoningConfig(effort="medium"),
        ))
        self.assertIs(fetch.seen[-1][1]["chat_template_kwargs"]["enable_thinking"], True)

        grok = provider.native.language_model("xai/grok-4.6")
        self.assertTrue(grok.capabilities.vision)
        self.assertTrue(grok.capabilities.reasoning)
        await grok.generate(ModelGenerateInput(messages=[
            ModelMessage(role="user", parts=[ImagePart(image="data:image/png;base64,AQID")]),
        ]))
        self.assertEqual(fetch.seen[-1][1]["messages"][0]["content"][0]["image_url"]["url"], "data:image/png;base64,AQID")
        with self.assertRaises(UnsupportedFeatureError):
            await grok.generate(ModelGenerateInput(
                messages=[ModelMessage(role="user", parts=[TextPart(text="think")])],
                reasoning=ReasoningConfig(effort="low"),
            ))

        ocr = provider.native.language_model("deepseek-ocr-maas")
        self.assertTrue(ocr.capabilities.vision)
        self.assertFalse(ocr.capabilities.tools)
        flash = provider.native.language_model("deepseek-v4.1-flash-maas")
        self.assertFalse(flash.capabilities.tools)
        self.assertFalse(flash.capabilities.structured_output)
        qwen = provider.native.language_model("qwen3-coder-480b-a35b-instruct-maas")
        await qwen.generate(ModelGenerateInput(
            messages=[ModelMessage(role="user", parts=[TextPart(text="call")])],
            tools={"lookup": tool(name="lookup", description="find", schema=Answer, execute=lambda item: item)},
        ))
        self.assertEqual(fetch.seen[-1][1]["model"], "qwen/qwen3-coder-480b-a35b-instruct-maas")
        self.assertEqual(fetch.seen[-1][1]["tool_choice"], "auto")

        legacy = provider.native.language_model("zai-org/glm-5.2-maas")
        with self.assertRaises(UnsupportedFeatureError):
            await legacy.generate(ModelGenerateInput(
                messages=[ModelMessage(role="user", parts=[TextPart(text="think")])],
                reasoning=ReasoningConfig(effort="low"),
            ))

    async def test_catalog_matches_retired_partner_rows_without_invented_prices(self):
        expected = {
            "openai/gpt-oss-20b-maas": "openai/gpt-oss-120b-maas",
            "meta/llama-3.3-70b-instruct-maas": "meta/llama-4-maverick-17b-128e-instruct-maas",
            "xai/grok-4.1-fast-reasoning": "xai/grok-4.6",
            "qwen/qwen3-235b-a22b-instruct-2507-maas": "qwen/qwen3-next-80b-a3b-instruct-maas",
            "zai-org/glm-4.7-maas": "zai-org/glm-5.2-maas",
            "deepseek-ai/deepseek-r1-0528-maas": "deepseek-ai/deepseek-v4.1-flash-maas",
            "deepseek-ai/deepseek-ocr-maas": "deepseek-ai/deepseek-v4.1-flash-maas",
            "intfloat/multilingual-e5-large-instruct-maas": "gemini-embedding-2",
        }
        for model_id, replacement in expected.items():
            entry = default_model_catalog.find("vertex", model_id)
            self.assertEqual(entry.availability, "retired")
            self.assertEqual(entry.replacement_model_id, replacement)
            self.assertIsNone(entry.pricing)
            self.assertEqual(entry.support_evidence, "offline-contract")
        flash = default_model_catalog.find("vertex", "deepseek-v4.1-flash-maas")
        self.assertEqual(flash.availability, "preview")
        self.assertIsNone(flash.pricing)
        self.assertFalse(flash.capabilities.tools)
        self.assertEqual(default_model_catalog.find("vertex", "lyria-002").api_surface, "media")
        self.assertEqual(default_model_catalog.find("vertex", "gemini-nano-banana-2.1").api_surface, "image")
        self.assertEqual(default_model_catalog.find("vertex", "multimodalembedding@001").api_surface, "embedding")

    async def test_e5_embeddings_are_text_only_openapi(self):
        seen = []

        async def fetch(url, **request):
            seen.append((url, request["json_body"]))
            return FakeResponse(200, {
                "data": [
                    {"index": 1, "embedding": [0.5, 1]},
                    {"index": 0, "embedding": [1, 2]},
                ],
                "usage": {"prompt_tokens": 4},
            })

        provider = create_vertex(access_token="t", project_id="p", fetch=fetch)
        model = provider.embedding_model("multilingual-e5-small-maas")
        self.assertEqual(model.model_id, "intfloat/multilingual-e5-small-maas")
        self.assertTrue(model.capabilities.embeddings)
        result = await model.embed(["alpha", "beta"])
        self.assertEqual(result.embeddings, [[1, 2], [0.5, 1]])
        self.assertEqual(result.usage.input_tokens, 4)
        self.assertTrue(seen[-1][0].endswith("/endpoints/openapi/embeddings"))
        self.assertEqual(seen[-1][1], {
            "model": "intfloat/multilingual-e5-small-maas",
            "input": ["alpha", "beta"],
            "encoding_format": "float",
        })
        with self.assertRaises(UnsupportedFeatureError):
            await model.embed([ImagePart(image="data:image/png;base64,AQID")])
        with self.assertRaises(UnsupportedFeatureError):
            await model.embed(["alpha"], {"output_dimensionality": 8})
        with self.assertRaises(ValidationError):
            await provider.embedding_model("intfloat/multilingual-e5-large-instruct-maas").embed(["alpha"])
        with self.assertRaises(ConfigurationError):
            create_vertex(api_key="k", express_mode=True).embedding_model("intfloat/multilingual-e5-small-maas")

    async def test_language_model_rejects_non_gemini_surfaces(self):
        provider = create_vertex(access_token="t", project_id="p", fetch=_ok())
        for model_id in (
            "claude-sonnet-5-5",
            "claude-fable-5-1",
            "virtual-try-on-001",
            "mistralai/mistral-ocr-2505",
            "intfloat/multilingual-e5-small-maas",
        ):
            with self.subTest(model_id=model_id):
                with self.assertRaises(UnsupportedFeatureError):
                    provider.native.language_model(model_id)
                with self.assertRaises(UnsupportedFeatureError):
                    provider.native.grounded_language_model(model_id)

    async def test_nano_banana_uses_vertex_image_generate_content(self):
        seen = []

        async def fetch(url, **request):
            seen.append((url, request["json_body"]))
            return FakeResponse(200, {"candidates": []})

        provider = create_vertex(access_token="t", project_id="p", fetch=fetch)
        await provider.images().generate(prompt="a red square", model="gemini-nano-banana-2.1")
        self.assertTrue(seen[-1][0].endswith("/publishers/google/models/gemini-nano-banana-2.1:generateContent"))
        self.assertEqual(seen[-1][1]["generationConfig"]["responseModalities"], ["IMAGE"])
