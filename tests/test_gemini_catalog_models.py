"""Gemini catalog IDs that reuse the existing Developer API clients."""
from __future__ import annotations

from unittest import IsolatedAsyncioTestCase

from zhivex_ai import create_gemini, default_model_catalog
from zhivex_ai.errors import UnsupportedFeatureError
from zhivex_ai.types import ImagePart, ModelGenerateInput, ModelMessage, ReasoningConfig, TextPart
from tests.test_gemini_provider import FakeResponse


def _text_response():
    return FakeResponse(
        status_code=200,
        payload={"candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}]},
    )


class GeminiCatalogModelTests(IsolatedAsyncioTestCase):
    async def test_only_embedding_2_accepts_multimodal_values(self):
        called = []

        async def fetch(url, **kwargs):
            called.append(url)
            return FakeResponse(status_code=200, payload={"embedding": {"values": [0.1]}})

        provider = create_gemini(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError) as caught:
            await provider.embedding_model("text-embedding-004").embed(
                [ImagePart(image="data:image/png;base64,AQID")]
            )
        self.assertIn("gemini-embedding-2", str(caught.exception))
        self.assertFalse(called)
        result = await provider.embedding_model("gemini-embedding-2").embed(
            [ImagePart(image="data:image/png;base64,AQID")]
        )
        self.assertEqual(result.embeddings, [[0.1]])
        self.assertIn("/models/gemini-embedding-2:embedContent", called[-1])

    async def test_gemini_3_preview_ids_use_thinking_level(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append((url, kwargs["json_body"]))
            return _text_response()

        provider = create_gemini(api_key="test", fetch=fetch)
        flash = provider.native.language_model("gemini-3-flash-preview")
        await flash.generate(ModelGenerateInput(
            messages=[ModelMessage(role="user", parts=[TextPart(text="think")])],
            reasoning=ReasoningConfig(effort="minimal"),
        ))
        self.assertIn("/models/gemini-3-flash-preview:generateContent", seen[-1][0])
        self.assertEqual(seen[-1][1]["generationConfig"]["thinkingConfig"], {"thinkingLevel": "minimal"})
        custom = provider.native.language_model("gemini-3.1-pro-preview-customtools")
        with self.assertRaises(UnsupportedFeatureError):
            await custom.generate(ModelGenerateInput(
                messages=[ModelMessage(role="user", parts=[TextPart(text="think")])],
                reasoning=ReasoningConfig(effort="minimal"),
            ))
        with self.assertRaises(UnsupportedFeatureError):
            await custom.generate(ModelGenerateInput(
                messages=[ModelMessage(role="user", parts=[TextPart(text="think")])],
                reasoning=ReasoningConfig(budget_tokens=64),
            ))
        await custom.generate(ModelGenerateInput(
            messages=[ModelMessage(role="user", parts=[TextPart(text="think")])],
            reasoning=ReasoningConfig(effort="high"),
        ))
        self.assertEqual(seen[-1][1]["generationConfig"]["thinkingConfig"], {"thinkingLevel": "high"})

        robotics = provider.native.language_model("gemini-robotics-er-1.6-preview")
        await robotics.generate(ModelGenerateInput(
            messages=[ModelMessage(role="user", parts=[TextPart(text="look")])],
        ))
        self.assertIn("/models/gemini-robotics-er-1.6-preview:generateContent", seen[-1][0])

    async def test_image_and_lite_video_use_existing_clients(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append((url, kwargs.get("json_body")))
            if "predictLongRunning" in url:
                return FakeResponse(status_code=200, payload={"name": "operations/lite", "done": False})
            return FakeResponse(status_code=200, payload={"candidates": []})

        provider = create_gemini(api_key="test", fetch=fetch)
        await provider.images().generate(prompt="square", model="gemini-nano-banana-2.1")
        self.assertIn("/models/gemini-nano-banana-2.1:generateContent", seen[-1][0])
        self.assertEqual(seen[-1][1]["generationConfig"]["responseModalities"], ["IMAGE"])
        await provider.images().generate(prompt="square")
        self.assertIn("/models/gemini-2.5-flash-image:generateContent", seen[-1][0])
        await provider.videos().generate(model="veo-3.1-lite-generate-preview", prompt="a short clip")
        self.assertIn("/models/veo-3.1-lite-generate-preview:predictLongRunning", seen[-1][0])

    def test_catalog_rows_match_lifecycle_without_invented_prices(self):
        flash = default_model_catalog.find("gemini", "gemini-3-flash-preview")
        custom = default_model_catalog.find("gemini", "gemini-3.1-pro-preview-customtools")
        image = default_model_catalog.find("gemini", "gemini-2.5-flash-image")
        banana = default_model_catalog.find("gemini", "gemini-nano-banana-2.1")
        embedding = default_model_catalog.find("gemini", "gemini-embedding-2")
        robotics = default_model_catalog.find("gemini", "gemini-robotics-er-1.6-preview")
        lite = default_model_catalog.find("gemini", "veo-3.1-lite-generate-preview")
        self.assertEqual(flash.availability, "preview")
        self.assertEqual(custom.availability, "preview")
        self.assertEqual(image.api_surface, "image")
        self.assertEqual(banana.api_surface, "image")
        self.assertTrue(embedding.capabilities.embeddings)
        self.assertTrue(embedding.capabilities.vision)
        self.assertEqual(robotics.availability, "retired")
        self.assertEqual(robotics.replacement_model_id, "gemini-robotics-er-2-preview")
        self.assertEqual(lite.availability, "retired")
        self.assertEqual(lite.replacement_model_id, "veo-3.1-generate-preview")
        for entry in (flash, custom, image, banana, embedding, robotics, lite):
            self.assertIsNone(entry.pricing)
            self.assertEqual(entry.support_evidence, "offline-contract")
        self.assertIsNone(default_model_catalog.find("gemini", "lyria-realtime-exp"))
