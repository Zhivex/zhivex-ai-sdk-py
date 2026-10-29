from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    ConfigurationError,
    UnsupportedFeatureError,
    ValidationError,
    create_azure_openai,
    create_xai,
    create_zai,
    xai_web_search_tool,
)
from zhivex_ai.catalog import default_model_catalog
from zhivex_ai.messages import user
from zhivex_ai.types import AudioInput, ImagePart, ModelGenerateInput, ModelMessage, ReasoningConfig, TextPart


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
            return self.body_bytes.decode()
        return json.dumps(self.payload)

    async def read(self) -> bytes:
        if self.body_bytes:
            return self.body_bytes
        return (self.body_text or json.dumps(self.payload or {})).encode()

    async def iter_lines(self) -> AsyncIterable[str]:
        for line in self.body_text.splitlines():
            yield line


def _completed(text: str = "ok") -> dict[str, Any]:
    return {
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }


class XAIProviderTests(IsolatedAsyncioTestCase):
    def test_missing_api_key_is_a_configuration_error(self) -> None:
        with self.assertRaises(ConfigurationError):
            create_xai(api_key="", base_url="https://api.x.ai/v1")

    def test_region_and_base_url_cannot_combine(self) -> None:
        with self.assertRaises(ConfigurationError):
            create_xai(api_key="test", base_url="https://api.x.ai/v1", region="us")

    async def test_grok_47_responses_sends_effort_and_encrypted_reasoning_when_store_is_off(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, *, headers: dict[str, str], json_body: dict[str, Any] | None = None, **kwargs: Any):
            requests.append({"url": url, "json": json_body, "headers": headers})
            return FakeResponse(status_code=200, payload=_completed())

        provider = create_xai(api_key="test-key", fetch=fetch)
        model = provider.native.language_model("grok-4.7")
        result = await model.generate(ModelGenerateInput(
            messages=[user("Prove it.")],
            reasoning=ReasoningConfig(effort="xhigh"),
            provider_options={"store": False, "conversation_id": "conv-1"},
        ))

        self.assertEqual(result.text, "ok")
        self.assertEqual(requests[0]["url"], "https://api.x.ai/v1/responses")
        self.assertEqual(requests[0]["json"]["model"], "grok-4.7")
        self.assertEqual(requests[0]["json"]["reasoning"], {"effort": "xhigh"})
        self.assertIn("reasoning.encrypted_content", requests[0]["json"]["include"])
        self.assertEqual(requests[0]["json"]["prompt_cache_key"], "conv-1")
        self.assertEqual(requests[0]["headers"]["authorization"], "Bearer test-key")
        self.assertFalse(model.capabilities.agent_capabilities.hosted_file_search)

    async def test_grok_47_rejects_chat_mode_and_sampling_penalties(self) -> None:
        async def fetch(*args: Any, **kwargs: Any):
            raise AssertionError("request must not be sent")

        model = create_xai(api_key="test", fetch=fetch).native.language_model("grok-4.7")
        with self.assertRaises(UnsupportedFeatureError):
            await model.generate(ModelGenerateInput(messages=[user("hi")], provider_options={"api_mode": "chat"}))
        with self.assertRaises(UnsupportedFeatureError):
            await model.generate(ModelGenerateInput(
                messages=[user("hi")],
                provider_options={"presence_penalty": 0.2},
            ))

    async def test_us_region_rejects_models_other_than_grok_46_and_47(self) -> None:
        async def fetch(*args: Any, **kwargs: Any):
            raise AssertionError("request must not be sent")

        model = create_xai(api_key="test", region="us", fetch=fetch).native.language_model("grok-4.5")
        with self.assertRaises(UnsupportedFeatureError):
            await model.generate(ModelGenerateInput(messages=[user("hi")]))

    async def test_chat_mode_uses_reasoning_effort_and_conversation_header(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, *, headers: dict[str, str], json_body: dict[str, Any] | None = None, **kwargs: Any):
            requests.append({"url": url, "json": json_body, "headers": headers})
            return FakeResponse(status_code=200, payload={
                "choices": [{"message": {"content": "chat ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            })

        model = create_xai(api_key="test", fetch=fetch).native.language_model("grok-4.3")
        result = await model.generate(ModelGenerateInput(
            messages=[user("hi")],
            reasoning=ReasoningConfig(effort="none"),
            provider_options={"api_mode": "chat", "conversation_id": "chat-1"},
        ))

        self.assertEqual(result.text, "chat ok")
        self.assertEqual(requests[0]["url"], "https://api.x.ai/v1/chat/completions")
        self.assertEqual(requests[0]["json"]["reasoning_effort"], "none")
        self.assertEqual(requests[0]["headers"]["x-grok-conv-id"], "chat-1")

    async def test_non_reasoning_model_rejects_reasoning_and_responses_hosted_tool_is_sent(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, *, json_body: dict[str, Any] | None = None, **kwargs: Any):
            requests.append(json_body)
            return FakeResponse(status_code=200, payload=_completed())

        provider = create_xai(api_key="test", fetch=fetch)
        plain = provider.native.language_model("grok-4.20-0309-non-reasoning")
        with self.assertRaises(UnsupportedFeatureError):
            await plain.generate(ModelGenerateInput(messages=[user("hi")], reasoning=ReasoningConfig(effort="low")))

        await provider.native.language_model("grok-4.6").generate(ModelGenerateInput(
            messages=[user("search")],
            tools={"web_search": xai_web_search_tool(allowed_domains=["docs.x.ai"])},
        ))
        self.assertEqual(requests[0]["tools"][0]["type"], "web_search")
        self.assertEqual(requests[0]["tools"][0]["filters"]["allowed_domains"], ["docs.x.ai"])

    def test_web_search_domains_cannot_combine(self) -> None:
        with self.assertRaises(UnsupportedFeatureError):
            xai_web_search_tool(allowed_domains=["a.example"], excluded_domains=["b.example"])

    async def test_image_quality_is_limited_to_imagine_2(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, *, json_body: dict[str, Any] | None = None, **kwargs: Any):
            requests.append({"url": url, "json": json_body})
            return FakeResponse(status_code=200, payload={"data": [{"url": "https://cdn.example/cat.jpg"}]})

        images = create_xai(api_key="test", fetch=fetch).native.images()
        with self.assertRaises(UnsupportedFeatureError):
            await images.generate(prompt="cat", model="grok-imagine-image", quality="low")
        result = await images.generate(
            prompt="cat",
            model="grok-imagine-image-2.0",
            quality="medium",
            size="2k",
            extra_body={"aspect_ratio": "16:9"},
        )
        self.assertEqual(result.images[0].url, "https://cdn.example/cat.jpg")
        self.assertEqual(requests[0]["url"], "https://api.x.ai/v1/images/generations")
        self.assertEqual(requests[0]["json"]["quality"], "medium")
        self.assertEqual(requests[0]["json"]["resolution"], "2k")
        self.assertEqual(requests[0]["json"]["aspect_ratio"], "16:9")

    async def test_video_resolution_limit_and_speech_route(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, *, json_body: dict[str, Any] | None = None, **kwargs: Any):
            requests.append({"url": url, "json": json_body})
            if url.endswith("/tts"):
                return FakeResponse(status_code=200, body_bytes=b"audio", headers={"content-type": "audio/mpeg"})
            return FakeResponse(status_code=200, payload={
                "request_id": "vid-1",
                "status": "done",
                "video": {"url": "https://cdn.example/clip.mp4"},
            })

        provider = create_xai(api_key="test", fetch=fetch)
        with self.assertRaises(UnsupportedFeatureError):
            await provider.native.videos().generate(prompt="wave", model="grok-imagine-video", extra_body={"resolution": "1080p"})
        operation = await provider.native.videos().generate(
            prompt="wave",
            model="grok-imagine-video-1.5",
            extra_body={"resolution": "1080p", "duration": 4},
        )
        self.assertTrue(operation.done)
        self.assertEqual(operation.response["url"], "https://cdn.example/clip.mp4")
        speech = await provider.native.speech_model("grok-tts").generate_speech(input="Hello", voice="eve")
        self.assertEqual(speech.audio, b"audio")
        self.assertEqual(requests[-1]["url"], "https://api.x.ai/v1/tts")
        self.assertEqual(requests[-1]["json"]["voice_id"], "eve")
        with self.assertRaises(UnsupportedFeatureError):
            provider.native.speech_model("grok-voice-2.0")

    async def test_transcription_rejects_prompt_and_posts_to_stt(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, *, body: Any = None, **kwargs: Any):
            requests.append({"url": url, "body": body})
            return FakeResponse(status_code=200, payload={"text": "hello"})

        model = create_xai(api_key="test", fetch=fetch).native.transcription_model("grok-voice-transcribe-2.0")
        with self.assertRaises(UnsupportedFeatureError):
            await model.transcribe(audio=AudioInput(data=b"abc", media_type="audio/wav"), prompt="hint")
        result = await model.transcribe(
            audio=AudioInput(data=b"abc", media_type="audio/wav"),
            provider_options={"keyterm": "Grok"},
        )
        self.assertEqual(result.text, "hello")
        self.assertEqual(requests[0]["url"], "https://api.x.ai/v1/stt")
        self.assertEqual(requests[0]["body"]["data"]["keyterm"], "Grok")


class ZAIProviderTests(IsolatedAsyncioTestCase):
    async def test_glm_53_enables_thinking_and_rejects_images_and_temperature(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, *, json_body: dict[str, Any] | None = None, **kwargs: Any):
            requests.append({"url": url, "json": json_body})
            return FakeResponse(status_code=200, payload={
                "choices": [{"message": {"content": "done", "reasoning_content": "think"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6, "prompt_tokens_details": {"cached_tokens": 1}},
            })

        provider = create_zai(api_key="test", fetch=fetch)
        model = provider.native.language_model("glm-5.3")
        with self.assertRaises(ValidationError):
            await model.generate(ModelGenerateInput(messages=[user("hi")], temperature=1.5))
        with self.assertRaises(UnsupportedFeatureError):
            await model.generate(ModelGenerateInput(messages=[ModelMessage(role="user", parts=[ImagePart(image="abcd", media_type="image/png")])]))
        result = await model.generate(ModelGenerateInput(
            messages=[user("Ship the patch.")],
            reasoning=ReasoningConfig(effort="max"),
        ))
        self.assertEqual(result.text, "done")
        self.assertEqual(result.usage.cached_input_tokens, 1)
        self.assertEqual(requests[0]["url"], "https://api.z.ai/api/paas/v4/chat/completions")
        self.assertEqual(requests[0]["json"]["thinking"], {"type": "enabled"})
        self.assertEqual(requests[0]["json"]["reasoning_effort"], "max")
        self.assertFalse(model.capabilities.vision)

    async def test_glm_53_flash_sends_images_and_preserves_thinking(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, *, json_body: dict[str, Any] | None = None, **kwargs: Any):
            requests.append(json_body)
            return FakeResponse(status_code=200, payload={"choices": [{"message": {"content": "seen"}, "finish_reason": "stop"}]})

        model = create_zai(api_key="test", endpoint="coding", fetch=fetch).native.language_model("glm-5.3-flash")
        await model.generate(ModelGenerateInput(messages=[ModelMessage(role="user", parts=[
            TextPart(text="What is this?"),
            ImagePart(image="https://cdn.example/cat.png", media_type="image/png"),
        ])]))
        self.assertTrue(model.capabilities.vision)
        self.assertEqual(requests[0]["thinking"], {"type": "enabled", "clear_thinking": False})
        self.assertEqual(requests[0]["messages"][0]["content"][1]["image_url"]["url"], "https://cdn.example/cat.png")

    async def test_glm_52_minimal_effort_disables_thinking(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, *, json_body: dict[str, Any] | None = None, **kwargs: Any):
            requests.append(json_body)
            return FakeResponse(status_code=200, payload={"choices": [{"message": {"content": "plain"}, "finish_reason": "stop"}]})

        model = create_zai(api_key="test", fetch=fetch).native.language_model("glm-5.2")
        await model.generate(ModelGenerateInput(messages=[user("hi")], reasoning=ReasoningConfig(effort="minimal")))
        self.assertEqual(requests[0]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", requests[0])
        with self.assertRaises(UnsupportedFeatureError):
            await model.generate(ModelGenerateInput(messages=[user("hi")], reasoning=ReasoningConfig(budget_tokens=10)))


class CatalogParityTests(IsolatedAsyncioTestCase):
    def test_xai_and_zai_catalog_prices(self) -> None:
        grok = default_model_catalog.find("xai", "grok-4.7")
        assert grok is not None and grok.pricing is not None
        self.assertEqual(grok.pricing.input_per_1m_tokens, 2)
        self.assertEqual(grok.pricing.cached_input_per_1m_tokens, 0.5)
        self.assertEqual(grok.pricing.output_per_1m_tokens, 6)
        self.assertEqual(grok.pricing.long_context_threshold_tokens, 200_000)
        image = default_model_catalog.find("xai", "grok-imagine-image-quality")
        assert image is not None
        self.assertEqual(image.availability, "deprecated")
        self.assertEqual(image.replacement_model_id, "grok-imagine-image-2.0")
        flash = default_model_catalog.find("zai", "glm-5.3-flash")
        assert flash is not None and flash.pricing is not None
        self.assertEqual(flash.pricing.input_per_1m_tokens, 0.15)
        self.assertIsNotNone(default_model_catalog.find("azure-openai", "gpt-6-sol"))
        self.assertEqual(default_model_catalog.find("azure-openai", "gpt-4o-mini").availability, "deprecated")

    async def test_azure_gpt6_families_use_openai_reasoning_gates(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(url: str, *, json_body: dict[str, Any] | None = None, **kwargs: Any):
            requests.append({"url": url, "json": json_body})
            return FakeResponse(status_code=200, payload=_completed("azure"))

        provider = create_azure_openai(api_key="test", endpoint="https://example.openai.azure.com", fetch=fetch)
        sol = provider.native.language_model("gpt-6-sol")
        with self.assertRaises(UnsupportedFeatureError):
            await sol.generate(ModelGenerateInput(messages=[user("hi")], temperature=0.2))
        await provider.native.language_model("gpt-6-luna").generate(ModelGenerateInput(
            messages=[user("hi")],
            reasoning=ReasoningConfig(effort="none"),
            temperature=0.2,
        ))
        self.assertEqual(requests[0]["json"]["model"], "gpt-6-luna")
        self.assertEqual(requests[0]["json"]["temperature"], 0.2)
        self.assertTrue(requests[0]["url"].endswith("/responses"))

    async def test_azure_speech_and_transcription_forward_new_model_ids(self) -> None:
        requests: list[str] = []

        async def fetch(url: str, *, json_body: dict[str, Any] | None = None, body: Any = None, **kwargs: Any):
            requests.append(url)
            if url.endswith("/audio/speech"):
                return FakeResponse(status_code=200, body_bytes=b"wav", headers={"content-type": "audio/mpeg"})
            return FakeResponse(status_code=200, payload={"text": "heard"})

        provider = create_azure_openai(api_key="test", endpoint="https://example.openai.azure.com", fetch=fetch)
        await provider.native.speech_model("gpt-4o-mini-tts").generate_speech(input="hello", voice="alloy")
        await provider.native.transcription_model("gpt-4o-mini-transcribe").transcribe(
            audio=AudioInput(data=b"abc", media_type="audio/wav"),
        )
        self.assertTrue(requests[0].endswith("/audio/speech"))
        self.assertTrue(requests[1].endswith("/audio/transcriptions"))
