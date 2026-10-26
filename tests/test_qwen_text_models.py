"""Qwen text families that share the existing Responses client."""
from __future__ import annotations

import json
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any
from unittest import IsolatedAsyncioTestCase

from zhivex_ai import (
    ImagePart,
    ModelMessage,
    ReasoningConfig,
    TextPart,
    UnsupportedFeatureError,
    create_qwen,
    default_model_catalog,
    qwen_web_search_tool,
    stream_text,
)
from zhivex_ai.providers.qwen import Qwen38LanguageModel, QwenStreamingLanguageModel
from zhivex_ai.types import ModelGenerateInput, StructuredOutputConfig


@dataclass
class FakeResponse:
    status_code: int
    payload: Any = None
    body_text: str = ""

    async def json(self) -> Any:
        return self.payload

    async def text(self) -> str:
        return self.body_text or json.dumps(self.payload)

    async def read(self) -> bytes:
        return (self.body_text or json.dumps(self.payload)).encode()

    async def iter_lines(self) -> AsyncIterable[str]:
        for line in self.body_text.splitlines():
            yield line


def _completed(text: str = "ok") -> dict[str, Any]:
    return {
        "id": "resp_qwen",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
    }


def _input(text: str = "hello") -> ModelGenerateInput:
    return ModelGenerateInput(messages=[ModelMessage(role="user", parts=[TextPart(text=text)])])


class QwenTextModelTests(IsolatedAsyncioTestCase):
    async def test_qwen_text_models_use_responses_reasoning_effort(self):
        requests: list[dict[str, Any]] = []

        async def fetch(url, **request):
            requests.append({"url": url, "json": request["json_body"]})
            return FakeResponse(200, _completed())

        provider = create_qwen(api_key="test", fetch=fetch)
        for model_id in (
            "qwen3.7-flash",
            "qwen3.7-flash-2026-07-15",
            "qwen3.8-27b",
            "deepseek-v4.1-flash",
            "deepseek-v4-pro",
            "deepseek-v4-pro-0813",
            "glm-5.2",
            "MiniMax-M2.5",
        ):
            with self.subTest(model_id=model_id):
                result = await provider.native.language_model(model_id).generate(
                    ModelGenerateInput(
                        messages=[ModelMessage(role="user", parts=[TextPart(text="hello")])],
                        reasoning=ReasoningConfig(effort="low"),
                    )
                )
                self.assertEqual(result.text, "ok")
                self.assertTrue(requests[-1]["url"].endswith("/responses"))
                self.assertEqual(requests[-1]["json"]["model"], model_id)
                self.assertEqual(requests[-1]["json"]["reasoning"], {"effort": "low"})
                entry = default_model_catalog.find("qwen", model_id)
                self.assertIsNotNone(entry)
                assert entry is not None
                self.assertEqual(entry.support_evidence, "offline-contract")
                self.assertTrue(entry.capabilities is not None and entry.capabilities.reasoning)
                self.assertTrue(entry.capabilities.tools)

    async def test_qwen38_dated_snapshot_stays_on_the_max_alias(self):
        requests: list[dict[str, Any]] = []

        async def fetch(url, **request):
            requests.append(request["json_body"])
            return FakeResponse(200, _completed())

        provider = create_qwen(api_key="test", fetch=fetch)
        model = provider.native.language_model("qwen3.8-max-2026-09-02")
        self.assertIsInstance(model, Qwen38LanguageModel)
        await model.generate(_input())
        self.assertEqual(requests[-1]["model"], "qwen3.8-max-2026-09-02")
        entry = default_model_catalog.find("qwen", "qwen3.8-max-2026-09-02")
        self.assertEqual(entry.model_id, "qwen3.8-max-0902")  # type: ignore[union-attr]

    async def test_qwen38_27b_sends_responses_image_input(self):
        requests: list[dict[str, Any]] = []

        async def fetch(url, **request):
            requests.append(request["json_body"])
            return FakeResponse(200, _completed("blue"))

        provider = create_qwen(api_key="test", fetch=fetch)
        result = await provider.native.language_model("qwen3.8-27b").generate(
            ModelGenerateInput(messages=[ModelMessage(role="user", parts=[
                TextPart(text="What color?"),
                ImagePart(image="data:image/png;base64,iVBORw0KGgo="),
            ])])
        )
        self.assertEqual(result.text, "blue")
        self.assertEqual(requests[-1]["input"][0]["content"][1], {
            "type": "input_image",
            "image_url": "data:image/png;base64,iVBORw0KGgo=",
        })

    async def test_qwen38_24t_is_streaming_and_thinking_only(self):
        requests: list[dict[str, Any]] = []

        async def fetch(url, **request):
            requests.append({"url": url, "json": request["json_body"], "stream": request["stream"]})
            return FakeResponse(
                200,
                body_text=(
                    'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
                    'data: {"type":"response.completed","response":{"id":"resp_stream","status":"completed"}}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        provider = create_qwen(api_key="test", fetch=fetch)
        model = provider.native.language_model("qwen3.8-2.4t-a95b")
        self.assertIsInstance(model, QwenStreamingLanguageModel)
        with self.assertRaises(UnsupportedFeatureError):
            await model.generate(_input())
        self.assertEqual(requests, [])
        with self.assertRaises(UnsupportedFeatureError):
            await model.stream(ModelGenerateInput(
                messages=[ModelMessage(role="user", parts=[TextPart(text="hello")])],
                reasoning=ReasoningConfig(effort="none"),
            ))
        with self.assertRaises(UnsupportedFeatureError):
            await model.stream(ModelGenerateInput(
                messages=[ModelMessage(role="user", parts=[TextPart(text="hello")])],
                provider_options={"enable_thinking": False},
            ))
        stream = stream_text(model=model, prompt="hello", reasoning=ReasoningConfig(effort="high"))
        result = await stream.collect()
        self.assertEqual(result.text, "ok")
        self.assertTrue(requests[-1]["stream"])
        self.assertEqual(requests[-1]["json"]["model"], "qwen3.8-2.4t-a95b")
        self.assertEqual(requests[-1]["json"]["reasoning"], {"effort": "high"})

    async def test_third_party_text_models_reject_unsupported_qwen_features(self):
        provider = create_qwen(api_key="test", fetch=lambda *args, **kwargs: None)
        for model_id in ("deepseek-v4-pro", "deepseek-v4-flash-0731", "glm-5.2", "MiniMax-M2.5"):
            with self.subTest(model_id=model_id, feature="builtin"):
                with self.assertRaises(UnsupportedFeatureError):
                    await provider.native.language_model(model_id).generate(ModelGenerateInput(
                        messages=[ModelMessage(role="user", parts=[TextPart(text="hello")])],
                        tools={"search": qwen_web_search_tool()},
                    ))
        for model_id in ("deepseek-v4-flash-0731", "MiniMax-M2.5"):
            with self.subTest(model_id=model_id, feature="structured"):
                with self.assertRaises(UnsupportedFeatureError):
                    await provider.native.language_model(model_id).generate(ModelGenerateInput(
                        messages=[ModelMessage(role="user", parts=[TextPart(text="hello")])],
                        structured_output=StructuredOutputConfig(schema={"type": "object"}, mode="native"),
                    ))
