from __future__ import annotations

import json
from unittest import IsolatedAsyncioTestCase

from pydantic import BaseModel

from zhivex_ai import create_vertex, generate_text, generate_object, stream_text, tool
from zhivex_ai.errors import ConfigurationError, ValidationError
from zhivex_ai.types import (
    ModelGenerateInput,
    ModelMessage,
    TextPart,
    ImagePart,
    ToolCallPart,
    ToolResultPart,
    ToolExecutionResult,
)
from tests.test_gemini_provider import FakeResponse


class Answer(BaseModel):
    value: str


class VertexChatTests(IsolatedAsyncioTestCase):
    async def test_mistral_image_and_function_roundtrip_on_both_factories(self):
        seen = []

        async def fetch(url, **kwargs):
            body = kwargs["json_body"]
            seen.append((url, body))
            if body["messages"][-1]["role"] == "tool":
                message = {"content": "verified"}
                reason = "stop"
            else:
                message = {"tool_calls": [{"id": "lookup123", "type": "function", "function": {"name": "lookup", "arguments": '{"value":"red"}'}}]}
                reason = "tool_calls"
            return FakeResponse(200, {"choices": [{"message": message, "finish_reason": reason}]})

        provider = create_vertex(access_token="synthetic", project_id="p", location="europe-west4", fetch=fetch)
        for model in (provider("mistral-medium-3"), provider.native.model_garden().language_model("mistralai/mistral-medium-3")):
            messages = [ModelMessage(role="user", parts=[TextPart(text="Look up this color"), ImagePart(image="data:image/png;base64,c3ludGhldGlj")])]
            result = await model.generate(ModelGenerateInput(messages=messages, tools={"lookup": tool(name="lookup", schema=Answer, execute=lambda x: x)}, tool_choice="auto"))
            call = next(p.tool_call for p in result.messages[0].parts if isinstance(p, ToolCallPart))
            self.assertEqual(call.input, {"value": "red"})
            self.assertEqual(seen[-1][1]["messages"][0]["content"][1]["image_url"]["url"], messages[0].parts[1].image)
            self.assertEqual(seen[-1][1]["tools"][0]["function"]["name"], "lookup")
            self.assertEqual(seen[-1][1]["tool_choice"], "auto")
            result = await model.generate(ModelGenerateInput(messages=[*messages, *result.messages, ModelMessage(role="tool", parts=[ToolResultPart(tool_result=ToolExecutionResult(tool_call_id=call.id, tool_name=call.name, output={"found": True}))])]))
            self.assertEqual(result.text, "verified")
            self.assertEqual(seen[-1][1]["messages"][-1]["tool_call_id"], "lookup123")
            self.assertEqual(json.loads(seen[-1][1]["messages"][-1]["content"]), {"found": True})
            self.assertTrue(seen[-1][0].endswith("/publishers/mistralai/models/mistral-medium-3:rawPredict"))
            self.assertFalse(model.capabilities.structured_output)

    async def test_mistral_portable_and_native_normalize_regional_raw_predict(self):
        seen = []
        closed = []

        class Response(FakeResponse):
            async def iter_lines(self):
                try:
                    for data in ({"choices": [{"delta": {"content": "O"}}]},
                                 {"choices": [{"delta": {"content": "K"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
                                 "[DONE]"):
                        yield "data: " + (data if isinstance(data, str) else json.dumps(data))
                        yield ""
                finally:
                    closed.append(True)

        async def fetch(url, **kwargs):
            seen.append((url, kwargs["json_body"]))
            return Response(200, {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

        provider = create_vertex(access_token="synthetic", project_id="p", location="us-central1", fetch=fetch)
        for name in ("mistral-medium-3", "mistral-small-2503", "codestral-2"):
            for model in (provider(name), provider("mistralai/" + name), provider.native.model_garden().language_model("mistralai/" + name)):
                self.assertEqual(model.provider, "vertex")
                self.assertEqual(model.capabilities.tools, name == "mistral-medium-3")
                self.assertEqual(model.capabilities.vision, name != "codestral-2")
                result = await generate_text(model=model, prompt="Synthetic")
                self.assertEqual(result.text, "OK")
                self.assertTrue(seen[-1][0].endswith(f"/publishers/mistralai/models/{name}:rawPredict"))
                result = await stream_text(model=model, prompt="Synthetic").collect()
                self.assertEqual(result.text, "OK")
                self.assertEqual(result.usage.output_tokens, 2)
                self.assertTrue(seen[-1][0].endswith(f"/publishers/mistralai/models/{name}:streamRawPredict"))
                self.assertEqual(seen[-1][1]["model"], name)
                self.assertNotIn("stream_options", seen[-1][1])
                if name == "mistral-small-2503":
                    await generate_text(model=model, messages=[ModelMessage(role="user", parts=[ImagePart(image="https://example.com/synthetic.png")])])
                    self.assertEqual(seen[-1][1]["messages"][0]["content"][0]["image_url"]["url"], "https://example.com/synthetic.png")
            with self.assertRaises(ConfigurationError):
                create_vertex(api_key="synthetic", express_mode=True)(name)
        self.assertEqual(len(closed), 9)

    async def test_native_thinking_toggle_preserves_boolean_and_other_template_options(self):
        seen = []
        async def fetch(url, **kwargs):
            seen.append(kwargs["json_body"])
            return FakeResponse(200, {"choices": [{"message": {"content": "323", "reasoning_content": "synthetic"}, "finish_reason": "stop"}]})
        garden = create_vertex(access_token="t", project_id="p", fetch=fetch).native.model_garden()
        for model_id in ("google/gemma-4-26b-a4b-it-maas", "zai-org/glm-5.2-maas"):
            for enabled in (False, True):
                options = {"chat_template_kwargs": {"enable_thinking": enabled, "other_option": "preserve"}}
                result = await garden.language_model(model_id).generate(ModelGenerateInput(messages=[ModelMessage(role="user", parts=[TextPart(text="compute")])], provider_options=options))
                self.assertIs(seen[-1]["chat_template_kwargs"]["enable_thinking"], enabled)
                self.assertEqual(seen[-1]["chat_template_kwargs"]["other_option"], "preserve")
                self.assertIsNot(seen[-1]["chat_template_kwargs"], options["chat_template_kwargs"])
                self.assertEqual(result.text, "323")
                self.assertEqual(result.messages[0].parts[1].data["reasoning_content"], "synthetic")

    async def test_tool_maas_json_and_function_contracts_on_both_factories(self):
        seen = []

        async def fetch(url, **kwargs):
            body = kwargs["json_body"]
            seen.append(body)
            if body.get("tools"):
                message = {"tool_calls": [{"id": "lookup-1", "function": {"name": "lookup", "arguments": '{"value":"ok"}'}}]}
                reason = "tool_calls"
            else:
                message = {"content": '{"value":"ok"}'}
                reason = "stop"
            return FakeResponse(200, {"choices": [{"message": message, "finish_reason": reason}]})

        provider = create_vertex(access_token="t", project_id="p", fetch=fetch)
        for model in (provider("glm-5.2-maas"), provider.native.model_garden().language_model("zai-org/glm-5.2-maas"), provider("gpt-oss-120b-maas"), provider.native.model_garden().language_model("openai/gpt-oss-120b-maas")):
            self.assertFalse(model.capabilities.vision)
            self.assertFalse(model.capabilities.reasoning)
            result = await generate_object(model=model, prompt="JSON", schema=Answer)
            self.assertEqual(result.object.value, "ok")
            self.assertEqual(seen[-1]["response_format"]["type"], "json_schema")
            result = await generate_text(model=model, prompt="lookup", tools={"lookup": tool(name="lookup", schema=Answer, execute=lambda x: x)}, tool_choice="auto")
            call = next(part.tool_call for message in result.steps[0].response.messages for part in message.parts if isinstance(part, ToolCallPart))
            self.assertEqual(call.input, {"value": "ok"})
            self.assertEqual(call.id, "lookup-1")
            self.assertEqual(seen[-1]["model"], model.model_id)
            self.assertEqual(seen[-1]["tools"][0]["function"]["name"], "lookup")
            self.assertEqual(seen[-1]["tool_choice"], "auto")

    async def test_known_maas_ids_route_to_vertex_chat_with_host_identity(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append((url, kwargs["json_body"]))
            return FakeResponse(200, {"choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}]})

        p = create_vertex(access_token="t", project_id="p", location="global", fetch=fetch)
        for full_id in (
            "openai/gpt-oss-120b-maas",
            "deepseek-ai/deepseek-v3.2-maas", "moonshotai/kimi-k2-thinking-maas",
            "zai-org/glm-5-maas", "zai-org/glm-5.2-maas", "qwen/qwen3-next-80b-a3b-instruct-maas",
            "minimaxai/minimax-m2-maas",
        ):
            from zhivex_ai import default_model_catalog

            entry = default_model_catalog.find("vertex", full_id.split("/", 1)[1])
            self.assertEqual(entry.model_id, full_id)
            self.assertEqual(entry.availability, "stable" if full_id == "openai/gpt-oss-120b-maas" else "preview" if full_id == "zai-org/glm-5.2-maas" else "deprecated")
            for identifier in (full_id, full_id.split("/", 1)[1]):
                model = p(identifier)
                self.assertEqual(model.provider, "vertex")
                self.assertEqual(model.capabilities.tools, full_id in {"zai-org/glm-5.2-maas", "openai/gpt-oss-120b-maas"})
                await generate_text(model=model, prompt="OK")
                self.assertTrue(seen[-1][0].endswith("/endpoints/openapi/chat/completions"))
                self.assertEqual(seen[-1][1]["model"], full_id)
        with self.assertRaises(ConfigurationError):
            create_vertex(api_key="k", express_mode=True)("deepseek-v3.2-maas")

    async def test_stream_cancellation_closes_transport_iterator(self):
        closed = []

        class Response(FakeResponse):
            async def iter_lines(self):
                try:
                    yield 'data: {"choices":[{"delta":{"content":"partial"}}]}'
                    yield ""
                finally:
                    closed.append(True)

        async def fetch(url, **kwargs):
            return Response(200)

        model = create_vertex(
            access_token="t", project_id="p", fetch=fetch
        ).native.language_model("google/gemma-4-26b-a4b-it-maas")
        events = await model.stream(
            ModelGenerateInput(
                messages=[ModelMessage(role="user", parts=[TextPart(text="synthetic")])]
            )
        )
        await anext(events)
        await events.aclose()
        self.assertEqual(closed, [True])

    async def test_gemma_routes_and_native_json_schema(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {"content": '{"value":"ok"}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 3,
                        "total_tokens": 5,
                    },
                },
            )

        provider = create_vertex(
            access_token="token", project_id="p", location="global", fetch=fetch
        )
        result = await generate_object(
            model=provider("gemma-4-26b-a4b-it-maas"), prompt="synthetic", schema=Answer
        )
        self.assertEqual(result.object.value, "ok")
        self.assertEqual(
            seen[0][0],
            "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi/chat/completions",
        )
        self.assertEqual(seen[0][1]["headers"]["authorization"], "Bearer token")
        body = seen[0][1]["json_body"]
        self.assertEqual(body["model"], "google/gemma-4-26b-a4b-it-maas")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertNotIn("contents", body)
        with self.assertRaises(ConfigurationError):
            create_vertex(api_key="test", express_mode=True)("gemma-4-26b-a4b-it-maas")

    async def test_client_function_call_roundtrip_and_provider_identity(self):
        seen = []
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"value":"x"}',
                                },
                            }
                        ],
                        "reasoning_content": "synthetic",
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        async def fetch(url, **kwargs):
            seen.append(kwargs["json_body"])
            return FakeResponse(200, response)

        model = create_vertex(
            access_token="token", project_id="p", fetch=fetch
        ).native.language_model("google/gemma-4-26b-a4b-it-maas")
        result = await model.generate(
            ModelGenerateInput(
                messages=[
                    ModelMessage(role="user", parts=[TextPart(text="synthetic")])
                ],
                tools={
                    "lookup": tool(name="lookup", schema=Answer, execute=lambda x: x)
                },
            )
        )
        call = next(
            p.tool_call for p in result.messages[0].parts if isinstance(p, ToolCallPart)
        )
        self.assertEqual(call.provider_metadata["provider"], "vertex")
        self.assertEqual(call.input, {"value": "x"})
        response = {
            "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]
        }
        await model.generate(
            ModelGenerateInput(
                messages=[
                    *result.messages,
                    ModelMessage(
                        role="tool",
                        parts=[
                            ToolResultPart(
                                tool_result=ToolExecutionResult(
                                    tool_call_id="c1",
                                    tool_name="lookup",
                                    output={"found": True},
                                )
                            )
                        ],
                    ),
                ]
            )
        )
        self.assertEqual(seen[-1]["messages"][0]["reasoning_content"], "synthetic")
        self.assertEqual(seen[-1]["messages"][1]["tool_call_id"], "c1")
        self.assertEqual(
            json.loads(seen[-1]["messages"][1]["content"]), {"found": True}
        )

    async def test_stream_accumulates_fragmented_tools_and_final_usage(self):
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "content": "OK",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"v":',
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": "1}"}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            },
        ]

        async def fetch(url, **kwargs):
            self.assertTrue(kwargs["stream"])
            return FakeResponse(
                200,
                body_text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                + "data: [DONE]\n\n",
            )

        model = create_vertex(
            access_token="token", project_id="p", fetch=fetch
        ).native.language_model("google/gemma-4-26b-a4b-it-maas")
        events = [
            event
            async for event in await model.stream(
                ModelGenerateInput(
                    messages=[
                        ModelMessage(role="user", parts=[TextPart(text="synthetic")])
                    ]
                )
            )
        ]
        self.assertEqual(events[0].text_delta, "OK")
        self.assertEqual(events[-2].tool_call.input, {"v": 1})
        self.assertEqual(events[-1].usage.total_tokens, 5)
        self.assertEqual(events[-1].provider_finish_reason, "tool_calls")

    async def test_unknown_deployment_does_not_inherit_gemini_capabilities(self):
        seen = []

        async def fetch(url, **kwargs):
            seen.append(url)
            return FakeResponse(
                200,
                {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]},
            )

        provider = create_vertex(access_token="token", project_id="p", fetch=fetch)
        model = provider.native.model_garden().language_model(
            "deployment-model", endpoint="123"
        )
        self.assertFalse(model.capabilities.tools)
        self.assertFalse(model.capabilities.vision)
        self.assertEqual(
            (await generate_text(model=model, prompt="synthetic")).text, "OK"
        )
        self.assertTrue(seen[0].endswith("/endpoints/123/chat/completions"))

    async def test_truncated_stream_is_not_success(self):
        async def fetch(url, **kwargs):
            return FakeResponse(
                200, body_text='data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            )

        model = create_vertex(access_token="token", project_id="p", fetch=fetch)(
            "google/gemma-4-26b-a4b-it-maas"
        )
        with self.assertRaises(ValidationError):
            await stream_text(model=model, prompt="synthetic").collect()
