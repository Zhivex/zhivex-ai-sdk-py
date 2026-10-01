from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
from unittest import IsolatedAsyncioTestCase, TestCase

from pydantic import BaseModel, ConfigDict

from zhivex_ai import (
    ReasoningConfig, UnsupportedFeatureError,
    create_openai, generate_text, stream_text, tool, user,
)
from zhivex_ai.catalog import ModelPricing, default_model_catalog
from zhivex_ai.errors import ValidationError, ZhivexAIError
from zhivex_ai.types import ModelGenerateInput
from tests.test_openai_provider import FakeResponse


def message(text, agent="/root", phase="final_answer"):
    result = {"type": "message", "id": text, "role": "assistant",
              "agent": {"agent_name": agent}, "content": [{"type": "output_text", "text": text}]}
    if phase is not None:
        result["phase"] = phase
    return result


def sse(events):
    return FakeResponse(200, body_text="".join("data: " + json.dumps(e) + "\n\n" for e in events))


class Lookup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class OpenAILatestModelTests(IsolatedAsyncioTestCase):
    async def test_each_new_model_generates_and_streams(self):
        for model_id in ("gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"):
            requests = []
            async def fetch(url, **kwargs):
                requests.append((url, kwargs["json_body"]))
                if kwargs.get("stream"):
                    return sse([{"type": "response.output_text.delta", "delta": "ok"},
                                {"type": "response.completed", "response": {"status": "completed"}}])
                return FakeResponse(200, payload={"status": "completed", "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]})
            with self.subTest(model=model_id):
                model = create_openai(api_key="test", fetch=fetch)(model_id)
                self.assertEqual((await generate_text(model=model, prompt="test")).text, "ok")
                async with stream_text(model=model, prompt="test") as result:
                    self.assertEqual((await result.collect()).text, "ok")
                self.assertTrue(all(url.endswith("/responses") and body["model"] == model_id for url, body in requests))

    async def test_model_guards_apply_before_fetch_in_all_request_paths(self):
        async def fetch(*args, **kwargs):
            self.fail("invalid request reached network")
        provider = create_openai(api_key="test", fetch=fetch)
        cases = [("gpt-6.1-sol", "none"), ("gpt-6.1-sol-2026-09-29", "minimal"),
                 ("gpt-6-sol", "minimal"), ("gpt-6-luna", "minimal")]
        for model_id, effort in cases:
            with self.subTest(model=model_id, effort=effort):
                input = ModelGenerateInput(messages=[], reasoning=ReasoningConfig(effort=effort))
                with self.assertRaises(UnsupportedFeatureError):
                    await provider.native.language_model(model_id).generate(input)
                with self.assertRaises(UnsupportedFeatureError):
                    await provider.native.language_model(model_id).stream(input)
                with self.assertRaises(UnsupportedFeatureError):
                    await provider.native.responses().create({"model": model_id, "input": "x", "reasoning": {"effort": effort}})

    async def test_sampling_allowed_only_with_supported_none_effort(self):
        requests = []
        async def fetch(url, **kwargs):
            requests.append(kwargs["json_body"])
            return FakeResponse(200, payload={"output": []})
        provider = create_openai(api_key="test", fetch=fetch)
        for model_id in ("gpt-6-sol", "gpt-6-luna"):
            await generate_text(model=provider(model_id), prompt="x", temperature=0.2, reasoning=ReasoningConfig(effort="none"))
            with self.assertRaises(UnsupportedFeatureError):
                await generate_text(model=provider(model_id), prompt="x", temperature=0.2)
        for options in ({"reasoning": {"effort": "none"}}, {"top_p": 0.8}, {"prompt_cache_retention": "24h"}):
            with self.assertRaises(UnsupportedFeatureError):
                await generate_text(model=provider.native.language_model("gpt-6.1-sol"), prompt="x", provider_options=options)
        self.assertEqual(len(requests), 2)

    async def test_multiagent_final_only_and_lossless_replay(self):
        requests = []
        items = [message("worker", "/root/worker"), message("working", phase="commentary"),
                 message("ambiguous", phase=None),
                 {"type": "message", "id": "missing-context", "role": "assistant", "content": [{"type": "output_text", "text": "unattributed"}]},
                 {"type": "message", "id": "missing-agent", "role": "assistant", "phase": "final_answer", "content": [{"type": "output_text", "text": "unknown agent"}]},
                 {"type": "multi_agent_call", "id": "spawn", "action": "spawn_agent", "agent": {"agent_name": "/root"}},
                 {"type": "multi_agent_call_output", "id": "spawn-out", "output": [], "agent": {"agent_name": "/root"}},
                 {"type": "agent_message", "agent": {"agent_name": "/root"}, "content": [{"type": "encrypted_content", "encrypted_content": "opaque"}]},
                 message("final")]
        async def fetch(url, **kwargs):
            requests.append(deepcopy(kwargs))
            return FakeResponse(200, payload={"status": "completed", "output": deepcopy(items)})
        model = create_openai(api_key="test", fetch=fetch).native.language_model("gpt-6.1-sol")
        options = {"multi_agent": {"enabled": True}, "store": False}
        result = await generate_text(model=model, prompt="work", provider_options=options)
        self.assertEqual(result.text, "final")
        self.assertFalse(any(p.type == "tool-call" for m in result.messages for p in m.parts))
        await generate_text(model=model, messages=[*result.messages, user("continue")], provider_options=options)
        self.assertEqual(requests[0]["headers"]["OpenAI-Beta"], "responses_multi_agent=v1")
        self.assertEqual(requests[1]["json_body"]["input"][1:-1], items)
        self.assertEqual(options, {"multi_agent": {"enabled": True}, "store": False})

    async def test_stream_multiagent_filters_text_executes_functions_once_and_replays(self):
        requests, executed = [], []
        call = {"type": "function_call", "id": "fc", "call_id": "lookup-1", "name": "lookup",
                "arguments": '{"name":"project"}', "agent": {"agent_name": "/root/worker"}}
        root = message("final")
        async def fetch(url, **kwargs):
            requests.append(deepcopy(kwargs))
            if not kwargs.get("stream"):
                return FakeResponse(200, payload={"status": "completed", "output": [root]})
            if len(requests) == 1:
                return sse([
                    {"type": "response.output_item.added", "output_index": 0, "item": message("", "/root/worker")},
                    {"type": "response.output_text.delta", "output_index": 0, "delta": "private worker"},
                    {"type": "response.output_item.done", "output_index": 0, "item": message("private worker", "/root/worker")},
                    {"type": "response.output_item.added", "output_index": 1, "item": call},
                    {"type": "response.output_item.done", "output_index": 1, "item": call},
                    {"type": "response.completed", "response": {"status": "completed"}},
                ])
            return sse([
                {"type": "response.output_item.added", "output_index": 0, "item": message("working", phase="commentary")},
                {"type": "response.output_text.delta", "output_index": 0, "delta": "working"},
                {"type": "response.output_item.done", "output_index": 0, "item": message("working", phase="commentary")},
                {"type": "response.output_item.added", "output_index": 1, "item": root},
                {"type": "response.output_text.delta", "output_index": 1, "delta": "final"},
                {"type": "response.output_item.done", "output_index": 1, "item": root},
                {"type": "response.completed", "response": {"status": "completed"}},
            ])
        def execute(input):
            executed.append(input.name)
            return {"status": "ok"}
        model = create_openai(api_key="test", fetch=fetch).native.language_model("gpt-6.1-sol")
        options = {"multi_agent": {"enabled": True}, "store": False}
        async with stream_text(model=model, prompt="work", provider_options=options,
                               tools={"lookup": tool(name="lookup", schema=Lookup, execute=execute)}, max_steps=3) as stream:
            result = await stream.collect()
        self.assertEqual(result.text, "final")
        self.assertEqual(executed, ["project"])
        self.assertEqual(sum(i.get("type") == "function_call" for i in requests[1]["json_body"]["input"]), 1)
        self.assertEqual(sum(i.get("type") == "function_call_output" for i in requests[1]["json_body"]["input"]), 1)
        # Replaying a collected stream must not duplicate final text as an ordinary assistant message.
        await model.generate(ModelGenerateInput(messages=result.messages, provider_options=options))
        replayed = requests[-1]["json_body"]["input"]
        self.assertEqual(sum(i.get("type") == "message" and i.get("id") == "final" for i in replayed), 1)
        self.assertFalse(any(i.get("type") == "message" and i.get("role") == "assistant" and "id" not in i for i in replayed))

    async def test_mixed_attributed_and_ordinary_history_preserves_both(self):
        from zhivex_ai.providers._openai_responses_normalization import _parse_responses_message
        requests = []
        async def fetch(url, **kwargs):
            requests.append(kwargs["json_body"])
            return FakeResponse(200, payload={"output": []})
        history = _parse_responses_message({"output": [message("progress", phase="commentary"),
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ordinary final"}]}]}, "openai")
        await create_openai(api_key="test", fetch=fetch).native.language_model("gpt-6.1-sol").generate(ModelGenerateInput(messages=[history]))
        self.assertEqual(requests[0]["input"][0]["content"][0]["text"], "ordinary final")
        self.assertEqual(requests[0]["input"][1], message("progress", phase="commentary"))

    async def test_multiagent_unsupported_options_fail_before_fetch(self):
        async def fetch(*args, **kwargs):
            self.fail("invalid multi-agent request reached network")
        client = create_openai(api_key="test", fetch=fetch).native.responses()
        for extra in ({"reasoning": {"summary": "auto"}}, {"max_tool_calls": 1},
                      {"multi_agent": {"enabled": True, "max_concurrent_subagents": False}},
                      {"multi_agent": []}):
            with self.subTest(extra=extra), self.assertRaises((ValidationError, UnsupportedFeatureError)):
                await client.create({"model": "gpt-6.1-sol", "multi_agent": {"enabled": True}, **extra})
        with self.assertRaises(UnsupportedFeatureError):
            await client.create({"model": "gpt-6-sol", "multi_agent": {"enabled": True}})
        with self.assertRaises(UnsupportedFeatureError):
            await client.compact({"model": "gpt-6.1-sol", "multi_agent": {"enabled": True}})

    async def test_stream_without_attribution_fails_closed(self):
        async def fetch(url, **kwargs):
            return sse([{"type": "response.output_text.delta", "delta": "unknown"}])
        model = create_openai(api_key="test", fetch=fetch).native.language_model("gpt-6.1-sol")
        iterator = await model.stream(ModelGenerateInput(provider_options={"multi_agent": {"enabled": True}}))
        with self.assertRaises(ZhivexAIError):
            async for _ in iterator:
                pass


class LatestCatalogTests(TestCase):
    def test_new_models_are_distinct_and_source_backed(self):
        cases = {"openai": ("gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"),
                 "anthropic": ("claude-opus-5-5", "claude-sonnet-5-5"),
                 "gemini": ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"),
                 "qwen": ("qwen3.7-text-rerank",)}
        for provider, ids in cases.items():
            for model_id in ids:
                with self.subTest(provider=provider, model=model_id):
                    entry = default_model_catalog.find(provider, model_id)
                    self.assertIsNotNone(entry)
                    self.assertEqual(entry.verified_at, "2026-09-30")
                    self.assertEqual(entry.support_evidence, "offline-contract")
                    self.assertTrue(entry.source_urls)
        rerank = default_model_catalog.find("qwen", "qwen3.7-text-rerank")
        self.assertEqual(rerank.regions, ("cn",))
        self.assertFalse(rerank.capabilities.embeddings)

    def test_long_context_cost_is_conservative_without_erasing_base_prices(self):
        entry = default_model_catalog.find("openai", "gpt-6.1-sol")
        pricing = entry.pricing
        self.assertEqual(pricing.input_per_1m_tokens, 2)
        self.assertEqual(pricing.output_per_1m_tokens, 10)
        self.assertEqual(pricing.long_context_threshold_tokens, 272_000)
        self.assertEqual(pricing.conservative_cost_per_1k_tokens(as_of=date(2026, 9, 30)), 0.015)
        self.assertIsNone(pricing.conservative_cost_per_1k_tokens(as_of=date(2026, 9, 28)))
        for kwargs in ({"long_context_threshold_tokens": True},
                       {"long_context_threshold_tokens": 0, "long_context_input_per_1m_tokens": 4},
                       {"long_context_input_per_1m_tokens": 4},
                       {"long_context_threshold_tokens": 10, "long_context_output_per_1m_tokens": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                ModelPricing(currency="USD", source_url="https://example.com", input_per_1m_tokens=2, **kwargs)
