"""Latest partner IDs remain Google-owned routes with native payloads."""
from __future__ import annotations

from copy import deepcopy
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from zhivex_ai import create_vertex, default_model_catalog
from zhivex_ai._http import BufferedResponse
from zhivex_ai.errors import ProviderHTTPError, UnsupportedFeatureError
from tests.test_gemini_provider import FakeResponse

# Base regional table. cache_write is the 5-minute write, not the 1-hour write.
_VERTEX_CLAUDE_RATES = {
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125, ("global", "us", "eu"), 100_000, 0.50, 2.50),
    "claude-sonnet-5-5": (2, 10, 0.10, 2.50, ("global", "us", "eu"), None, None, None),
    "claude-opus-5": (5, 25, 0.50, 6.25, ("global", "us", "eu"), None, None, None),
    "claude-sonnet-5": (2, 10, 0.20, 2.50, ("global", "us", "eu"), None, None, None),
    "claude-fable-5-1": (10, 50, 0.25, 12.50, ("global", "us", "eu"), None, None, None),
    "claude-opus-4-6": (5, 25, 0.50, 6.25, ("global", "us-east5", "europe-west1", "asia-southeast1"), None, None, None),
    "claude-sonnet-4-6": (3, 15, 0.30, 3.75, ("global", "us-east5", "europe-west1", "asia-southeast1"), None, None, None),
}


class VertexLatestModelsTests(IsolatedAsyncioTestCase):
    async def test_claude_55_native_messages_in_documented_locations(self):
        payload = {"id": "synthetic", "content": [{"type": "thinking", "thinking": "reason", "signature": "signed"},
                                                  {"type": "text", "text": "answer"}], "usage": {"input_tokens": 12}}
        for model in ("claude-sonnet-5-5", "claude-opus-5-5"):
            for location, host in (("global", "aiplatform.googleapis.com"), ("us", "aiplatform.us.rep.googleapis.com"),
                                   ("eu", "aiplatform.eu.rep.googleapis.com")):
                with self.subTest(model=model, location=location):
                    fetch = AsyncMock(return_value=FakeResponse(200, payload))
                    body = {"model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32,
                            "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"},
                            "anthropic_version": "caller-version"}
                    original = deepcopy(body)
                    result = await create_vertex(access_token="synthetic", project_id="project", location=location,
                                                 fetch=fetch).native.model_garden().anthropic_messages(model=model, body=body)
                    self.assertEqual(result, payload)
                    self.assertEqual(body, original)
                    self.assertEqual(fetch.call_args.args[0], f"https://{host}/v1/projects/project/locations/{location}/publishers/anthropic/models/{model}:rawPredict")
                    sent = fetch.call_args.kwargs["json_body"]
                    self.assertNotIn("model", sent)
                    self.assertEqual(sent["anthropic_version"], "vertex-2023-10-16")
                    self.assertEqual(sent["thinking"], {"type": "adaptive"})
                    self.assertEqual(sent["output_config"], {"effort": "high"})
                    self.assertEqual(fetch.call_args.kwargs["headers"]["authorization"], "Bearer synthetic")

    async def test_claude_55_streaming_keeps_native_stream_for_caller(self):
        for model in ("claude-sonnet-5-5", "claude-opus-5-5"):
            with self.subTest(model=model):
                response = BufferedResponse(200, b'data: {"type":"message_stop"}\n\n', {"content-type": "text/event-stream"})
                fetch = AsyncMock(return_value=response)
                body = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 32,
                        "thinking": {"type": "adaptive"}, "stream": True}
                result = await create_vertex(access_token="synthetic", project_id="project", location="global",
                                             fetch=fetch).native.model_garden().anthropic_messages(model=model, body=body)
                self.assertIs(result, response)
                self.assertTrue(fetch.call_args.args[0].endswith(f"/{model}:streamRawPredict"))
                self.assertTrue(fetch.call_args.kwargs["stream"])

    async def test_vertex_claude_native_messages_keep_thinking_and_cache(self):
        payload = {"id": "synthetic", "content": [{"type": "text", "text": "answer"}]}
        for model in _VERTEX_CLAUDE_RATES:
            thinking = {"type": "between_tools"} if model == "claude-sonnet-5-5" else {"type": "adaptive"}
            with self.subTest(model=model):
                fetch = AsyncMock(return_value=FakeResponse(200, payload))
                body = {
                    "messages": [{
                        "role": "user",
                        "content": [{
                            "type": "text", "text": "hello",
                            "cache_control": {"type": "ephemeral", "ttl": "1h"},
                        }],
                    }],
                    "max_tokens": 32,
                    "thinking": thinking,
                    "output_config": {"effort": "high"},
                    "anthropic_version": "caller-version",
                }
                original = deepcopy(body)
                await create_vertex(
                    access_token="synthetic", project_id="project", location="global", fetch=fetch,
                ).native.model_garden().anthropic_messages(model=model, body=body)
                self.assertEqual(body, original)
                self.assertTrue(fetch.call_args.args[0].endswith(
                    f"/publishers/anthropic/models/{model}:rawPredict"
                ))
                sent = fetch.call_args.kwargs["json_body"]
                self.assertEqual(sent["anthropic_version"], "vertex-2023-10-16")
                self.assertEqual(sent["thinking"], thinking)
                self.assertEqual(sent["output_config"], {"effort": "high"})
                self.assertEqual(
                    sent["messages"][0]["content"][0]["cache_control"],
                    {"type": "ephemeral", "ttl": "1h"},
                )
                provider = create_vertex(access_token="synthetic", project_id="project", fetch=fetch)
                with self.assertRaises(UnsupportedFeatureError):
                    provider.native.language_model(model)

    def test_vertex_claude_catalog_uses_base_regional_prices(self):
        for model, rates in _VERTEX_CLAUDE_RATES.items():
            with self.subTest(model=model):
                input_rate, output_rate, cache_hit, cache_write, regions, threshold, long_input, long_output = rates
                entry = default_model_catalog.find("vertex", model)
                self.assertIsNotNone(entry)
                assert entry is not None
                self.assertEqual(entry.support_evidence, "offline-contract")
                self.assertEqual(entry.regions, regions)
                self.assertIsNone(entry.cost_per_1k_tokens)
                pricing = entry.pricing
                self.assertIsNotNone(pricing)
                assert pricing is not None
                self.assertEqual(pricing.input_per_1m_tokens, input_rate)
                self.assertEqual(pricing.output_per_1m_tokens, output_rate)
                self.assertEqual(pricing.cached_input_per_1m_tokens, cache_hit)
                self.assertEqual(pricing.cache_write_per_1m_tokens, cache_write)
                self.assertEqual(pricing.long_context_threshold_tokens, threshold)
                self.assertEqual(pricing.long_context_input_per_1m_tokens, long_input)
                self.assertEqual(pricing.long_context_output_per_1m_tokens, long_output)
                self.assertTrue(entry.capabilities is not None and entry.capabilities.reasoning)
                self.assertTrue(entry.capabilities.web_search)

    async def test_claude_partner_permission_errors_remain_explicit(self):
        fetch = AsyncMock(return_value=BufferedResponse(403, b'{"error":{"message":"permission denied"}}'))
        with self.assertRaises(ProviderHTTPError) as caught:
            await create_vertex(access_token="synthetic", project_id="project", location="global", fetch=fetch).native.model_garden().anthropic_messages(
                model="claude-sonnet-5-5", body={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 32})
        self.assertEqual(caught.exception.status, 403)
        self.assertEqual(fetch.await_count, 1)

    async def test_vertex_existing_tts_retains_prebuilt_voice_contract(self):
        payload = {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "audio/pcm", "data": "YQ=="}}]}}]}
        fetch = AsyncMock(return_value=FakeResponse(200, payload))
        model = create_vertex(access_token="synthetic", project_id="project", location="global", fetch=fetch).native.speech_model("gemini-3.1-flash-tts-preview")
        output = await model.generate_speech(input="Hello", voice="Kore")
        self.assertEqual(output.audio, b"a")
        self.assertTrue(fetch.call_args.args[0].endswith("/publishers/google/models/gemini-3.1-flash-tts-preview:generateContent"))
        self.assertEqual(fetch.call_args.kwargs["json_body"]["generationConfig"]["speechConfig"],
                         {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}})
