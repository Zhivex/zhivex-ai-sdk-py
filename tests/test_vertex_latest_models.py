"""Latest partner IDs remain Google-owned routes with native payloads."""
from __future__ import annotations

from copy import deepcopy
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from zhivex_ai import create_vertex
from zhivex_ai._http import BufferedResponse
from zhivex_ai.errors import ProviderHTTPError
from tests.test_gemini_provider import FakeResponse


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
