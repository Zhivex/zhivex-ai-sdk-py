from __future__ import annotations

import base64
from urllib.parse import parse_qs, urlparse
from unittest import IsolatedAsyncioTestCase

from zhivex_ai import create_gemini, create_vertex, create_openai, generate_speech
from zhivex_ai._http import BufferedResponse
from zhivex_ai.errors import ParseError, ProviderHTTPError, ValidationError
from zhivex_ai.experimental.gemini import GeminiVoicesClient
from zhivex_ai.types import RetryOptions
import json


class GeminiVoicesTests(IsolatedAsyncioTestCase):
    async def test_design_and_stateless_replication_use_documented_payload_without_mutation(self):
        calls = []
        async def fetch(url, **kwargs):
            calls.append((url, kwargs))
            result = {"key": "voicekey_test", "expire_time": "later"} if kwargs["json_body"].get("store") is False else {"id": "voice_test", "sample_audio": {"data": "YQ=="}, "usage": {"total_tokens": 4}}
            return BufferedResponse(200, json.dumps(result).encode())
        provider = create_gemini(api_key="test-key", fetch=fetch)
        client = provider.native.voices()
        self.assertIsInstance(client, GeminiVoicesClient)
        self.assertIs(client, provider.native.voices())
        body = {"voice": {"type": "prompted", "prompted": {"input": "Warm narrator"}}}
        voice = await client.create(body, RetryOptions(timeout_ms=42))
        self.assertNotIn("store", body)
        self.assertEqual(voice["usage"], {"total_tokens": 4})
        self.assertTrue(calls[0][1]["json_body"]["store"])
        self.assertEqual(calls[0][1]["timeout_ms"], 42)
        self.assertNotIn("test-key", calls[0][0])
        self.assertEqual(calls[0][1]["headers"]["x-goog-api-key"], "test-key")
        await client.design(prompt="Friendly", model="gemini-3.8-flash-tts", display_name="Narrator")
        self.assertEqual(calls[1][1]["json_body"]["voice"]["prompted"], {"input": "Friendly"})
        result = await client.replicate(source_audio=b"source", consent_audio=b"consent", store=False)
        self.assertEqual(result["key"], "voicekey_test")
        self.assertEqual(result["expire_time"], "later")
        audio = calls[-1][1]["json_body"]["voice"]["replicated"]
        self.assertEqual(base64.b64decode(audio["consent_audio"]["data"]), b"consent")

    async def test_pagination_preserves_filters_and_encodes_tokens(self):
        calls = []
        async def fetch(url, **kwargs):
            calls.append((url, kwargs))
            page = {"voices": [{"id": "voice_a"}], "next_page_token": "a+b/&"} if len(calls) == 1 else {"voices": [{"id": "Puck"}]}
            return BufferedResponse(200, json.dumps(page).encode())
        client = create_gemini(api_key="test", fetch=fetch).native.voices()
        params = {"language_code": ["en-US", "es-ES"], "type": ["prebuilt", "prompted"], "search": "Warm & deep", "page_size": 1}
        results = [v async for v in client.iter_voices(params)]
        self.assertEqual([v["id"] for v in results], ["voice_a", "Puck"])
        self.assertNotIn("page_token", params)
        query = parse_qs(urlparse(calls[1][0]).query)
        self.assertEqual(query["page_token"], ["a+b/&"])
        self.assertEqual(query["language_code"], params["language_code"])
        self.assertEqual(query["type"], params["type"])
        self.assertEqual(calls[1][1]["method"], "GET")
        self.assertIsNone(calls[1][1]["json_body"])

    async def test_get_delete_only_stored_resources_and_empty_delete_response(self):
        calls = []
        async def fetch(url, **kwargs):
            calls.append((url, kwargs))
            return BufferedResponse(204, b"") if kwargs["method"] == "DELETE" else BufferedResponse(200, b'{"id":"voice_a"}')
        client = create_gemini(api_key="test", fetch=fetch).native.voices()
        self.assertEqual((await client.get("voices/voice_a"))["id"], "voice_a")
        await client.delete("voice_a")
        self.assertTrue(calls[-1][0].endswith("/voices/voice_a"))
        self.assertEqual(calls[-1][1]["method"], "DELETE")
        for resource in ("Puck", "voicekey_secret", "voice_a/../voice_b", "voice_a?key=bad", "https://host/voice_a"):
            with self.subTest(resource=resource), self.assertRaises(ValidationError):
                await client.get(resource)
        self.assertEqual(len(calls), 2)
        with self.assertRaises(AttributeError):
            create_vertex(access_token="test", project_id="test", fetch=fetch).native.voices()

    async def test_validation_happens_before_fetch(self):
        async def fetch(url, **kwargs):
            self.fail("Invalid input reached transport")
        client = create_gemini(api_key="test", fetch=fetch).native.voices()
        invalid = [{}, {"voice": {"type": "prebuilt"}}, {"store": False, "voice": {"type": "prompted", "prompted": {"input": "x"}}},
                   {"voice": {"type": "prompted", "prompted": {"input": " "}}},
                   {"voice": {"type": "replicated", "replicated": {"source_audio": {"mime_type": "audio/wav", "data": "YQ=="}}}},
                   {"voice": {"type": "replicated", "replicated": {"source_audio": {"mime_type": "audio/wav", "data": "!!!"}, "consent_audio": {"mime_type": "audio/wav", "data": "YQ=="}}}}]
        for body in invalid:
            with self.subTest(body=body), self.assertRaises(ValidationError):
                await client.create(body)
        for params in ({"page_size": 0}, {"page_size": True}, {"search": "ñ" * 1025}, {"unknown": True}):
            with self.subTest(params=params), self.assertRaises(ValidationError):
                await client.list(params)
        with self.assertRaises(ValidationError):
            await client.replicate(source_audio=b"source", consent_audio=b"")

    async def test_http_errors_and_repeated_tokens_fail_explicitly(self):
        async def failed(url, **kwargs):
            return BufferedResponse(403, b'{"error":"permission"}')
        client = create_gemini(api_key="test", fetch=failed).native.voices()
        with self.assertRaises(ProviderHTTPError) as raised:
            await client.list()
        self.assertEqual(raised.exception.status, 403)
        async def repeated(url, **kwargs):
            return BufferedResponse(200, b'{"voices":[],"next_page_token":"repeat"}')
        client = create_gemini(api_key="test", fetch=repeated).native.voices()
        with self.assertRaises(ParseError):
            [voice async for voice in client.iter_voices()]

    async def test_gemini_38_speech_uses_unified_voice_but_legacy_and_vertex_keep_prebuilt(self):
        calls = []
        async def fetch(url, **kwargs):
            calls.append(kwargs["json_body"])
            return BufferedResponse(200, b'{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"audio/wav","data":"YQ=="}}]}}]}')
        gemini = create_gemini(api_key="test", fetch=fetch)
        for model_id in ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"):
            result = await generate_speech(model=gemini.speech_model(model_id), input="Hello", voice="voice_test")
            self.assertEqual(result.audio, b"a")
            self.assertEqual(calls[-1]["generationConfig"]["speechConfig"]["voiceConfig"], {"voice": "voice_test"})
        await generate_speech(model=gemini.speech_model("gemini-2.5-flash-preview-tts"), input="Hello", voice="Kore")
        self.assertEqual(calls[-1]["generationConfig"]["speechConfig"]["voiceConfig"], {"prebuiltVoiceConfig": {"voiceName": "Kore"}})
        vertex = create_vertex(access_token="test", project_id="test", fetch=fetch)
        await generate_speech(model=vertex.speech_model("gemini-3.8-flash-tts"), input="Hello", voice="Kore")
        self.assertEqual(calls[-1]["generationConfig"]["speechConfig"]["voiceConfig"], {"prebuiltVoiceConfig": {"voiceName": "Kore"}})

    async def test_secret_repr_unknown_provider_and_malformed_responses(self):
        async def fetch(url, **kwargs):
            return BufferedResponse(200, b'{}')
        client = create_gemini(api_key="private-key", fetch=fetch).native.voices()
        self.assertNotIn("private-key", repr(client))
        with self.assertRaises(AttributeError):
            create_openai(api_key="test", fetch=fetch).native.voices()
        for payload in (b'[]', b'null', b'bad json', b'{"voices":{}}', b'{"voices":["x"]}', b'{"next_page_token":[]}'):
            async def malformed(url, **kwargs):
                return BufferedResponse(200, payload)
            client = create_gemini(api_key="test", fetch=malformed).native.voices()
            with self.subTest(payload=payload), self.assertRaises(ParseError):
                [voice async for voice in client.iter_voices()]
        with self.assertRaises(ValidationError):
            [voice async for voice in client.iter_voices({"page_token": []})]
