from copy import deepcopy
from unittest import IsolatedAsyncioTestCase

from zhivex_ai import create_vertex
from zhivex_ai.errors import ProviderHTTPError, ValidationError
from zhivex_ai.types import RetryOptions
from tests.test_gemini_provider import FakeResponse


class VertexFIMTests(IsolatedAsyncioTestCase):
    async def test_fim_routes_native_body_and_preserves_stream_ownership(self):
        seen = []
        response = FakeResponse(200, {"choices": [{"message": {"content": "return 42"}}], "usage": {"completion_tokens": 3}})

        async def fetch(url, **kwargs):
            seen.append((url, deepcopy(kwargs)))
            kwargs["json_body"]["metadata"]["test"] = "changed"
            return response

        garden = create_vertex(access_token="synthetic", project_id="p", location="europe-west4", fetch=fetch).native.model_garden()
        for stream in (False, True):
            body = {"prompt": "def answer():\n", "suffix": "\n# end", "stream": stream, "stop": ["#"], "max_tokens": 32, "metadata": {"test": "original"}, "model": "cannot-redirect"}
            original = deepcopy(body)
            result = await garden.codestral_fim(body, model="codestral-2@001", options=RetryOptions(timeout_ms=1234, max_retries=0))
            self.assertEqual(body, original)
            action = "streamRawPredict" if stream else "rawPredict"
            self.assertTrue(seen[-1][0].endswith(f"/publishers/mistralai/models/codestral-2@001:{action}"))
            self.assertEqual(seen[-1][1]["json_body"], {**body, "model": "codestral-2"})
            self.assertEqual(seen[-1][1]["timeout_ms"], 1234)
            if stream:
                self.assertIs(result, response)
                self.assertTrue(seen[-1][1]["stream"])
            else:
                self.assertEqual(result, response.payload)

    async def test_invalid_payload_is_rejected_and_http_failure_is_preserved(self):
        seen = []
        async def fetch(url, **kwargs):
            seen.append(url)
            return FakeResponse(404, {"error": "missing"})
        garden = create_vertex(access_token="synthetic", project_id="p", location="us-central1", fetch=fetch).native.model_garden()
        for body in ({}, {"prompt": 3}, {"prompt": "", "suffix": []}, {"prompt": "", "stream": "false"}, {"prompt": "", "messages": []}):
            with self.subTest(body=body), self.assertRaises(ValidationError):
                await garden.codestral_fim(body)
        self.assertEqual(seen, [])
        for stream in (False, True):
            with self.assertRaises(ProviderHTTPError) as caught:
                await garden.codestral_fim({"prompt": "synthetic", "stream": stream})
            self.assertEqual(caught.exception.status, 404)
        self.assertEqual(len(seen), 2)
