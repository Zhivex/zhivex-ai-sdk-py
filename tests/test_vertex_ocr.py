from copy import deepcopy
from unittest import IsolatedAsyncioTestCase

from zhivex_ai import create_vertex
from zhivex_ai.errors import ProviderHTTPError, ValidationError
from tests.test_gemini_provider import FakeResponse


class VertexOCRTests(IsolatedAsyncioTestCase):
    async def test_document_and_image_preserve_native_payload_and_page_metadata(self):
        seen = []
        payload = {"pages": [{"index": 0, "markdown": "# Synthetic", "images": [{"id": "img-0", "image_base64": "synthetic"}], "dimensions": {"width": 64}}], "usage_info": {"pages_processed": 1}, "model": "mistral-ocr-2505"}
        async def fetch(url, **kwargs):
            seen.append((url, deepcopy(kwargs)))
            kwargs["json_body"]["document"]["changed"] = True
            return FakeResponse(200, payload)
        garden = create_vertex(access_token="synthetic", project_id="p", location="us-central1", fetch=fetch).native.model_garden()
        for kind, value in (("document_url", "data:application/pdf;base64,c3ludGhldGlj"), ("image_url", "https://example.com/synthetic.png")):
            body = {"document": {"type": kind, kind: value}, "pages": [0], "include_image_base64": True, "model": "cannot-redirect"}
            original = deepcopy(body)
            result = await garden.mistral_ocr(body)
            self.assertEqual(result, payload)
            self.assertEqual(body, original)
            self.assertEqual(seen[-1][1]["json_body"], {**original, "model": "mistral-ocr-2505"})
            self.assertTrue(seen[-1][0].endswith("/publishers/mistralai/models/mistral-ocr-2505:rawPredict"))

    async def test_bad_documents_and_stream_rejected_before_dispatch(self):
        seen = []
        async def fetch(url, **kwargs):
            seen.append(url)
            return FakeResponse(404, {"error": "missing"})
        garden = create_vertex(access_token="synthetic", project_id="p", fetch=fetch).native.model_garden()
        valid = {"document": {"type": "document_url", "document_url": "https://example.com/synthetic.pdf"}}
        for body in ({}, {"document": []}, {"document": {"type": "document_url", "document_url": 3}}, {"document": {"type": "file", "file": "a"}}, {**valid, "stream": True}, {**valid, "stream": "false"}):
            with self.subTest(body=body), self.assertRaises(ValidationError):
                await garden.mistral_ocr(body)
        self.assertFalse(seen)
        with self.assertRaises(ProviderHTTPError) as caught:
            await garden.mistral_ocr(valid)
        self.assertEqual(caught.exception.status, 404)
        self.assertEqual(len(seen), 1)
