from unittest import IsolatedAsyncioTestCase
from copy import deepcopy

from zhivex_ai import create_vertex
from zhivex_ai.errors import ValidationError
from tests.test_gemini_provider import FakeResponse


class VertexRagTests(IsolatedAsyncioTestCase):
    async def test_engine_config_uses_configured_region_and_preserves_payload(self):
        from zhivex_ai.types import RetryOptions
        seen = []
        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return FakeResponse(200, {"name": "operations/config", "done": False})
        rag = create_vertex(access_token="t", project_id="p", location="us-central1", fetch=fetch).native.rag()
        body = {"ragManagedDbConfig": {"scaled": {}}}
        await rag.get_engine_config(RetryOptions(timeout_ms=1000, max_retries=0))
        operation = await rag.update_engine_config(body)
        expected = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/ragEngineConfig"
        self.assertEqual([url for url, _ in seen], [expected, expected])
        self.assertEqual([kwargs["method"] for _, kwargs in seen], ["GET", "PATCH"])
        self.assertEqual(seen[0][1]["timeout_ms"], 1000)
        self.assertIsNone(seen[0][1]["json_body"])
        self.assertEqual(seen[1][1]["json_body"], body)
        self.assertIsNot(seen[1][1]["json_body"], body)
        self.assertFalse(operation["done"])

    async def test_native_routes_payloads_and_pagination(self):
        seen = []
        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return FakeResponse(200, {"done": True, "nextPageToken": "next"})
        provider = create_vertex(access_token="token", project_id="p", location="us-central1", fetch=fetch)
        rag = provider.native.rag()
        self.assertIs(rag, provider.native.rag())
        base = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1"
        body = {"displayName": "test"}
        original = deepcopy(body)
        await rag.create(body)
        await rag.get("c")
        page = await rag.list(page_size=2, page_token="a+b")
        await rag.update("c", body)
        await rag.import_files("c", {"importRagFilesConfig": {"gcsSource": {"uris": ["gs://bucket/file"]}}})
        await rag.list_files("c", page_token="a+b")
        await rag.get_file("c", "f")
        await rag.delete_file("c", "f")
        query = {"query": {"text": "test"}, "vertexRagStore": {"ragResources": [{"ragCorpus": "projects/p/locations/us-central1/ragCorpora/c"}]}}
        await rag.retrieve_contexts(query)
        await rag.delete("c", force=True)
        operation = await rag.wait_operation("projects/p/locations/us-central1/operations/op")
        expected = [
            ("/ragCorpora", "POST"), ("/ragCorpora/c", "GET"),
            ("/ragCorpora?pageSize=2&pageToken=a%2Bb", "GET"),
            ("/ragCorpora/c", "PATCH"),
            ("/ragCorpora/c/ragFiles:import", "POST"),
            ("/ragCorpora/c/ragFiles?pageToken=a%2Bb", "GET"),
            ("/ragCorpora/c/ragFiles/f", "GET"), ("/ragCorpora/c/ragFiles/f", "DELETE"),
            (":retrieveContexts", "POST"), ("/ragCorpora/c?force=true", "DELETE"),
            ("/operations/op", "GET"),
        ]
        self.assertEqual([(url, kwargs.get("method", "POST")) for url, kwargs in seen], [(base + path, method) for path, method in expected])
        self.assertTrue(all(kwargs["headers"]["authorization"] == "Bearer token" for _, kwargs in seen))
        self.assertEqual(seen[8][1]["json_body"], query)
        self.assertEqual(body, original)
        self.assertEqual(page["nextPageToken"], "next")
        self.assertTrue(operation["done"])

    async def test_operation_error_is_preserved_and_polling_continues_until_done(self):
        responses = [{"name": "operations/op"}, {"name": "operations/op", "done": True, "error": {"code": 7, "message": "denied"}}]
        async def fetch(url, **kwargs):
            return FakeResponse(200, responses.pop(0))
        rag = create_vertex(access_token="t", project_id="p", fetch=fetch).native.rag()
        operation = await rag.wait_operation("operations/op", poll_interval_ms=1)
        self.assertEqual(operation["error"]["code"], 7)
        self.assertFalse(responses)

    async def test_rejects_destinations_actions_and_express(self):
        async def fetch(*args, **kwargs):
            self.fail("invalid resource reached fetch")
        rag = create_vertex(access_token="t", project_id="p", fetch=fetch).native.rag()
        for value in ("https://example.com", "c:delete", "../c", "c?x=1", "ragCorpora/c/ragFiles/f"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                await rag.get(value)
        with self.assertRaises(ValidationError):
            await rag.get_file("c", "f:delete")
        with self.assertRaises(ValidationError):
            await rag.get_operation("ragCorpora/c")
        with self.assertRaises(ValidationError):
            await rag.update("c", {}, update_mask="display_name")
        with self.assertRaises(AttributeError):
            create_vertex(api_key="key", express_mode=True).native.rag()
