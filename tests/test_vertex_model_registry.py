from copy import deepcopy
from unittest import IsolatedAsyncioTestCase
from urllib.parse import parse_qs, urlsplit

from zhivex_ai import create_vertex
from zhivex_ai.errors import ValidationError
from tests.test_gemini_provider import FakeResponse


class VertexModelRegistryTests(IsolatedAsyncioTestCase):
    async def test_registry_is_project_scoped_and_preserves_upload_and_pagination(self):
        seen = []
        async def fetch(url, **kwargs):
            seen.append((url, deepcopy(kwargs)))
            if kwargs.get("json_body"):
                kwargs["json_body"]["mutated"] = True
            return FakeResponse(200, {"nextPageToken": "next", "done": True})
        garden = create_vertex(access_token="synthetic", project_id="p", location="us-central1", fetch=fetch).native.model_garden()
        body = {"modelId": "my-model", "model": {"artifactUri": "gs://synthetic/model", "containerSpec": {"imageUri": "example/image"}}, "parentModel": "projects/p/locations/us-central1/models/parent"}
        await garden.upload_model(body)
        self.assertTrue(seen[-1][0].endswith("/projects/p/locations/us-central1/models:upload"))
        self.assertEqual(seen[-1][1]["json_body"], body)
        self.assertNotIn("mutated", body)
        await garden.get_registered_model("my-model@stable")
        self.assertTrue(seen[-1][0].endswith("/models/my-model@stable"))
        result = await garden.list_registered_models(page_size=2, page_token="a+b", filter='labels.test="yes"', read_mask="name,displayName")
        self.assertEqual(result["nextPageToken"], "next")
        self.assertEqual(parse_qs(urlsplit(seen[-1][0]).query)["pageToken"], ["a+b"])
        self.assertEqual(len(seen), 3)
        await garden.update_registered_model("my-model", {"description": "new"}, update_mask="description")
        self.assertEqual(seen[-1][1]["method"], "PATCH")
        await garden.delete_registered_model("my-model")
        self.assertEqual(seen[-1][1]["method"], "DELETE")
        self.assertEqual(len(seen), 5)
        await garden.wait_operation("projects/p/locations/us-central1/models/my-model/operations/op")
        self.assertEqual(seen[-1][1]["method"], "GET")

    async def test_registry_rejects_resource_and_action_injection_before_dispatch(self):
        async def fetch(*args, **kwargs):
            self.fail("invalid model dispatched")
        garden = create_vertex(access_token="synthetic", project_id="p", fetch=fetch).native.model_garden()
        for model in ("../other", "id:delete", "id?x=1", "https://example.com/model", "projects/p/models/m"):
            for method in (garden.get_registered_model, garden.delete_registered_model):
                with self.assertRaises(ValidationError):
                    await method(model)
