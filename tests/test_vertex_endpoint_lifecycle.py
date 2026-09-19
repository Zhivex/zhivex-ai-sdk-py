from copy import deepcopy
from unittest import IsolatedAsyncioTestCase

from zhivex_ai import create_vertex
from zhivex_ai.errors import ValidationError
from tests.test_gemini_provider import FakeResponse


class VertexEndpointLifecycleTests(IsolatedAsyncioTestCase):
    async def test_native_payloads_actions_and_no_implicit_deployment(self):
        seen = []
        async def fetch(url, **kwargs):
            seen.append((url, deepcopy(kwargs)))
            if kwargs.get("json_body"):
                kwargs["json_body"]["changed"] = True
            return FakeResponse(200, {"name": "projects/p/locations/us-central1/operations/op"})
        garden = create_vertex(access_token="synthetic", project_id="p", location="us-central1", fetch=fetch).native.model_garden()
        base = "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1"
        body = {"displayName": "synthetic", "labels": {"test": "true"}}
        await garden.create_endpoint(body, endpoint_id="test-endpoint")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[-1][0], base + "/endpoints?endpointId=test-endpoint")
        self.assertNotIn("changed", body)
        await garden.update_endpoint("test-endpoint", body, update_mask="displayName,labels")
        self.assertEqual(seen[-1][1]["method"], "PATCH")
        self.assertTrue(seen[-1][0].endswith("?updateMask=displayName%2Clabels"))
        deploy = {"deployedModel": {"model": "projects/p/locations/us-central1/models/m", "dedicatedResources": {"minReplicaCount": 1}}, "trafficSplit": {"0": 100}}
        await garden.deploy_model("test-endpoint", deploy)
        self.assertEqual(seen[-1][0], base + "/endpoints/test-endpoint:deployModel")
        self.assertEqual(seen[-1][1]["json_body"], deploy)
        await garden.undeploy_model("test-endpoint", {"deployedModelId": "d", "trafficSplit": {}})
        self.assertTrue(seen[-1][0].endswith(":undeployModel"))
        await garden.delete_endpoint("test-endpoint")
        self.assertEqual(seen[-1][1]["method"], "DELETE")
        self.assertIsNone(seen[-1][1]["json_body"])
        self.assertEqual(len(seen), 5)
        for bad in ("../other", "id:deployModel", "https://example.com", "id?x=1"):
            with self.assertRaises(ValidationError):
                await garden.deploy_model(bad, {})
        self.assertEqual(len(seen), 5)

    async def test_poll_terminal_error_and_timeout_never_resubmit(self):
        seen = []
        responses = [{}, {"done": True, "error": {"code": 8, "message": "quota"}}]
        async def fetch(url, **kwargs):
            seen.append((url, kwargs))
            return FakeResponse(200, responses.pop(0) if responses else {})
        garden = create_vertex(access_token="synthetic", project_id="p", fetch=fetch).native.model_garden()
        result = await garden.wait_operation("endpoints/e/operations/op", poll_interval_ms=1)
        self.assertEqual(result["error"]["code"], 8)
        with self.assertRaises(TimeoutError):
            await garden.wait_operation("operations/op", poll_interval_ms=1, timeout_ms=5)
        self.assertTrue(all(kwargs.get("method") == "GET" for _, kwargs in seen))
        for bad in ("operations/op:cancel", "https://example.com/operations/op", "projects/p/locations/r/operations/../other"):
            with self.assertRaises(ValidationError):
                await garden.get_operation(bad)
