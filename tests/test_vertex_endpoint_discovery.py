from argparse import Namespace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from scripts import verify_vertex_endpoint_discovery as runner


class VertexEndpointDiscoveryTests(IsolatedAsyncioTestCase):
    async def test_repeated_cursor_is_failure_not_complete_inventory(self):
        for registry in (False, True):
            with self.subTest(registry=registry), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = Namespace(wheel=root / "wheel", output=root / "report", project="p", location=["us-central1"], registered_models=registry)
                client = SimpleNamespace(list_endpoints=AsyncMock(return_value={"nextPageToken": "repeat"}),
                                         list_registered_models=AsyncMock(return_value={"nextPageToken": "repeat"}))
                provider = SimpleNamespace(native=SimpleNamespace(model_garden=lambda: client))
                with patch.object(runner, "verify_wheel", return_value="digest"), \
                     patch.object(runner.google.auth, "default", return_value=(object(), None)), \
                     patch.object(runner, "create_vertex", return_value=provider):
                    self.assertEqual(await runner.run(args), 1)
                selected = client.list_registered_models if registry else client.list_endpoints
                self.assertEqual(selected.await_count, 2)
                item = json.loads(args.output.read_text())["locations"]["us-central1"]
                self.assertEqual(item["list"], "failed")

    async def test_registry_empty_success_does_not_infer_get_or_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(wheel=root / "wheel", output=root / "report", project="p", location=["us-central1"], registered_models=True)
            client = SimpleNamespace(list_registered_models=AsyncMock(return_value={}), get_registered_model=AsyncMock())
            provider = SimpleNamespace(native=SimpleNamespace(model_garden=lambda: client))
            with patch.object(runner, "verify_wheel", return_value="digest"), \
                 patch.object(runner.google.auth, "default", return_value=(object(), None)), \
                 patch.object(runner, "create_vertex", return_value=provider):
                self.assertEqual(await runner.run(args), 0)
            client.get_registered_model.assert_not_awaited()
            item = json.loads(args.output.read_text())["locations"]["us-central1"]
            self.assertEqual(item["model_count"], 0)
            self.assertEqual(item["get"], "not-run-no-model")
