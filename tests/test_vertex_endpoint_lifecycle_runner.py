from argparse import Namespace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("google.auth", reason="Vertex runner tests require the vertex extra")
pytest.importorskip("google.auth.transport.requests", reason="Vertex ADC requires requests")

from scripts import verify_vertex_endpoint_lifecycle as runner
from zhivex_ai.errors import ProviderHTTPError


class EndpointLifecycleRunnerTests(IsolatedAsyncioTestCase):
    async def test_unknown_creation_refuses_resubmission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(wheel=root / "wheel", state=root / "state", output=root / "report", project="p", location="us-central1")
            args.state.write_text(json.dumps({"identity": {"project": "p", "location": args.location, "wheel_sha256": "digest"}}))
            with patch.object(runner, "verify_wheel", return_value="digest"), patch.object(runner.google.auth, "default") as auth:
                with self.assertRaises(SystemExit):
                    await runner.run(args)
                auth.assert_not_called()

    async def test_failed_update_cleans_owned_empty_endpoint_but_never_deployed_one(self):
        for deployed in (False, True):
            with self.subTest(deployed=deployed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = Namespace(wheel=root / "wheel", state=root / "state", output=root / "report", project="p", location="us-central1")
                args.state.write_text(json.dumps({"identity": {"project": "p", "location": args.location, "wheel_sha256": "digest"}, "endpoint_id": "owned", "create_operation": "operations/create", "created": True, "safe_to_delete": True}))
                client = AsyncMock()
                client.get_endpoint.side_effect = [{"displayName": "owned", "deployedModels": [{"id": "d"}] if deployed else []}, ProviderHTTPError("missing", 404)]
                client.update_endpoint.side_effect = RuntimeError("update failure")
                client.delete_endpoint.return_value = {"name": "operations/delete"}
                client.wait_operation.return_value = {"done": True}
                provider = SimpleNamespace(native=SimpleNamespace(model_garden=lambda: client))
                with patch.object(runner, "verify_wheel", return_value="digest"), \
                     patch.object(runner.google.auth, "default", return_value=(object(), None)), \
                     patch.object(runner, "create_vertex", return_value=provider):
                    self.assertEqual(await runner.run(args), 1)
                client.create_endpoint.assert_not_awaited()
                if deployed:
                    client.delete_endpoint.assert_not_awaited()
                else:
                    self.assertEqual(client.delete_endpoint.call_args.args, ("owned",))
                    self.assertTrue(json.loads(args.output.read_text())["cleanup_complete"])
