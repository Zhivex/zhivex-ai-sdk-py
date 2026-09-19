from argparse import Namespace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from scripts import verify_vertex_rag_lifecycle as runner


class VertexRagLifecycleTests(IsolatedAsyncioTestCase):
    async def test_existing_unknown_creation_cannot_resubmit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(wheel=root / "wheel", project="p", location="us-central1", state=root / "state", output=root / "report")
            state = {"identity": {"project": "p", "location": args.location, "wheel_sha256": "digest"}}
            args.state.write_text(json.dumps(state))
            with patch.object(runner, "verify_wheel", return_value="digest"), patch.object(runner.google.auth, "default") as auth:
                with self.assertRaisesRegex(SystemExit, "Unknown creation"):
                    await runner.run(args)
                auth.assert_not_called()
            self.assertEqual(json.loads(args.state.read_text()), state)

    async def test_failed_update_still_deletes_exact_created_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(wheel=root / "wheel", project="p", location="us-central1", state=root / "state", output=root / "report")
            corpus = "projects/p/locations/us-central1/ragCorpora/owned"
            state = {"identity": {"project": "p", "location": args.location, "wheel_sha256": "digest"}, "display_name": "owned", "create_operation": "operations/create", "corpus": corpus}
            args.state.write_text(json.dumps(state))
            rag = AsyncMock()
            rag.get.return_value = {"displayName": "owned"}
            rag.update.side_effect = RuntimeError("update failure")
            rag.delete.return_value = {"name": "operations/delete"}
            rag.wait_operation.return_value = {"done": True}
            provider = SimpleNamespace(native=SimpleNamespace(rag=lambda: rag))
            with patch.object(runner, "verify_wheel", return_value="digest"), patch.object(runner.google.auth, "default", return_value=(object(), None)), patch.object(runner, "create_vertex", return_value=provider):
                self.assertEqual(await runner.run(args), 1)
            rag.create.assert_not_called()
            self.assertEqual(rag.delete.call_args.args, (corpus,))
            report = json.loads(args.output.read_text())
            self.assertEqual(report["operations"]["update"], "failed")
            self.assertTrue(report["cleanup_complete"])
