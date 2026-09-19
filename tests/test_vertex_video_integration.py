from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tempfile
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

import pytest

pytest.importorskip("google.auth", reason="Vertex runner tests require the vertex extra")
pytest.importorskip("google.auth.transport.requests", reason="Vertex ADC requires requests")

from scripts import verify_vertex_video_integration as runner


class VertexVideoCheckpointTests(IsolatedAsyncioTestCase):
    async def test_extension_identity_and_modes_fail_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.mp4"
            video.write_bytes(b"\x00\x00\x00\x18ftyp" + b"x" * 100)
            args = Namespace(wheel=root / "wheel", output=root / "report", state=root / "state", project="p", location="us-central1", model="veo-3.1-generate-001", image=None, last_frame=None, reference=[], video=video)
            checkpoint = {"wheel_sha256": "digest", "operation_name": "existing-operation", "identity": {
                "project": "p", "location": args.location, "model": args.model,
                "video_sha256": "old-digest",
            }}
            args.state.write_text(json.dumps(checkpoint))
            with patch.object(runner, "verify_wheel", return_value="digest"), patch.object(runner.google.auth, "default") as auth:
                with self.assertRaisesRegex(SystemExit, "another target"):
                    await runner.run(args)
                for field in ("image", "last_frame", "reference"):
                    setattr(args, field, [root / "missing.png"] if field == "reference" else root / "missing.png")
                    with self.assertRaisesRegex(SystemExit, "cannot be combined"):
                        await runner.run(args)
                    setattr(args, field, [] if field == "reference" else None)
                video.write_bytes(b"invalid")
                with self.assertRaisesRegex(SystemExit, "MP4 input"):
                    await runner.run(args)
                auth.assert_not_called()
            self.assertEqual(json.loads(args.state.read_text()), checkpoint)
            self.assertFalse(args.output.exists())

    async def test_reference_identity_and_incompatible_modes_fail_before_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.png"
            reference.write_bytes(b"\x89PNG\r\n\x1a\nreference")
            args = Namespace(wheel=root / "wheel", output=root / "report", state=root / "state", project="p", location="us-central1", model="veo-3.1-generate-001", image=None, last_frame=None, reference=[reference])
            checkpoint = {"wheel_sha256": "digest", "operation_name": "existing-operation", "identity": {
                "project": "p", "location": args.location, "model": args.model,
                "reference_sha256": ["old-digest"],
            }}
            args.state.write_text(json.dumps(checkpoint))
            with patch.object(runner, "verify_wheel", return_value="digest"), patch.object(runner.google.auth, "default") as auth:
                with self.assertRaisesRegex(SystemExit, "another target"):
                    await runner.run(args)
                args.image = reference
                with self.assertRaisesRegex(SystemExit, "cannot be combined"):
                    await runner.run(args)
                args.image = None
                args.reference = [reference] * 4
                with self.assertRaisesRegex(SystemExit, "at most three"):
                    await runner.run(args)
                auth.assert_not_called()
            self.assertEqual(json.loads(args.state.read_text()), checkpoint)

    async def test_changed_last_frame_rejects_resume_before_auth_or_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            last = root / "last.png"
            first.write_bytes(b"\x89PNG\r\n\x1a\nfirst")
            last.write_bytes(b"\x89PNG\r\n\x1a\nlast")
            args = Namespace(wheel=root / "wheel", output=root / "report", state=root / "state", project="p", location="us-central1", model="veo-3.1-fast-generate-001", image=first, last_frame=last)
            checkpoint = {"wheel_sha256": "digest", "operation_name": "existing-operation", "identity": {
                "project": "p", "location": args.location, "model": args.model,
                "image_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
                "last_frame_sha256": "old-digest",
            }}
            args.state.write_text(json.dumps(checkpoint))
            with patch.object(runner, "verify_wheel", return_value="digest"), patch.object(runner.google.auth, "default") as auth:
                with self.assertRaisesRegex(SystemExit, "another target"):
                    await runner.run(args)
                auth.assert_not_called()
            self.assertEqual(json.loads(args.state.read_text()), checkpoint)
            self.assertFalse(args.output.exists())

    async def test_last_frame_requires_first_frame_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(wheel=root / "wheel", output=root / "report", state=root / "state", project="p", location="us-central1", model="veo", image=None, last_frame=root / "last.png")
            with patch.object(runner, "verify_wheel", return_value="digest"), self.assertRaisesRegex(SystemExit, "requires --image"):
                await runner.run(args)
            self.assertFalse(args.state.exists())
