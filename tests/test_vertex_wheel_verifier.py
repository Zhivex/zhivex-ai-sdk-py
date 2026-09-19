"""Prevent stale or incomplete wheel evidence from passing source verification."""
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch
import zipfile

from scripts import verify_vertex_integration as verifier


class VertexWheelVerifierTests(TestCase):
    def test_file_sets_and_bytes_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            installed = Path(directory) / "installed" / "zhivex_ai"
            source = root / "src" / "zhivex_ai"
            installed.mkdir(parents=True)
            source.mkdir(parents=True)
            wheel = Path(directory) / "sdk.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("zhivex_ai/__init__.py", "# exact\n")
            for folder in (installed, source):
                (folder / "__init__.py").write_text("# exact\n")
            with patch.object(verifier, "ROOT", root), patch.object(verifier.sdk, "__file__", str(installed / "__init__.py")):
                self.assertEqual(len(verifier.verify_wheel(wheel)), 64)
                for folder in (installed, source):
                    extra = folder / "unpackaged.py"
                    extra.write_text("# stale\n")
                    with self.assertRaisesRegex(SystemExit, "file set"):
                        verifier.verify_wheel(wheel)
                    extra.unlink()
                (installed / "__init__.py").write_text("# modified\n")
                with self.assertRaisesRegex(SystemExit, "does not match"):
                    verifier.verify_wheel(wheel)
