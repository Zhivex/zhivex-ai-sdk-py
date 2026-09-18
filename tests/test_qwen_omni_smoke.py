from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import zipfile

from scripts import smoke_qwen_omni as smoke


class OmniInstalledSmokeTests(TestCase):
    def test_rejects_checkout_import(self):
        with patch.object(smoke.zhivex_ai, "__file__", str(smoke.ROOT / "src/zhivex_ai/__init__.py")):
            with self.assertRaisesRegex(RuntimeError, "checkout"):
                smoke.verify_installed_wheel(Path("unused.whl"))

    def test_checks_installed_bytes_and_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "zhivex_ai"
            package.mkdir()
            module = package / "__init__.py"
            module.write_bytes(b"original")
            wheel = root / "zhivex_ai_sdk-0.25.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("zhivex_ai/__init__.py", b"original")
                archive.writestr("zhivex_ai_sdk-0.25.0.dist-info/METADATA", "Name: zhivex-ai-sdk\nVersion: 0.25.0\n")
            with patch.object(smoke.zhivex_ai, "__file__", str(module)), patch.object(smoke.metadata, "version", return_value="0.25.0"):
                result = smoke.verify_installed_wheel(wheel)
                self.assertEqual(result["package_version"], "0.25.0")
                self.assertEqual(len(result["sha256"]), 64)
                module.write_bytes(b"modified")
                with self.assertRaisesRegex(RuntimeError, "bytes differ"):
                    smoke.verify_installed_wheel(wheel)

    def test_rejects_renamed_wheel_with_wrong_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / "zhivex_ai/__init__.py"
            wheel = root / "zhivex_ai_sdk-0.25.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("zhivex_ai_sdk-0.25.0.dist-info/METADATA", "Name: zhivex-ai-sdk\nVersion: 0.24.0\n")
            with patch.object(smoke.zhivex_ai, "__file__", str(module)), patch.object(smoke.metadata, "version", return_value="0.25.0"):
                with self.assertRaisesRegex(RuntimeError, "metadata differs"):
                    smoke.verify_installed_wheel(wheel)
