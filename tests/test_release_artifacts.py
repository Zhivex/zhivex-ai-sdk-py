from __future__ import annotations

import ast
import io
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from unittest import TestCase
import zipfile

from scripts import verify_release_artifacts


ROOT = Path(__file__).resolve().parents[1]


class ReleaseArtifactToolingTests(TestCase):
    @staticmethod
    def _write_wheel(path: Path, *, metadata_version: str, project_name: str = "zhivex-ai-sdk") -> None:
        metadata = f"Metadata-Version: 2.4\nName: {project_name}\nVersion: {metadata_version}\n"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                f"zhivex_ai_sdk-{metadata_version}.dist-info/METADATA",
                metadata,
            )

    @staticmethod
    def _write_sdist(path: Path, *, metadata_version: str, project_name: str = "zhivex-ai-sdk") -> None:
        payload = (
            "[build-system]\n"
            'requires = ["hatchling"]\n'
            'build-backend = "hatchling.build"\n\n'
            "[project]\n"
            f'name = "{project_name}"\n'
            f'version = "{metadata_version}"\n'
        ).encode("utf-8")
        member = tarfile.TarInfo(f"zhivex_ai_sdk-{metadata_version}/pyproject.toml")
        member.size = len(payload)
        with tarfile.open(path, "w:gz") as archive:
            archive.addfile(member, io.BytesIO(payload))

    def test_release_artifact_verifier_is_parseable_and_mentions_required_extras(self) -> None:
        script = ROOT / "scripts/verify_release_artifacts.py"
        source = script.read_text("utf-8")
        ast.parse(source)
        for extra in ["postgres", "mcp", "api", "a2a", "ag-ui", "otel", "docx"]:
            self.assertIn(f'"{extra}"', source)
        self.assertIn("validate_release", source)
        self.assertIn("tool_executions", source)

    def test_release_artifact_selection_requires_exact_version_and_clean_dist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            dist = Path(temporary_dir)
            expected = dist / "zhivex_ai_sdk-0.12.1-py3-none-any.whl"
            stale = dist / "zhivex_ai_sdk-0.11.0-py3-none-any.whl"
            self._write_wheel(expected, metadata_version="0.12.1")
            self._write_wheel(stale, metadata_version="0.11.0")

            with self.assertRaisesRegex(RuntimeError, "Remove stale or mismatched artifacts"):
                verify_release_artifacts._select_release_artifact(dist, version="0.12.1", kind="wheel")

            stale.unlink()
            selected = verify_release_artifacts._select_release_artifact(dist, version="0.12.1", kind="wheel")

        self.assertEqual(selected.name, expected.name)

    def test_release_artifact_selection_rejects_only_wrong_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            dist = Path(temporary_dir)
            self._write_sdist(dist / "zhivex_ai_sdk-0.11.0.tar.gz", metadata_version="0.11.0")

            with self.assertRaisesRegex(RuntimeError, "Expected exactly one sdist artifact for version 0.12.1"):
                verify_release_artifacts._select_release_artifact(dist, version="0.12.1", kind="sdist")

    def test_wheel_metadata_must_match_release_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            wheel = Path(temporary_dir) / "zhivex_ai_sdk-0.12.1-py3-none-any.whl"
            self._write_wheel(wheel, metadata_version="0.11.0")

            with self.assertRaisesRegex(RuntimeError, "metadata version mismatch"):
                verify_release_artifacts._verify_wheel_metadata(wheel, expected_version="0.12.1")

    def test_sdist_pyproject_must_match_release_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            sdist = Path(temporary_dir) / "zhivex_ai_sdk-0.12.1.tar.gz"
            self._write_sdist(sdist, metadata_version="0.11.0")

            with self.assertRaisesRegex(RuntimeError, "metadata version mismatch"):
                verify_release_artifacts._verify_sdist_metadata(sdist, expected_version="0.12.1")

    def test_release_evidence_collector_and_plan_are_present(self) -> None:
        script = ROOT / "scripts/collect_release_evidence.py"
        source = script.read_text("utf-8")
        ast.parse(source)
        for gate in ["compile", "public contract", "artifact install smoke"]:
            self.assertIn(gate, source)

        version = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["version"]
        release_plan = (ROOT / f"docs/releases/{version}.md").read_text("utf-8")
        self.assertIn(version, release_plan)
        self.assertIn("make release-evidence", release_plan)
        self.assertIn("Postgres run store", release_plan)

    def test_release_evidence_records_candidate_metadata_and_artifact_hashes(self) -> None:
        script = ROOT / "scripts/collect_release_evidence.py"
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "evidence.md"
            process = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dry-run",
                    "--only",
                    "compile",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stdout)
            evidence = output.read_text("utf-8")

        self.assertRegex(evidence, r"Git commit: `[0-9a-f]{40}`")
        self.assertRegex(evidence, r"Working tree: `(clean|dirty)`")
        self.assertRegex(evidence, r"Python: `\d+\.\d+\.\d+")
        self.assertIn("## Artifact SHA256", evidence)
        self.assertIn("pip-audit==", evidence)

    def test_release_tag_verifier_rejects_version_mismatches(self) -> None:
        script = ROOT / "scripts/verify_release_tag.py"
        version = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]["version"]
        accepted = subprocess.run(
            [sys.executable, str(script), "--tag", f"v{version}"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        rejected = subprocess.run(
            [sys.executable, str(script), "--tag", "v0.0.0"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stdout)
        self.assertEqual(rejected.returncode, 1, rejected.stdout)

    def test_security_dependency_floors_exclude_known_vulnerable_ranges(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
        extras = pyproject["project"]["optional-dependencies"]
        self.assertIn("python-dotenv>=1.2.2", extras["dev"])
        self.assertIn("pytest>=9.0.3", extras["dev"])
        self.assertIn("pip-audit>=2.10.1", extras["dev"])
        self.assertIn("setuptools>=83.0.0", extras["dev"])
        self.assertIn("mcp>=1.28.1", extras["mcp"])
        self.assertIn("a2a-sdk[fastapi]>=1.1.2,<2", extras["a2a"])
        self.assertIn("ag-ui-protocol>=0.1.19,<0.2", extras["ag-ui"])

    def test_makefile_release_check_runs_install_verification(self) -> None:
        makefile = (ROOT / "Makefile").read_text("utf-8")
        self.assertIn("release-install-check:", makefile)
        self.assertIn("release-evidence:", makefile)
        self.assertIn("scripts/verify_release_artifacts.py", makefile)
        self.assertIn("scripts/collect_release_evidence.py", makefile)
        self.assertIn("tests/test_agent_safety_runtime.py", makefile)
        self.assertIn("tests/test_tool_timeout_safety.py", makefile)
        self.assertIn("release-check: check test-release build release-install-check security-check", makefile)
        self.assertIn("rm -rf dist build", makefile)
        self.assertIn("security-check:", makefile)
        self.assertIn("pip_audit . --strict", makefile)
        self.assertIn("pip_audit --local --skip-editable", makefile)

    def test_sdist_includes_root_docs_linked_from_packaged_docs(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
        sdist = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]
        include = set(sdist["include"])
        force_include = sdist["force-include"]

        for path in [
            "/CHANGELOG.md",
            "/CONTRIBUTING.md",
            "/PRODUCTION_APIS.md",
            "/README.md",
            "/SECURITY.md",
            "/STABILITY.md",
            "/SUPPORT.md",
            "/VERSIONING.md",
            "/docs",
        ]:
            self.assertIn(path, include)
        self.assertEqual(force_include[".env.example"], ".env.example")

    def test_ci_matrix_and_publish_workflows_include_release_verification(self) -> None:
        ci = (ROOT / ".github/workflows/ci.yml").read_text("utf-8")
        publish = (ROOT / ".github/workflows/publish-pypi.yml").read_text("utf-8")
        test_publish = (ROOT / ".github/workflows/publish-testpypi.yml").read_text("utf-8")

        for version in ['"3.11"', '"3.12"', '"3.13"', '"3.14"']:
            self.assertIn(version, ci)
        self.assertIn("scripts/verify_release_artifacts.py", ci)
        for workflow in [ci, publish, test_publish]:
            action_refs = re.findall(r"uses:\s+[^@\s]+@([^\s#]+)", workflow)
            self.assertTrue(action_refs)
            for action_ref in action_refs:
                self.assertRegex(action_ref, r"^[0-9a-f]{40}$")

        for workflow in [ci, publish, test_publish]:
            self.assertIn("persist-credentials: false", workflow)
        self.assertIn("python -m pip_audit --skip-editable", ci)
        self.assertEqual(ci.count('"setuptools>=83.0.0"'), 3)
        self.assertIn('"setuptools>=83.0.0"', publish)
        self.assertIn('"setuptools>=83.0.0"', test_publish)
        for workflow, environment in [(publish, "pypi"), (test_publish, "testpypi")]:
            self.assertIn("actions/upload-artifact@", workflow)
            self.assertIn("actions/download-artifact@", workflow)
            self.assertIn("needs: build", workflow)
            self.assertIn(f"name: {environment}", workflow)
            self.assertEqual(workflow.count("id-token: write"), 1)
            self.assertIn("environment: release-smoke", workflow)
            self.assertIn("ZHIVEX_SMOKE_USE_INSTALLED: \"1\"", workflow)
            self.assertIn("ZHIVEX_RELEASE_SMOKE_PROVIDERS", workflow)
            self.assertIn("python scripts/run_live_smoke.py", workflow)
            self.assertIn("needs: [build, live-agent-smoke]", workflow)

        self.assertIn("scripts/verify_release_tag.py", publish)
        for workflow in [publish, test_publish]:
            self.assertIn("fetch-depth: 0", workflow)
            self.assertIn('git merge-base --is-ancestor "$GITHUB_SHA" refs/remotes/origin/main', workflow)
            self.assertIn("make PYTHON=python release-check", workflow)
            self.assertIn(".[dev,postgres,mcp,api,a2a,ag-ui,otel,docx]", workflow)
