from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import tomllib
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]


class ReleaseArtifactToolingTests(TestCase):
    def test_release_artifact_verifier_is_parseable_and_mentions_required_extras(self) -> None:
        script = ROOT / "scripts/verify_release_artifacts.py"
        source = script.read_text("utf-8")
        ast.parse(source)
        for extra in ["postgres", "mcp", "api", "otel", "docx"]:
            self.assertIn(f'"{extra}"', source)

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
        self.assertIn("mcp>=1.28.1", extras["mcp"])

    def test_makefile_release_check_runs_install_verification(self) -> None:
        makefile = (ROOT / "Makefile").read_text("utf-8")
        self.assertIn("release-install-check:", makefile)
        self.assertIn("release-evidence:", makefile)
        self.assertIn("scripts/verify_release_artifacts.py", makefile)
        self.assertIn("scripts/collect_release_evidence.py", makefile)
        self.assertIn("release-check: build release-install-check", makefile)
        self.assertIn("rm -rf dist build", makefile)
        self.assertIn("security-check:", makefile)
        self.assertIn("pip_audit --skip-editable", makefile)

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
        for workflow in [ci, publish, test_publish]:
            self.assertIn("scripts/verify_release_artifacts.py", workflow)
            action_refs = re.findall(r"uses:\s+[^@\s]+@([^\s#]+)", workflow)
            self.assertTrue(action_refs)
            for action_ref in action_refs:
                self.assertRegex(action_ref, r"^[0-9a-f]{40}$")

        for workflow in [ci, publish, test_publish]:
            self.assertIn("persist-credentials: false", workflow)
        self.assertIn("python -m pip_audit --skip-editable", ci)
        for workflow, environment in [(publish, "pypi"), (test_publish, "testpypi")]:
            self.assertIn("actions/upload-artifact@", workflow)
            self.assertIn("actions/download-artifact@", workflow)
            self.assertIn("needs: build", workflow)
            self.assertIn(f"name: {environment}", workflow)
            self.assertEqual(workflow.count("id-token: write"), 1)

        self.assertIn("scripts/verify_release_tag.py", publish)
