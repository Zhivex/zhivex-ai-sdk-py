from __future__ import annotations

import ast
from pathlib import Path
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

    def test_makefile_release_check_runs_install_verification(self) -> None:
        makefile = (ROOT / "Makefile").read_text("utf-8")
        self.assertIn("release-install-check:", makefile)
        self.assertIn("scripts/verify_release_artifacts.py", makefile)
        self.assertIn("release-check: build release-install-check", makefile)

    def test_sdist_includes_root_docs_linked_from_packaged_docs(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
        sdist = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]
        include = set(sdist["include"])
        force_include = sdist["force-include"]

        for path in [
            "/CHANGELOG.md",
            "/CONTRIBUTING.md",
            "/MATURITY_PLAN.md",
            "/PRODUCTION_APIS.md",
            "/RELEASING.md",
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
