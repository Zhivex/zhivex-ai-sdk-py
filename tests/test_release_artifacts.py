from __future__ import annotations

import ast
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from unittest import TestCase
from unittest.mock import patch
import zipfile

from scripts import audit_dependencies, collect_release_evidence, verify_release_artifacts


ROOT = Path(__file__).resolve().parents[1]


class ReleaseArtifactToolingTests(TestCase):
    def test_ci_and_publish_workflows_provision_the_pinned_uv_release_dependency(self) -> None:
        setup_uv = (
            "uses: astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d "
            "# v10.0.1"
        )
        for workflow_name in ["ci.yml", "publish-pypi.yml", "publish-testpypi.yml"]:
            workflow = (ROOT / ".github/workflows" / workflow_name).read_text("utf-8")
            self.assertIn(setup_uv, workflow)
            self.assertIn('version: "0.12.4"', workflow)

        ci_workflow = (ROOT / ".github/workflows/ci.yml").read_text("utf-8")
        self.assertIn("make PYTHON=python security-check", ci_workflow)

    def test_publish_workflows_use_metadata_25_compatible_publisher(self) -> None:
        publisher = (
            "uses: pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33 "
            "# v1.14.2"
        )
        for workflow_name in ["publish-pypi.yml", "publish-testpypi.yml"]:
            workflow = (ROOT / ".github/workflows" / workflow_name).read_text("utf-8")
            self.assertIn(publisher, workflow)

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
        self.assertIn("AgentEvaluationTrialResult", source)
        self.assertIn("WorkflowLeaseManager", source)
        self.assertIn("ResponsesEventStore", source)
        self.assertIn("create_postgres_workflow_checkpoint_store", source)
        self.assertIn("create_postgres_workflow_lease_manager", source)

    def test_installed_postgres_workflow_smoke_uses_real_durable_backends_and_cleanup(self) -> None:
        smoke = verify_release_artifacts._postgres_workflow_smoke_code()

        self.assertIn("asyncpg.create_pool", smoke)
        self.assertIn("create_postgres_workflow_checkpoint_store", smoke)
        self.assertIn("create_postgres_workflow_lease_manager", smoke)
        self.assertIn('result.status == "completed"', smoke)
        self.assertIn('metadata["execution_lease"]["fencing_token"] == 1', smoke)
        self.assertIn("migrate_workflow_run_checkpoint", smoke)
        self.assertIn("workflow-checkpoint-schema-migrated", smoke)
        self.assertIn("DROP TABLE IF EXISTS", smoke)
        self.assertIn('print("postgres-workflow-ok")', smoke)
        ast.parse(smoke)

    def test_required_postgres_workflow_smoke_fails_without_a_dsn(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ZHIVEX_TEST_POSTGRES_DSN"):
                verify_release_artifacts._run_postgres_workflow_smoke(
                    Path("python"),
                    required=True,
                )

    def test_installed_artifact_smoke_uses_the_focused_namespaces(self) -> None:
        smoke = verify_release_artifacts._smoke_code("0.20.0")

        for namespace in [
            "zhivex_ai.evals",
            "zhivex_ai.experimental",
            "zhivex_ai.integrations.protocols",
            "zhivex_ai.integrations.responses",
            "zhivex_ai.workflows",
        ]:
            self.assertIn(f"from {namespace} import", smoke)
        self.assertIn('assert "create_meta" in STABLE_EXPORTS', smoke)
        self.assertIn('assert "meta_hosted_tool" in BETA_EXPORTS', smoke)
        self.assertIn('assert "WorkflowGraph" in STABLE_EXPORTS', smoke)
        self.assertIn('assert "migrate_workflow_checkpoint" in STABLE_EXPORTS', smoke)
        self.assertIn('assert "create_temporal_workflow_adapter" in BETA_EXPORTS', smoke)
        self.assertIn("workflow-checkpoint-v1-to-v2", smoke)

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
        self.assertIn(collect_release_evidence.PATH_SANITIZATION_NOTE, evidence)

    def test_release_evidence_sanitizes_local_paths_portably(self) -> None:
        raw = "\n".join(
            [
                "repo=/Users/alice/work/zhivex/src/example.py",
                "repo-uri=file:///Users/alice/work/zhivex/dist/package.whl",
                "home=/Users/alice/Library/Caches/pip",
                "tmp=/private/tmp/zhivex-release-a1b2c3/venv/bin/python",
                "python=/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/asyncio.py",
                r"windows=C:\Users\alice\work\zhivex\src\example.py",
            ]
        )

        sanitized = collect_release_evidence._sanitize_evidence_output(
            raw,
            repo_root="/Users/alice/work/zhivex",
            home_dir="/Users/alice",
            temp_dirs=("/private/tmp",),
            python_prefixes=("/Library/Frameworks/Python.framework/Versions/3.14",),
        )
        sanitized = collect_release_evidence._sanitize_evidence_output(
            sanitized,
            repo_root=r"C:\Users\alice\work\zhivex",
            home_dir=r"C:\Users\alice",
            temp_dirs=(r"C:\Users\alice\AppData\Local\Temp",),
            python_prefixes=(r"C:\Python314",),
        )

        self.assertIn("repo=<repo>/src/example.py", sanitized)
        self.assertIn("repo-uri=<repo>/dist/package.whl", sanitized)
        self.assertIn("home=<home>/Library/Caches/pip", sanitized)
        self.assertIn("tmp=<tmp>/<run>/venv/bin/python", sanitized)
        self.assertIn("python=<python>/lib/python3.14/asyncio.py", sanitized)
        self.assertIn("windows=<repo>/src/example.py", sanitized)
        self.assertNotIn("alice", sanitized)
        self.assertNotIn("file:///", sanitized)

    def test_checked_in_release_evidence_contains_no_personal_local_paths(self) -> None:
        evidence = (ROOT / "docs/releases/0.16.0-evidence.md").read_text("utf-8")

        self.assertIn(collect_release_evidence.PATH_SANITIZATION_NOTE, evidence)
        self.assertNotIn(str(Path.home()), evidence)
        self.assertNotIn("/Users/", evidence)
        self.assertNotIn("/private/tmp/", evidence)
        self.assertNotIn("/var/folders/", evidence)
        self.assertNotIn("/Library/Frameworks/", evidence)
        self.assertNotRegex(evidence, r"[A-Za-z]:\\Users\\")

    def test_reconstructed_017_evidence_does_not_claim_retroactive_qwen_certification(self) -> None:
        evidence = (ROOT / "docs/releases/0.17.0-evidence.md").read_text("utf-8")

        self.assertIn("d7d0dc592c3f3dbc78bc8c4e0edf58e880f8c711", evidence)
        self.assertIn("run 31121324133", evidence)
        self.assertIn("run 31121403493", evidence)
        self.assertIn("OpenAI | yes | yes", evidence)
        self.assertIn("Qwen | no | no", evidence)
        self.assertIn("no live Qwen 3.8 Max certification", evidence)
        self.assertIn("cannot turn that later observation into evidence that existed before publication", evidence)

    def test_release_evidence_can_sanitize_an_existing_file_without_rerunning_gates(self) -> None:
        script = ROOT / "scripts/collect_release_evidence.py"
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "evidence.md"
            output.write_text(
                "\n".join(
                    [
                        "# Evidence",
                        "",
                        "- Mode: `executed`",
                        "",
                        "| compile | passed | `make compile` |",
                        f"traceback: {ROOT}/src/zhivex_ai/agent.py",
                        "sha256: abc123",
                        "",
                    ]
                ),
                "utf-8",
            )
            process = subprocess.run(
                [sys.executable, str(script), "--sanitize-existing", "--output", str(output)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            evidence = output.read_text("utf-8")

        self.assertEqual(process.returncode, 0, process.stdout)
        self.assertIn(collect_release_evidence.PATH_SANITIZATION_NOTE, evidence)
        self.assertIn("| compile | passed | `make compile` |", evidence)
        self.assertIn("traceback: <repo>/src/zhivex_ai/agent.py", evidence)
        self.assertIn("sha256: abc123", evidence)
        self.assertNotIn(str(ROOT), evidence)

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
        for extra in ["dev", "mcp", "a2a"]:
            self.assertIn("cryptography>=50.0.0", extras[extra])

    def test_dependency_audit_resolves_and_audits_all_extras_without_reresolution(self) -> None:
        requirements = Path("/tmp/all-extras-requirements.txt")
        compile_command = audit_dependencies._compile_command(
            uv="uv",
            requirements=requirements,
            cache_dir=Path("/tmp/uv-cache"),
        )
        audit_command = audit_dependencies._audit_command(
            requirements=requirements,
            cache_dir=Path("/tmp/pip-audit-cache"),
        )

        self.assertIn("--all-extras", compile_command)
        self.assertIn("--universal", compile_command)
        self.assertIn("--generate-hashes", compile_command)
        self.assertIn("--quiet", compile_command)
        self.assertIn("--no-emit-package", compile_command)
        self.assertIn("zhivex-ai-sdk", compile_command)
        self.assertIn("--no-deps", audit_command)
        self.assertIn("--require-hashes", audit_command)
        self.assertIn("--disable-pip", audit_command)
        self.assertIn("--cache-dir", audit_command)

    def test_makefile_release_check_runs_install_verification(self) -> None:
        makefile = (ROOT / "Makefile").read_text("utf-8")
        self.assertIn("release-install-check:", makefile)
        self.assertIn("release-evidence:", makefile)
        self.assertIn("scripts/verify_release_artifacts.py", makefile)
        self.assertIn("RELEASE_ARTIFACT_FLAGS", makefile)
        self.assertIn("scripts/collect_release_evidence.py", makefile)
        self.assertIn("tests/test_agent_safety_runtime.py", makefile)
        self.assertIn("tests/test_tool_timeout_safety.py", makefile)
        self.assertIn("release-check: check test-release build release-install-check security-check", makefile)
        self.assertIn("rm -rf dist build", makefile)
        self.assertIn("security-check:", makefile)
        self.assertIn("scripts/audit_dependencies.py", makefile)
        self.assertNotIn("pip_audit . --strict", makefile)
        self.assertIn("pip_audit --local --skip-editable", makefile)

    def test_sdist_includes_root_docs_linked_from_packaged_docs(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
        wheel_force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
        sdist = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]
        include = set(sdist["include"])
        force_include = sdist["force-include"]

        self.assertEqual(wheel_force_include["src/zhivex_ai/py.typed"], "zhivex_ai/py.typed")
        self.assertEqual(wheel_force_include["src/zhivex_ai/__init__.pyi"], "zhivex_ai/__init__.pyi")

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
        self.assertIn("--require-postgres-workflow-smoke", ci)
        for workflow in [ci, publish, test_publish]:
            action_refs = re.findall(r"uses:\s+[^@\s]+@([^\s#]+)", workflow)
            self.assertTrue(action_refs)
            for action_ref in action_refs:
                self.assertRegex(action_ref, r"^[0-9a-f]{40}$")

        for workflow in [ci, publish, test_publish]:
            self.assertIn("persist-credentials: false", workflow)
        self.assertIn("make PYTHON=python security-check", ci)
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
            self.assertIn("ZHIVEX_SMOKE_PROVIDERS: openai,meta", workflow)
            self.assertIn("ZHIVEX_SMOKE_OPENAI_MODEL: gpt-5.6-luna", workflow)
            self.assertIn(
                "ZHIVEX_SMOKE_META_MODEL: muse-spark-1.2-contributor",
                workflow,
            )
            self.assertIn("ZHIVEX_SMOKE_META_CERTIFICATION: \"1\"", workflow)
            self.assertIn("docs/releases/0.20.0-smoke-policy.json", workflow)
            self.assertIn("ZHIVEX_SMOKE_ARTIFACT_PATH: dist", workflow)
            self.assertIn("release-smoke-evidence.json", workflow)
            self.assertIn("name: release-smoke-evidence", workflow)
            self.assertIn("python scripts/run_live_smoke.py", workflow)
            self.assertIn("needs: [build, live-agent-smoke]", workflow)

        self.assertIn("scripts/verify_release_tag.py", publish)
        for workflow in [publish, test_publish]:
            self.assertIn("fetch-depth: 0", workflow)
            self.assertIn('git merge-base --is-ancestor "$GITHUB_SHA" refs/remotes/origin/main', workflow)
            self.assertIn(
                "make PYTHON=python RELEASE_ARTIFACT_FLAGS=--require-postgres-workflow-smoke release-check",
                workflow,
            )
            self.assertIn(".[dev,postgres,mcp,api,a2a,ag-ui,otel,docx]", workflow)

        policy = json.loads(
            (ROOT / "docs/releases/0.20.0-smoke-policy.json").read_text("utf-8")
        )
        self.assertEqual(
            policy["required_providers"]["openai"]["model"],
            "gpt-5.6-luna",
        )
        self.assertEqual(
            policy["required_providers"]["meta"]["model"],
            "muse-spark-1.2-contributor",
        )
