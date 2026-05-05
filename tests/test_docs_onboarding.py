from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]


class DocsOnboardingTests(TestCase):
    def test_onboarding_docs_exist_and_are_linked_from_readme(self) -> None:
        expected = [
            "docs/QUICKSTART.md",
            "docs/PROVIDERS.md",
            "docs/AGENTS.md",
            "docs/WORKFLOWS.md",
            "docs/GATEWAY.md",
            "docs/OBSERVABILITY.md",
            "docs/TROUBLESHOOTING.md",
            "docs/PARITY_MATRIX.md",
            "docs/MIGRATING_FROM_TYPESCRIPT.md",
            "PRODUCTION_APIS.md",
            "CONTRIBUTING.md",
            ".env.example",
        ]
        readme = (ROOT / "README.md").read_text("utf-8")
        for relative in expected:
            self.assertTrue((ROOT / relative).exists(), relative)
        for relative in expected[:9]:
            self.assertIn(relative, readme)

    def test_env_example_covers_live_smoke_environment_variables(self) -> None:
        smoke_source = (ROOT / "scripts/run_live_smoke.py").read_text("utf-8")
        env_example = (ROOT / ".env.example").read_text("utf-8")
        names = sorted(set(re.findall(r'os\.getenv\("([A-Z0-9_]+)"', smoke_source)))
        ignored = {"PATH"}
        missing = [name for name in names if name not in ignored and f"{name}=" not in env_example]
        self.assertEqual(missing, [])

    def test_examples_readme_has_verification_index_for_key_examples(self) -> None:
        examples = (ROOT / "examples/README.md").read_text("utf-8")
        for command in [
            ".venv/bin/python examples/agents/structured_workflow_outputs.py",
            ".venv/bin/python examples/agents/workflow_resume.py",
            ".venv/bin/python examples/agents/artifact_document_workflow.py",
            ".venv/bin/python examples/agents/research_report_workflow.py",
            ".venv/bin/python examples/text/tier1_providers.py",
            "uvicorn examples.integrations.fastapi_chat_api:app --reload",
        ]:
            self.assertIn(command, examples)
        self.assertIn("## Verification Index", examples)

    def test_parity_matrix_tracks_required_maturity_columns(self) -> None:
        matrix = (ROOT / "docs/PARITY_MATRIX.md").read_text("utf-8")
        for label in ["Implemented", "Documented", "Offline-tested", "Live-smoked", "Stability"]:
            self.assertIn(label, matrix)
        self.assertIn("DeepSeek is deferred from Python GA", matrix)
        self.assertIn("vLLM remains a tier-1 Python provider", matrix)
        data = json.loads((ROOT / "docs/parity_matrix.json").read_text("utf-8"))
        self.assertEqual(data["columns"], ["implemented", "documented", "offline_tested", "live_smoked", "stability"])
        self.assertIn("Security and operations guides", {row["area"] for row in data["areas"]})

    def test_docs_do_not_reference_missing_local_markdown_files(self) -> None:
        markdown_files = [
            *ROOT.glob("*.md"),
            *ROOT.glob("docs/**/*.md"),
            ROOT / "examples/README.md",
        ]
        missing: list[str] = []
        link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+\.md)(?:#[^)]+)?\)")
        for path in markdown_files:
            text = path.read_text("utf-8")
            for raw_target in link_pattern.findall(text):
                if raw_target.startswith(("http://", "https://")):
                    continue
                target = (path.parent / raw_target).resolve()
                if not target.exists():
                    missing.append(f"{path.relative_to(ROOT)} -> {raw_target}")
        self.assertEqual(missing, [])

    def test_makefile_exposes_docs_target(self) -> None:
        makefile = (ROOT / "Makefile").read_text("utf-8")
        self.assertIn("test-docs:", makefile)
        self.assertIn("tests/test_docs_onboarding.py", makefile)

    def test_smoke_env_names_can_be_parsed_from_script(self) -> None:
        source = (ROOT / "scripts/run_live_smoke.py").read_text("utf-8")
        ast.parse(source)
