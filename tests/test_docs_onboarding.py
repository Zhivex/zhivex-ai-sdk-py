from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]


class DocsOnboardingTests(TestCase):
    def test_examples_use_only_the_top_level_public_package(self) -> None:
        deep_imports: list[str] = []
        for path in sorted((ROOT / "examples").rglob("*.py")):
            tree = ast.parse(path.read_text("utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("zhivex_ai."):
                    deep_imports.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} imports {node.module}"
                    )
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("zhivex_ai."):
                            deep_imports.append(
                                f"{path.relative_to(ROOT)}:{node.lineno} imports {alias.name}"
                            )

        self.assertEqual(deep_imports, [])

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
            ".venv/bin/python examples/text/meta_text.py",
            "uvicorn examples.integrations.fastapi_chat_api:app --reload",
        ]:
            self.assertIn(command, examples)
        self.assertIn("## Verification Index", examples)

    def test_agent_guide_links_detailed_guides_and_has_minimal_tool_flow(self) -> None:
        guide = (ROOT / "docs/AGENTS.md").read_text("utf-8")

        for target in [
            "./agents/approvals.md",
            "./agents/durable-state.md",
            "./agents/tool-registries.md",
        ]:
            self.assertIn(target, guide)
        for symbol in ["AgentRunResult", "AgentStreamResult", "run_agent", "tool"]:
            self.assertIn(f"`{symbol}", guide)
        self.assertIn('export OPENAI_API_KEY="your-api-key"', guide)

    def test_filesystem_mcp_examples_are_pinned_scoped_and_policy_gated(self) -> None:
        sources = [
            (ROOT / "README.md").read_text("utf-8"),
            (ROOT / "examples/agents/mcp_tools.py").read_text("utf-8"),
        ]

        for source in sources:
            self.assertIn('command="bunx"', source)
            self.assertIn("@modelcontextprotocol/server-filesystem@2026.7.10", source)
            self.assertIn("approval_policy=", source)
            self.assertIn("READ_ONLY_FILESYSTEM_TOOLS", source)
            self.assertNotIn('command="npx"', source)
            self.assertNotIn('args=["-y", "@modelcontextprotocol/server-filesystem", "."]', source)

    def test_parity_matrix_tracks_required_maturity_columns(self) -> None:
        matrix = (ROOT / "docs/PARITY_MATRIX.md").read_text("utf-8")
        for label in ["Implemented", "Documented", "Offline-tested", "Live-smoked", "Stability"]:
            self.assertIn(label, matrix)
        self.assertIn("DeepSeek now participates in the tier-1 portable provider contract", matrix)
        self.assertIn("vLLM remains a tier-1 Python provider", matrix)
        data = json.loads((ROOT / "docs/parity_matrix.json").read_text("utf-8"))
        self.assertEqual(data["columns"], ["implemented", "documented", "offline_tested", "live_smoked", "stability"])
        by_area = {row["area"]: row for row in data["areas"]}
        for area in [
            "Agent evaluations and CI gates",
            "Agent protocols and hosting",
            "General CLI and local playground",
            "Security and operations guides",
        ]:
            self.assertIn(area, by_area)
            self.assertIn(f"| {area} |", matrix)
        workflow = by_area["Workflow orchestration and durable graphs"]
        for capability in ["resume/fork/cancel", "execution leases", "heartbeat", "fencing"]:
            self.assertIn(capability, workflow["implemented"])
            self.assertIn(capability, matrix)

    def test_current_release_docs_do_not_regress_to_the_previous_workflow_boundary(self) -> None:
        workflows = (ROOT / "docs/WORKFLOWS.md").read_text("utf-8")
        self.assertIn("`0.16.0` still does not provide an automatic checkpoint migration engine", workflows)
        self.assertNotIn("`0.15.0` does not provide an automatic checkpoint migration engine", workflows)

    def test_examples_readme_lists_the_complete_live_smoke_scope(self) -> None:
        examples = (ROOT / "examples/README.md").read_text("utf-8")
        expected = (
            "OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex, Qwen, Kimi, "
            "DeepSeek, vLLM, and optional local Ollama"
        )
        self.assertIn(expected, examples)

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
