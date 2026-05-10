from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import zhivex_ai
from zhivex_ai.api_stability import STABLE_EXPORTS


DOCUMENTED_STABLE_EXPORTS = {
    "create_openai",
    "create_anthropic",
    "create_azure_openai",
    "create_gemini",
    "create_vertex",
    "create_vllm",
    "generate_text",
    "stream_text",
    "generate_object",
    "stream_object",
    "generate_grounded_text",
    "embed",
    "embed_many",
    "embed_content",
    "embed_content_many",
    "Agent",
    "AgentRegistry",
    "AgentRuntime",
    "AgentSession",
    "AgentSkillActivatedEvent",
    "AgentSkillSkippedEvent",
    "ToolRegistry",
    "run_agent",
    "stream_agent",
    "resume_agent",
    "create_agent_session",
    "load_agent_session",
    "SkillDefinition",
    "SkillDependency",
    "SkillRegistry",
    "skill",
    "load_skill",
    "discover_skills",
    "set_agent_session_skills",
    "get_agent_session_skills",
    "clear_agent_session_skills",
    "create_postgres_agent_memory_store",
    "create_postgres_checkpoint_store",
    "discover_mcp_tools",
    "mcp_stdio_server",
    "mcp_http_server",
    "create_mcp_tool_registry",
    "GatewayAttempt",
    "GatewayConfig",
    "GatewayError",
    "GatewayImageAttachment",
    "GatewayMessage",
    "GatewayModelTarget",
    "GatewayObjectResponse",
    "GatewayResponse",
    "create_gateway",
    "ProviderHTTPError",
    "ConfigurationError",
    "ValidationError",
    "UnsupportedFeatureError",
    "HTTPResponse",
    "stream_sse",
    "to_sse_response",
    "to_sse_stream",
    "to_text_stream",
    "to_text_stream_response",
    "to_ui_message_stream_response",
}


class PublicContractTests(TestCase):
    def test_stable_exports_are_available_from_top_level_package(self) -> None:
        exported = set(zhivex_ai.__all__)
        self.assertTrue(DOCUMENTED_STABLE_EXPORTS.issubset(exported))

    def test_stability_doc_lists_the_stable_exports(self) -> None:
        stability = (ROOT / "STABILITY.md").read_text("utf-8")
        for symbol in sorted(DOCUMENTED_STABLE_EXPORTS):
            self.assertIn(f"`{symbol}`", stability)

    def test_stable_manifest_matches_public_contract(self) -> None:
        self.assertEqual(DOCUMENTED_STABLE_EXPORTS, set(STABLE_EXPORTS))

    def test_core_docs_link_to_each_other(self) -> None:
        readme = (ROOT / "README.md").read_text("utf-8")
        stability = (ROOT / "STABILITY.md").read_text("utf-8")
        support = (ROOT / "SUPPORT.md").read_text("utf-8")
        versioning = (ROOT / "VERSIONING.md").read_text("utf-8")
        changelog = (ROOT / "CHANGELOG.md").read_text("utf-8")

        self.assertIn("[STABILITY.md](./STABILITY.md)", readme)
        self.assertIn("[VERSIONING.md](./VERSIONING.md)", readme)
        self.assertIn("[SUPPORT.md](./SUPPORT.md)", readme)
        self.assertIn("[CHANGELOG.md](./CHANGELOG.md)", readme)

        self.assertIn("[README.md](./README.md)", stability)
        self.assertIn("[VERSIONING.md](./VERSIONING.md)", stability)
        self.assertIn("[CHANGELOG.md](./CHANGELOG.md)", stability)

        self.assertIn("[README.md](./README.md)", support)
        self.assertIn("[STABILITY.md](./STABILITY.md)", support)
        self.assertIn("[VERSIONING.md](./VERSIONING.md)", support)
        self.assertIn("[CHANGELOG.md](./CHANGELOG.md)", support)

        self.assertIn("[README.md](./README.md)", versioning)
        self.assertIn("[STABILITY.md](./STABILITY.md)", versioning)
        self.assertIn("[CHANGELOG.md](./CHANGELOG.md)", versioning)

        self.assertIn("[README.md](./README.md)", changelog)
        self.assertIn("[STABILITY.md](./STABILITY.md)", changelog)
        self.assertIn("[VERSIONING.md](./VERSIONING.md)", changelog)

    def test_readme_keeps_realtime_marked_as_experimental(self) -> None:
        readme = (ROOT / "README.md").read_text("utf-8")
        self.assertIn("Experimental realtime/live voice sessions plus `stream_live_agent()`", readme)

    def test_beta_package_signal_is_consistent_in_metadata_and_docs(self) -> None:
        readme = (ROOT / "README.md").read_text("utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text("utf-8")

        self.assertIn("beta package", readme)
        self.assertIn('version = "0.7.0"', pyproject)
        self.assertIn('Development Status :: 4 - Beta', pyproject)

    def test_readme_mentions_beta_packaged_skills_and_docx_extra(self) -> None:
        readme = (ROOT / "README.md").read_text("utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text("utf-8")

        self.assertIn("beta skill-package layer", readme)
        self.assertIn('pip install "zhivex-ai-sdk[docx]"', readme)
        self.assertIn('zhivex-skills = "zhivex_ai.skill_cli:main"', pyproject)
