from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import zhivex_ai
from zhivex_ai.api_stability import BETA_EXPORTS, STABLE_EXPORTS
from zhivex_ai.errors import ValidationError, WorkflowConflictError
from zhivex_ai.workflows import __all__ as WORKFLOW_EXPORTS


DOCUMENTED_STABLE_EXPORTS = {
    "HTTPTransport",
    "aclose_default_clients",
    "AgentCheckpoint",
    "AgentTrace",
    "EmbedOutput",
    "EmbeddingContent",
    "EmbeddingModel",
    "FinishReason",
    "GenerateGroundedTextOutput",
    "GenerateObjectOutput",
    "GenerateTextOutput",
    "GroundedLanguageModel",
    "LanguageModel",
    "ModelMessage",
    "StreamEvent",
    "StreamObjectResult",
    "StreamTextResult",
    "TokenUsage",
    "ToolCall",
    "create_openai",
    "create_anthropic",
    "create_azure_openai",
    "create_deepseek",
    "create_gemini",
    "create_kimi",
    "create_meta",
    "create_qwen",
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
    "AgentContext",
    "AgentHooks",
    "AgentHandoff",
    "AgentMiddleware",
    "AgentMiddlewareNext",
    "AgentObserver",
    "AgentRegistry",
    "AgentReplayEvent",
    "AgentReplayResult",
    "AgentRuntime",
    "AgentSession",
    "AgentChildRun",
    "AgentRunResult",
    "AgentRunRequest",
    "AgentEventDeliveryError",
    "AgentRunCancelled",
    "AgentRunSnapshot",
    "AgentRunState",
    "AgentRunStatus",
    "AgentRunStep",
    "AgentRunStore",
    "AgentRunTreeCancellationResult",
    "AgentStreamResult",
    "AgentSkillActivatedEvent",
    "AgentSkillSkippedEvent",
    "AgentToolApprovalEvent",
    "ApprovalDecision",
    "DynamicInstructions",
    "PendingApproval",
    "ToolApprovalRequest",
    "ToolDefinition",
    "ToolExecutionContext",
    "ToolExecutionError",
    "ToolExecutionOptions",
    "ToolExecutionResult",
    "ToolRegistry",
    "ToolSet",
    "handoff_to",
    "tool",
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
    "create_postgres_agent_run_store",
    "create_postgres_checkpoint_store",
    "PostgresAgentRunStore",
    "serialize_agent_run_state",
    "deserialize_agent_run_state",
    "agent_run_state_to_json",
    "agent_run_state_from_json",
    "cancel_agent_run",
    "cancel_agent_run_tree",
    "create_agent_run_snapshot",
    "replay_agent_run",
    "get_pending_agent_approvals",
    "resume_agent_run",
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
    "ToolExecutionOutcomeUnknown",
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

WORKFLOW_STABLE_EXPORTS = {
    name
    for name in WORKFLOW_EXPORTS
    if name
    not in {
        "create_dbos_workflow_adapter",
        "create_prefect_workflow_adapter",
        "create_restate_workflow_adapter",
        "create_temporal_workflow_adapter",
    }
}
DOCUMENTED_STABLE_EXPORTS |= WORKFLOW_STABLE_EXPORTS | {"JsonValue"}

COMPATIBILITY_ROOT_EXPORTS_SHA256 = "deb06b95a40a64967058ee7ca8690354f4d8504f62e94b22b527bb1e23172cd7"


class PublicContractTests(TestCase):
    def test_package_root_does_not_expand_without_an_explicit_contract_update(self) -> None:
        payload = "\n".join(sorted(zhivex_ai.__all__)).encode("utf-8")

        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            COMPATIBILITY_ROOT_EXPORTS_SHA256,
            "Keep new Beta and Experimental APIs in focused namespaces. If the root must change, "
            "review STABILITY.md, VERSIONING.md, CHANGELOG.md, the public stub, and this snapshot together.",
        )

    def test_meta_factory_is_stable_while_native_helpers_remain_lazy_beta_exports(self) -> None:
        meta_exports = {
            "create_meta",
            "meta_hosted_tool",
            "meta_tool_search_tool",
            "meta_web_search_tool",
        }
        sys.modules.pop("zhivex_ai.providers.meta", None)
        for name in meta_exports:
            zhivex_ai.__dict__.pop(name, None)

        self.assertTrue(meta_exports.issubset(zhivex_ai.__all__))
        self.assertIn("create_meta", STABLE_EXPORTS)
        self.assertTrue((meta_exports - {"create_meta"}).issubset(BETA_EXPORTS))
        self.assertNotIn("zhivex_ai.providers.meta", sys.modules)

        factory = zhivex_ai.create_meta

        self.assertIn("zhivex_ai.providers.meta", sys.modules)
        self.assertIs(factory, sys.modules["zhivex_ai.providers.meta"].create_meta)
        for name in meta_exports - {"create_meta"}:
            self.assertIs(getattr(zhivex_ai, name), getattr(sys.modules["zhivex_ai.providers.meta"], name))

    def test_example_fixture_types_keep_their_documented_stability(self) -> None:
        beta_fixture_types = {"GenerateResult", "ModelGenerateInput"}

        self.assertTrue((beta_fixture_types | {"JsonValue"}).issubset(zhivex_ai.__all__))
        self.assertTrue(beta_fixture_types.issubset(BETA_EXPORTS))
        self.assertIn("JsonValue", STABLE_EXPORTS)
        for name in beta_fixture_types | {"JsonValue"}:
            self.assertIsNotNone(getattr(zhivex_ai, name))

    def test_workflow_errors_are_typed_stable_top_level_exports(self) -> None:
        workflow_errors = {
            "WorkflowConflictError",
            "WorkflowDefinitionMismatchError",
            "WorkflowInterruptError",
            "WorkflowLeaseLostError",
            "WorkflowRunNotFoundError",
        }

        self.assertTrue(workflow_errors.issubset(zhivex_ai.__all__))
        self.assertTrue(workflow_errors.issubset(STABLE_EXPORTS))
        for name in workflow_errors:
            error_type = getattr(zhivex_ai, name)
            self.assertTrue(issubclass(error_type, ValidationError))
            self.assertIs(error_type, getattr(sys.modules["zhivex_ai.errors"], name))
        self.assertTrue(issubclass(zhivex_ai.WorkflowLeaseLostError, WorkflowConflictError))

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
        self.assertIn('version = "0.23.0"', pyproject)
        self.assertIn('Development Status :: 4 - Beta', pyproject)

    def test_readme_mentions_beta_packaged_skills_and_docx_extra(self) -> None:
        readme = (ROOT / "README.md").read_text("utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text("utf-8")

        self.assertIn("beta skill-package layer", readme)
        self.assertIn('pip install "zhivex-ai-sdk[docx]"', readme)
        self.assertIn('zhivex-skills = "zhivex_ai.skill_cli:main"', pyproject)
