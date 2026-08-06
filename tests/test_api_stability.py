from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import zhivex_ai
from zhivex_ai.api_stability import (
    API_STABILITY,
    BETA_EXPORTS,
    EXPERIMENTAL_EXPORTS,
    STABLE_EXPORTS,
)


KNOWN_LEVELS = {"stable", "beta", "experimental"}
KNOWN_CATEGORIES = {
    "agent",
    "catalog",
    "errors",
    "foundation",
    "gateway",
    "messages",
    "middleware",
    "observability",
    "provider",
    "provider-support",
    "protocol",
    "realtime",
    "safety",
    "skills",
    "transport",
    "types",
    "ui",
    "workflow",
}

TRANSITIVE_STABLE_EXPORTS = {
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
}


class ApiStabilityTests(TestCase):
    def test_inevitable_transitive_contracts_are_stable(self) -> None:
        self.assertTrue(TRANSITIVE_STABLE_EXPORTS.issubset(STABLE_EXPORTS))
        for name in TRANSITIVE_STABLE_EXPORTS:
            self.assertEqual(API_STABILITY[name].level, "stable")

    def test_workflow_errors_are_beta_error_entries(self) -> None:
        workflow_errors = {
            "WorkflowConflictError",
            "WorkflowDefinitionMismatchError",
            "WorkflowInterruptError",
            "WorkflowLeaseLostError",
            "WorkflowRunNotFoundError",
        }

        for name in workflow_errors:
            self.assertEqual(API_STABILITY[name].level, "beta")
            self.assertEqual(API_STABILITY[name].category, "errors")

    def test_every_public_export_has_stability_metadata(self) -> None:
        self.assertEqual(set(zhivex_ai.__all__), set(API_STABILITY))

    def test_manifest_sets_are_explicit_and_non_overlapping(self) -> None:
        stable = set(STABLE_EXPORTS)
        beta = set(BETA_EXPORTS)
        experimental = set(EXPERIMENTAL_EXPORTS)

        self.assertFalse(stable & beta)
        self.assertFalse(stable & experimental)
        self.assertFalse(beta & experimental)
        self.assertEqual(set(zhivex_ai.__all__), stable | beta | experimental)

    def test_manifest_uses_known_levels_and_categories(self) -> None:
        for entry in API_STABILITY.values():
            self.assertIn(entry.level, KNOWN_LEVELS)
            self.assertIn(entry.category, KNOWN_CATEGORIES)

    def test_manifest_does_not_mask_missing_module_mappings(self) -> None:
        for name, entry in API_STABILITY.items():
            self.assertEqual(entry.name, name)
            self.assertIn(name, zhivex_ai._EXPORTS)

    def test_stable_exports_are_documented_in_stability_doc(self) -> None:
        stability = (ROOT / "STABILITY.md").read_text("utf-8")
        for symbol in sorted(STABLE_EXPORTS):
            self.assertIn(f"`{symbol}`", stability)

    def test_experimental_exports_are_not_listed_as_stable(self) -> None:
        stability = (ROOT / "STABILITY.md").read_text("utf-8")
        stable_section = stability.split("## Beta", 1)[0]

        for symbol in sorted(EXPERIMENTAL_EXPORTS):
            self.assertNotIn(f"`{symbol}`", stable_section)

    def test_stability_doc_points_to_manifest_as_drift_gate(self) -> None:
        stability = (ROOT / "STABILITY.md").read_text("utf-8")
        versioning = (ROOT / "VERSIONING.md").read_text("utf-8")

        self.assertIn("api_stability.py", stability)
        self.assertIn("api_stability.py", versioning)
