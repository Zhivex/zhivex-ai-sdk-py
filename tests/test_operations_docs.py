from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OperationsDocsTests(unittest.TestCase):
    def test_phase8_docs_exist_and_are_linked(self) -> None:
        expected = [
            "SECURITY.md",
            "docs/OPERATIONS.md",
            "docs/THREAT_MODEL.md",
            "docs/OBSERVABILITY.md",
            "docs/PRODUCTION.md",
            "docs/RC_READINESS.md",
        ]
        readme = (ROOT / "README.md").read_text("utf-8")
        support = (ROOT / "SUPPORT.md").read_text("utf-8")
        for relative in expected:
            self.assertTrue((ROOT / relative).exists(), relative)
        for relative in ["SECURITY.md", "docs/OPERATIONS.md", "docs/THREAT_MODEL.md", "docs/OBSERVABILITY.md"]:
            self.assertIn(relative, readme)
        self.assertIn("SECURITY.md", support)

    def test_rc_readiness_doc_records_final_release_gate(self) -> None:
        text = (ROOT / "docs/RC_READINESS.md").read_text("utf-8")
        for phrase in [
            "Required RC Evidence",
            "make check",
            "make release-check",
            "1.0.0 Gate",
            "Beta Areas After RC",
            "src/zhivex_ai/api_stability.py",
            "docs/RELEASE_EVIDENCE.md",
        ]:
            self.assertIn(phrase, text)

    def test_observability_docs_standardize_correlation_and_otel(self) -> None:
        text = (ROOT / "docs/OBSERVABILITY.md").read_text("utf-8")
        for phrase in [
            "OpenTelemetry",
            "request_id",
            "session_id",
            "run_id",
            "gateway_attempt_id",
            "ProviderHTTPError.retryable",
            "ProviderHTTPError.retry_after_ms",
            "TokenUsage",
            "examples/integrations/operations_hardening.py",
        ]:
            self.assertIn(phrase, text)

    def test_operations_docs_cover_failure_and_runtime_patterns(self) -> None:
        text = (ROOT / "docs/OPERATIONS.md").read_text("utf-8")
        for phrase in [
            "Retry And Backoff",
            "Circuit Breakers",
            "Provider Error Normalization",
            "Cost And Budgets",
            "Concurrency And Cancellation",
            "Serverless And Workers",
            "retry_after_ms",
            "budget guards",
            "redacted provider error details",
        ]:
            self.assertIn(phrase, text)

    def test_security_docs_cover_risky_capabilities(self) -> None:
        security = (ROOT / "SECURITY.md").read_text("utf-8")
        threat_model = (ROOT / "docs/THREAT_MODEL.md").read_text("utf-8")
        combined = f"{security}\n{threat_model}"
        for phrase in [
            "Secrets",
            "Data Retention",
            "Tool Execution",
            "MCP And Hosted Tools",
            "remote MCP",
            "file access",
            "shell-like",
            "code execution",
            "human approval",
            "redaction",
        ]:
            self.assertIn(phrase, combined)

    def test_makefile_docs_target_includes_phase8_docs(self) -> None:
        makefile = (ROOT / "Makefile").read_text("utf-8")
        self.assertIn("tests/test_docs_onboarding.py", makefile)
        self.assertIn("tests/test_operations_docs.py", makefile)


if __name__ == "__main__":
    unittest.main()
