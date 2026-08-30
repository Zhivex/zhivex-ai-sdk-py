from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OperationsDocsTests(unittest.TestCase):
    def test_phase8_docs_exist_and_are_linked(self) -> None:
        expected = [
            "SECURITY.md",
            "docs/OPERATIONS.md",
            "docs/OBSERVABILITY.md",
            "docs/PRODUCTION.md",
        ]
        readme = (ROOT / "README.md").read_text("utf-8")
        support = (ROOT / "SUPPORT.md").read_text("utf-8")
        for relative in expected:
            self.assertTrue((ROOT / relative).exists(), relative)
        for relative in ["SECURITY.md", "docs/OPERATIONS.md", "docs/OBSERVABILITY.md"]:
            self.assertIn(relative, readme)
        self.assertIn("SECURITY.md", support)

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

    def test_observability_docs_define_terminal_gateway_attempt_contract(self) -> None:
        text = (ROOT / "docs/OBSERVABILITY.md").read_text("utf-8")
        for phrase in [
            'phase="finished"',
            "terminal=True",
            "attemptId",
            "errorType",
            "exactly one terminal payload",
            "Observer exceptions are non-authoritative",
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
        for phrase in [
            "Secrets",
            "Data Retention",
            "Tool Execution",
            "MCP And Hosted Tools",
            "remote MCP",
            "filesystem",
            "shell-like",
            "code execution",
            "human approval",
            "redaction",
        ]:
            self.assertIn(phrase, security)

    def test_fastapi_examples_do_not_return_raw_provider_errors(self) -> None:
        for relative in [
            "examples/integrations/fastapi_chat_api.py",
            "examples/integrations/fastapi_streaming_api.py",
            "examples/integrations/fastapi_gateway_api.py",
        ]:
            source = (ROOT / relative).read_text("utf-8")
            self.assertNotIn('"message": str(error)', source, relative)
            self.assertIn('"message": "Upstream provider request failed."', source, relative)
            self.assertIn('"provider_status": error.status', source, relative)

    def test_makefile_docs_target_includes_phase8_docs(self) -> None:
        makefile = (ROOT / "Makefile").read_text("utf-8")
        self.assertIn("tests/test_docs_onboarding.py", makefile)
        self.assertIn("tests/test_operations_docs.py", makefile)


if __name__ == "__main__":
    unittest.main()
