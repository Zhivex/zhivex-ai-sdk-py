from __future__ import annotations

import unittest

from examples.integrations.operations_hardening import run_operations_hardening_demo


class OperationsHardeningExampleTests(unittest.IsolatedAsyncioTestCase):
    async def test_operations_hardening_example_runs_offline(self) -> None:
        summary = await run_operations_hardening_demo()

        self.assertEqual(summary.request_id, "req_offline_001")
        self.assertEqual(summary.session_id, "sess_offline_001")
        self.assertEqual(summary.run_id, "run_offline_001")
        self.assertIn("[REDACTED]", summary.redacted_prompt)
        self.assertNotIn("user@example.com", summary.redacted_prompt)
        self.assertTrue(summary.budget_blocked)
        self.assertTrue(summary.retryable_error)
        self.assertEqual(summary.retry_after_ms, 250)
        self.assertEqual(summary.telemetry_events, ["generate-start", "generate-error", "generate-start", "generate-finish"])
        self.assertEqual(summary.circuit_transitions, ["open", "half-open", "closed"])
        self.assertEqual(summary.response_text, "request correlated and guarded")
        self.assertEqual(summary.agent_safety_policy, "review_sensitive")


if __name__ == "__main__":
    unittest.main()
