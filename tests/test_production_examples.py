from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

from examples.production.worker_resume import run_worker_resume_demo


ROOT = Path(__file__).resolve().parents[1]


class ProductionExampleTests(unittest.IsolatedAsyncioTestCase):
    def test_fastapi_agent_api_example_is_parseable(self) -> None:
        source = (ROOT / "examples/production/fastapi_agent_api.py").read_text("utf-8")
        tree = ast.parse(source)
        self.assertIn("fail_on_missing_adapter=True", source)
        self.assertIn("DATABASE_URL is required", source)
        self.assertIn("idempotency_key", source)

        approval_response = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ApprovalResponse"
        )
        approval_fields = {
            statement.target.id
            for statement in approval_response.body
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
        }
        self.assertEqual(
            {"approval_id", "tool_name", "reason", "permissions", "created_at_ms"},
            approval_fields,
        )

        list_approvals = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "list_run_approvals"
        )
        resolve_approval = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "resolve_run_approval"
        )
        list_approval_names = {node.id for node in ast.walk(list_approvals) if isinstance(node, ast.Name)}
        resolve_approval_names = {node.id for node in ast.walk(resolve_approval) if isinstance(node, ast.Name)}
        self.assertIn("get_pending_agent_approvals", list_approval_names)
        self.assertIn("resume_agent_run", resolve_approval_names)

    async def test_worker_resume_example_runs_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = await run_worker_resume_demo(str(Path(directory) / "worker.sqlite3"))

        self.assertEqual(summary.first_run_id, summary.reused_run_id)
        self.assertEqual(summary.resumed_text, "resumed job completed")
        self.assertEqual(summary.replay_status, "completed")


if __name__ == "__main__":
    unittest.main()
