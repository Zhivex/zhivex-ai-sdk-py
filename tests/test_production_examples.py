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
        self.assertIn("Depends(require_api_identity)", source)
        self.assertIn("RequestBodyLimitMiddleware", source)
        self.assertIn("secrets.compare_digest", source)
        self.assertIn("ZHIVEX_AGENT_TABLE_PREFIX", source)
        self.assertIn("max_length=32_768", source)
        self.assertIn("content_length > self.max_body_bytes", source)
        self.assertIn("received_bytes > self.max_body_bytes", source)
        self.assertIn("status_code=429", source)

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

        protected_endpoints = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name in {"run_agent_endpoint", "list_run_approvals", "resolve_run_approval", "gateway_endpoint"}
        }
        self.assertEqual(len(protected_endpoints), 4)
        for endpoint in protected_endpoints.values():
            dependency_names = {
                argument.id
                for call in ast.walk(endpoint)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "Depends"
                for argument in call.args
                if isinstance(argument, ast.Name)
            }
            self.assertIn("require_api_identity", dependency_names)

        agent_request = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentRequest")
        agent_request_fields = {
            statement.target.id
            for statement in agent_request.body
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
        }
        self.assertEqual(agent_request_fields, {"prompt", "idempotency_key"})

    async def test_worker_resume_example_runs_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = await run_worker_resume_demo(str(Path(directory) / "worker.sqlite3"))

        self.assertEqual(summary.first_run_id, summary.reused_run_id)
        self.assertEqual(summary.resumed_text, "resumed job completed")
        self.assertEqual(summary.replay_status, "completed")


if __name__ == "__main__":
    unittest.main()
