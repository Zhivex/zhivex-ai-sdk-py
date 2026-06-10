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
        ast.parse(source)
        self.assertIn("fail_on_missing_adapter=True", source)
        self.assertIn("DATABASE_URL is required", source)
        self.assertIn("idempotency_key", source)

    async def test_worker_resume_example_runs_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = await run_worker_resume_demo(str(Path(directory) / "worker.sqlite3"))

        self.assertEqual(summary.first_run_id, summary.reused_run_id)
        self.assertEqual(summary.resumed_text, "resumed job completed")
        self.assertEqual(summary.replay_status, "completed")


if __name__ == "__main__":
    unittest.main()
