"""Offline installed-package smoke covers the new public architecture contracts."""
from __future__ import annotations

import ast
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from scripts import verify_release_artifacts


class InstalledArchitectureSmokeTests(TestCase):
    def test_architecture_smoke_executes_offline_durable_deadline_and_retention_flows(self) -> None:
        code = verify_release_artifacts._architecture_smoke_code()
        ast.parse(code)
        # An accidental provider request must fail this offline release gate.
        with patch("httpx.AsyncClient.send", side_effect=AssertionError("artifact smoke must remain offline")):
            exec(compile(code, "<installed-architecture-smoke>", "exec"), {})

    def test_base_gate_executes_architecture_smoke_with_artifact_interpreter(self) -> None:
        python = Path("/isolated-artifact/bin/python")
        commands = []
        with patch.object(verify_release_artifacts, "_run", side_effect=commands.append):
            verify_release_artifacts._run_base_smoke(python, expected_version=verify_release_artifacts._package_version())
        architecture_commands = [
            command for command in commands
            if len(command) == 4 and command[1:3] == ["-I", "-c"] and "installed-architecture-ok" in command[3]
        ]
        self.assertEqual(len(architecture_commands), 1)
        self.assertEqual(architecture_commands[0][0], str(python))

    def test_postgres_gate_exercises_owned_and_borrowed_agent_pool_lifecycles(self) -> None:
        code = verify_release_artifacts._postgres_workflow_smoke_code()
        ast.parse(code)
        for operation in (
            "create_postgres_agent_memory_store", "create_postgres_checkpoint_store",
            "create_postgres_agent_run_store", "await owned_runs.close()",
            "await memory.close()", "await agent_checkpoints.close()", "await runs.close()",
            "await borrowed_connection.fetchval", '"agent_memory"', '"agent_checkpoints"', '"runs"',
        ):
            self.assertIn(operation, code)
        self.assertIn("async with create_postgres_agent_run_store", code)
        self.assertIn('persisted.status == "completed"', code)
        self.assertIn('restored_agent.output_text == "postgres-agent-ok"', code)
