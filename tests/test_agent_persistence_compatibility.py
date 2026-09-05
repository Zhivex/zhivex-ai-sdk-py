"""Characterize data emitted by the actual pre-extraction 0.22.0 wheel."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
from unittest import IsolatedAsyncioTestCase

from zhivex_ai import create_sqlite_agent_memory_store, create_sqlite_checkpoint_store
from zhivex_ai._agent_persistence import (
    _deserialize_agent_checkpoint,
    _deserialize_agent_memory_state,
    _serialize_agent_checkpoint,
    _serialize_agent_memory_state,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/agent_persistence_0_22.json").read_text())


class AgentPersistenceCompatibilityTests(IsolatedAsyncioTestCase):
    def test_previous_wheel_serialization_roundtrips_without_format_drift(self):
        self.assertEqual(FIXTURE["source_package_version"], "0.22.0")
        memory = _deserialize_agent_memory_state(FIXTURE["memory"])
        checkpoint = _deserialize_agent_checkpoint(FIXTURE["checkpoint"])
        self.assertEqual(_serialize_agent_memory_state(memory), FIXTURE["memory"])
        self.assertEqual(_serialize_agent_checkpoint(checkpoint), FIXTURE["checkpoint"])
        self.assertIsNone(checkpoint.response.raw_response)
        self.assertFalse(checkpoint.is_final)

    async def test_current_stores_load_pre_extraction_sqlite_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = str(Path(temporary) / "legacy.sqlite")
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE zhivex_agent_memory (namespace TEXT, session_id TEXT, state_json TEXT, updated_at_ms INTEGER, PRIMARY KEY(namespace, session_id))")
                connection.execute("INSERT INTO zhivex_agent_memory VALUES (?, ?, ?, ?)", ("default", "session-compat", json.dumps(FIXTURE["memory"]), 1234))
                connection.execute("CREATE TABLE zhivex_agent_checkpoints (namespace TEXT, run_id TEXT, session_id TEXT, step_index INTEGER, saved_at_ms INTEGER, is_final INTEGER, checkpoint_json TEXT)")
                connection.execute("INSERT INTO zhivex_agent_checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)", ("default", "run-compat", "session-compat", 2, 1234, 0, json.dumps(FIXTURE["checkpoint"])))
            memory = await create_sqlite_agent_memory_store(database).load("session-compat")
            checkpoint = await create_sqlite_checkpoint_store(database).get_latest(session_id="session-compat")
            self.assertEqual(memory.summary, "Apollo checkpoint")
            self.assertEqual(_serialize_agent_checkpoint(checkpoint), FIXTURE["checkpoint"])

    async def test_typed_tool_approval_survives_recreated_store_and_executes_once(self):
        from datetime import date
        from pydantic import BaseModel, ConfigDict
        from zhivex_ai import (
            Agent, ApprovalDecision, ModelMessage, ToolCall, ValidationError,
            create_sqlite_agent_run_store, resume_agent_run, run_agent, tool, tool_call_part,
        )
        from zhivex_ai.evals import GenerateResult, create_mock_language_model

        class Input(BaseModel):
            model_config = ConfigDict(extra="forbid")
            project: str
            due: date

        Input.model_rebuild(_types_namespace={"date": date})
        executions = []

        async def review(request):
            self.assertIsInstance(request.tool_input, Input)
            return ApprovalDecision.require_human(approval_id="typed-approval")

        def execute(data):
            self.assertIsInstance(data, Input)
            executions.append(data)
            return {"project": data.project}

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "state.sqlite3")
            definition = tool(name="lookup", schema=Input, execute=execute, requires_approval=True)
            agent = Agent(name="typed", run_store=create_sqlite_agent_run_store(path),
                          approval_policy=review, tools={"lookup": definition},
                          model=create_mock_language_model(responses=[GenerateResult(
                              messages=[ModelMessage(role="assistant", parts=[tool_call_part(ToolCall(
                                  id="typed-call", name="lookup", input={"project": "Apollo", "due": "2026-09-05"}
                              ))])], finish_reason="tool-calls")]))
            pending = await run_agent(agent=agent, prompt="synthetic")
            self.assertEqual(pending.state.status, "suspended")
            self.assertEqual(executions, [])
            restored = await create_sqlite_agent_run_store(path).load(pending.run_id)
            self.assertEqual(restored.pending_approvals[0].arguments,
                             {"project": "Apollo", "due": "2026-09-05"})
            resumed_agent = Agent(name="typed", run_store=create_sqlite_agent_run_store(path),
                                  approval_policy=review, tools={"lookup": definition},
                                  model=create_mock_language_model(responses=[GenerateResult(text="done", finish_reason="stop")]))
            resumed = await resume_agent_run(agent=resumed_agent, run_id=pending.run_id, approval_id="typed-approval")
            self.assertEqual(resumed.text, "done")
            self.assertEqual(len(executions), 1)
            self.assertEqual(executions[0].due, date(2026, 9, 5))
            with self.assertRaises(ValidationError):
                await resume_agent_run(agent=resumed_agent, run_id=pending.run_id, approval_id="typed-approval")
            self.assertEqual(len(executions), 1)
