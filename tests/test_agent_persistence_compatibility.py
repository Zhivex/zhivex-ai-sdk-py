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
