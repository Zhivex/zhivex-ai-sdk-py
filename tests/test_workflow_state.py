from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zhivex_ai.errors import ValidationError
from zhivex_ai.workflow_state import (
    WORKFLOW_CHECKPOINT_SCHEMA_VERSION,
    InMemoryWorkflowCheckpointStore,
    PostgresWorkflowCheckpointStore,
    SQLiteWorkflowCheckpointStore,
    WorkflowCheckpoint,
    WorkflowInterrupt,
    WorkflowNodeCheckpoint,
    WorkflowTransition,
    create_in_memory_workflow_checkpoint_store,
    create_postgres_workflow_checkpoint_store,
    create_sqlite_workflow_checkpoint_store,
    deserialize_workflow_checkpoint,
    serialize_workflow_checkpoint,
    workflow_checkpoint_from_json,
    workflow_checkpoint_to_json,
)


def checkpoint(
    *,
    checkpoint_id: str = "cp-0",
    run_id: str = "run-1",
    sequence: int = 0,
    idempotency_key: str | None = "workflow-request-1",
    status: str = "running",
) -> WorkflowCheckpoint:
    return WorkflowCheckpoint(
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        workflow_name="loan-review",
        definition_version="2026-07-31",
        definition_digest="sha256:definition",
        sequence=sequence,
        status=status,  # type: ignore[arg-type]
        session_id="session-1",
        parent_run_id="parent-1",
        idempotency_key=idempotency_key,
        state={"amount": 125_000, "customer": {"name": "Apollo"}},
        nodes={
            "intake": WorkflowNodeCheckpoint(
                node_name="intake",
                status="completed",
                attempt=1,
                idempotency_key="run-1:intake",
                child_run_id="agent-run-1",
                output={"rating": "low"},
                started_at_ms=10,
                finished_at_ms=20,
                metadata={"provider": "mock"},
            ),
            "decision": WorkflowNodeCheckpoint(node_name="decision"),
        },
        edge_decisions={"intake->decision": True, "intake->reject": False},
        ready_nodes=["decision"],
        pending_interrupt=WorkflowInterrupt(
            interrupt_id="interrupt-1",
            node_name="decision",
            reason="manager review",
            payload={"amount": 125_000},
            created_at_ms=25,
            phase="before",
            metadata={"queue": "credit"},
        ),
        transition=WorkflowTransition(
            type="workflow-interrupted",
            at_ms=25,
            node_name="decision",
            from_status="running",
            to_status="suspended",
            detail={"reason": "manager review"},
        ),
        forked_from_run_id="source-run",
        forked_from_checkpoint_id="source-cp",
        resume_values={"interrupt-0": {"approved": True}},
        created_at_ms=25,
        updated_at_ms=25,
        metadata={"tenant": "acme"},
    )


class WorkflowCheckpointSerializationTests(unittest.TestCase):
    def test_round_trip_preserves_full_checkpoint(self) -> None:
        original = checkpoint()

        payload = serialize_workflow_checkpoint(original)
        restored = deserialize_workflow_checkpoint(payload)
        from_json = workflow_checkpoint_from_json(workflow_checkpoint_to_json(original))

        self.assertEqual(restored, original)
        self.assertEqual(from_json, original)
        self.assertEqual(restored.pending_interrupt.phase, "before")
        self.assertEqual(restored.nodes["intake"].output, {"rating": "low"})

    def test_rejects_future_schema_versions(self) -> None:
        payload = serialize_workflow_checkpoint(checkpoint())
        payload["schema_version"] = WORKFLOW_CHECKPOINT_SCHEMA_VERSION + 1

        with self.assertRaisesRegex(ValidationError, "future schema_version"):
            deserialize_workflow_checkpoint(payload)

    def test_rejects_node_key_drift_and_non_boolean_edges(self) -> None:
        mismatched = checkpoint()
        mismatched.nodes = {"other": WorkflowNodeCheckpoint(node_name="intake")}
        with self.assertRaisesRegex(ValidationError, "does not match"):
            serialize_workflow_checkpoint(mismatched)

        invalid_edge = checkpoint()
        invalid_edge.edge_decisions = {"intake->decision": 1}  # type: ignore[dict-item]
        with self.assertRaisesRegex(ValidationError, "must be a boolean"):
            serialize_workflow_checkpoint(invalid_edge)

    def test_rejects_non_object_json(self) -> None:
        with self.assertRaisesRegex(ValidationError, "JSON object"):
            workflow_checkpoint_from_json("[]")

    def test_requires_complete_fork_lineage(self) -> None:
        fork = checkpoint()
        fork.forked_from_checkpoint_id = None
        with self.assertRaisesRegex(ValidationError, "fork lineage"):
            serialize_workflow_checkpoint(fork)

    def test_rejects_runtime_objects_and_non_finite_numbers_recursively(self) -> None:
        runtime_value = checkpoint()
        runtime_value.state["client"] = object()  # type: ignore[assignment]
        with self.assertRaisesRegex(ValidationError, "non-JSON value"):
            serialize_workflow_checkpoint(runtime_value)

        non_finite = checkpoint()
        non_finite.nodes["intake"].metadata["score"] = float("nan")
        with self.assertRaisesRegex(ValidationError, "NaN or infinity"):
            workflow_checkpoint_to_json(non_finite)


class InMemoryWorkflowCheckpointStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_append_load_history_and_idempotency_lookup(self) -> None:
        store = create_in_memory_workflow_checkpoint_store()
        self.assertIsInstance(store, InMemoryWorkflowCheckpointStore)
        initial = await store.append(checkpoint())
        updated_checkpoint = checkpoint(checkpoint_id="cp-1", sequence=1, status="suspended")
        updated_checkpoint.state["review"] = "waiting"
        updated = await store.append(updated_checkpoint, expected_sequence=0)

        latest = await store.load_latest("run-1")
        loaded = await store.load_checkpoint("cp-0")
        by_key = await store.find_by_idempotency_key("workflow-request-1")
        history = await store.list_checkpoints("run-1")

        self.assertEqual(initial.sequence, 0)
        self.assertEqual(updated.sequence, 1)
        self.assertEqual(latest, updated)
        self.assertEqual(loaded.sequence, 0)
        self.assertEqual(by_key, updated)
        self.assertEqual([item.sequence for item in history], [0, 1])

        latest.state["mutated"] = True
        self.assertNotIn("mutated", (await store.load_latest("run-1")).state)

    async def test_append_requires_compare_and_swap_sequence(self) -> None:
        store = create_in_memory_workflow_checkpoint_store()
        await store.append(checkpoint())

        with self.assertRaisesRegex(ValidationError, "expected_sequence is required"):
            await store.append(checkpoint(checkpoint_id="cp-1", sequence=1))
        with self.assertRaisesRegex(ValidationError, "sequence conflict"):
            await store.append(checkpoint(checkpoint_id="cp-1", sequence=1), expected_sequence=2)
        with self.assertRaisesRegex(ValidationError, "next checkpoint sequence"):
            await store.append(checkpoint(checkpoint_id="cp-2", sequence=2), expected_sequence=0)

        self.assertEqual(len(await store.list_checkpoints("run-1")), 1)

    async def test_idempotency_key_cannot_be_claimed_by_another_run(self) -> None:
        store = create_in_memory_workflow_checkpoint_store()
        await store.append(checkpoint())

        with self.assertRaisesRegex(ValidationError, "already claimed"):
            await store.append(checkpoint(checkpoint_id="other-cp", run_id="run-2"))

    async def test_terminal_run_cannot_be_appended(self) -> None:
        store = create_in_memory_workflow_checkpoint_store()
        await store.append(checkpoint(status="completed"))

        with self.assertRaisesRegex(ValidationError, "is terminal"):
            await store.append(
                checkpoint(checkpoint_id="cp-1", sequence=1, status="completed"),
                expected_sequence=0,
            )


class SQLiteWorkflowCheckpointStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_survives_reinstantiation_and_preserves_append_only_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "workflow.sqlite3")
            first = create_sqlite_workflow_checkpoint_store(path, namespace="tenant-a")
            self.assertIsInstance(first, SQLiteWorkflowCheckpointStore)
            await first.append(checkpoint())
            await first.append(
                checkpoint(checkpoint_id="cp-1", sequence=1, status="completed"),
                expected_sequence=0,
            )

            reopened = create_sqlite_workflow_checkpoint_store(path, namespace="tenant-a")
            latest = await reopened.load_latest("run-1")
            original = await reopened.load_checkpoint("cp-0")
            by_key = await reopened.find_by_idempotency_key("workflow-request-1")
            history = await reopened.list_checkpoints("run-1")

            self.assertEqual(latest.sequence, 1)
            self.assertEqual(latest.status, "completed")
            self.assertEqual(original.sequence, 0)
            self.assertEqual(by_key.sequence, 1)
            self.assertEqual([item.sequence for item in history], [0, 1])

    async def test_namespace_isolation_and_cas_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "workflow.sqlite3")
            tenant_a = create_sqlite_workflow_checkpoint_store(path, namespace="tenant-a")
            tenant_b = create_sqlite_workflow_checkpoint_store(path, namespace="tenant-b")
            await tenant_a.append(checkpoint())

            self.assertIsNone(await tenant_b.load_latest("run-1"))
            with self.assertRaisesRegex(ValidationError, "sequence conflict"):
                await tenant_a.append(checkpoint(checkpoint_id="cp-1", sequence=1), expected_sequence=1)
            self.assertEqual((await tenant_a.load_latest("run-1")).sequence, 0)

    async def test_idempotency_constraint_is_durable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "workflow.sqlite3")
            store = create_sqlite_workflow_checkpoint_store(path)
            await store.append(checkpoint())

            with self.assertRaisesRegex(ValidationError, "uniqueness constraint"):
                await store.append(checkpoint(checkpoint_id="other-cp", run_id="run-2"))


class PostgresWorkflowCheckpointStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_asyncpg_import_is_deferred_until_first_operation(self) -> None:
        store = create_postgres_workflow_checkpoint_store("postgres://example")
        self.assertIsInstance(store, PostgresWorkflowCheckpointStore)

        with patch.dict(sys.modules, {"asyncpg": None}):
            with self.assertRaisesRegex(RuntimeError, "optional dependency"):
                await store.load_latest("run-1")

    def test_rejects_unsafe_table_prefix(self) -> None:
        with self.assertRaisesRegex(ValidationError, "SQL identifier"):
            create_postgres_workflow_checkpoint_store("postgres://example", table_prefix="bad-prefix")


if __name__ == "__main__":
    unittest.main()
