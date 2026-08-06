from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any

from zhivex_ai import Agent, WorkflowStep, create_mock_language_model
from zhivex_ai.types import GenerateResult
from zhivex_ai.workflow_graph import (
    WorkflowBuilder,
    WorkflowGraph,
    fork_workflow,
    resume_workflow,
)
from zhivex_ai.workflow_state import (
    InMemoryWorkflowCheckpointStore,
    create_in_memory_workflow_checkpoint_store,
    deserialize_workflow_checkpoint,
    serialize_workflow_checkpoint,
    workflow_checkpoint_to_json,
)


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "workflow_checkpoint_v0_15_schema_v1.json"
)
V0_15_DEFINITION_DIGEST = (
    "90974584f4e6b21883eb94d9fad8c0c9d8c697a0af85bc7c212e07188f5d0587"
)


def _load_fixture() -> dict[str, Any]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("Workflow compatibility fixture must be a JSON object.")
    return payload


def _agent_with_text(text: str) -> Agent:
    return Agent(
        name="v015-publisher",
        model=create_mock_language_model(
            responses=[GenerateResult(text=text, finish_reason="stop")]
        ),
    )


def _workflow(store: InMemoryWorkflowCheckpointStore, *, output: str) -> WorkflowGraph:
    return (
        WorkflowBuilder("compatibility-workflow", definition_version="2026-07-31")
        .add_step(
            WorkflowStep("publish", _agent_with_text(output), output_key="published"),
            entrypoint=True,
        )
        .interrupt_before("publish", reason="Published v0.15 review")
        .build(checkpoint_store=store)
    )


async def _restore_fixture_history(
    store: InMemoryWorkflowCheckpointStore,
) -> tuple[str, str]:
    fixture = _load_fixture()
    checkpoints = fixture["checkpoints"]
    for payload in checkpoints:
        checkpoint = deserialize_workflow_checkpoint(payload)
        await store.append(
            checkpoint,
            expected_sequence=checkpoint.sequence - 1 if checkpoint.sequence else None,
        )
    latest = checkpoints[-1]
    return latest["run_id"], latest["checkpoint_id"]


class WorkflowV015CheckpointSerializationCompatibilityTests(unittest.TestCase):
    def test_deserializes_and_canonically_reserializes_published_v015_schema_v1(
        self,
    ) -> None:
        fixture = _load_fixture()
        self.assertEqual(fixture["producer"], "zhivex-ai-sdk==0.15.0")
        self.assertEqual(fixture["checkpoint_schema_version"], 1)

        latest_payload = fixture["checkpoints"][-1]
        for payload in fixture["checkpoints"]:
            self.assertEqual(payload["schema_version"], 1)
            self.assertNotIn("execution_lease", payload["metadata"])
            for node in payload["nodes"].values():
                self.assertNotIn("suspension", node)

        restored = deserialize_workflow_checkpoint(latest_payload)

        self.assertEqual(restored.schema_version, 1)
        self.assertEqual(restored.status, "suspended")
        self.assertEqual(restored.state, {"prepared": "prepared-by-v015"})
        self.assertIsNone(restored.nodes["publish"].suspension)

        expected_current_payload = copy.deepcopy(latest_payload)
        expected_current_payload["nodes"]["publish"]["suspension"] = None
        self.assertEqual(
            serialize_workflow_checkpoint(restored), expected_current_payload
        )
        self.assertEqual(
            workflow_checkpoint_to_json(restored),
            json.dumps(
                expected_current_payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )


class WorkflowV015CheckpointRuntimeCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_runtime_resumes_v015_checkpoint_history(self) -> None:
        store = create_in_memory_workflow_checkpoint_store()
        run_id, _ = await _restore_fixture_history(store)
        graph = _workflow(store, output="published-by-current-runtime")
        latest = await store.load_latest(run_id)

        self.assertEqual(graph.definition_digest, V0_15_DEFINITION_DIGEST)
        assert latest is not None
        assert latest.pending_interrupt is not None
        interrupt_id = latest.pending_interrupt.interrupt_id

        resumed = await resume_workflow(
            graph,
            run_id,
            interrupt_id=interrupt_id,
            resume_value={"approved": True},
        )

        self.assertEqual(resumed.status, "completed")
        self.assertEqual(resumed.state["prepared"], "prepared-by-v015")
        self.assertEqual(resumed.state["published"], "published-by-current-runtime")
        assert resumed.checkpoint is not None
        self.assertEqual(
            resumed.checkpoint.resume_values[interrupt_id],
            {"approved": True},
        )

    async def test_current_runtime_forks_v015_checkpoint_history(self) -> None:
        store = create_in_memory_workflow_checkpoint_store()
        run_id, checkpoint_id = await _restore_fixture_history(store)
        graph = _workflow(store, output="published-on-current-fork")

        forked = await fork_workflow(
            graph,
            run_id,
            checkpoint_id=checkpoint_id,
            state_updates={"branch": "compatibility-test"},
            idempotency_key="v015-current-runtime-fork",
        )

        self.assertEqual(forked.status, "suspended")
        self.assertNotEqual(forked.run_id, run_id)
        self.assertEqual(forked.forked_from_run_id, run_id)
        assert forked.checkpoint is not None
        self.assertEqual(forked.checkpoint.forked_from_checkpoint_id, checkpoint_id)
        self.assertEqual(forked.state["prepared"], "prepared-by-v015")
        self.assertEqual(forked.state["branch"], "compatibility-test")
        assert forked.checkpoint.pending_interrupt is not None

        resumed_fork = await resume_workflow(
            graph,
            forked.run_id,
            interrupt_id=forked.checkpoint.pending_interrupt.interrupt_id,
            resume_value={"approved": True},
        )

        self.assertEqual(resumed_fork.status, "completed")
        self.assertEqual(resumed_fork.state["published"], "published-on-current-fork")
        self.assertEqual(resumed_fork.forked_from_run_id, run_id)


if __name__ == "__main__":
    unittest.main()
