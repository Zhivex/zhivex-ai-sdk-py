from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from zhivex_ai import (
    Agent,
    WorkflowBuilder,
    WorkflowCheckpointStore,
    WorkflowGraph,
    WorkflowStep,
    create_mock_language_model,
    create_sqlite_workflow_checkpoint_store,
    fork_workflow,
    resume_workflow,
)
from zhivex_ai.types import GenerateResult


@dataclass(slots=True)
class DurableGraphSummary:
    original_run_id: str
    initial_status: str
    resumed_status: str
    resumed_decision: str
    forked_run_id: str
    forked_from_run_id: str
    forked_status: str
    forked_decision: str
    original_checkpoints: int
    forked_checkpoints: int


def _agent(name: str, text: str) -> Agent:
    return Agent(
        name=name,
        model=create_mock_language_model(
            responses=[GenerateResult(text=text, finish_reason="stop")],
        ),
    )


def _build_workflow(
    store: WorkflowCheckpointStore,
    *,
    intake_text: str,
    review_text: str,
    decision_text: str,
) -> WorkflowGraph:
    return (
        WorkflowBuilder("durable_loan_review", definition_version="0.15.0")
        .add_step(
            WorkflowStep("intake", _agent("intake_agent", intake_text), output_key="application"),
            entrypoint=True,
        )
        .add_step(WorkflowStep("review", _agent("review_agent", review_text), output_key="review"))
        .add_step(WorkflowStep("decide", _agent("decision_agent", decision_text), output_key="decision"))
        .add_edge("intake", "review")
        .add_edge("review", "decide")
        .interrupt_before("review", reason="A human must approve the extracted application.")
        .build(checkpoint_store=store)
    )


async def run_durable_graph_workflow_demo() -> DurableGraphSummary:
    with tempfile.TemporaryDirectory(prefix="zhivex-workflow-") as directory:
        database_path = str(Path(directory) / "workflow-checkpoints.sqlite3")
        store = create_sqlite_workflow_checkpoint_store(database_path, namespace="demo")
        first_graph = _build_workflow(
            store,
            intake_text="application-extracted",
            review_text="unused-before-interrupt",
            decision_text="unused-before-interrupt",
        )
        suspended = await first_graph.run(
            prompt="Review the application.",
            idempotency_key="loan-review-001",
        )
        assert suspended.checkpoint is not None
        assert suspended.checkpoint.pending_interrupt is not None
        source_checkpoint_id = suspended.checkpoint.checkpoint_id
        interrupt_id = suspended.checkpoint.pending_interrupt.interrupt_id

        # A new store and graph instance model a new worker process using the
        # same persisted SQLite checkpoint history.
        reopened_store = create_sqlite_workflow_checkpoint_store(database_path, namespace="demo")
        resumed_graph = _build_workflow(
            reopened_store,
            intake_text="unused-completed-step",
            review_text="review-approved",
            decision_text="offer-approved",
        )
        resumed = await resume_workflow(
            resumed_graph,
            suspended.run_id,
            interrupt_id=interrupt_id,
            resume_value={"approved": True, "reviewer": "human-1"},
        )

        # Forking preserves completed work and lineage but creates a new run.
        fork_graph = _build_workflow(
            reopened_store,
            intake_text="unused-completed-step",
            review_text="review-conservative",
            decision_text="manual-review",
        )
        forked = await fork_workflow(
            fork_graph,
            suspended.run_id,
            checkpoint_id=source_checkpoint_id,
            state_updates={"scenario": "conservative"},
            idempotency_key="loan-review-001-conservative",
        )
        assert forked.checkpoint is not None
        assert forked.checkpoint.pending_interrupt is not None
        forked_resumed = await resume_workflow(
            fork_graph,
            forked.run_id,
            interrupt_id=forked.checkpoint.pending_interrupt.interrupt_id,
            resume_value={"approved": True, "reviewer": "human-2"},
        )

        original_history = await reopened_store.list_checkpoints(suspended.run_id)
        forked_history = await reopened_store.list_checkpoints(forked.run_id)
        return DurableGraphSummary(
            original_run_id=suspended.run_id,
            initial_status=suspended.status,
            resumed_status=resumed.status,
            resumed_decision=str(resumed.state["decision"]),
            forked_run_id=forked.run_id,
            forked_from_run_id=str(forked_resumed.forked_from_run_id),
            forked_status=forked_resumed.status,
            forked_decision=str(forked_resumed.state["decision"]),
            original_checkpoints=len(original_history),
            forked_checkpoints=len(forked_history),
        )


async def main() -> None:
    summary = await run_durable_graph_workflow_demo()
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
