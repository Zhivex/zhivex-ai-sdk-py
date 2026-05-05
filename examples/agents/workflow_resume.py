from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    Agent,
    AgentSession,
    SequentialAgent,
    WorkflowStep,
    create_agent_session,
    create_in_memory_agent_run_store,
    create_mock_language_model,
)
from zhivex_ai.agent_state import AgentRunStore
from zhivex_ai.types import GenerateResult, JsonValue


@dataclass(slots=True)
class WorkflowRecord:
    process_id: str
    state: dict[str, JsonValue] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)
    workflow_run_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WorkflowResumeSummary:
    completed_steps: list[str]
    decision: str
    workflow_runs: int


def _agent(name: str, text: str) -> Agent:
    return Agent(
        name=name,
        model=create_mock_language_model(responses=[GenerateResult(text=text, finish_reason="stop")]),
    )


def _remaining_steps(record: WorkflowRecord) -> list[WorkflowStep]:
    candidates = [
        WorkflowStep("validate", _agent("validator", "valid"), input_template="Validate {application}.", output_key="validation"),
        WorkflowStep(
            "decide",
            _agent("decider", "approved"),
            input_template="Decide with {application} and {validation}.",
            output_key="decision",
        ),
    ]
    return [step for step in candidates if step.name not in record.completed_steps]


async def _run_remaining(record: WorkflowRecord, *, run_store: AgentRunStore) -> WorkflowRecord:
    session: AgentSession = create_agent_session(state=dict(record.state))
    steps = _remaining_steps(record)
    if not steps:
        return record

    workflow = SequentialAgent(name="resume_pipeline", steps=steps, run_store=run_store)
    result = await workflow.run(session=session)
    record.state = dict(result.state)
    record.completed_steps.extend(item.name for item in result.step_results if item.status == "completed")
    record.workflow_run_ids.append(result.run_id)
    return record


async def run_workflow_resume_demo() -> WorkflowResumeSummary:
    run_store = create_in_memory_agent_run_store()
    record = WorkflowRecord(
        process_id="WF-RESUME-1",
        state={"application": "Apollo Tools requests working capital."},
        completed_steps=["extract"],
    )

    await _run_remaining(record, run_store=run_store)
    await _run_remaining(record, run_store=run_store)

    return WorkflowResumeSummary(
        completed_steps=list(record.completed_steps),
        decision=str(record.state["decision"]),
        workflow_runs=len(record.workflow_run_ids),
    )


async def main() -> None:
    summary = await run_workflow_resume_demo()
    print(
        {
            "completed_steps": summary.completed_steps,
            "decision": summary.decision,
            "workflow_runs": summary.workflow_runs,
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
