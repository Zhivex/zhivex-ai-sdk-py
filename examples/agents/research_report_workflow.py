from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (
    Agent,
    ParallelAgent,
    SequentialAgent,
    WorkflowStep,
    create_agent_session,
    create_agent_trace_artifact,
    create_in_memory_agent_run_store,
    create_mock_language_model,
    replay_agent_run,
)
from zhivex_ai.types import GenerateResult


@dataclass(slots=True)
class ResearchReportSummary:
    report: str
    research_keys: list[str]
    replay_events: list[str]
    trace_preview: str


def _agent(name: str, text: str) -> Agent:
    return Agent(
        name=name,
        model=create_mock_language_model(responses=[GenerateResult(text=text, finish_reason="stop")]),
    )


async def run_research_report_workflow_demo() -> ResearchReportSummary:
    run_store = create_in_memory_agent_run_store()
    session = create_agent_session()
    research = ParallelAgent(
        name="research_fanout",
        run_store=run_store,
        steps=[
            WorkflowStep("market", _agent("market_researcher", "Demand is up 12%."), prompt="Research market.", output_key="market"),
            WorkflowStep("risk", _agent("risk_researcher", "Supply risk is low."), prompt="Research risk.", output_key="risk"),
        ],
    )
    synthesis = SequentialAgent(
        name="report_synthesis",
        run_store=run_store,
        steps=[
            WorkflowStep(
                "report",
                _agent("reporter", "Apollo expansion should proceed."),
                input_template="Write report using {market} and {risk}.",
                output_key="report",
            )
        ],
    )

    research_result = await research.run(session=session)
    report_result = await synthesis.run(session=research_result.session)
    replay = replay_agent_run(report_result.state_snapshot)
    trace = create_agent_trace_artifact(report_result.state_snapshot)

    return ResearchReportSummary(
        report=str(report_result.state["report"]),
        research_keys=sorted(key for key in report_result.state if key in {"market", "risk"}),
        replay_events=[event.type for event in replay.timeline],
        trace_preview=trace.output_preview,
    )


async def main() -> None:
    summary = await run_research_report_workflow_demo()
    print(
        {
            "report": summary.report,
            "research_keys": summary.research_keys,
            "replay_events": summary.replay_events,
            "trace_preview": summary.trace_preview,
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
