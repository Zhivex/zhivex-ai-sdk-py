from __future__ import annotations

import asyncio
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import Agent
from zhivex_ai.evals import GenerateResult, create_mock_language_model
from zhivex_ai.workflows import SequentialAgent, WorkflowStep


@dataclass(slots=True)
class DocumentWorkflowSummary:
    title: str
    artifact_name: str
    artifact_preview: str
    state_keys: list[str]


def _agent(name: str, text: str) -> Agent:
    return Agent(
        name=name,
        model=create_mock_language_model(responses=[GenerateResult(text=text, finish_reason="stop")]),
    )


async def run_artifact_document_workflow_demo() -> DocumentWorkflowSummary:
    workflow = SequentialAgent(
        name="document_pipeline",
        steps=[
            WorkflowStep("outline", _agent("outliner", "Apollo migration report"), prompt="Create outline.", output_key="title"),
            WorkflowStep(
                "draft",
                _agent("writer", "Status: green\nRisk: low\nNext: migrate billing worker"),
                input_template="Draft report for {title}.",
                output_key="draft",
            ),
            WorkflowStep(
                "review",
                _agent("reviewer", "approved"),
                input_template="Review report titled {title}.",
                output_key="review",
            ),
        ],
    )

    result = await workflow.run()
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_path = Path(temp_dir) / "apollo-migration-report.md"
        artifact_path.write_text(f"# {result.state['title']}\n\n{result.state['draft']}\n", encoding="utf-8")
        preview = artifact_path.read_text("utf-8").splitlines()[0]

    return DocumentWorkflowSummary(
        title=str(result.state["title"]),
        artifact_name="apollo-migration-report.md",
        artifact_preview=preview,
        state_keys=sorted(result.state),
    )


async def main() -> None:
    summary = await run_artifact_document_workflow_demo()
    print(
        {
            "title": summary.title,
            "artifact_name": summary.artifact_name,
            "artifact_preview": summary.artifact_preview,
            "state_keys": summary.state_keys,
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
