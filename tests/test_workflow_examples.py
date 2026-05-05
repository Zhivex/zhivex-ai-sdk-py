from __future__ import annotations

import unittest

from examples.agents.artifact_document_workflow import run_artifact_document_workflow_demo
from examples.agents.research_report_workflow import run_research_report_workflow_demo
from examples.agents.structured_workflow_outputs import run_structured_workflow_demo
from examples.agents.workflow_resume import run_workflow_resume_demo


class WorkflowExampleTests(unittest.IsolatedAsyncioTestCase):
    async def test_structured_outputs_are_validated_by_application_code(self) -> None:
        summary = await run_structured_workflow_demo()

        self.assertEqual(summary.company, "Apollo Tools")
        self.assertEqual(summary.rating, "low")
        self.assertEqual(summary.max_offer, 125_000)
        self.assertEqual(summary.state_keys, ["intake_json", "risk_json"])

    async def test_workflow_resume_runs_only_remaining_steps(self) -> None:
        summary = await run_workflow_resume_demo()

        self.assertEqual(summary.completed_steps, ["extract", "validate", "decide"])
        self.assertEqual(summary.decision, "approved")
        self.assertEqual(summary.workflow_runs, 1)

    async def test_artifact_document_workflow_keeps_file_storage_app_owned(self) -> None:
        summary = await run_artifact_document_workflow_demo()

        self.assertEqual(summary.title, "Apollo migration report")
        self.assertEqual(summary.artifact_name, "apollo-migration-report.md")
        self.assertEqual(summary.artifact_preview, "# Apollo migration report")
        self.assertEqual(summary.state_keys, ["draft", "review", "title"])

    async def test_research_report_workflow_replays_synthesis_trace(self) -> None:
        summary = await run_research_report_workflow_demo()

        self.assertEqual(summary.report, "Apollo expansion should proceed.")
        self.assertEqual(summary.research_keys, ["market", "risk"])
        self.assertIn("workflow-step-finish", summary.replay_events)
        self.assertIn("Apollo expansion should proceed.", summary.trace_preview)


if __name__ == "__main__":
    unittest.main()
