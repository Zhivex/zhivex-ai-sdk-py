from __future__ import annotations

import unittest

from examples.agents.hr_candidate_selection_agent import (
    InMemoryHiringProcessStore,
    HiringProcessRecord,
    SAMPLE_BIAS_RISK_CANDIDATE,
    SAMPLE_BIAS_RISK_INTERVIEW_TRANSCRIPT,
    SAMPLE_INCOMPLETE_CANDIDATE,
    SAMPLE_JOB,
    SAMPLE_STRONG_CANDIDATE,
    SAMPLE_STRONG_INTERVIEW_TRANSCRIPT,
    finalize_recruiter_review,
    resume_hiring_process,
    start_hiring_process,
    submit_interview,
)
from zhivex_ai import create_in_memory_agent_run_store, replay_agent_run


class HrCandidateSelectionExampleTests(unittest.IsolatedAsyncioTestCase):
    async def test_strong_candidate_reaches_completed_recruiter_decision(self) -> None:
        process_store = InMemoryHiringProcessStore()
        run_store = create_in_memory_agent_run_store()

        started = await start_hiring_process(
            process_id="HR-2025-001",
            job_payload=SAMPLE_JOB,
            candidate_payload=SAMPLE_STRONG_CANDIDATE,
            process_store=process_store,
            run_store=run_store,
        )
        interviewed = await submit_interview(
            process_id="HR-2025-001",
            transcript=SAMPLE_STRONG_INTERVIEW_TRANSCRIPT,
            process_store=process_store,
            run_store=run_store,
        )
        finalized = await finalize_recruiter_review(
            process_id="HR-2025-001",
            approved=True,
            process_store=process_store,
            run_store=run_store,
        )

        self.assertEqual(started.record.status, "pending_interview")
        self.assertEqual(interviewed.record.status, "pending_recruiter_review")
        self.assertEqual(finalized.record.status, "completed")
        self.assertEqual(
            list(finalized.record.steps),
            [
                "job_description",
                "resume_intake",
                "screening",
                "interview_plan",
                "interview_evaluation",
                "recommendation",
                "fairness_review",
                "recruiter_decision",
            ],
        )
        self.assertTrue(finalized.record.steps["fairness_review"]["data"]["valid"])
        self.assertTrue(finalized.record.steps["recruiter_decision"]["data"]["approved"])

    async def test_incomplete_resume_repairs_then_resumes_to_completed(self) -> None:
        process_store = InMemoryHiringProcessStore()
        run_store = create_in_memory_agent_run_store()

        stopped = await start_hiring_process(
            process_id="HR-2025-002",
            job_payload=SAMPLE_JOB,
            candidate_payload=SAMPLE_INCOMPLETE_CANDIDATE,
            process_store=process_store,
            run_store=run_store,
        )

        self.assertEqual(stopped.record.status, "needs_repair")
        self.assertIn("Missing required field: years_experience", stopped.record.issues)
        self.assertNotIn("interview_plan", stopped.record.steps)
        self.assertNotIn("recommendation", stopped.record.steps)

        repaired = await process_store.apply_repair("HR-2025-002", {"years_experience": 6})
        self.assertEqual(repaired.status, "active")
        self.assertEqual(repaired.steps["resume_intake"]["data"]["years_experience"], 6)

        resumed = await resume_hiring_process(
            process_id="HR-2025-002",
            process_store=process_store,
            run_store=run_store,
        )
        interviewed = await submit_interview(
            process_id="HR-2025-002",
            transcript=SAMPLE_STRONG_INTERVIEW_TRANSCRIPT,
            process_store=process_store,
            run_store=run_store,
        )
        finalized = await finalize_recruiter_review(
            process_id="HR-2025-002",
            approved=True,
            process_store=process_store,
            run_store=run_store,
        )

        self.assertEqual(resumed.record.status, "pending_interview")
        self.assertEqual(interviewed.record.status, "pending_recruiter_review")
        self.assertEqual(finalized.record.status, "completed")

    async def test_bias_risk_recommendation_fails_fairness_review(self) -> None:
        process_store = InMemoryHiringProcessStore()
        run_store = create_in_memory_agent_run_store()

        started = await start_hiring_process(
            process_id="HR-2025-003",
            job_payload=SAMPLE_JOB,
            candidate_payload=SAMPLE_BIAS_RISK_CANDIDATE,
            process_store=process_store,
            run_store=run_store,
        )
        interviewed = await submit_interview(
            process_id="HR-2025-003",
            transcript=SAMPLE_BIAS_RISK_INTERVIEW_TRANSCRIPT,
            process_store=process_store,
            run_store=run_store,
        )

        self.assertEqual(started.record.status, "pending_interview")
        self.assertEqual(interviewed.record.status, "failed")
        self.assertFalse(interviewed.record.steps["fairness_review"]["data"]["valid"])
        self.assertNotIn("recruiter_decision", interviewed.record.steps)
        self.assertTrue(any("young" in issue for issue in interviewed.record.issues))

    async def test_store_contract_returns_copies_and_applies_repair(self) -> None:
        store = InMemoryHiringProcessStore()
        await store.save(HiringProcessRecord(process_id="HR-1", steps={"resume_intake": {"data": {}}}))

        loaded = await store.load("HR-1")
        self.assertIsNotNone(loaded)
        loaded.status = "failed"  # type: ignore[union-attr]
        loaded.steps["resume_intake"]["data"]["years_experience"] = 1  # type: ignore[union-attr]
        self.assertEqual((await store.load("HR-1")).status, "active")  # type: ignore[union-attr]
        self.assertNotIn("years_experience", (await store.load("HR-1")).steps["resume_intake"]["data"])  # type: ignore[union-attr]

        repaired = await store.apply_repair("HR-1", {"years_experience": 5})
        self.assertEqual(repaired.status, "active")
        self.assertEqual(repaired.steps["resume_intake"]["status"], "completed")
        self.assertEqual(repaired.steps["resume_intake"]["data"]["years_experience"], 5)

    async def test_workflow_trace_replay_contains_step_events(self) -> None:
        process_store = InMemoryHiringProcessStore()
        run_store = create_in_memory_agent_run_store()

        result = await start_hiring_process(
            process_id="HR-TRACE",
            job_payload=SAMPLE_JOB,
            candidate_payload=SAMPLE_STRONG_CANDIDATE,
            process_store=process_store,
            run_store=run_store,
        )

        snapshot = result.workflow_results[0].state_snapshot
        self.assertIsNotNone(snapshot)
        replay = replay_agent_run(snapshot)
        event_types = [event.type for event in replay.timeline]

        self.assertIn("workflow-step-finish", event_types)
        self.assertIn("run-finish", event_types)


if __name__ == "__main__":
    unittest.main()
