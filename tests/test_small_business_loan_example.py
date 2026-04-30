from __future__ import annotations

import unittest

from examples.agents.small_business_loan_agent import (
    InMemoryLoanProcessStore,
    LoanProcessRecord,
    SAMPLE_COMPLETE_APPLICATION,
    SAMPLE_INCOMPLETE_APPLICATION,
    finalize_loan_process,
    resume_loan_process,
    start_loan_process,
)
from zhivex_ai import create_in_memory_agent_run_store, replay_agent_run


class SmallBusinessLoanExampleTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_application_reaches_completed_decision(self) -> None:
        process_store = InMemoryLoanProcessStore()
        run_store = create_in_memory_agent_run_store()

        started = await start_loan_process(
            process_id="SBL-2025-00142",
            application_payload=SAMPLE_COMPLETE_APPLICATION,
            process_store=process_store,
            run_store=run_store,
        )
        finalized = await finalize_loan_process(
            process_id="SBL-2025-00142",
            approved=True,
            process_store=process_store,
            run_store=run_store,
        )

        self.assertEqual(started.record.status, "pending_approval")
        self.assertEqual(finalized.record.status, "completed")
        self.assertEqual(
            list(finalized.record.steps),
            ["document_extraction", "underwriting", "pricing", "loan_decision", "validation"],
        )
        self.assertTrue(finalized.record.steps["loan_decision"]["data"]["approved"])
        self.assertEqual(finalized.record.issues, [])

    async def test_incomplete_application_stops_then_repairs_and_resumes(self) -> None:
        process_store = InMemoryLoanProcessStore()
        run_store = create_in_memory_agent_run_store()

        stopped = await start_loan_process(
            process_id="SBL-2025-00391",
            application_payload=SAMPLE_INCOMPLETE_APPLICATION,
            process_store=process_store,
            run_store=run_store,
        )

        self.assertEqual(stopped.record.status, "needs_repair")
        self.assertIn("Missing required field: loan_amount_requested", stopped.record.issues)
        self.assertNotIn("pricing", stopped.record.steps)
        self.assertNotIn("loan_decision", stopped.record.steps)

        repaired = await process_store.apply_repair("SBL-2025-00391", {"loan_amount_requested": 150_000})
        self.assertEqual(repaired.status, "active")
        self.assertEqual(repaired.steps["document_extraction"]["data"]["loan_amount_requested"], 150_000)

        resumed = await resume_loan_process(
            process_id="SBL-2025-00391",
            process_store=process_store,
            run_store=run_store,
        )
        finalized = await finalize_loan_process(
            process_id="SBL-2025-00391",
            approved=True,
            process_store=process_store,
            run_store=run_store,
        )

        self.assertEqual(resumed.record.status, "pending_approval")
        self.assertEqual(finalized.record.status, "completed")
        self.assertIn("underwriting", finalized.record.steps)
        self.assertIn("pricing", finalized.record.steps)

    async def test_store_contract_returns_copies_and_applies_repair(self) -> None:
        store = InMemoryLoanProcessStore()
        await store.save(LoanProcessRecord(process_id="SBL-1", steps={"document_extraction": {"data": {}}}))

        loaded = await store.load("SBL-1")
        self.assertIsNotNone(loaded)
        loaded.status = "failed"  # type: ignore[union-attr]
        loaded.steps["document_extraction"]["data"]["loan_amount_requested"] = 1  # type: ignore[union-attr]
        self.assertEqual((await store.load("SBL-1")).status, "active")  # type: ignore[union-attr]
        self.assertNotIn("loan_amount_requested", (await store.load("SBL-1")).steps["document_extraction"]["data"])  # type: ignore[union-attr]

        repaired = await store.apply_repair("SBL-1", {"loan_amount_requested": 200_000})
        self.assertEqual(repaired.status, "active")
        self.assertEqual(repaired.steps["document_extraction"]["status"], "completed")
        self.assertEqual(repaired.steps["document_extraction"]["data"]["loan_amount_requested"], 200_000)

    async def test_workflow_trace_replay_contains_step_events(self) -> None:
        process_store = InMemoryLoanProcessStore()
        run_store = create_in_memory_agent_run_store()

        result = await start_loan_process(
            process_id="SBL-TRACE",
            application_payload=SAMPLE_COMPLETE_APPLICATION,
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
