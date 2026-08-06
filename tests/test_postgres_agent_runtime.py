from __future__ import annotations

import asyncio
import os
import re
import uuid
from dataclasses import replace
from unittest import IsolatedAsyncioTestCase, skipUnless

from zhivex_ai import (
    Agent,
    AgentRunCancelled,
    AgentRunState,
    ApprovalDecision,
    PendingApproval,
    ToolApprovalRequest,
    WorkflowCheckpoint,
    WorkflowBuilder,
    WorkflowConflictError,
    WorkflowLeaseLostError,
    WorkflowStep,
    WorkflowTransition,
    cancel_agent_run,
    cancel_workflow,
    create_agent_session,
    create_mock_language_model,
    create_postgres_agent_memory_store,
    create_postgres_agent_run_store,
    create_postgres_checkpoint_store,
    create_postgres_workflow_checkpoint_store,
    create_postgres_workflow_lease_manager,
    create_text_message,
    fail_agent_run_resume_claim,
    fork_workflow,
    resume_agent_run,
    resume_workflow,
    run_agent,
    tool,
)
from zhivex_ai.errors import ValidationError
from zhivex_ai.types import GenerateResult, ModelGenerateInput, ModelMessage, ToolCall, ToolCallPart
from zhivex_ai.workflow_state import WORKFLOW_POSTGRES_SCHEMA_VERSION


_POSTGRES_DSN = os.getenv("ZHIVEX_TEST_POSTGRES_DSN")
_SAFE_PREFIX_RE = re.compile(r"^zhivex_it_[a-f0-9]{12}$")


@skipUnless(_POSTGRES_DSN, "requires ZHIVEX_TEST_POSTGRES_DSN")
class PostgresAgentRuntimeIntegrationTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        assert _POSTGRES_DSN is not None
        self.dsn = _POSTGRES_DSN
        self.prefix = f"zhivex_it_{uuid.uuid4().hex[:12]}"
        self.assertRegex(self.prefix, _SAFE_PREFIX_RE)
        self.memory = create_postgres_agent_memory_store(self.dsn, table_prefix=self.prefix)
        self.checkpoints = create_postgres_checkpoint_store(self.dsn, table_prefix=self.prefix)
        self.runs = create_postgres_agent_run_store(self.dsn, table_prefix=self.prefix)
        import asyncpg

        self.workflow_pool = await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=1,
            max_size=8,
        )
        self.workflow_checkpoints = create_postgres_workflow_checkpoint_store(
            table_prefix=self.prefix,
            pool=self.workflow_pool,
        )
        self.workflow_leases = create_postgres_workflow_lease_manager(
            table_prefix=self.prefix,
            pool=self.workflow_pool,
        )

    async def asyncTearDown(self) -> None:
        if not _SAFE_PREFIX_RE.fullmatch(self.prefix):
            raise AssertionError("refusing to clean Postgres tables for an unsafe integration-test prefix")
        import asyncpg

        await self.workflow_checkpoints.close()
        await self.workflow_leases.close()
        await self.workflow_pool.close()
        connection = await asyncpg.connect(self.dsn)
        try:
            for suffix in (
                "agent_memory",
                "agent_checkpoints",
                "runs",
                "workflow_runs",
                "workflow_checkpoints",
                "workflow_leases",
                "workflow_schema",
            ):
                table = f'{self.prefix}_{suffix}'
                await connection.execute(f'DROP TABLE IF EXISTS "{table}"')
        finally:
            await connection.close()

    async def test_agent_run_persists_memory_checkpoint_and_run_state(self) -> None:
        model = create_mock_language_model(
            responses=[
                GenerateResult(
                    text="postgres-agent-ok",
                    messages=[create_text_message("assistant", "postgres-agent-ok")],
                    finish_reason="stop",
                )
            ]
        )
        agent = Agent(
            name="postgres_integration_agent",
            model=model,
            memory=self.memory,
            checkpoint_store=self.checkpoints,
            run_store=self.runs,
        )
        session = create_agent_session(id=f"{self.prefix}-session")

        result = await run_agent(
            agent=agent,
            session=session,
            prompt="Persist this run.",
            idempotency_key=f"{self.prefix}-run",
        )

        restored_memory = await self.memory.load(session.id)
        restored_checkpoint = await self.checkpoints.get_latest(run_id=result.run_id)
        restored_run = await self.runs.load(result.run_id)
        self.assertEqual(result.text, "postgres-agent-ok")
        self.assertTrue(restored_memory.messages)
        assert restored_checkpoint is not None
        self.assertTrue(restored_checkpoint.is_final)
        assert restored_run is not None
        self.assertEqual(restored_run.status, "completed")
        self.assertEqual(restored_run.output_text, "postgres-agent-ok")

    async def test_workflow_graph_resumes_from_postgres_checkpoint_history(self) -> None:
        def graph(*, publish_text: str):
            return (
                WorkflowBuilder("postgres_workflow", definition_version="1")
                .add_step(
                    WorkflowStep(
                        "draft",
                        Agent(
                            name="workflow_draft",
                            model=create_mock_language_model(
                                responses=[GenerateResult(text="draft", finish_reason="stop")]
                            ),
                        ),
                        output_key="draft",
                    ),
                    entrypoint=True,
                )
                .add_step(
                    WorkflowStep(
                        "publish",
                        Agent(
                            name="workflow_publish",
                            model=create_mock_language_model(
                                responses=[GenerateResult(text=publish_text, finish_reason="stop")]
                            ),
                        ),
                        output_key="published",
                    )
                )
                .add_edge("draft", "publish")
                .interrupt_after("draft", reason="integration review")
                .build(
                    checkpoint_store=self.workflow_checkpoints,
                    lease_manager=self.workflow_leases,
                )
            )

        suspended = await graph(publish_text="unused").run(
            idempotency_key=f"{self.prefix}-workflow"
        )
        assert suspended.checkpoint is not None
        assert suspended.checkpoint.pending_interrupt is not None
        resumed = await resume_workflow(
            graph(publish_text="published"),
            suspended.run_id,
            interrupt_id=suspended.checkpoint.pending_interrupt.interrupt_id,
            resume_value={"approved": True},
        )

        history = await self.workflow_checkpoints.list_checkpoints(suspended.run_id)
        self.assertEqual(suspended.status, "suspended")
        self.assertEqual(resumed.status, "completed")
        self.assertEqual(resumed.state["draft"], "draft")
        self.assertEqual(resumed.state["published"], "published")
        self.assertGreater(len(history), 5)

    async def test_workflow_lease_fences_concurrent_and_expired_owners(self) -> None:
        run_id = f"{self.prefix}-leased-workflow"
        managers = [
            create_postgres_workflow_lease_manager(self.dsn, table_prefix=self.prefix)
            for _ in range(12)
        ]

        try:
            claims = await asyncio.gather(
                *(
                    manager.acquire(
                        run_id,
                        owner_id=f"worker-{index}",
                        ttl_ms=100,
                        now_ms=1_000,
                    )
                    for index, manager in enumerate(managers)
                )
            )

            winners = [lease for lease in claims if lease is not None]
            self.assertEqual(len(winners), 1)
            first = winners[0]
            self.assertEqual(first.fencing_token, 1)

            renewed = await self.workflow_leases.renew(
                run_id,
                token=first.token,
                ttl_ms=100,
                now_ms=1_050,
            )
            assert renewed is not None
            self.assertEqual(renewed.expires_at_ms, 1_150)
            self.assertIsNone(
                await self.workflow_leases.renew(
                    run_id,
                    token="stale-token",
                    ttl_ms=100,
                    now_ms=1_060,
                )
            )
            self.assertIsNone(
                await self.workflow_leases.acquire(
                    run_id,
                    owner_id="early-worker",
                    ttl_ms=100,
                    now_ms=1_149,
                )
            )

            replacement = await self.workflow_leases.acquire(
                run_id,
                owner_id="replacement-worker",
                ttl_ms=100,
                now_ms=1_150,
            )
            assert replacement is not None
            self.assertEqual(replacement.fencing_token, 2)
            self.assertFalse(await self.workflow_leases.release(run_id, token=first.token))
            self.assertTrue(await self.workflow_leases.release(run_id, token=replacement.token))
        finally:
            await asyncio.gather(*(manager.close() for manager in managers))

    async def test_workflow_fenced_append_rejects_stale_postgres_owner(self) -> None:
        run_id = f"{self.prefix}-fenced-append"
        initial = WorkflowCheckpoint(
            checkpoint_id=f"{run_id}-cp-0",
            run_id=run_id,
            workflow_name="postgres-fencing",
            definition_version="1",
            definition_digest="digest-1",
            idempotency_key=f"{run_id}-key",
            transition=WorkflowTransition(type="workflow-start", at_ms=1),
        )
        await self.workflow_checkpoints.append(initial)
        first = await self.workflow_leases.acquire(run_id, owner_id="worker-a", ttl_ms=10_000)
        assert first is not None
        self.assertTrue(await self.workflow_leases.release(run_id, token=first.token))
        replacement_lease = await self.workflow_leases.acquire(
            run_id,
            owner_id="worker-b",
            ttl_ms=10_000,
        )
        assert replacement_lease is not None
        candidate = replace(
            initial,
            checkpoint_id=f"{run_id}-cp-1",
            sequence=1,
            transition=WorkflowTransition(type="workflow-progress", at_ms=2),
        )

        with self.assertRaises(WorkflowLeaseLostError):
            await self.workflow_checkpoints.append_fenced(
                candidate,
                expected_sequence=0,
                lease_manager=self.workflow_leases,
                lease=first,
            )
        latest = await self.workflow_checkpoints.load_latest(run_id)
        assert latest is not None
        self.assertEqual(latest.sequence, 0)

        committed = await self.workflow_checkpoints.append_fenced(
            candidate,
            expected_sequence=0,
            lease_manager=self.workflow_leases,
            lease=replacement_lease,
        )
        self.assertEqual(committed.sequence, 1)

    async def test_workflow_postgres_uses_server_clock_schema_version_and_namespaces(self) -> None:
        run_id = f"{self.prefix}-namespace-run"
        async with self.workflow_pool.acquire() as connection:
            before_ms = int(
                await connection.fetchval(
                    "SELECT FLOOR(EXTRACT(EPOCH FROM clock_timestamp()) * 1000)::BIGINT"
                )
            )
        lease = await self.workflow_leases.acquire(run_id, owner_id="clock-worker", ttl_ms=10_000)
        assert lease is not None
        async with self.workflow_pool.acquire() as connection:
            after_ms = int(
                await connection.fetchval(
                    "SELECT FLOOR(EXTRACT(EPOCH FROM clock_timestamp()) * 1000)::BIGINT"
                )
            )
        self.assertLessEqual(before_ms, lease.acquired_at_ms)
        self.assertLessEqual(lease.acquired_at_ms, after_ms)

        tenant_a_store = create_postgres_workflow_checkpoint_store(
            table_prefix=self.prefix,
            namespace="tenant-a",
            pool=self.workflow_pool,
        )
        tenant_b_store = create_postgres_workflow_checkpoint_store(
            table_prefix=self.prefix,
            namespace="tenant-b",
            pool=self.workflow_pool,
        )
        tenant_a_leases = create_postgres_workflow_lease_manager(
            table_prefix=self.prefix,
            namespace="tenant-a",
            pool=self.workflow_pool,
        )
        tenant_b_leases = create_postgres_workflow_lease_manager(
            table_prefix=self.prefix,
            namespace="tenant-b",
            pool=self.workflow_pool,
        )
        shared = WorkflowCheckpoint(
            checkpoint_id="shared-cp",
            run_id="shared-run",
            workflow_name="namespaced",
            definition_version="1",
            definition_digest="digest-1",
            idempotency_key="shared-key",
        )
        await tenant_a_store.append(shared)
        await tenant_b_store.append(shared)
        tenant_a_lease = await tenant_a_leases.acquire(
            "shared-run",
            owner_id="tenant-a-worker",
            ttl_ms=10_000,
        )
        tenant_b_lease = await tenant_b_leases.acquire(
            "shared-run",
            owner_id="tenant-b-worker",
            ttl_ms=10_000,
        )
        assert tenant_a_lease is not None and tenant_b_lease is not None
        self.assertEqual(tenant_a_lease.fencing_token, 1)
        self.assertEqual(tenant_b_lease.fencing_token, 1)
        self.assertEqual((await tenant_a_store.load_latest("shared-run")).checkpoint_id, "shared-cp")
        self.assertEqual((await tenant_b_store.load_latest("shared-run")).checkpoint_id, "shared-cp")

        await tenant_a_store.close()
        await tenant_b_store.close()
        await tenant_a_leases.close()
        await tenant_b_leases.close()
        async with self.workflow_pool.acquire() as connection:
            versions = await connection.fetch(
                f'SELECT component, version FROM "{self.prefix}_workflow_schema" ORDER BY component'
            )
            still_open = await connection.fetchval("SELECT 1")
        self.assertEqual(
            [(row["component"], row["version"]) for row in versions],
            [("checkpoint", WORKFLOW_POSTGRES_SCHEMA_VERSION), ("lease", WORKFLOW_POSTGRES_SCHEMA_VERSION)],
        )
        self.assertEqual(still_open, 1)

    async def test_active_workflow_worker_observes_postgres_cancellation(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_step(_context):
            started.set()
            await release.wait()
            return "late-result"

        graph = (
            WorkflowBuilder("postgres-workflow-cancel")
            .add_step(WorkflowStep("work", executor=slow_step, output_key="result"), entrypoint=True)
            .build(
                checkpoint_store=self.workflow_checkpoints,
                lease_manager=self.workflow_leases,
            )
        )
        key = f"{self.prefix}-workflow-cancel"
        worker = asyncio.create_task(graph.run(idempotency_key=key))
        await asyncio.wait_for(started.wait(), timeout=2)
        running = await self.workflow_checkpoints.find_by_idempotency_key(key)
        assert running is not None

        try:
            cancelled = await cancel_workflow(graph, running.run_id, reason="operator-stop")
        finally:
            release.set()
        worker_result = await worker

        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(worker_result.status, "cancelled")
        self.assertNotIn("result", worker_result.state)

    async def test_concurrent_postgres_resume_is_lease_serialized(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def reviewed_step(_context):
            started.set()
            await release.wait()
            return "approved"

        graph = (
            WorkflowBuilder("postgres-concurrent-resume")
            .add_step(
                WorkflowStep("review", executor=reviewed_step, output_key="decision"),
                entrypoint=True,
            )
            .interrupt_before("review")
            .build(
                checkpoint_store=self.workflow_checkpoints,
                lease_manager=self.workflow_leases,
            )
        )
        suspended = await graph.run(idempotency_key=f"{self.prefix}-resume-source")
        pending = suspended.checkpoint.pending_interrupt
        assert pending is not None
        winner = asyncio.create_task(
            resume_workflow(
                graph,
                suspended.run_id,
                interrupt_id=pending.interrupt_id,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=2)

        try:
            with self.assertRaises(WorkflowConflictError):
                await resume_workflow(
                    graph,
                    suspended.run_id,
                    interrupt_id=pending.interrupt_id,
                )
        finally:
            release.set()
        completed = await winner
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.state["decision"], "approved")

    async def test_concurrent_postgres_forks_claim_one_idempotent_run(self) -> None:
        async def complete(_context):
            return "done"

        graph = (
            WorkflowBuilder("postgres-concurrent-fork")
            .add_step(WorkflowStep("work", executor=complete, output_key="result"), entrypoint=True)
            .build(
                checkpoint_store=self.workflow_checkpoints,
                lease_manager=self.workflow_leases,
            )
        )
        source = await graph.run()
        key = f"{self.prefix}-same-fork"
        outcomes = await asyncio.gather(
            *(fork_workflow(graph, source.run_id, idempotency_key=key) for _ in range(8)),
            return_exceptions=True,
        )
        unexpected = [
            outcome
            for outcome in outcomes
            if isinstance(outcome, BaseException) and not isinstance(outcome, WorkflowConflictError)
        ]
        successes = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
        self.assertFalse(unexpected)
        self.assertTrue(successes)
        winner = await self.workflow_checkpoints.find_by_idempotency_key(key)
        assert winner is not None
        self.assertEqual({result.run_id for result in successes}, {winner.run_id})

    async def test_dsn_owned_workflow_pools_restart_cleanly(self) -> None:
        namespace = f"restart-{self.prefix}"
        store = create_postgres_workflow_checkpoint_store(
            self.dsn,
            table_prefix=self.prefix,
            namespace=namespace,
            pool_min_size=0,
            pool_max_size=2,
        )
        leases = create_postgres_workflow_lease_manager(
            self.dsn,
            table_prefix=self.prefix,
            namespace=namespace,
            pool_min_size=0,
            pool_max_size=2,
        )
        checkpoint = WorkflowCheckpoint(
            checkpoint_id="restart-cp",
            run_id="restart-run",
            workflow_name="restart",
            definition_version="1",
            definition_digest="restart-digest",
        )
        await store.append(checkpoint)
        lease = await leases.acquire("restart-run", owner_id="restart-worker", ttl_ms=10_000)
        assert lease is not None
        await store.close()
        await leases.close()

        reopened_store = create_postgres_workflow_checkpoint_store(
            self.dsn,
            table_prefix=self.prefix,
            namespace=namespace,
            pool_min_size=0,
            pool_max_size=2,
        )
        reopened_leases = create_postgres_workflow_lease_manager(
            self.dsn,
            table_prefix=self.prefix,
            namespace=namespace,
            pool_min_size=0,
            pool_max_size=2,
        )
        try:
            restored = await reopened_store.load_latest("restart-run")
            assert restored is not None
            self.assertEqual(restored.checkpoint_id, checkpoint.checkpoint_id)
            self.assertEqual(restored.run_id, checkpoint.run_id)
            self.assertTrue(
                await reopened_leases.validate(
                    "restart-run",
                    token=lease.token,
                    fencing_token=lease.fencing_token,
                )
            )
        finally:
            await reopened_store.close()
            await reopened_leases.close()

    async def test_idempotency_claim_has_one_winner_across_concurrent_connections(self) -> None:
        key = f"{self.prefix}-idempotency"
        stores = [create_postgres_agent_run_store(self.dsn, table_prefix=self.prefix) for _ in range(12)]
        candidates = [
            AgentRunState(
                run_id=f"{self.prefix}-candidate-{index}",
                agent_name="postgres_integration_agent",
                provider="test",
                model_id="test",
                idempotency_key=key,
                updated_at_ms=index + 1,
            )
            for index in range(12)
        ]

        claimed = await asyncio.gather(
            *(store.claim_idempotency_key(state) for store, state in zip(stores, candidates, strict=True))
        )

        winning_run_ids = {state.run_id for state in claimed}
        self.assertEqual(len(winning_run_ids), 1)
        restored = await self.runs.find_by_idempotency_key(key)
        assert restored is not None
        self.assertEqual(restored.run_id, claimed[0].run_id)

    async def test_pending_approval_claim_has_one_concurrent_winner(self) -> None:
        run_id = f"{self.prefix}-approval-run"
        approval_id = f"{self.prefix}-approval"
        await self.runs.save(
            AgentRunState(
                run_id=run_id,
                agent_name="postgres_integration_agent",
                provider="test",
                model_id="test",
                status="suspended",
                pending_approvals=[
                    PendingApproval(
                        id=approval_id,
                        name="write_record",
                        tool_fingerprint="integration-test-fingerprint",
                    )
                ],
            )
        )

        claims = await asyncio.gather(
            *(
                self.runs.claim_pending_approval(
                    run_id,
                    approval_id,
                    claim_token=f"claim-{index}",
                    claimed_at_ms=1_000 + index,
                )
                for index in range(12)
            )
        )

        winners = [state for state in claims if state is not None]
        self.assertEqual(len(winners), 1)
        restored = await self.runs.load(run_id)
        assert restored is not None
        self.assertEqual(restored.status, "running")
        resume_claim = restored.metadata["resume_claim"]
        assert isinstance(resume_claim, dict)
        self.assertEqual(resume_claim["approval_id"], approval_id)

    async def test_cancel_is_atomic_against_a_stale_worker_completion(self) -> None:
        run_id = f"{self.prefix}-cancel-run"
        await self.runs.save(
            AgentRunState(
                run_id=run_id,
                agent_name="postgres_integration_agent",
                provider="test",
                model_id="test",
            )
        )
        stale_worker = await self.runs.load(run_id)
        assert stale_worker is not None

        cancelled = await cancel_agent_run(self.runs, run_id, reason="operator-stop", now_ms=2_000)
        assert cancelled is not None
        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(cancelled.revision, 1)

        stale_worker.status = "completed"
        stale_worker.output_text = "late result"
        with self.assertRaisesRegex(ValidationError, "revision conflict"):
            await self.runs.save(stale_worker)

        restored = await self.runs.load(run_id)
        assert restored is not None
        self.assertEqual(restored.status, "cancelled")
        self.assertEqual(restored.output_text, "")

    async def test_active_agent_worker_surfaces_postgres_cancellation(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        base_model = create_mock_language_model(
            responses=[
                GenerateResult(
                    text="late-result",
                    messages=[create_text_message("assistant", "late-result")],
                    finish_reason="stop",
                )
            ]
        )

        class BlockingModel:
            provider = base_model.provider
            model_id = base_model.model_id
            capabilities = base_model.capabilities

            async def generate(self, input: ModelGenerateInput) -> GenerateResult:
                started.set()
                await release.wait()
                return await base_model.generate(input)

        key = f"{self.prefix}-active-cancel"
        agent = Agent(name="postgres_cancel_agent", model=BlockingModel(), run_store=self.runs)
        worker = asyncio.create_task(run_agent(agent=agent, prompt="wait", idempotency_key=key))
        await asyncio.wait_for(started.wait(), timeout=2)
        running = await self.runs.find_by_idempotency_key(key)
        assert running is not None
        cancelled = await cancel_agent_run(self.runs, running.run_id, reason="operator-stop")
        assert cancelled is not None
        release.set()

        with self.assertRaises(AgentRunCancelled):
            await worker
        restored = await self.runs.load(running.run_id)
        assert restored is not None
        self.assertEqual(restored.status, "cancelled")

    async def test_postgres_approval_resume_executes_tool_once_and_completes_parent(self) -> None:
        executions = 0

        async def suspend_policy(_request: ToolApprovalRequest) -> ApprovalDecision:
            return ApprovalDecision.require_human("manager review", approval_id="postgres-approval")

        def execute(input: dict[str, str]) -> dict[str, str]:
            nonlocal executions
            executions += 1
            return {"item": input["item"], "approved": "yes"}

        model = create_mock_language_model(
            responses=[
                GenerateResult(
                    messages=[
                        ModelMessage(
                            role="assistant",
                            parts=[
                                ToolCallPart(
                                    tool_call=ToolCall(
                                        id="postgres-tool-call",
                                        name="lookup",
                                        input={"item": "apollo"},
                                    )
                                )
                            ],
                        )
                    ],
                    finish_reason="tool-calls",
                ),
                GenerateResult(
                    text="approved-result",
                    messages=[create_text_message("assistant", "approved-result")],
                    finish_reason="stop",
                ),
            ]
        )
        agent = Agent(
            name="postgres_approval_agent",
            model=model,
            run_store=self.runs,
            approval_policy=suspend_policy,
            tools={
                "lookup": tool(
                    name="lookup",
                    schema=dict[str, str],
                    execute=execute,
                    requires_approval=True,
                )
            },
        )
        suspended = await run_agent(agent=agent, prompt="lookup")
        self.assertEqual(suspended.state.status, "suspended")  # type: ignore[union-attr]

        resumed = await resume_agent_run(
            agent=agent,
            run_id=suspended.run_id,
            approval_id="postgres-approval",
        )

        self.assertEqual(resumed.text, "approved-result")
        self.assertEqual(executions, 1)
        parent = await self.runs.load(suspended.run_id)
        assert parent is not None
        self.assertEqual(parent.status, "completed")
        self.assertEqual(len(parent.child_runs), 1)

    async def test_resume_claim_reconciliation_requires_exact_claim_token(self) -> None:
        run_id = f"{self.prefix}-reconcile-run"
        approval_id = f"{self.prefix}-reconcile-approval"
        await self.runs.save(
            AgentRunState(
                run_id=run_id,
                agent_name="postgres_integration_agent",
                provider="test",
                model_id="test",
                status="suspended",
                pending_approvals=[PendingApproval(id=approval_id, name="write_record")],
            )
        )
        claimed = await self.runs.claim_pending_approval(
            run_id,
            approval_id,
            claim_token="lease-owner",
            claimed_at_ms=1_000,
        )
        assert claimed is not None

        wrong_owner = await fail_agent_run_resume_claim(
            self.runs,
            run_id,
            claim_token="other-worker",
            reason="expired",
            now_ms=2_000,
        )
        self.assertIsNone(wrong_owner)
        reconciled = await fail_agent_run_resume_claim(
            self.runs,
            run_id,
            claim_token="lease-owner",
            reason="lease expired; operator review required",
            now_ms=2_000,
        )
        assert reconciled is not None
        self.assertEqual(reconciled.status, "failed")
        self.assertEqual(reconciled.revision, 2)
