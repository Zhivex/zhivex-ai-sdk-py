from __future__ import annotations

import json
import unittest
from dataclasses import replace

from zhivex_ai.errors import ValidationError
from zhivex_ai.workflow_adapters import (
    CallbackWorkflowAdapter,
    WorkflowAdapter,
    WorkflowStepExecutorRegistry,
    WorkflowStepOutcome,
    WorkflowStepRequest,
    create_dbos_workflow_adapter,
    create_prefect_workflow_adapter,
    create_restate_workflow_adapter,
    create_temporal_workflow_adapter,
)


def step_request(**overrides) -> WorkflowStepRequest:
    values = {
        "workflow_name": "loan-review",
        "definition_version": "0.15.0",
        "definition_digest": "sha256:definition-v1",
        "workflow_run_id": "wf-123",
        "node_id": "risk-review",
        "executor_ref": "agents.risk-review",
        "input": {"amount": 1000},
        "state": {"customer_id": "customer-1"},
        "metadata": {"tenant": "bank-ar"},
        "checkpoint_id": "checkpoint-1",
        "correlation_ids": {"trace_id": "trace-1"},
    }
    values.update(overrides)
    return WorkflowStepRequest(**values)


class WorkflowAdapterEnvelopeTests(unittest.TestCase):
    def test_request_json_round_trip_and_retry_idempotency(self) -> None:
        request = step_request(attempt=1)
        retried = replace(request, attempt=9)

        restored = WorkflowStepRequest.from_json(request.to_json())

        self.assertEqual(restored, request)
        self.assertEqual(retried.step_idempotency_key, request.step_idempotency_key)
        self.assertNotEqual(
            replace(request, activation_index=1).step_idempotency_key,
            request.step_idempotency_key,
        )
        self.assertEqual(json.loads(request.to_json())["step_idempotency_key"], request.step_idempotency_key)

    def test_request_rejects_tampered_identity_and_runtime_objects(self) -> None:
        payload = step_request().to_dict()
        payload["step_idempotency_key"] = "tampered"

        with self.assertRaisesRegex(ValidationError, "idempotency"):
            WorkflowStepRequest.from_dict(payload)
        with self.assertRaisesRegex(ValidationError, "runtime objects"):
            step_request(input={"callback": lambda: None})

    def test_outcome_json_round_trip(self) -> None:
        request = step_request()
        outcome = WorkflowStepOutcome.for_request(
            request,
            status="completed",
            output={"risk": "low"},
            state_patch={"risk": "low"},
            child_run_id="run-1",
        )

        self.assertEqual(WorkflowStepOutcome.from_json(outcome.to_json()), outcome)

    def test_suspension_and_cancellation_outcomes_round_trip(self) -> None:
        request = step_request()
        suspended = WorkflowStepOutcome.for_request(
            request,
            status="suspended",
            suspension={"signal": "manager-review", "resume_token": "review-1"},
        )
        cancelled = WorkflowStepOutcome.for_request(request, status="cancelled")

        self.assertEqual(WorkflowStepOutcome.from_json(suspended.to_json()), suspended)
        self.assertEqual(WorkflowStepOutcome.from_json(cancelled.to_json()), cancelled)


class WorkflowStepExecutorRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_registry_dispatches_sync_and_async_executors(self) -> None:
        registry = WorkflowStepExecutorRegistry()

        def sync_executor(request: WorkflowStepRequest) -> WorkflowStepOutcome:
            return WorkflowStepOutcome.for_request(request, status="completed", output="sync")

        async def async_executor(request: WorkflowStepRequest) -> WorkflowStepOutcome:
            return WorkflowStepOutcome.for_request(request, status="completed", output="async")

        registry.register(
            executor_ref="agents.risk-review",
            definition_digest="sha256:definition-v1",
            executor=sync_executor,
        )
        registry.register(
            executor_ref="agents.risk-review",
            definition_digest="sha256:definition-v2",
            executor=async_executor,
        )

        self.assertEqual((await registry.dispatch(step_request())).output, "sync")
        self.assertEqual(
            (
                await registry.dispatch(
                    step_request(definition_digest="sha256:definition-v2")
                )
            ).output,
            "async",
        )

    async def test_registry_fails_closed_for_missing_executor_and_digest(self) -> None:
        registry = WorkflowStepExecutorRegistry()
        registry.register(
            executor_ref="agents.risk-review",
            definition_digest="sha256:definition-v1",
            executor=lambda request: WorkflowStepOutcome.for_request(
                request,
                status="completed",
            ),
        )

        with self.assertRaisesRegex(ValidationError, "Unknown workflow executor_ref"):
            await registry.dispatch(step_request(executor_ref="agents.unknown"))
        with self.assertRaisesRegex(ValidationError, "digest mismatch"):
            await registry.dispatch(step_request(definition_digest="sha256:unknown"))

    async def test_registry_rejects_outcome_for_another_step(self) -> None:
        registry = WorkflowStepExecutorRegistry()

        def wrong_executor(request: WorkflowStepRequest) -> WorkflowStepOutcome:
            return replace(
                WorkflowStepOutcome.for_request(request, status="completed"),
                node_id="another-node",
            )

        registry.register(
            executor_ref="agents.risk-review",
            definition_digest="sha256:definition-v1",
            executor=wrong_executor,
        )

        with self.assertRaisesRegex(ValidationError, "node identity"):
            await registry.dispatch(step_request())


class CallbackWorkflowAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_callback_adapter_dispatches_user_callback(self) -> None:
        seen: list[str] = []

        async def callback(request: WorkflowStepRequest) -> WorkflowStepOutcome:
            seen.append(request.step_idempotency_key)
            return WorkflowStepOutcome.for_request(
                request,
                status="completed",
                output={"backend": "custom"},
            )

        adapter = CallbackWorkflowAdapter(backend="custom", callback=callback)
        request = step_request()

        result = await adapter.dispatch(request)

        self.assertIsInstance(adapter, WorkflowAdapter)
        self.assertEqual(result.output, {"backend": "custom"})
        self.assertEqual(seen, [request.step_idempotency_key])

    async def test_factories_are_dependency_free_and_capabilities_are_conservative(self) -> None:
        def callback(request: WorkflowStepRequest) -> WorkflowStepOutcome:
            return WorkflowStepOutcome.for_request(request, status="completed")

        dbos = create_dbos_workflow_adapter(callback)
        temporal = create_temporal_workflow_adapter(callback)
        prefect = create_prefect_workflow_adapter(callback)
        restate = create_restate_workflow_adapter(callback)

        self.assertEqual([dbos.backend, temporal.backend, prefect.backend, restate.backend], ["dbos", "temporal", "prefect", "restate"])
        self.assertTrue(dbos.capabilities.durable_steps)
        self.assertTrue(dbos.capabilities.explicit_resume)
        self.assertTrue(dbos.capabilities.fork)
        self.assertTrue(temporal.capabilities.signals)
        self.assertFalse(temporal.capabilities.explicit_resume)
        self.assertFalse(prefect.capabilities.durable_steps)
        self.assertTrue(prefect.capabilities.native_step_retries)
        self.assertTrue(restate.capabilities.durable_steps)
        self.assertFalse(restate.capabilities.fork)

        request = step_request()
        for adapter in (dbos, temporal, prefect, restate):
            self.assertEqual((await adapter.dispatch(request)).status, "completed")
