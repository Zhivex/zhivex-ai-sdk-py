# Durable Workflow Graphs

Zhivex AI SDK provides beta orchestration primitives for backend workflows whose shape is known before execution. `WorkflowGraph` adds validated DAG execution, persisted routing decisions, append-only checkpoints, explicit interruption and resume, fork lineage, and step-level retry policy to the existing sequential, parallel, and loop agents.

The SDK owns orchestration mechanics. The application continues to own business policy, authorization, vertical records, approval UI, artifact storage, retention, and external integrations.

## Stability

The complete workflow surface introduced in `0.15.0` remains beta in `0.16.0`:

- Existing orchestration: `SequentialAgent`, `ParallelAgent`, `LoopAgent`, `WorkflowStep`, `WorkflowRunResult`, `WorkflowStepResult`, workflow status/error aliases, `run_workflow`, `workflow_step`, and `validate_workflow_expectations`
- Graphs: `WorkflowBuilder`, `WorkflowGraph`, `GraphWorkflow`, `WorkflowEdge`, `WorkflowContext`, graph callable/phase aliases, `resume_workflow`, `fork_workflow`, and `cancel_workflow`
- Functional graph steps: `WorkflowFunctionContext`, `WorkflowFunctionResult`, and `WorkflowFunctionExecutor`
- Durable state: checkpoint schema/status aliases, `WorkflowCheckpoint`, `WorkflowNodeCheckpoint`, `WorkflowInterrupt`, `WorkflowTransition`, `WorkflowCheckpointStore`, serialization helpers, and the in-memory, SQLite, and Postgres workflow checkpoint stores
- Execution ownership: `WorkflowExecutionLease`, `WorkflowLeaseManager`, and the in-memory, SQLite, and Postgres lease managers/factories
- Step retries: `WorkflowRetryPolicy`
- External runtime contracts: adapter schema/capability types, `WorkflowStepRequest`, `WorkflowStepOutcome`, `WorkflowStepExecutor`, `WorkflowStepExecutorRegistry`, `CallbackWorkflowAdapter`, and the DBOS, Temporal, Prefect, and Restate callback-adapter factories

Use top-level imports from `zhivex_ai`. Avoid deep imports. These APIs may evolve between minor releases and are not part of the stable compatibility contract yet.

## Choose The Orchestration Model

Use the smallest model that represents the workflow:

- `SequentialAgent` for a fixed linear pipeline.
- `ParallelAgent` for one independent fan-out wave.
- `LoopAgent` for bounded refinement where repetition is intentional.
- `WorkflowGraph` for branching, fan-out/fan-in, durable transition history, explicit interrupts, resume, or fork.

`WorkflowGraph` is acyclic. Use `LoopAgent` for loops instead of hiding cycles inside a graph definition.

## Build A Graph

```python
from zhivex_ai import Agent, WorkflowBuilder, WorkflowStep

workflow = (
    WorkflowBuilder("loan_review", definition_version="2026-07-31")
    .add_step(
        WorkflowStep("extract", extractor, prompt="Extract the application", output_key="application"),
        entrypoint=True,
    )
    .add_step(
        WorkflowStep("risk", risk_agent, input_template="Review {application}", output_key="risk"),
    )
    .add_step(
        WorkflowStep("decide", decision_agent, input_template="Decide from {risk}", output_key="decision"),
    )
    .add_edge("extract", "risk")
    .add_edge("risk", "decide")
    .build()
)

result = await workflow.run(idempotency_key="loan-123")
```

The builder validates non-empty unique step names, edge identities, references, entrypoints, reachability, globally unique output/metadata keys, positive concurrency, and acyclicity before execution.

Ready nodes execute as a bounded parallel wave. Their declared `output_key`, `metadata_key`, and adapter `state_patch` values are merged before outgoing conditions are evaluated. Set `max_concurrency` on `build(...)` or `WorkflowGraph(...)` to constrain a wave.

## Functional Steps

A graph step can execute an `Agent` or a sync/async Python function. Functional steps are useful for deterministic validation, routing preparation, database reads through injected dependencies, or application-owned activities that do not need a model call:

```python
from zhivex_ai import WorkflowFunctionResult, WorkflowStep

async def calculate_risk(context):
    score = await context.deps.risk_service.score(context.state["application"])
    return WorkflowFunctionResult(
        output={"score": score},
        state_patch={"risk_band": "low" if score < 20 else "review"},
        metadata={"calculator": "risk-v3"},
    )

risk_step = WorkflowStep(
    "calculate_risk",
    executor=calculate_risk,
    output_key="risk",
)
```

`WorkflowFunctionContext` contains workflow/run/step identity, attempt, stable step idempotency key, rendered input, durable state, resume values, and ephemeral `deps`. A function returns a JSON value or `WorkflowFunctionResult`; its output, state patch, and metadata must contain only finite JSON values. Runtime dependencies are never checkpointed.

Functional executors are supported only by `WorkflowGraph`; existing `SequentialAgent`, `ParallelAgent`, and `LoopAgent` steps still require an `Agent`. Each graph step must define exactly one of `agent` or `executor` unless a callback adapter owns dispatch through `executor_ref`.

A functional step can be retried by `WorkflowRetryPolicy`, so writes must use `context.idempotency_key` at the destination. Checkpointing cannot make an arbitrary database/API side effect atomic with workflow progress; use destination idempotency, an outbox, or reconciliation.

## Conditional Branches

An edge condition can be synchronous or asynchronous and receives `WorkflowContext`:

```python
workflow = (
    WorkflowBuilder("route")
    .add_step(WorkflowStep("classify", classifier, output_key="route"), entrypoint=True)
    .add_step(WorkflowStep("approve", approver, output_key="decision"))
    .add_step(WorkflowStep("review", reviewer, output_key="review"))
    .add_edge(
        "classify",
        "approve",
        name="low-risk",
        condition=lambda context: context.source_output == "low",
    )
    .add_edge(
        "classify",
        "review",
        name="manual-review",
        condition=lambda context: context.source_output != "low",
    )
    .build()
)
```

Each edge decision is persisted before downstream dispatch. Resume and recovery reuse the recorded boolean instead of evaluating the condition again. Conditions should still be deterministic and side-effect free because their callable identity contributes to the definition digest.

## Canonical Checkpoints

`WorkflowCheckpoint` is the canonical durable workflow record. A new immutable checkpoint is appended for every recorded transition, including start, step start, step finish, retry, routing, interrupt, resume, fork, skip, recovery, and workflow finish.

Each checkpoint includes:

- workflow name, definition version, and definition digest
- run, session, parent, checkpoint, sequence, and idempotency identities
- JSON workflow state and per-node status/output/error metadata
- persisted edge decisions and currently ready nodes
- pending interrupt and resume values
- transition details and fork lineage

Append operations use `expected_sequence` as an optimistic compare-and-swap boundary. A stale writer cannot overwrite or reorder workflow history. `WorkflowRunResult.checkpoint` exposes the canonical latest checkpoint; `state_snapshot` remains an `AgentRunState` projection for the existing replay and observability path.

Checkpoint stores have different operational guarantees:

- `create_in_memory_workflow_checkpoint_store()` is deterministic and concurrency-safe inside one process, but it does not survive process exit.
- `create_sqlite_workflow_checkpoint_store(path, namespace=...)` persists append-only history on one local filesystem and survives worker reconstruction. Coordinate filesystem ownership and backup outside the SDK.
- `create_postgres_workflow_checkpoint_store(dsn, table_prefix=...)` uses the optional `postgres` extra and transactional sequence/idempotency constraints for shared workers. Validate it against the deployment database before production use.

Persist only JSON data. Dependencies, clients, credentials, agents, and callables are runtime objects and must be supplied again when resuming.

## Execution Leases And Fencing

For shared or restartable workers, attach a lease manager separately from the
checkpoint store:

```python
from zhivex_ai import (
    WorkflowBuilder,
    create_postgres_workflow_checkpoint_store,
    create_postgres_workflow_lease_manager,
)

workflow = WorkflowBuilder("publishing").add_step(...).build(
    checkpoint_store=create_postgres_workflow_checkpoint_store(dsn),
    lease_manager=create_postgres_workflow_lease_manager(dsn),
    lease_ttl_ms=30_000,
    lease_heartbeat_ms=10_000,
)
```

The graph acquires one run lease, renews it in the background, verifies it
before durable progress, and releases it at the end. `resume_workflow(...)` and
`fork_workflow(...)` use the same ownership boundary. A takeover is allowed
only after expiry and increments a monotonic fencing token; the former owner is
then refused before it can append stale progress. Checkpoints retain the owner
reference and fencing token for correlation, never the secret lease token.

Use the in-memory manager only in one process and the SQLite manager only where
one shared filesystem/database has reviewed locking semantics. The Postgres
manager is the shared-worker implementation, but it still requires integration
and contention tests against the deployment database and pool/proxy topology.
Leases reduce concurrent orchestration writers; external side effects still
need their own idempotency or fencing support.

## Interrupt And Resume

Declare safe interruption boundaries before or after a node:

```python
from zhivex_ai import WorkflowBuilder, resume_workflow

workflow = (
    WorkflowBuilder("publishing", definition_version="1")
    .add_step(WorkflowStep("draft", writer, output_key="draft"), entrypoint=True)
    .add_step(WorkflowStep("publish", publisher, output_key="published"))
    .add_edge("draft", "publish")
    .interrupt_after("draft", reason="Human review")
    .build(checkpoint_store=checkpoint_store)
)

suspended = await workflow.run(idempotency_key="publication-42")
pending = suspended.checkpoint.pending_interrupt

resumed = await resume_workflow(
    workflow,
    suspended.run_id,
    interrupt_id=pending.interrupt_id,
    resume_value={"approved": True, "reviewer": "user-7"},
    state_updates={"approval_record_id": "approval-99"},
)
```

The caller must acknowledge the exact pending `interrupt_id`. A reconstructed worker must rebuild the same workflow definition and attach the same persistent checkpoint store. Resume fails closed when the workflow name, `definition_version`, or computed definition digest differs.

When a graph node is suspended by a local-tool approval, `resume_workflow(...)` also accepts `approval_id`, `approved`, `reason`, and `node_name`; it delegates the child continuation to the stable agent approval-resume contract. Runtime `deps` are never serialized and must be supplied again.

Interrupts occur at declared workflow boundaries or when a node explicitly suspends through the agent/adapter contract. They do not forcibly preempt synchronous code or undo an external side effect already in progress.

Adapter suspension metadata is persisted in the node checkpoint and cleared
only when the step resumes. An adapter `cancelled` outcome remains a cancelled
workflow instead of being converted into a generic failure.

## Cooperative Cancellation

```python
from zhivex_ai import cancel_workflow

cancelled = await cancel_workflow(workflow, run_id, reason="Operator request")
```

Cancellation appends a canonical `workflow-cancelled` transition, marks running
or suspended nodes cancelled, skips pending nodes, and causes an active worker
to observe the cancelled checkpoint before later progress. It is cooperative:
it cannot preempt synchronous code or roll back an external effect already in
flight.

## Fork From A Checkpoint

Fork creates a new run from a selected immutable checkpoint:

```python
from zhivex_ai import fork_workflow

forked = await fork_workflow(
    workflow,
    source_run_id,
    checkpoint_id=selected_checkpoint_id,
    state_updates={"scenario": "conservative"},
    idempotency_key="loan-123-conservative",
)
```

The fork records `forked_from_run_id` and `forked_from_checkpoint_id`, retains completed/skipped node history from the selected point, resets running or suspended work to pending, and creates a new run id. Declared interrupts can trigger again on the fork and require explicit acknowledgement.

Fork does not roll back external effects produced before the selected checkpoint. Side-effecting tools and activities need application-owned idempotency keys, reconciliation, and compensation policy.

## Step Retries And Idempotency

`WorkflowStep.max_retries` keeps its existing meaning: it configures model/provider retries inside one `run_agent(...)` invocation. Use `WorkflowRetryPolicy` when the complete logical workflow step may be attempted again:

```python
from zhivex_ai import WorkflowRetryPolicy, WorkflowStep

step = WorkflowStep(
    "risk",
    risk_agent,
    output_key="risk",
    retry_policy=WorkflowRetryPolicy(
        max_attempts=3,
        backoff_ms=250,
        max_backoff_ms=2_000,
    ),
)
```

Without `retry_if`, step retries are limited to retryable `ProviderHTTPError` failures. A custom sync or async predicate can opt into other errors. The logical step idempotency identity remains stable across attempts and is propagated to agent tool execution, but the application must use it when deduplicating writes. Do not retry an operation whose external outcome is unknown unless the destination supports idempotency or reconciliation.

## External Runtime Adapter Contracts

The adapter layer defines JSON envelopes and callback boundaries for dispatching a graph step through an application-owned workflow engine integration:

- `WorkflowStepRequest` carries definition/run/node identities, stable step idempotency key, checkpoint id, state revision, input, state, metadata, and correlation ids.
- `WorkflowStepOutcome` returns completed, failed, suspended, or cancelled status plus JSON output, state patch, metadata, error, suspension, and child-run identity.
- `WorkflowStepExecutorRegistry` resolves only an explicitly registered `executor_ref` and matching definition digest.
- `CallbackWorkflowAdapter` supports sync or async dispatch callbacks.

`create_dbos_workflow_adapter(...)`, `create_temporal_workflow_adapter(...)`, `create_prefect_workflow_adapter(...)`, and `create_restate_workflow_adapter(...)` create dependency-free callback contracts with conservative capability metadata. They are not embedded engine clients, workers, schedulers, or certified integrations. The application must install, configure, operate, and integration-test the selected engine, then implement the callback with that engine's real activity/task API.

## Recovery And Operational Boundaries

- Re-entering `WorkflowGraph.run(...)` with the same workflow idempotency key returns a terminal or suspended result. If the latest checkpoint is still `running`, the call fails closed by default.
- With a lease manager configured, `run(..., recover_running=True)` can take over only after the prior lease expires; the new fencing token prevents the old owner from appending. Without a lease manager, the caller must first reconcile that the prior worker is gone. Recovery appends `workflow-recovered` and resets recorded running nodes to pending.
- Recovery can re-dispatch a node whose external outcome is unknown. Every agent tool, functional executor, and callback activity that writes externally must deduplicate with its stable logical step idempotency key or reconcile before retrying.
- Definition version and digest drift fail closed. `0.15.0` does not provide an automatic checkpoint migration engine; migration is an explicit application operation with its own audit record.
- SQLite and Postgres stores keep append-only checkpoint history. Retention, archival, encryption, tenant authorization, backups, and deletion remain deployment responsibilities.
- Checkpoints may contain prompts, model output, approval values, and business state. Treat them as sensitive records and avoid placing secrets in workflow state or metadata.
- Adapter callbacks, local tools, and agent nodes can produce external effects. Durable orchestration is not a distributed transaction.

## Existing Workflow Agents

The existing declarative agents remain supported and beta:

- `SequentialAgent` runs steps in order.
- `ParallelAgent` isolates a fan-out wave and merges only declared output/metadata keys.
- `LoopAgent` runs bounded refinement with `max_iterations` and an optional stop condition.

They preserve their existing `WorkflowStep` fields and error policies. Attach an `AgentRunStore` for their final replay projection. Use `WorkflowGraph` when per-transition checkpoint history, resume, or fork is required.

## Reference Examples

Offline examples:

- `examples/agents/durable_graph_workflow.py`
- `examples/agents/sequential_workflow.py`
- `examples/agents/parallel_workflow.py`
- `examples/agents/loop_workflow.py`
- `examples/agents/structured_workflow_outputs.py`
- `examples/agents/workflow_resume.py`
- `examples/agents/artifact_document_workflow.py`
- `examples/agents/research_report_workflow.py`
- `examples/agents/small_business_loan_agent.py`
- `examples/agents/hr_candidate_selection_agent.py`

The durable graph example uses mock models and SQLite, so it runs without provider credentials while still exercising process-style reconstruction, explicit resume, and fork lineage.
