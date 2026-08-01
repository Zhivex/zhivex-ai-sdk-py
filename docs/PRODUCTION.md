# Production Agent Operations

This guide covers practical backend defaults for production agent services.

## Request Boundary

Create one application request id per inbound request and carry it through:

- HTTP logs
- agent `metadata`
- approval records
- trace exports
- run-store metadata

Use `idempotency_key` with `run_agent(...)` or `stream_agent(...)` when a user action can be retried by the client or job runner. Built-in stores claim the key atomically: the SDK returns the existing completed or in-progress run identity when the key already exists and does not start a second model loop. Custom production stores must implement atomic `claim_idempotency_key(...)`.

Authenticate and authorize before constructing an agent or gateway request. Bind each credential to a server-owned tenant partition and provider/model policy; do not treat `X-Tenant-ID`, model IDs, tool names, or run IDs supplied by a client as authorization. Enforce request-body and field-size limits while the body is read, plus distributed rate/concurrency limits before provider work begins. The production FastAPI example demonstrates a fail-closed single-tenant bearer boundary and an in-process limiter; replace the limiter with a shared gateway or datastore implementation across replicas.

## Storage Defaults

Use Postgres for production memory and checkpoint persistence:

```python
from zhivex_ai import create_postgres_agent_memory_store, create_postgres_checkpoint_store

memory = create_postgres_agent_memory_store(dsn)
checkpoints = create_postgres_checkpoint_store(dsn)
```

Use run stores when you need idempotency, cancellation-tree records, replay, audit snapshots, or pending approvals. SQLite and in-memory stores are beta/local development choices.

## Approval Ownership

The SDK emits approval requests, applies `approval_policy`, and can persist suspended runs with pending approvals in the configured run store. Your application owns:

- the user/admin approval UI
- role checks
- audit records
- escalation and timeout policy

For synchronous policies, return an `ApprovalDecision` or bool from `approval_policy`. A tool marked `requires_approval=True` is denied when no policy is configured. For asynchronous HITL systems, return `ApprovalDecision.require_human(...)`, load the pending approval with `get_pending_agent_approvals(...)`, and call `resume_agent_run(...)` when the user responds. Built-in run stores claim that pending approval atomically before any tool execution; duplicate workers are rejected instead of repeating a side effect. Custom stores used for approval resume must implement the same atomic claim contract.

## Timeouts And Retries

Set endpoint-level timeout budgets in the API layer and pass them into `run_agent(...)`, `stream_agent(...)`, or the underlying model calls. Treat provider retries as upstream resilience, not as a substitute for API request budgets.

Retry only transient provider or network failures. `ProviderHTTPError.retryable` is the application-facing signal for upstream retryability, and `ProviderHTTPError.retry_after_ms` should be honored when present. Do not retry `ValidationError`, `UnsupportedFeatureError`, policy denials, malformed prompts, or authentication/configuration failures.

Gateway calls expose `max_retries`, `retry_backoff_ms`, retry number, target rank, and retryability through `GatewayConfig(on_attempt=...)`. Pair gateway retries with an application idempotency key when the user action can be submitted more than once.

For production gateway routes, set `GatewayConfig(fail_on_missing_adapter=True)` when every configured provider target is expected to have an adapter at startup. This prevents an accidentally missing primary adapter from being hidden by a successful fallback. Skipped targets now emit `on_attempt` payloads, so missing adapters, capability skips, vision skips, and cost-budget skips can be correlated with request logs.

Gateway routes do not fall back on provider refusals by default. A result with `finish_reason="refusal"` or `provider_finish_reason="refusal"` is recorded as an attempt and returned from the selected target; set `GatewayConfig(fallback_on_refusal=True)` when an application should explicitly retry refusals on fallback targets.

Use `create_circuit_breaker_middleware(...)` around provider calls that should fail fast during an outage. Log state changes with request id, provider, model, failure count, and breaker status.

## Provider Error Normalization

At the API boundary, map SDK errors consistently:

- `ValidationError` -> bad request
- `UnsupportedFeatureError` -> unsupported request or disabled capability
- `ConfigurationError` -> deployment/configuration failure
- `ProviderHTTPError(retryable=True)` -> retryable upstream failure
- `ProviderHTTPError(retryable=False)` -> non-retryable upstream failure

Never return raw provider response bodies to users. Redact error bodies before support logs or traces.

## Cost And Budget Guards

Persist provider, model, `run_id`, `session_id`, input tokens, output tokens, and total tokens when `TokenUsage` is available. Use those fields for app-owned billing and tenant reporting.

Use `create_budget_guard(...)` and `create_safety_policy(...)` for runtime ceilings on steps, tool calls, tool errors, and token usage. Budget guards are safety tripwires; tenant quotas and billing meters still belong in your application.

## Concurrency And Cancellation

The SDK is async-first. Keep concurrency bounded per provider/model and give every production call a timeout budget. Tool implementations should accept cancellation from the worker or HTTP layer and avoid unbounded network calls.

Tool timeouts are an uncertainty boundary, not proof that an external action was rolled back. The SDK raises `ToolExecutionOutcomeUnknown` and stops the agent loop when a tool exceeds `ToolExecutionOptions.timeout_ms`. Use `ToolExecutionContext.idempotency_key` with the downstream service to reconcile the operation before retrying; Python cannot terminate a synchronous callable that is already running in a worker thread.

Use run stores when you need cancellation records, idempotency, replay, or auditability. Built-in stores cancel atomically and use state revisions so a late worker completion cannot overwrite `cancelled`. Cancellation does not terminate a provider request, thread, or external side effect already in flight; workers and tools must still cooperate to stop promptly.

## Serverless Vs Workers

Use serverless handlers for short generation, streaming, or gateway requests that fit within platform timeouts. Use long-running workers for tool-heavy agents, human approval handoffs, resumable workflows, large file processing, and jobs that need durable checkpoints.

Serverless handlers should avoid production reliance on in-memory stores. Workers should use durable run/checkpoint stores, idempotency keys, bounded concurrency, cooperative cancellation, and queue-owned retry/dead-letter policy.

## Observability

Record:

- `run_id`
- `session_id`
- `agent.name`
- provider/model
- idempotency key
- request id
- approval ids
- tool names and permission tags

Use `create_otel_agent_observer(...)` for OpenTelemetry spans, and `create_agent_trace_artifact(...)` / `summarize_agent_trace(...)` for persisted run-state analysis.

## Recovery

Use `resume_agent(...)` for session/checkpoint recovery. Use `resume_agent_run(...)` to continue a suspended run after a pending approval is approved or denied. For beta durable workflow graphs, reconstruct the exact definition with the same persistent store, then use `resume_workflow(...)` for one suspended run or `fork_workflow(...)` for an auditable new lineage from a selected checkpoint. Re-entering a still-running idempotent graph fails closed; set `recover_running=True` only after lease/operator reconciliation proves the previous worker is gone.

Use `replay_agent_run(...)` to inspect the agent-state projection without re-running providers. Use workflow checkpoint history for authoritative graph recovery and routing decisions. Use `cancel_agent_run_tree(...)` to cancel the root and descendants visible while it traverses the agent run store. Each stored transition is atomic, but external effects and multi-record traversal are not a distributed transaction: stop new dispatch, use stable destination idempotency keys, and reconcile again after workers settle.

## Release Evidence

For release candidates, run `make release-evidence` to write the local release gate output to `docs/releases/<version>-evidence.md`. Treat live provider smoke as a separate environment-dependent record and list every skipped provider with its missing credential or model environment variable.

## Security

For secrets, data retention, MCP, hosted tools, file access, shell-like capabilities, and secure tool defaults, see [../SECURITY.md](../SECURITY.md).
