# Production Agent Operations

This guide covers practical backend defaults for production agent services.

## Request Boundary

Create one application request id per inbound request and carry it through:

- HTTP logs
- agent `metadata`
- approval records
- trace exports
- run-store metadata

Use `idempotency_key` with `run_agent(...)` or `stream_agent(...)` when a user action can be retried by the client or job runner. The SDK returns the existing completed run state when the key already exists in the configured run store.

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

For synchronous policies, return an `ApprovalDecision` or bool from `approval_policy`. For asynchronous HITL systems, return `ApprovalDecision.require_human(...)`, load the pending approval with `get_pending_agent_approvals(...)`, and call `resume_agent_run(...)` when the user responds.

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

Use run stores when you need cancellation records, idempotency, replay, or auditability. `cancel_agent_run_tree(...)` marks stored run trees as cancelled; active workers and tools must still cooperate to interrupt work.

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

Use `resume_agent(...)` for session/checkpoint recovery. Use `resume_agent_run(...)` to continue a suspended run after a pending approval is approved or denied. Use `replay_agent_run(...)` to inspect what happened without re-running providers. Use `cancel_agent_run_tree(...)` to mark stored run trees as cancelled; application workers must still cooperate to interrupt active tasks.

## Release Evidence

For release candidates, run `make release-evidence` to write the local release gate output to `docs/releases/<version>-evidence.md`. Treat live provider smoke as a separate environment-dependent record and list every skipped provider with its missing credential or model environment variable.

## Security

For secrets, data retention, MCP, hosted tools, file access, shell-like capabilities, and secure tool defaults, see [../SECURITY.md](../SECURITY.md).
