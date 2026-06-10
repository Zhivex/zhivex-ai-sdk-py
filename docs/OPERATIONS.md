# Production Operations

This guide is the operations runbook for services built on the Zhivex AI SDK.

## Correlation IDs

Carry one application request ID through every boundary. Use stable field names so logs, traces, human approval UIs, and gateway dashboards can be joined.

| Field | Owner | Use |
| --- | --- | --- |
| `request_id` | application | inbound HTTP request, job, or user action |
| `session_id` | application or SDK session | conversation and checkpoint grouping |
| `run_id` | SDK agent runtime | one agent execution or resumed run |
| `gateway_attempt_id` | application log field | one gateway target attempt; derive from request ID, provider, model, target rank, and retry |
| `idempotency_key` | application | safe client/job retries for the same user action |

Put these IDs in agent metadata, gateway `on_attempt` logs, OpenTelemetry spans, approval records, and support tickets.

## Retry And Backoff

Provider adapters normalize HTTP failures into `ProviderHTTPError` where possible. `retryable` is true for transient statuses such as 408, 429, and 5xx unless the provider adapter overrides it. `retry_after_ms` is parsed from `Retry-After` headers when available.

Recommended pattern:

- keep an endpoint-level timeout budget in the application
- use bounded retries with backoff for provider calls and gateway fallbacks
- honor `retry_after_ms` when present
- do not retry validation errors, unsupported feature errors, auth failures, or policy denials
- make user actions idempotent before enabling client or worker retries

Gateway routing exposes `max_retries`, `retry_backoff_ms`, `retryable`, `retry`, and target rank through `GatewayConfig(on_attempt=...)`.

## Circuit Breakers

Use `create_circuit_breaker_middleware(...)` around expensive or flaky model calls. Record state transitions with provider, model, request ID, status, and failure count.

Circuit breakers should protect:

- provider outages
- repeated gateway target failures
- model deployments with intermittent capacity
- downstream services called by tools

Keep the breaker threshold low enough to reduce cascading failures but high enough to avoid opening on a single isolated user error.

## Cost And Budgets

Use `TokenUsage` from model and agent results for cost reporting. Persist provider, model, input tokens, output tokens, total tokens, and run ID.

Use `create_budget_guard(...)` and `create_safety_policy(...)` for agent-level ceilings. These budget guards can stop runaway runs before they become cost incidents:

- max steps
- max tool calls
- max tool errors
- max input, output, or total tokens

Budget guards are runtime tripwires. They do not replace billing meters, prepaid balances, tenant quotas, or app-owned cost allocation.

## Concurrency And Cancellation

The SDK is async-first. Production services should:

- set request and job timeouts before calling `run_agent(...)`, `stream_agent(...)`, `generate_text(...)`, or gateway calls
- propagate cancellation from HTTP disconnects or worker shutdowns
- keep tool implementations cooperative and timeout-aware
- use run stores when cancellation, replay, idempotency, pending approvals, or auditability matters
- use `cancel_agent_run_tree(...)` to mark stored run trees as cancelled

Stored cancellation records do not stop already-running provider calls or arbitrary app tools by themselves. Workers must check their own cancellation signals.

## Serverless And Workers

Serverless handlers fit short request/response generation, gateway calls, and small agent runs. Use durable workers for long-running agents, tool-heavy workflows, human approval handoffs, large file processing, and resumable jobs.

Serverless defaults:

- short endpoint timeout
- no in-memory run store for production state
- request-scoped provider clients
- trace/log export before returning

Worker defaults:

- durable run store, pending approval resume, and checkpoint store
- idempotency key per job
- cooperative cancellation
- bounded concurrency per provider/model
- dead-letter and retry policy owned by the queue

## Provider Error Normalization

Map SDK errors at your API boundary:

- `ValidationError` -> client error
- `UnsupportedFeatureError` -> client or configuration error
- `ConfigurationError` -> deployment/configuration error
- `ProviderHTTPError(retryable=True)` -> retryable upstream error
- `ProviderHTTPError(retryable=False)` -> non-retryable upstream error

Do not return raw provider response bodies to end users. Log redacted provider error details with request ID and provider/model metadata.

## Related Guides

- [PRODUCTION.md](./PRODUCTION.md)
- [OBSERVABILITY.md](./OBSERVABILITY.md)
- [../SECURITY.md](../SECURITY.md)
