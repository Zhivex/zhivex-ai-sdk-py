# Observability

This guide covers the operational observability hooks in Zhivex AI SDK and the recommended way to use them in production services.

Related examples:

- `examples/integrations/observability.py`
- `examples/integrations/operations_hardening.py`
- [../PRODUCTION_APIS.md](../PRODUCTION_APIS.md)
- [../STABILITY.md](../STABILITY.md)
- [AGENTS.md](./AGENTS.md)
- [OPERATIONS.md](./OPERATIONS.md)

## Install

```bash
pip install "zhivex-ai-sdk[otel]"
```

For local repo work:

```bash
make dev
```

## Foundation Telemetry

`create_telemetry_middleware(...)` emits lifecycle events around model calls. The event payload includes model identity, input, start time, finish time, latency, output, or error.

The telemetry middleware is a Beta API. It is suitable for evaluated production adoption behind an application-owned observability boundary, but its exact event ergonomics may change between minor releases with changelog coverage.

Recommended log fields:

- `request_id` from your HTTP or worker layer
- provider and model ID
- event type
- latency
- finish reason or exception class

Do not log raw prompts, tool inputs, provider payloads, or response bodies unless your application has redacted and classified them.

## OpenTelemetry

Install the optional extra when exporting OpenTelemetry spans:

```bash
pip install "zhivex-ai-sdk[otel]"
```

Use `create_otel_agent_observer(...)` to connect agent runs, guardrails, tools, handoffs, summaries, and errors to your tracer. Your service should set resource attributes such as service name, deployment environment, tenant, and region through the OpenTelemetry SDK. The SDK-level observer should receive already-approved metadata, not secrets.

Per-tool guardrail events use the existing `guardrail` event with `scope="tool"`, `tool_name`, `tool_stage`, and whether a replacement occurred. They intentionally omit tool input, tool output, runtime dependencies, and underlying policy exception details; keep any additional guardrail metadata low-cardinality and non-sensitive.

`create_otel_agent_observer(...)` and its concrete observer type are Beta. Keep exporter configuration and your durable telemetry schema application-owned rather than treating SDK span names or attributes as an independently stable storage contract.

Recommended span attributes:

- `request_id`
- `session_id`
- `run_id`
- `agent.name`
- provider and model
- idempotency key
- tool name and permission tags
- approval ID when a tool waits for human review

## Gateway Attempts

`GatewayConfig(on_attempt=...)` receives provider, model, success status, latency, retryability, retry number, target rank, and error text. Use it for fallback dashboards and incident debugging.

Standardize gateway attempt logs with:

- `request_id`
- `gateway_attempt_id`
- provider and model
- target rank
- retry number
- success status
- retryable status
- latency
- redacted error class or message

## Agent Tracing

`create_otel_agent_observer(...)` connects the agent runtime to an OpenTelemetry tracer for agent runs, guardrails, tools, and handoffs.

Useful correlation fields:

- `request_id`
- `run.id`
- `session.id`
- `agent.name`
- provider and model

Use `create_agent_trace_artifact(...)`, `summarize_agent_trace(...)`, and `replay_agent_run(...)` for persisted run-state analysis without re-running providers.

## Durable Workflow Transitions

`WorkflowGraph` records an append-only `WorkflowCheckpoint` for every durable transition. Use checkpoint history as the workflow audit timeline and `WorkflowRunResult.state_snapshot` as the compatibility projection into existing agent replay/trace tooling.

Recommended workflow correlation fields:

- workflow `run_id`, name, `definition_version`, and definition digest
- checkpoint id and monotonically increasing sequence
- transition type and timestamp
- node name, node status, and logical attempt
- logical step idempotency key and child agent run id
- interrupt id and phase, without its raw payload
- source run/checkpoint ids for forks
- request, session, tenant, and trace ids supplied by the application
- adapter backend and executor reference when a callback adapter is used
- execution-lease owner reference and monotonic fencing token, never the secret lease token

Alert on repeated sequence conflicts, definition mismatch, exhausted step retries, long-lived suspended runs, repeated recovery of a `running` node, and forks without an application audit reason. A checkpoint append confirms orchestration progress; it does not prove that an external side effect committed. Correlate destination idempotency/reconciliation evidence separately.

The runtime agent observer emits `zhivex.agent.run`, `zhivex.agent.model`,
`zhivex.agent.tool`, and `zhivex.agent.handoff` spans with safe identity,
status, duration, finish-reason, and token-usage attributes. Trace artifacts and
summaries also expose `started_at_ms`, `finished_at_ms`, and `duration_ms` when
the persisted state contains both timestamps. Applications should add tenant,
request, and business correlation at their boundary without adding prompt or
tool payload content.

Do not export full checkpoint state, node output, resume values, adapter envelopes, or interrupt payloads as span attributes. These fields can contain prompts, model output, approval data, and regulated business records.

## Protocol Correlation

A2A, AG-UI, and Responses identifiers are external transport identifiers. Correlate them with the authenticated application identity and internal agent run; do not use them as authorization tokens.

Recommended fields:

- protocol and route/binding
- authenticated tenant and subject identifiers or irreversible references
- A2A task/context IDs, AG-UI thread/run IDs, or Responses response/item IDs
- internal agent `run_id`, `session_id`, and agent/model alias
- provider and provider model from server configuration
- request size, duration, terminal status, and normalized error class

Keep prompts, AG-UI state/forwarded props, A2A message parts/artifacts, Responses input/output, and tool arguments/results out of default span attributes. If content logging is explicitly approved, apply DLP/redaction and the deployment's retention policy before export.

## Hooks, Middleware, And Events

These extension surfaces have distinct responsibilities:

- `AgentHooks` observes in-process agent, physical model-call, tool, approval, handoff, and error lifecycle points.
- `AgentMiddleware` wraps a complete root run and can enforce application boundaries or return an application cache hit.
- foundation `wrap_language_model(...)` / generation middleware decorates a model independently of the agent runtime.
- `AgentObserver` creates operational spans.
- `AgentEvent` remains the ordered event-stream and trace-history contract.

Do not use a lifecycle hook as an authorization policy or durable event sink. Approval policy remains authoritative, and event delivery should be reconciled by `run_id`. Hook payloads can contain prompts, tool inputs, outputs, and the in-process dependency object; apply the same redaction rules as model telemetry and never stringify or export `context.deps`.

## Operating Pattern

For production services:

- log one `request_id` per inbound request
- propagate `session_id`, `run_id`, and `gateway_attempt_id` when available
- keep provider/model on every model-level event
- log gateway attempts separately from final responses
- treat SDK validation errors as application errors
- preserve retryability so alerts distinguish bad requests from upstream instability
- record `ProviderHTTPError.retryable` and `ProviderHTTPError.retry_after_ms`
- emit cost fields from `TokenUsage` where available
- redact prompts, tool inputs, provider payloads, traces, and error bodies according to application policy
