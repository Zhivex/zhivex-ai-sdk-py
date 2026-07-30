# Observability

This guide covers the operational observability hooks in Zhivex AI SDK and the recommended way to use them in production services.

Related examples:

- `examples/integrations/observability.py`
- `examples/integrations/operations_hardening.py`
- [../PRODUCTION_APIS.md](../PRODUCTION_APIS.md)
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
