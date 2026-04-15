# Observability

This guide covers the operational observability hooks that already exist in Zhivex AI SDK and the recommended way to use them in production services.

Related documents:

- [README.md](./README.md)
- [PRODUCTION_APIS.md](./PRODUCTION_APIS.md)
- [`examples/integrations/observability.py`](./examples/integrations/observability.py)

## Install

For OpenTelemetry-based agent tracing:

```bash
pip install "zhivex-ai-sdk[otel]"
```

For local work inside this repository:

```bash
make dev
.venv/bin/python -m pip install opentelemetry-api opentelemetry-sdk
```

## Foundation telemetry middleware

`create_telemetry_middleware(...)` emits lifecycle events around `generate(...)` calls.

The event payload always includes:

- `type`
- `model`
- `input`
- `startedAt`

Finish and error events also include:

- `finishedAt`
- `latencyMs`

Finish events include:

- `output`

Error events include:

- `error`

Recommended production log fields:

- request id from your HTTP layer
- provider and model id from `event["model"]`
- event type
- latency
- finish reason or exception class when available

## Gateway attempt hooks

`GatewayConfig(on_attempt=...)` lets you record every gateway attempt and retry decision.

The callback receives a dictionary with these keys:

- `provider`
- `modelId`
- `ok`
- `latencyMs`
- `errorMessage`
- `retryable`
- `retry`
- `targetRank`

Recommended uses:

- structured logs for fallback routing
- retry dashboards
- latency distributions by provider and model
- incident debugging when the selected provider changes unexpectedly

## Agent tracing with OpenTelemetry

`create_otel_agent_observer(...)` connects the agent runtime to an OpenTelemetry tracer.

Use it when you want spans around:

- agent runs
- guardrails
- tool execution
- handoffs

When you use agents in production, the most useful correlation fields are:

- request id from the API layer
- `run.id`
- `session.id`
- `agent.name`

If you store or emit agent traces, keep `run_id` and `session_id` available in your application logs so a single request can be followed across HTTP logs, telemetry middleware, and agent events.

## Recommended operating pattern

For production services:

- log one request id per inbound HTTP request
- include provider and model in every model-level event
- record gateway attempts separately from final responses
- capture SDK validation errors as application errors, not provider outages
- capture provider retryability so alerts distinguish bad requests from upstream instability

The integration example in [`examples/integrations/observability.py`](./examples/integrations/observability.py) shows a compact starting point for middleware and gateway logs.
