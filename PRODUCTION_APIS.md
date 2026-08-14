# Production APIs

This guide shows the recommended starting point for exposing Zhivex AI SDK through an HTTP API.

Related examples:

- [`examples/integrations/fastapi_chat_api.py`](./examples/integrations/fastapi_chat_api.py)
- [`examples/integrations/fastapi_streaming_api.py`](./examples/integrations/fastapi_streaming_api.py)
- [`examples/integrations/fastapi_gateway_api.py`](./examples/integrations/fastapi_gateway_api.py)
- [docs/GATEWAY.md](./docs/GATEWAY.md)
- [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md)
- [docs/OPERATIONS.md](./docs/OPERATIONS.md)
- [docs/PROTOCOLS.md](./docs/PROTOCOLS.md)
- [SECURITY.md](./SECURITY.md)

## Install

Install the SDK and the API integration extras:

```bash
pip install "zhivex-ai-sdk[api]"
```

For local work inside this repository:

```bash
make dev
.venv/bin/python -m pip install fastapi uvicorn
```

## Recommended pattern

For production-facing API servers:

- authenticate every non-health route and fail closed when credentials are not configured
- bind authenticated identities to a tenant-owned data partition; never trust a tenant header by itself
- enforce request-body, field-length, concurrency, and rate limits before invoking a provider
- import supported APIs from `zhivex_ai`
- prefer the current tier-1 providers for stable production API paths: OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex, Qwen, Kimi/Moonshot, DeepSeek, and vLLM
- keep Beta providers such as Meta Model API behind an application-owned allowlist, pinned model ID, privacy-tier policy, and provider-specific integration evidence
- validate request bodies with Pydantic
- pass `timeout_ms` explicitly from the API layer into SDK calls
- map SDK exceptions into stable HTTP error responses
- keep provider construction and fallback policy in one place
- put the SDK behind your own service layer before exposing it to clients

## Error mapping

The examples in this repository use the following default mapping:

- `ValidationError`, `ParseError`, `UnsupportedFeatureError` -> `400`
- `ProviderHTTPError` -> `503` when retryable, otherwise `502`
- `ConfigurationError` -> `500`
- any unexpected exception -> `500`

This keeps the public API stable even when upstream providers return provider-specific response shapes or status codes.

If you need a pass-through proxy instead of an application API, you can expose upstream status codes directly. For most product APIs, the safer default is to normalize them.

## Timeouts

The examples default to `30_000 ms` and pass that timeout into SDK calls. That gives the API layer one place to define request budgets.

For production services, prefer:

- one timeout budget per endpoint
- shorter timeouts for interactive chat routes
- longer timeouts only for explicitly long-running operations

## Streaming

The SDK already exposes transport helpers that map cleanly to FastAPI:

- `to_text_stream_response(...)` for plain text streaming
- `to_ui_message_stream_response(...)` for SSE-style UI message streams

The streaming example adapts those helpers into `fastapi.responses.StreamingResponse` so the API layer stays thin.

## Gateway APIs

When you want fallback routing in the API layer:

- create the gateway once at request time or through a dependency
- keep primary and fallback targets explicit
- return the selected provider and model in the JSON response
- treat routing as application policy, not client policy

The gateway example uses OpenAI as primary and Anthropic as fallback, but the pattern is the same for any supported provider set. DeepSeek can participate in text-only routes through `create_deepseek()`; gateway vision requests skip it rather than dropping image inputs.

For direct Anthropic Opus 5 routes, use the fixed `claude-opus-5` ID and let adaptive thinking remain the default unless the endpoint deliberately selects a supported effort. Do not configure non-default sampling, assistant prefill, server-side Web Fetch, or Priority Tier for this model. Keep Bedrock routing separate because the current Bedrock Converse adapter does not claim Opus 5.

For the strongest compatibility story, prefer tier-1 providers for the API paths you want to treat as part of your long-term contract.

For DeepSeek-backed APIs, pin a current V4 model ID and let the adapter own thinking/tool compatibility. Do not expose arbitrary `provider_options` to clients, and do not route image, embedding, audio, moderation, or hosted-tool workloads to DeepSeek because those surfaces are outside this SDK's DeepSeek contract.

For vLLM-backed APIs, keep the app contract tied to SDK primitives rather than vLLM custom endpoints. Text, streaming, structured output/tools, embeddings, transcription, and realtime ASR are supported through the OpenAI-compatible server when the served model/task supports them; custom endpoints such as tokenize, rerank, classify, and score should stay behind app-owned code if needed.

For direct Meta Model API routes, pin the Standard `muse-spark-1.2` model unless an explicit data-governance decision authorizes the Contributor tier. Keep `tool_choice` on `auto`, validate tool arguments in the application, and do not convert offline adapter tests into a production-readiness claim. Meta remains Beta and non-Tier-1 until an exact provider/model/artifact/SHA live smoke and the broader promotion criteria are satisfied.

## Agent APIs

For agent-backed API servers, keep application policy outside the SDK:

- construct authenticated, tenant-scoped clients in the request/worker layer and pass them through `deps=`
- supply dependencies again when resuming a suspended run; they are intentionally not persisted
- use `output_type` for response-shape validation, while keeping regulated decisions and business policy in application-owned services
- keep lifecycle hooks bounded and redacted; use approval policy for authorization and `AgentObserver` for tracing

- use `idempotency_key` for retryable user actions
- attach memory/checkpoint stores when sessions need recovery
- attach run stores when you need replay, snapshots, cancellation records, or pending approvals
- keep human authorization, notification, and audit policy in application storage
- pass request ids through agent metadata and logs

See [docs/AGENTS.md](./docs/AGENTS.md) and [docs/PRODUCTION.md](./docs/PRODUCTION.md) for the full runtime and production guidance.

## Protocol APIs

The current `0.18.1` line includes beta A2A v1, AG-UI, and Responses-compatible adapters. Use them behind the same production controls as any other public agent API:

- Resolve A2A skills and Responses `model` values to a server-owned allowlist of configured agents. Never construct providers from caller input.
- Authenticate before agent execution and derive tenant/task/thread/run ownership from the authenticated tenant and subject. A protocol ID, model alias, or tenant header is not authorization.
- Inject an official A2A task store, request-context builder/owner resolver, and queue manager for multi-process or durable deployments.
- Treat the Responses endpoint as a strict text/message create and stream subset. Optional GET/event replay exists only with an application-owned, tenant-scoped `ResponsesEventStore`; do not promise unsupported tool, background, retrieval, cancellation, or continuation endpoints.
- Resolve `HostedAgentRunOptions` from authenticated application state so sessions, dependencies, runtime, and idempotency never come from caller-controlled serialized objects.
- Use the official AG-UI encoder and keep UI state, resume/interrupt authorization, and reconnect persistence in the application.
- Apply request, field, concurrency, rate, provider-token, and output limits before public exposure. Redact protocol payloads and tool results in logs.
- Correlate external A2A task/context IDs, AG-UI thread/run IDs, or Responses IDs with the internal agent run ID and authenticated identity.

`create_agent_playground_app(...)` and `zhivex playground` are local development tools. The CLI refuses non-loopback binds; neither surface supplies authentication, TLS, distributed limits, tenant persistence, or approval UI. Do not deploy them as a public console.

See [docs/PROTOCOLS.md](./docs/PROTOCOLS.md) for supported routes, extras, wire boundaries, and limitations.

## Workflow APIs

Durable workflow graphs were introduced in `0.15.0` and remain beta in the current `0.18.1` line; expose them behind an application-owned API contract rather than returning SDK checkpoint objects directly.

Recommended endpoint boundaries:

- A start endpoint validates the business request, supplies a server-owned workflow definition and idempotency key, and returns the workflow `run_id`, status, and an application status URL.
- A status endpoint loads the latest checkpoint through a tenant-scoped repository and returns an allowlisted projection. Workflow state, prompts, model output, interrupt payloads, and metadata may contain sensitive data.
- A resume endpoint authorizes the specific pending action and acknowledges the exact `interrupt_id` or agent `approval_id`. Do not treat possession of a run id as authorization.
- A fork endpoint is a privileged operation. Record the source run/checkpoint, caller, reason, state overrides, and new run id in the business audit log.
- A cancellation endpoint authorizes the specific run and records caller/reason before invoking cooperative `cancel_workflow(...)`.

Use a stable application idempotency key for start and fork requests. `WorkflowCheckpointStore.append(...)` protects checkpoint ordering with an expected sequence, but it does not make an external business write transactional with workflow progress. Side-effecting nodes must use destination-supported idempotency, an outbox, or explicit reconciliation/compensation.

Functional graph executors receive ephemeral `deps` plus a stable logical step idempotency key and may return only finite JSON durable results. Treat them like retryable activities: a database or API write must consume that key or use an outbox. Do not place credentials, clients, transactions, or authorization objects in `WorkflowFunctionResult`.

`InMemoryWorkflowCheckpointStore` and `InMemoryWorkflowLeaseManager` are suitable only for tests and one-process demos. SQLite survives local process reconstruction but is not the recommended shared boundary for multiple API replicas. Pair the optional Postgres checkpoint and lease managers, or application implementations of both protocols, for shared workers and run contention/expiry/takeover tests against the actual deployment database.

Reconstruct the same `WorkflowGraph` definition before resume or fork. The SDK rejects mismatched workflow names, definition versions, and definition digests. Supply runtime `deps` again; never place database clients, credentials, authorization objects, or other runtime dependencies into checkpoint state.

Idempotent re-entry does not take over a `running` workflow automatically. It fails closed unless `recover_running=True`. With a `WorkflowLeaseManager`, takeover succeeds only after lease expiry and increments a fencing token; without one, the operator must prove the previous worker is gone. Recovery records `workflow-recovered`, but external writes still require the logical step idempotency key, destination-supported fencing, or reconciliation.

The DBOS, Temporal, Prefect, and Restate adapter factories expose versioned callback request/outcome contracts only. They do not start those engines or certify their persistence, retry, signal, or worker behavior. Keep the real engine client and worker integration in the application layer and test it end to end.

See [docs/WORKFLOWS.md](./docs/WORKFLOWS.md) for the complete beta contract and [`examples/agents/durable_graph_workflow.py`](./examples/agents/durable_graph_workflow.py) for an offline SQLite reconstruction/resume/fork example.

The production agent example requires `ZHIVEX_AGENT_API_TOKEN`, `ZHIVEX_TENANT_ID`, and a server-owned `ZHIVEX_AGENT_MODEL`; it uses a fixed server-side Postgres table prefix, limits request bodies and Pydantic fields, and applies a small process-local rate limit. Put a distributed limiter at the API gateway when running more than one process or replica. Clients must send both `Authorization: Bearer ...` and the matching `X-Tenant-ID`; a user-controlled tenant header is not an authorization mechanism on its own, and clients do not select provider model IDs.
