# Agent Protocols And Hosting

Zhivex AI SDK `0.16.0` introduces beta adapters for A2A v1, AG-UI, and a constrained OpenAI Responses-compatible hosting surface. The configured `Agent` remains the source of runtime behavior; protocols are transport boundaries, not new authorization or business-policy engines.

## Installation

```bash
pip install "zhivex-ai-sdk[api]"       # Responses host and playground
pip install "zhivex-ai-sdk[a2a]"       # official a2a-sdk server
pip install "zhivex-ai-sdk[ag-ui]"     # official AG-UI event encoder
```

The A2A extra is pinned to the supported `a2a-sdk` 1.x line and the AG-UI extra to `ag-ui-protocol` 0.1.x. Both protocol surfaces remain beta and may track upstream minor changes in a future Zhivex minor release.

Import these adapters from `zhivex_ai.integrations.protocols` and `zhivex_ai.integrations.responses`. Existing top-level imports remain available for compatibility, but new code should make the Beta integration boundary explicit.

## A2A v1

```python
from zhivex_ai.integrations.protocols import (
    A2AAgentExecutor,
    create_a2a_agent_card,
    create_a2a_app,
)

card = create_a2a_agent_card(
    agent,
    url="https://agents.example.com/a2a",
    version="1.0.0",
    description="Answer support questions.",
)
app = create_a2a_app(
    executor=A2AAgentExecutor(
        agent,
        run_options_resolver=resolve_run_options,
        on_protocol_event=record_protocol_event,
    ),
    card=card,
    authorize=authorize,
    task_store=durable_task_store,
    request_context_builder=tenant_context_builder,
    queue_manager=queue_manager,
)
```

The server uses the official A2A Python SDK and exposes:

- `GET /.well-known/agent-card.json`
- JSON-RPC v1 at `POST /a2a/rpc`
- HTTP+JSON v1 under `/a2a`, including message send/stream and task operations

Clients must send `A2A-Version: 1.0`. v0.3 compatibility is intentionally disabled. The in-process `A2AAgentExecutor.send_message()` / `stream_message()` helpers are useful for contract tests; public network hosting should use `create_a2a_app()` so the official SDK owns protobuf JSON, task lifecycle, JSON-RPC envelopes, and SSE framing.

The default official in-memory task store is process-local. `task_store`,
`request_context_builder`, and `queue_manager` are passed to the official A2A
request handler for application-owned durability, ownership, and queueing.
Replace them before multi-replica or multi-tenant production use. Derive
ownership from authenticated tenant and subject; a task ID or caller-provided
user name is not authorization. Authentication is required on every route
except the exact public `GET /.well-known/agent-card.json` route, and actual
request bytes are bounded even when `Content-Length` is absent or false.

## AG-UI

```python
from zhivex_ai.integrations.protocols import stream_agent_ag_ui, to_ag_ui_sse_response

events = stream_agent_ag_ui(
    agent=agent,
    prompt="Summarize the case",
    thread_id="thread-123",
    run_options_resolver=resolve_run_options,
    on_protocol_event=record_protocol_event,
)
response = to_ag_ui_sse_response(events)
```

The adapter maps run, text, tool-call, tool-result, finish, and error events. `to_ag_ui_sse_response()` validates and encodes them with the official `ag-ui-protocol` `EventEncoder`; it is separate from the SDK-owned UI message transport in `zhivex_ai.ui`.

Applications own thread persistence, resume payload authorization, human interrupts, state snapshots/deltas, reconnect policy, and client rendering. `ProtocolLimits` bounds thread/run identifiers and prompt text. Failures expose `"Agent execution failed."` unless an application-owned `error_mapper` returns another reviewed public message. Do not place secrets or unrestricted tool results in forwarded props or UI state.

## Trusted run context and protocol events

`ProtocolRunOptionsResolver` receives a `ProtocolInvocation` and returns
`HostedAgentRunOptions` with a server-created session, ephemeral `deps`, stable
idempotency key, and optional runtime. Resolve these only from authenticated
application state; never accept serialized dependencies or runtime objects from
the protocol payload. `ProtocolInvocation.request` and `.payload` are
content-bearing trusted callback inputs and must not be persisted automatically.

`ProtocolEventCallback` receives a sanitized lifecycle record containing
protocol/action/status, external identifiers, alias, internal run id, and error
class where available. It intentionally excludes prompt, request, dependencies,
tool payloads, and exception text. `ProtocolErrorMapper` may expose a reviewed
public error; the default never returns raw exception details.

## Responses-compatible hosting

```python
from zhivex_ai.integrations.protocols import ProtocolLimits
from zhivex_ai.integrations.responses import InMemoryResponsesEventStore, create_responses_app

app = create_responses_app(
    agents={"support": support_agent},
    authorize=lambda request: request.headers.get("authorization") == expected_token,
    run_options_resolver=resolve_run_options,
    limits=ProtocolLimits(max_request_bytes=512_000, max_text_chars=32_000),
    event_store=InMemoryResponsesEventStore(),  # development only
)
```

`POST /v1/responses` accepts only a server-owned `model` alias, string or
message-list text input, optional instructions, and a boolean `stream`.
Unknown request, message, and content fields are rejected instead of ignored.
`ProtocolLimits` bounds actual request bytes, aliases, message/part counts, and
aggregate input text. Streaming emits the Responses lifecycle from
`response.created` through `response.completed`, including monotonic
`sequence_number`, SSE `id`, typed events, `Cache-Control: no-cache,
no-transform`, and disabled proxy buffering.

When a `ResponsesEventStore` is configured, completed/failed response objects
are available at `GET /v1/responses/{response_id}` and events at
`GET /v1/responses/{response_id}/events`. `after_sequence` or `Last-Event-ID`
replays only later events. `InMemoryResponsesEventStore` is process-local and
ignores tenant ownership; production implementations must scope every operation
using the authenticated `ProtocolInvocation`, provide retention, and coordinate
concurrent writers across replicas.

This is a compatibility host for agent text output, not a clone of every
OpenAI Responses feature. It rejects non-message items, non-text content,
hosted tools, `background`, request-controlled `store`, retrieval, cancellation,
and provider-native continuation semantics. Put those behind explicit
application APIs if needed.

## Security and production checklist

- Supply `authorize=` or enforce equivalent authentication middleware; protocol IDs are never credentials.
- Resolve aliases to server-configured agents. Do not let clients construct providers or pass raw provider options.
- Enforce tenant ownership in A2A task/context/queue infrastructure and Responses session/run/event stores.
- Keep the default 1 MiB request limit or choose a reviewed lower limit; add field, rate, concurrency, and output limits at the edge.
- Bind request IDs, tenant/subject, protocol, external task/thread/response ID, internal run ID, agent alias, provider, and model in traces.
- Redact prompts, tool inputs/results, artifacts, and error details before logs or protocol metadata.
- The `zhivex playground` command refuses non-loopback hosts and has no authentication. Never wrap the raw playground app as a public console.
