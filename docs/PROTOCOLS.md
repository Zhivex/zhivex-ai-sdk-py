# Agent Protocols And Hosting

Zhivex AI SDK `0.16.0` introduces beta adapters for A2A v1, AG-UI, and a constrained OpenAI Responses-compatible hosting surface. The configured `Agent` remains the source of runtime behavior; protocols are transport boundaries, not new authorization or business-policy engines.

## Installation

```bash
pip install "zhivex-ai-sdk[api]"       # Responses host and playground
pip install "zhivex-ai-sdk[a2a]"       # official a2a-sdk server
pip install "zhivex-ai-sdk[ag-ui]"     # official AG-UI event encoder
```

The A2A extra is pinned to the supported `a2a-sdk` 1.x line and the AG-UI extra to `ag-ui-protocol` 0.1.x. Both protocol surfaces remain beta and may track upstream minor changes in a future Zhivex minor release.

## A2A v1

```python
from zhivex_ai import (
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
app = create_a2a_app(executor=A2AAgentExecutor(agent), card=card)
```

The server uses the official A2A Python SDK and exposes:

- `GET /.well-known/agent-card.json`
- JSON-RPC v1 at `POST /a2a/rpc`
- HTTP+JSON v1 under `/a2a`, including message send/stream and task operations

Clients must send `A2A-Version: 1.0`. v0.3 compatibility is intentionally disabled. The in-process `A2AAgentExecutor.send_message()` / `stream_message()` helpers are useful for contract tests; public network hosting should use `create_a2a_app()` so the official SDK owns protobuf JSON, task lifecycle, JSON-RPC envelopes, and SSE framing.

The default official in-memory task store is process-local. Replace it and provide a deployment-owned `ServerCallContextBuilder`/owner resolver before multi-replica or multi-tenant production use. Derive ownership from authenticated tenant and subject; a task ID or caller-provided user name is not authorization.

## AG-UI

```python
from zhivex_ai import stream_agent_ag_ui, to_ag_ui_sse_response

events = stream_agent_ag_ui(
    agent=agent,
    prompt="Summarize the case",
    thread_id="thread-123",
)
response = to_ag_ui_sse_response(events)
```

The adapter maps run, text, tool-call, tool-result, finish, and error events. `to_ag_ui_sse_response()` validates and encodes them with the official `ag-ui-protocol` `EventEncoder`; it is separate from the SDK-owned UI message transport in `zhivex_ai.ui`.

Applications own thread persistence, resume payload authorization, human interrupts, state snapshots/deltas, reconnect policy, and client rendering. Do not place secrets or unrestricted tool results in forwarded props or UI state.

## Responses-compatible hosting

```python
from zhivex_ai import create_responses_app

app = create_responses_app(
    agents={"support": support_agent},
    authorize=lambda request: request.headers.get("authorization") == expected_token,
)
```

`POST /v1/responses` accepts a server-owned `model` alias, string or message-list input, optional instructions, and `stream=true`. Streaming emits the Responses lifecycle from `response.created` through `response.completed`, including monotonic `sequence_number` fields and typed SSE event names.

This is a compatibility host for agent text output, not a clone of every OpenAI Responses feature. It currently rejects non-message input items and non-text content. It does not expose arbitrary provider credentials, provider model IDs, hosted tools, background mode, response persistence, retrieval, cancellation, or provider-native continuation semantics. Put those behind explicit application APIs if needed.

## Security and production checklist

- Supply `authorize=` or enforce equivalent authentication middleware; protocol IDs are never credentials.
- Resolve aliases to server-configured agents. Do not let clients construct providers or pass raw provider options.
- Enforce tenant ownership in task, session, run, and artifact stores.
- Keep the default 1 MiB request limit or choose a reviewed lower limit; add field, rate, concurrency, and output limits at the edge.
- Bind request IDs, tenant/subject, protocol, external task/thread/response ID, internal run ID, agent alias, provider, and model in traces.
- Redact prompts, tool inputs/results, artifacts, and error details before logs or protocol metadata.
- The playground binds to loopback by default and has no authentication. Never expose it directly to a public network.
