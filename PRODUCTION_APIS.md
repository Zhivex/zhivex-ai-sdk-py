# Production APIs

This guide shows the recommended starting point for exposing Zhivex AI SDK through an HTTP API.

Related examples:

- [`examples/integrations/fastapi_chat_api.py`](./examples/integrations/fastapi_chat_api.py)
- [`examples/integrations/fastapi_streaming_api.py`](./examples/integrations/fastapi_streaming_api.py)
- [`examples/integrations/fastapi_gateway_api.py`](./examples/integrations/fastapi_gateway_api.py)
- [docs/GATEWAY.md](./docs/GATEWAY.md)
- [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md)
- [docs/OPERATIONS.md](./docs/OPERATIONS.md)
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

- import supported APIs from `zhivex_ai`
- prefer the current tier-1 providers for stable production API paths: OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex, Qwen, Kimi/Moonshot, and vLLM
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

The gateway example uses OpenAI as primary and Anthropic as fallback, but the pattern is the same for any supported provider set. Anthropic and vLLM are now part of the tier-1 text-generation story as well.

For the strongest compatibility story, prefer tier-1 providers for the API paths you want to treat as part of your long-term contract.

For vLLM-backed APIs, keep the app contract tied to SDK primitives rather than vLLM custom endpoints. Text, streaming, structured output/tools, embeddings, transcription, and realtime ASR are supported through the OpenAI-compatible server when the served model/task supports them; custom endpoints such as tokenize, rerank, classify, and score should stay behind app-owned code if needed.

## Agent APIs

For agent-backed API servers, keep application policy outside the SDK:

- use `idempotency_key` for retryable user actions
- attach memory/checkpoint stores when sessions need recovery
- attach run stores when you need replay, snapshots, or cancellation records
- keep human approval queues and authorization in application storage
- pass request ids through agent metadata and logs

See [docs/AGENTS.md](./docs/AGENTS.md) and [docs/PRODUCTION.md](./docs/PRODUCTION.md) for the full runtime and production guidance.
