# Stability

Zhivex AI SDK uses three stability levels so production integrators can understand which surfaces are intended to remain predictable over time.

Supported public imports should come from `zhivex_ai`. Deep imports from internal modules are not part of the stable contract unless this document names an explicit exception.

Related documents:

- [README.md](./README.md)
- [VERSIONING.md](./VERSIONING.md)
- [CHANGELOG.md](./CHANGELOG.md)

## Stable

These APIs are the supported public contract for application code and production integrations:

- Provider factories: `create_openai`, `create_anthropic`, `create_azure_openai`, `create_gemini`, `create_vertex`
- Text generation: `generate_text`, `stream_text`
- Structured output: `generate_object`, `stream_object`
- Grounded text: `generate_grounded_text`
- Embeddings: `embed`, `embed_many`
- Agent runtime: `Agent`, `run_agent`, `stream_agent`, `resume_agent`
- Gateway: `GatewayAttempt`, `GatewayConfig`, `GatewayError`, `GatewayImageAttachment`, `GatewayMessage`, `GatewayModelTarget`, `GatewayObjectResponse`, `GatewayResponse`, `create_gateway`
- Core errors: `ProviderHTTPError`, `ConfigurationError`, `ValidationError`, `UnsupportedFeatureError`
- HTTP and SSE helpers: `HTTPResponse`, `stream_sse`, `to_sse_response`, `to_sse_stream`, `to_text_stream`, `to_text_stream_response`, `to_ui_message_stream_response`

The stable surface is intentionally narrow. It reflects the most defendable cross-provider experience and the main API-building primitives in this SDK today.

## Beta

These APIs are supported and documented, but they may still change between minor releases as the SDK matures:

- Middleware helpers
- Model catalog helpers
- Postgres-backed memory and checkpoint stores
- MCP helpers and MCP-backed registries

Beta APIs still require changelog coverage when they change, but they do not carry the same compatibility guarantees as the stable surface.

## Experimental

These areas are available for evaluation, but they should not be treated as a long-term compatibility contract yet:

- Realtime and live voice flows, including `stream_live_agent()`
- Provider-native hosted tools and escape hatches that do not map cleanly to the portable contract
- Provider areas currently marked as `native-only` or `compatibility` in the support matrix

Experimental areas may change faster than the rest of the SDK. Production adopters should isolate usage behind their own service layer before depending on them.

## Provider scope

The current tier-1 provider story for the stable surface is:

- OpenAI
- Anthropic
- Azure OpenAI
- Gemini
- Vertex

In this repository, tier-1 means the provider is part of the stable surface story, production API guidance, and support-matrix contract checks.

Anthropic is included in the tier-1 set for text-generation API paths. Embeddings, transcription, and speech remain outside the current Anthropic provider surface in this SDK.

Other providers remain useful, but they should be evaluated with the support matrix and the stability level of the specific feature area in mind.
