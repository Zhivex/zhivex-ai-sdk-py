# Python SDK Parity Matrix

This matrix records the Python SDK maturity boundary against the broader Zhivex SDK platform story. It is a product-readiness guide, not a promise that every TypeScript API must exist in Python.

Machine-readable companion: [parity_matrix.json](./parity_matrix.json).

Legend:

- Implemented: code exists in this repo
- Documented: top-level or `docs/` guidance exists
- Offline-tested: deterministic tests cover the behavior without provider credentials
- Live-smoked: optional smoke path exists for real providers
- Stability: stable, beta, or experimental according to `STABILITY.md`

| Area | Implemented | Documented | Offline-tested | Live-smoked | Stability |
| --- | --- | --- | --- | --- | --- |
| Public API manifest | Yes | Yes | Yes | N/A | stable contract |
| Text and streaming | Yes | Yes | Yes | Yes | stable |
| Structured output | Yes | Yes | Yes | Yes | stable |
| Embeddings | Yes | Yes | Yes | provider-dependent | stable |
| Grounded text | Yes | Yes | Yes | provider-dependent | stable |
| Gateway fallback | Yes | Yes | Yes | provider-dependent | stable |
| Tier-1 provider adapters | Yes | Yes | Yes | optional | stable provider story |
| Agent core runtime | Yes | Yes | Yes | provider-dependent | stable |
| Agent run stores and replay | Yes | Yes | Yes | N/A | beta |
| Human approval policies | Yes | Yes | Yes | provider-dependent | beta |
| MCP helpers and registries | Yes | Yes | partial | optional | stable helper path |
| Declarative workflows | Yes | Yes | Yes | N/A | beta |
| Packaged skills | Yes | Yes | Yes | N/A | beta |
| Realtime/live agents | Yes | partial | Yes | optional | experimental |
| Google native media/cache clients | Yes | provider docs | provider tests | optional | beta/native |
| Release install verification | Yes | Yes | Yes | N/A | beta release gate |
| Security and operations guides | Yes | Yes | Yes | N/A | beta guidance |

## GA Boundary

Python GA should include the stable public API surface, tier-1 provider story, agent core runtime, gateway contracts, production API patterns, and release/install verification. DeepSeek is deferred from Python GA. vLLM remains a tier-1 Python provider for SDK primitives exposed by its OpenAI-compatible server.

The SDK orchestrates. Applications own business policy, durable vertical storage, approval queues, compliance systems, and provider selection policy.
