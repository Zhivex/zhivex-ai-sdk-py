# Python SDK Product Boundary And Parity Matrix

This matrix records the Python SDK maturity boundary against the broader Zhivex SDK platform story. Python parity means equivalent outcomes for the portable agent contract; it does not mean copying every TypeScript API, integration, or package into the Python core. See [SCOPE.md](./SCOPE.md) for the normative product boundary and non-goals.

The default product journey is one portable agent backed by foundation primitives, provider adapters, durable execution, approvals, replay, gateway routing, and production operations. The repository can contain broader capabilities without making all of them part of that core promise.

Machine-readable companion: [parity_matrix.json](./parity_matrix.json).

Legend:

- Implemented: code exists in this repo
- Documented: top-level or `docs/` guidance exists
- Offline-tested: deterministic tests cover the behavior without provider credentials
- Live-smoked: optional smoke path exists for real providers
- Stability: stable, beta, or experimental according to `STABILITY.md`

## Core Agent And Production Path

These areas define the Python product boundary and the path toward GA.

| Area | Implemented | Documented | Offline-tested | Live-smoked | Stability |
| --- | --- | --- | --- | --- | --- |
| Public API manifest | Yes | Yes | Yes | N/A | stable contract |
| Text and streaming | Yes | Yes | Yes | Yes | stable |
| Structured output | Yes | Yes | Yes | Yes | stable |
| Embeddings | Yes | Yes | Yes | provider-dependent | stable |
| Grounded text | Yes | Yes | Yes | provider-dependent | stable |
| Gateway fallback | Yes | Yes | Yes | provider-dependent | stable |
| Tier-1 provider adapters | Yes | Yes | Yes | optional | stable provider story |
| Meta Model API adapter | Yes: Stable `create_meta` + Standard `muse-spark-1.2` portable text/tools; native extras remain Beta | Yes | Yes | opt-in path; release certification pending exact artifact evidence | stable tier-1 scope with beta native extras |
| Agent core runtime | Yes | Yes | Yes | provider-dependent | stable |
| Typed agent DX (deps, outputs, hooks, middleware) | Yes | Yes | Yes | provider-dependent | stable |
| Agent run stores and replay | Yes | Yes | Yes | N/A | stable for Postgres/run-state/replay; beta for local stores |
| Durable human approvals | Yes | Yes | Yes | provider-dependent | stable for local-tool pending approvals; beta for provider-managed approvals/UI chunks |
| Production API and worker examples | Yes | Yes | Yes | optional | beta guidance |
| Release install verification | Yes | Yes | Yes | N/A | beta release gate |
| Security and operations guides | Yes | Yes | Yes | N/A | beta guidance |

## Optional Extensions And Incubating Capabilities

These areas remain available for teams that need them. Their presence does not expand the package-wide GA boundary. Applications should use the focused namespaces and isolate Beta or Experimental contracts behind app-owned interfaces; Stable workflows still require application-owned authorization, storage policy, and side-effect controls.

| Area | Implemented | Documented | Offline-tested | Live-smoked | Stability |
| --- | --- | --- | --- | --- | --- |
| MCP helpers and registries | Yes | Yes | partial | optional | stable helper path |
| Workflow orchestration and durable graphs | Yes: sequential/parallel/loop, DAG, functional steps, versioned checkpoints/migration, resume/fork/cancel, execution leases, heartbeat, and fencing | Yes | Yes; in-memory/SQLite ownership, compatibility, migration, and recovery covered | Installed-wheel Postgres checkpoint/lease integration is mandatory in CI/release | stable core; named external-engine factories beta |
| Agent evaluations and CI gates | Yes: repeated trials, bounded concurrency, custom metrics, baselines, regression gates, JSON/JUnit artifacts, and trace-derived datasets | Yes | Yes | optional provider-backed evaluation | beta |
| Agent protocols and hosting | Yes: A2A v1, AG-UI, constrained Responses hosting, trusted run context, safe errors, limits, and optional stores/replay | Yes | Yes; official protocol packages exercised | provider-dependent | beta |
| General CLI and local playground | Yes: inspect, run, eval, Responses/A2A serve, and loopback-only playground | Yes | Yes | N/A | beta |
| Packaged skills | Yes | Yes | Yes | N/A | beta |
| Realtime/live agents | Yes | partial | Yes | optional | experimental |
| Google native media/cache clients | Yes | provider docs | provider tests | optional | beta/native |

## GA Boundary

Python GA should include the stable public API surface, tier-1 provider story, agent core runtime, durable execution and approvals, gateway contracts, production API patterns, and release/install verification. DeepSeek now participates in the tier-1 portable provider contract for text, streaming, structured output, tools, and reasoning. vLLM remains a tier-1 Python provider for SDK primitives exposed by its OpenAI-compatible server.

Workflow orchestration, evaluations, protocol hosting, the general CLI/playground, packaged skills, provider-native media clients, and realtime do not need to become Stable for the agent core to reach GA. They can mature independently as optional or incubating surfaces.

Meta Model API participates in the GA/Tier-1 boundary only through `create_meta()` and Standard `muse-spark-1.2` portable text generation, streaming, structured output, callable tools, and agent tool loops. Contributor models and native extensions mature independently as Beta. Tier-1 remains a contract-supported classification; exact release certification is a separate artifact evidence gate.

The SDK orchestrates. Applications own business policy, durable vertical storage, approval UI/authorization, compliance systems, and provider selection policy.
