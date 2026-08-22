# Versioning

Zhivex AI SDK follows a strict versioning policy for the documented stable surface even while the package remains in the `0.x` range.

Related documents:

- [README.md](./README.md)
- [STABILITY.md](./STABILITY.md)
- [SUPPORT.md](./SUPPORT.md)
- [CHANGELOG.md](./CHANGELOG.md)

## Policy

- Stable APIs must not change in silently breaking ways.
- Breaking changes to stable APIs require advance deprecation notice.
- Breaking changes to stable APIs require an entry in [CHANGELOG.md](./CHANGELOG.md).
- Breaking changes to stable APIs require migration guidance in release notes.
- Beta APIs may evolve between minor releases, but user-visible changes must still be documented in the changelog.
- Experimental APIs may change without strong compatibility guarantees, but they must remain clearly labeled in documentation.

## Stable surface rules

The stable surface is defined in [STABILITY.md](./STABILITY.md) and enforced by `src/zhivex_ai/api_stability.py`. For that surface:

- Prefer additive changes over behavioral changes.
- Do not rename, remove, or repurpose public stable exports without a deprecation path.
- Keep top-level imports from `zhivex_ai` as the primary public entrypoint.
- Update `src/zhivex_ai/api_stability.py` and its drift tests whenever `zhivex_ai.__all__` changes.
- New stable provider factories are additive public API changes and must land with provider contracts, support metadata, docs, examples, and artifact-install verification. `create_deepseek` follows that policy; `0.19.0` applies it to `create_meta` with a deliberately narrow Standard `muse-spark-1.2` portable text/tool contract.
- Additive fields on `Agent`, `AgentContext`, `AgentRunRequest`, `AgentRunResult`, and `ToolExecutionContext` must preserve existing positional construction and string-only `result.text` behavior. Generic typing and `result.output` are additive in `0.14.0`.

## Namespace policy

The package root is a compatibility aggregator for the current Beta release line, while focused namespaces expose optional product areas:

- `zhivex_ai.evals`
- `zhivex_ai.workflows`
- `zhivex_ai.integrations.protocols`
- `zhivex_ai.experimental`

Existing top-level imports keep their documented stability level. Recommending a focused namespace is not a removal or promotion. New Beta or Experimental symbols should not be added to the package root by default; adding one requires an explicit compatibility rationale, stability-manifest entry, changelog coverage, documentation, and contract tests.

If a legacy top-level Beta or Experimental import is eventually retired, it must first receive a documented replacement and a deprecation period. Stable top-level imports continue to follow the stricter deprecation workflow below.

## Deprecation workflow

When a stable API needs to change:

1. Mark the API as deprecated in docs and release notes.
2. Provide the replacement path.
3. Record the change in [CHANGELOG.md](./CHANGELOG.md).
4. Include migration guidance for downstream users.
5. Remove the deprecated API only in a planned breaking release after the warning period.

## Beta and experimental expectations

Beta APIs are intended for early adoption with documented change management. The packaged-skill layer, in-memory and SQLite agent run stores, native subagent tools, declarative workflow agents, durable workflow graphs/checkpoints/stores, workflow resume/fork, workflow callback adapters, evaluation helpers/experiments, A2A and AG-UI adapters, Responses-compatible hosting, the general CLI/local playground, trace artifacts, redaction policies, budget guards, and UI approval chunks follow this beta policy.

The current beta-only areas are narrower than the full agent story. Foundation model middleware helpers and model catalog helpers remain beta. The agent runtime—including typed context/output contracts, dynamic instructions, lifecycle hooks, run middleware, and the observer protocol—session helpers, agent skills, MCP helpers, MCP-backed registries, Postgres-backed agent stores, run-state serialization, cancellation, replay, run-snapshot helpers, and durable local-tool approval resume are now part of the documented stable surface and follow the stable-surface rules above.

`AgentRunStore.save(...)` remains compatible with implementations returning `None`; built-in stores return the authoritative persisted state and revision. Atomic capabilities such as `claim_idempotency_key(...)`, `claim_pending_approval(...)`, `cancel_run(...)`, and `fail_resume_claim(...)` are checked only when the corresponding feature is used. Custom production stores must implement those operations transactionally to enable those features.

The `0.15.0` workflow additions preserve existing `WorkflowStep`, `SequentialAgent`, `ParallelAgent`, `LoopAgent`, `run_workflow(...)`, and string-output behavior. New `WorkflowStep` fields are additive. `WorkflowStep.max_retries` continues to configure retries inside `run_agent(...)`; complete logical step retries use the separate beta `WorkflowRetryPolicy` contract. An optional functional `executor` is valid only in `WorkflowGraph`; existing declarative workflow agents continue to require an `Agent`.

Durable workflow compatibility is bound to both `definition_version` and the computed definition digest. Resume and fork reject drift rather than re-evaluating a possibly different graph. Applications must choose a new definition version whenever steps, routing conditions, interrupt points, retry behavior, or executor identities change. `0.15.0` does not provide automatic checkpoint migration; an application that migrates durable state must validate it explicitly, preserve source history and lineage, and record the migration outside the normal append sequence.

`WorkflowStep.definition_revision` and edge `definition_revision` are additive beta identity fields for configuration that Python source inspection cannot see reliably, such as agent settings, closure values, prompt-template versions, or application policy revisions. When omitted, the legacy `0.16.0` digest is preserved exactly. Set them deliberately and change them whenever that hidden behavior changes; a changed revision fails resume and fork closed through `WorkflowDefinitionMismatchError`.

`WORKFLOW_CHECKPOINT_SCHEMA_VERSION` and `WORKFLOW_ADAPTER_SCHEMA_VERSION` version the beta serialized contracts independently from the package version. Future readers reject unsupported schema versions. The compatibility suite keeps a real `0.15.0` schema-v1 checkpoint fixture and proves canonical reserialization, resume, and fork against the current reader. Postgres workflow stores also keep a checked backend schema-version ledger; an unexpected version fails closed instead of guessing a migration. There is still no automatic application checkpoint migration engine. Callback-adapter factories for DBOS, Temporal, Prefect, and Restate do not create a compatibility guarantee for those third-party runtimes; only the Zhivex request/outcome envelope is versioned here.

The `0.16.0` protocol surfaces track explicit upstream lines: A2A protocol v1 through `a2a-sdk>=1.1.2,<2`, and AG-UI through `ag-ui-protocol>=0.1.19,<0.2`. A future Zhivex minor may follow an upstream minor contract change with changelog and migration guidance. The Responses-compatible host covers the documented text/message create, streaming, stored-result, and event-replay subset only when an application-owned `ResponsesEventStore` is configured; it does not establish compatibility for every OpenAI Responses request item, tool, background operation, cancellation, or provider-native continuation endpoint. Protocol run-option, limit, error, event, and storage contracts remain Beta.

Evaluation experiment artifacts are beta application contracts identified by `AGENT_EVALUATION_ARTIFACT_SCHEMA_VERSION`. Variant, metric, gate, trial, usage, cost, latency, and redacted-trajectory fields may gain additive metadata between minor releases. Trial ordering is dataset order then repetition order. Existing single-trial callers retain the `repetitions=1` and `max_concurrency=1` behavior; `0.16.0` adds repeated trials and experiments additively.

Workflow lease managers and `cancel_workflow(...)` remain Beta. Lease tokens are opaque ownership credentials and are never serialized into checkpoints; monotonic fencing tokens are correlation and stale-writer controls, not secrets. In `0.17.0`, matching built-in checkpoint and lease backends validate ownership and append progress atomically, Postgres uses server time and checked backend schema metadata, and operational failures have typed subclasses of `ValidationError`. Checkpoint and adapter payload schema versions remain unchanged because compatibility with the published schema-v1 payload is verified directly. There is still no automatic application checkpoint migration engine.

Experimental APIs are intended for evaluation. They should be consumed behind an application-owned abstraction if production teams need to try them before they graduate.

## Current maturity target

This policy governs the current `Beta` phase of the SDK. The goal is to keep the documented stable surface predictable while release evidence is captured from built artifacts before a future stable release.
