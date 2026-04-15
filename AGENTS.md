# AGENTS.md

This file is a working guide for coding agents and contributors operating in `zhivex-ai-sdk-py`.

## Project Snapshot

- Package: `zhivex-ai-sdk`
- Language: Python `>=3.11`
- Build backend: `hatchling`
- Core positioning: async-first, agent-first, multi-provider SDK
- Source root: `src/zhivex_ai`
- Current package maturity: `Beta`

Primary product docs live in:

- `README.md`
- `STABILITY.md`
- `VERSIONING.md`
- `SUPPORT.md`
- `CHANGELOG.md`
- `PRODUCTION_APIS.md`
- `OBSERVABILITY.md`

Read those before changing public behavior.

## Repository Layout

- `src/zhivex_ai/`: SDK implementation
- `src/zhivex_ai/providers/`: provider adapters and provider-specific helpers
- `tests/`: contract, runtime, transport, gateway, and provider tests
- `examples/`: runnable reference examples by feature area
- `scripts/`: support scripts such as support-matrix generation and live smoke runs

High-value modules:

- `src/zhivex_ai/__init__.py`: canonical public export surface
- `src/zhivex_ai/agent.py`: agent runtime, sessions, handoffs, tools, approvals
- `src/zhivex_ai/generate_text.py`: normalized text generation and streaming
- `src/zhivex_ai/generate_object.py`: structured output APIs
- `src/zhivex_ai/gateway.py`: fallback routing and gateway contracts
- `src/zhivex_ai/transport.py`: HTTP/SSE helpers
- `src/zhivex_ai/provider_support.py`: support-matrix rendering helpers

## Public API Rules

This repo has a deliberately narrow stable surface.

- Supported production imports should come from `zhivex_ai`, not deep internal paths.
- Treat `src/zhivex_ai/__init__.py` as the public contract boundary.
- `tests/test_public_contract.py` enforces that the documented stable exports remain available.
- If you change the stable surface, update `README.md`, `STABILITY.md`, `VERSIONING.md`, `CHANGELOG.md`, and the contract tests in the same change.
- Do not silently rename, remove, or repurpose stable exports.

Stable APIs currently center on:

- tier-1 provider factories: OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex
- foundation primitives: text, streaming, structured output, embeddings, grounded text
- agent runtime: `Agent`, `run_agent`, `stream_agent`, `resume_agent`
- gateway contracts
- core transport helpers and main SDK errors

Beta and experimental areas must stay clearly labeled in docs and examples.

## Provider Design Rules

The provider story in this SDK is intentional:

- Prefer thin provider adapters over provider-specific logic leaking into shared APIs.
- Preserve the distinction between `provider(...)` or `provider.portable.*` and `provider.native.*`.
- The portable layer should only accept the SDK-owned cross-provider contract.
- Provider-specific hosted tools and escape hatches belong in native paths.
- Tier-1 support claims must stay aligned with `README.md`, `STABILITY.md`, `SUPPORT.md`, and the generated support matrix.

When changing provider capabilities:

- update the adapter
- update or add tests for supported behavior
- update examples if the public usage pattern changes
- regenerate any provider support documentation if needed

## Change Guidance By Area

### Foundation APIs

If you touch `generate_text`, `stream_text`, `generate_object`, `stream_object`, `embed`, or grounded text:

- preserve normalized request and result shapes
- keep validation and error behavior consistent across providers
- add or update tests in `tests/test_core.py` and relevant provider tests

### Agent Runtime

If you touch `agent.py` or runtime behavior:

- verify run, resume, and streaming flows
- cover handoffs, tool execution, approvals, memory, summaries, and checkpoints as applicable
- update `tests/test_agent.py`, `tests/test_agent_extensions.py`, or related tests

### Gateway And Transport

If you touch gateway or HTTP/SSE helpers:

- preserve API-facing response semantics
- keep production examples aligned with the recommended patterns in `PRODUCTION_APIS.md`
- update `tests/test_gateway.py`, `tests/test_transport.py`, and any affected example apps

### Observability

If you touch telemetry, tracing, or gateway attempt hooks:

- keep `OBSERVABILITY.md` accurate
- preserve useful correlation fields such as request IDs, session IDs, run IDs, provider, and model metadata

### Docs And Examples

Examples are part of the product story, not filler.

- keep examples runnable and representative of current APIs
- prefer updating an existing example over adding a near-duplicate
- if a user-facing workflow changes, update the relevant example and top-level docs together

## Development Workflow

Set up the repo:

```bash
make dev
```

Main validation commands:

```bash
make test
make test-cov
make lint
make typecheck
make compile
make check
```

Release-oriented validation:

```bash
make release-check
```

Live provider smoke runner:

```bash
make smoke
```

Only run smoke tests when the required credentials and model IDs are configured.

## Testing Expectations

For non-trivial changes, run the narrowest useful tests first, then broader validation if the area is cross-cutting.

Good defaults:

- public surface changes: `tests/test_public_contract.py`
- foundation behavior: `tests/test_core.py`
- agent runtime: `tests/test_agent.py` and `tests/test_agent_extensions.py`
- provider behavior: the relevant `tests/test_*_provider.py`
- transport and HTTP helpers: `tests/test_transport.py` and `tests/test_http.py`
- gateway logic: `tests/test_gateway.py`

Before release-oriented changes, prefer `make check`.

## Documentation Discipline

User-visible changes should usually update more than code.

- Update `CHANGELOG.md` for user-visible behavior changes.
- Update `README.md` when the recommended usage or support story changes.
- Update `STABILITY.md` when stability classification changes.
- Update `SUPPORT.md` when the support scope changes.
- Update `VERSIONING.md` when deprecation or compatibility policy changes.
- Update `PRODUCTION_APIS.md` or `OBSERVABILITY.md` when operational guidance changes.

## Practical Guardrails

- Prefer top-level imports in docs and examples: `from zhivex_ai import ...`
- Keep async-first patterns intact.
- Avoid introducing provider-specific branching into shared surfaces unless the contract explicitly allows it.
- Do not expand stable guarantees casually; this repo documents them on purpose.
- If a change is experimental, say so in docs and examples.
- If a stable change requires migration, document it immediately.

## When In Doubt

Start from the smallest change that preserves:

- the top-level public contract
- cross-provider consistency
- tier-1 provider reliability
- runnable examples
- accurate product documentation

If you cannot keep all five aligned in the same change, the change is probably incomplete.
