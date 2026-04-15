# Zhivex AI SDK Maturity Plan

This document turns the current "works well and tests pass" state of the SDK into a concrete path toward a mature product that can be positioned confidently for final production APIs.

## Current Baseline

- Package builds successfully and the test suite is green locally.
- Core areas already exist: provider adapters, foundation APIs, agent runtime, gateway, middleware, transport helpers, and release automation.
- The package now declares `Development Status :: 4 - Beta`, so the product maturity signal is aligned with the documented public surface and current engineering gates.

## Product Goal

Move the SDK from `alpha` to a state where we can honestly claim:

- stable public API for the main use cases
- predictable behavior on the supported provider set
- production-ready API integration guidance
- observable and debuggable runtime behavior
- disciplined releases with migration guidance

## Stable Scope

The first maturity milestone should explicitly define the stable product surface.

### Proposed stable surface

- `create_openai`, `create_anthropic`, `create_azure_openai`, `create_gemini`, `create_vertex`
- `generate_text`, `stream_text`
- `generate_object`, `stream_object`
- `generate_grounded_text`
- `embed`, `embed_many`
- `Agent`, `run_agent`, `stream_agent`, `resume_agent`
- `create_gateway` and the main gateway dataclasses
- `ProviderHTTPError`, `ConfigurationError`, `ValidationError`, `UnsupportedFeatureError`
- HTTP and SSE helpers from `transport.py`

### Proposed beta surface

- middleware helpers
- model catalog
- Postgres-backed memory and checkpoint stores
- MCP-related helpers

### Proposed experimental surface

- realtime/live voice
- provider-native hosted tools that do not map cleanly to the portable contract
- providers currently marked as `native-only` or `compatibility` for major feature areas

## Phase Plan

## P0: Release Safety And Product Boundaries

Goal: stop accidental breakage and define what the SDK supports.

### Work items

1. Add a public API policy.
   Files:
   - `README.md`
   - new `STABILITY.md`
   - `src/zhivex_ai/__init__.py`

   Changes:
   - document what is stable, beta, and experimental
   - make `__init__.py` the canonical public import surface
   - discourage deep imports from internal modules

2. Add versioning and deprecation policy.
   Files:
   - new `VERSIONING.md`
   - `RELEASING.md`
   - `README.md`

   Changes:
   - define SemVer policy
   - require migration notes for breaking changes
   - require deprecation window for stable APIs

3. Add changelog discipline.
   Files:
   - new `CHANGELOG.md`
   - `RELEASING.md`

   Changes:
   - keep user-visible changes per release
   - split into Added, Changed, Fixed, Removed, Deprecated

### Exit criteria

- stable and experimental surfaces are documented
- every release has changelog entries
- deprecations have a documented path and timeline

## P1: CI Quality Gates

Goal: make regressions hard to ship.

### Work items

1. Add linting.
   Files:
   - `pyproject.toml`
   - `.github/workflows/ci.yml`

   Changes:
   - add `ruff`
   - fail CI on lint errors

2. Add static type checking.
   Files:
   - `pyproject.toml`
   - `.github/workflows/ci.yml`

   Changes:
   - add `pyright` or `mypy`
   - start with the public API and core runtime modules
   - tighten gradually instead of blocking the whole repo on day one

3. Add coverage reporting.
   Files:
   - `pyproject.toml`
   - `.github/workflows/ci.yml`

   Changes:
   - measure coverage for `src/zhivex_ai`
   - set an initial floor, then raise it over time

4. Split CI into clear gates.
   Files:
   - `.github/workflows/ci.yml`

   Changes:
   - `compile`
   - `lint`
   - `typecheck`
   - `test`
   - `build`

### Exit criteria

- CI fails on lint or type regressions
- coverage is visible and enforced
- release candidates pass all gates, not just tests

## P1: Production API Integration

Goal: make the SDK easy to adopt correctly in backend services.

### Work items

1. Publish a reference FastAPI integration.
   Files:
   - new `examples/integrations/fastapi_chat_api.py`
   - new `examples/integrations/fastapi_streaming_api.py`
   - `examples/README.md`
   - `README.md`

   Changes:
   - request validation with Pydantic
   - timeout and cancellation handling
   - standard error mapping
   - SSE and plain text streaming examples

2. Add error handling guidance.
   Files:
   - new `docs/API_PATTERNS.md` or `PRODUCTION_APIS.md`
   - `README.md`

   Changes:
   - map SDK errors to HTTP status codes
   - explain retries, provider failures, and fallback behavior
   - define default timeout recommendations

3. Add deployment-ready examples.
   Files:
   - new `examples/integrations/fastapi_gateway_api.py`
   - `examples/README.md`

   Changes:
   - one direct provider example
   - one structured output example
   - one gateway fallback example

### Exit criteria

- a backend team can build a production-style API without inventing the integration pattern
- error handling and streaming patterns are documented and tested

## P1: Supported Provider Strategy

Goal: narrow the support promise until the product story is honest and defendable.

### Work items

1. Define tier-1 providers.
   Proposed tier-1:
   - OpenAI
   - Anthropic
   - Azure OpenAI
   - Gemini
   - Vertex

   Files:
   - `README.md`
   - `src/zhivex_ai/provider_support.py`
   - provider-specific docs if added

   Changes:
   - explicitly mark tier-1 versus compatibility providers
   - clarify which claims apply only to tier-1

2. Add contract tests per tier-1 provider.
   Files:
   - new `tests/contracts/`
   - provider tests under `tests/`

   Changes:
   - common test matrix for text, streaming, structured output, tools, embeddings where supported
   - assert consistent output semantics and errors

### Exit criteria

- marketing claims match the supported matrix
- provider behavior is tested against the portable contract

## P2: Runtime Observability And Operations

Goal: make incidents diagnosable.

### Work items

1. Promote observability to a first-class story.
   Files:
   - `src/zhivex_ai/observability.py`
   - `src/zhivex_ai/agent.py`
   - `src/zhivex_ai/gateway.py`
   - `README.md`

   Changes:
   - document and expose tracing hooks clearly
   - standardize event naming
   - include provider, model, latency, retry, fallback, and token metadata

2. Add structured logging guidance.
   Files:
   - new `PRODUCTION_APIS.md`
   - examples

   Changes:
   - request IDs
   - session IDs
   - agent run IDs
   - gateway attempt correlation

3. Tighten gateway decisioning.
   Files:
   - `src/zhivex_ai/gateway.py`
   - `tests/test_gateway.py`

   Changes:
   - reduce reliance on string heuristics for model ranking
   - prefer explicit capability and catalog metadata
   - document fallback and retry semantics

### Exit criteria

- production failures can be traced through logs and telemetry
- gateway decisions are explainable and test-backed

## P2: Security And Tool Execution

Goal: make agent and tool execution safer by default.

### Work items

1. Audit default tool execution behavior.
   Files:
   - `src/zhivex_ai/agent.py`
   - `src/zhivex_ai/types.py`
   - `README.md`

   Changes:
   - ensure dangerous behavior is opt-in
   - document approval policies and secure defaults

2. Add security guidance.
   Files:
   - new `SECURITY.md`
   - `README.md`

   Changes:
   - secrets handling
   - network and shell tool risks
   - provider credential guidance
   - data retention considerations

### Exit criteria

- secure defaults are documented
- risky features are clearly labeled and require deliberate opt-in

## P2: Release And Maintenance Discipline

Goal: make adoption low-risk for downstream teams.

### Work items

1. Expand release checklist.
   Files:
   - `RELEASING.md`
   - `.github/workflows/publish-pypi.yml`
   - `.github/workflows/publish-testpypi.yml`

   Changes:
   - require changelog
   - require green CI gates
   - require migration notes when relevant

2. Add support policy.
   Files:
   - new `SUPPORT.md`

   Changes:
   - how long versions are supported
   - what kinds of fixes qualify for patch releases
   - support expectations for experimental areas

### Exit criteria

- downstream users know what to expect from upgrades
- releases are consistent and auditable

## Suggested Immediate Patch Set

If we want the highest impact with the least amount of work, do these next:

1. Add `CHANGELOG.md`, `STABILITY.md`, and `VERSIONING.md`.
2. Add `ruff` and `coverage` to CI.
3. Add a production-style `FastAPI` example with streaming.
4. Rework the README to distinguish tier-1 providers from compatibility providers.
5. Move realtime wording under an explicit experimental banner.

## Definition Of Done For "Mature Product"

We can consider the SDK mature enough to market for final production APIs when all of the following are true:

- stable API surface is documented and mostly frozen
- CI enforces lint, typecheck, tests, coverage, and build
- tier-1 providers have contract coverage
- production API examples exist and are maintained
- observability guidance is built-in, not implied
- release notes and migration paths exist
- the package can honestly operate as a `Beta` release today and later graduate to `Stable`
