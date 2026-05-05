# Zhivex AI SDK Python Maturity Roadmap

This file is the working roadmap for taking `zhivex-ai-sdk-py` from a strong beta SDK to a mature product for providers, agents, workflows, and production backend integrations.

It is intentionally written as an AI-friendly handoff document. Future Codex sessions should update the checkboxes as work lands, add evidence under each phase, and keep the roadmap grounded in the real repository state.

## How To Use This File

- Treat each phase as a delivery track with explicit exit criteria.
- Mark a checkbox only when the code, docs, and verification evidence exist in the repo.
- Add short evidence bullets with file paths, commands, commits, or test outputs.
- Do not mark a phase complete only because the feature exists. It must also be documented and covered by an appropriate gate.
- Keep this file in sync with `README.md`, `STABILITY.md`, `VERSIONING.md`, `SUPPORT.md`, `CHANGELOG.md`, and provider capability metadata.

## Current Verified Baseline

Last local verification:

- [x] `make check` passed in this checkout.
- [x] Result: `388 passed, 1 skipped, 4 subtests passed`.
- [x] Coverage: `82.87%`, above the current `80%` floor.
- [x] `compile`, `ruff`, `mypy`, support-matrix check, and coverage tests all passed.

Current strengths:

- [x] Async-first package published as beta in `pyproject.toml`.
- [x] Public import surface exists through `zhivex_ai`.
- [x] Foundation APIs exist: text, streaming, structured output, embeddings, audio, grounded text, routing, gateway, middleware, transport helpers.
- [x] Agent runtime exists: agents, handoffs, subagent tools, registries, memory/session stores, guardrails, approvals, traces, stream events.
- [x] Durable run state exists: in-memory, SQLite, Postgres run stores, cancellation, serialization, replay/evaluation helpers.
- [x] Declarative workflow agents exist: sequential, parallel, loop, shared state, output keys, templated step inputs, error policies.
- [x] Provider support matrix is generated from runtime metadata.
- [x] Offline business reference apps exist and are tested.

Current maturity status:

- [x] Parity matrix and GA boundary decisions are documented.
- [x] Docs/DX onboarding includes quickstart, troubleshooting, `.env.example`, and examples index polish.
- [x] Release process includes automated "install what we ship" verification.
- [x] Observability, security, and production operations guidance are first-class product documentation.

Current release-readiness work:

- [x] All maturity phases are complete in this file.
- [x] RC/GA evidence expectations are documented in `docs/RC_READINESS.md`.
- [ ] Final human release evidence must be filled in `docs/RELEASE_EVIDENCE.md` from the exact commit being tagged.

## Product Goal

The Python SDK is mature when a backend team can safely build provider-backed and agent-backed production services with:

- stable public APIs for the main use cases
- clear beta/experimental boundaries for advanced features
- verified provider behavior across the supported tier-1 set
- dependable agent runtime semantics for tools, approvals, persistence, replay, and tracing
- production examples for API, worker, and workflow use cases
- disciplined releases, migration notes, and package-install verification

## Phase 0: Baseline, Scope, And Product Boundary

Status: Complete.

Goal: define exactly what "mature Python SDK" means, separate feature breadth from release readiness, and avoid copying TypeScript blindly.

Checklist:

- [x] Record current offline verification in this file.
- [x] Confirm the Python SDK already has broad runtime coverage.
- [x] Add or update a machine-readable Python-vs-TypeScript parity matrix.
- [x] Define the target GA provider list.
- [x] Decide whether DeepSeek belongs in Python GA parity or remains deferred.
- [x] Decide whether vLLM should remain a Python differentiator or also drive TS parity.
- [x] Define the release gate for each release class: dev, beta, RC, GA.
- [x] Add a clear "SDK orchestrates, application owns policy/storage" boundary to production docs.

Evidence:

- `src/zhivex_ai/api_stability.py` is the machine-readable public API/stability contract.
- `docs/PARITY_MATRIX.md` and `docs/parity_matrix.json` record implemented, documented, offline-tested, live-smoked, and stability columns for the product areas.
- Tier-1 provider scope is documented in `README.md`, `SUPPORT.md`, and `docs/providers/tier-1.md`.
- DeepSeek is deferred from Python GA; vLLM remains a tier-1 Python provider for SDK primitives exposed by its OpenAI-compatible server.
- Release gates and RC/GA evidence are documented in `RELEASING.md`, `docs/RELEASE_EVIDENCE.md`, and `docs/RC_READINESS.md`.
- Production docs preserve the boundary that the SDK orchestrates while applications own policy, storage, compliance, approvals, and provider selection.

Suggested files:

- `MATURITY_PLAN.md`
- `README.md`
- `docs/PARITY_MATRIX.md`
- `docs/PRODUCTION.md`
- `STABILITY.md`
- `SUPPORT.md`

Exit criteria:

- [x] A new Codex can read the repo and know which providers, APIs, and agent features are promised.
- [x] The parity matrix distinguishes "implemented", "documented", "tested offline", "live-smoked", and "stable".
- [x] The roadmap is not dependent on hidden conversation context.

Verification:

- `make check`
- Manual review of `README.md`, `STABILITY.md`, `SUPPORT.md`, and `docs/PARITY_MATRIX.md`

## Phase 1: Public API Contract And Stability Enforcement

Status: Complete.

Goal: make the public surface versionable, reviewable, and protected against accidental breaking changes.

Checklist:

- [x] Public import surface exists through `src/zhivex_ai/__init__.py`.
- [x] Stability docs exist.
- [x] Versioning docs exist.
- [x] Support docs exist.
- [x] Add an API stability manifest equivalent in spirit to TypeScript `getApiStability`.
- [x] Classify exported symbols as `stable`, `beta`, `experimental`, or `internal`.
- [x] Add tests that fail when public exports drift without manifest updates.
- [x] Add tests that fail when a `stable` API is removed or reclassified without migration notes.
- [x] Add `py.typed` if missing, and verify it is included in the built wheel.
- [x] Document deep-import policy and mark internal modules clearly.
- [x] Add a deprecation helper or documented deprecation pattern for stable APIs.

Evidence:

- `src/zhivex_ai/api_stability.py` classifies all 462 top-level exports as `stable`, `beta`, or `experimental`.
- `tests/test_api_stability.py` verifies manifest drift, level/category validity, stable-doc coverage, and experimental exports staying out of the Stable section.
- `tests/test_public_contract.py` now checks the documented stable contract against `STABLE_EXPORTS`.
- `src/zhivex_ai/py.typed` exists and is included in `dist/zhivex_ai_sdk-0.5.0-py3-none-any.whl`.
- Focused verification: `.venv/bin/python -m pytest tests/test_public_contract.py tests/test_api_stability.py -q` -> `14 passed`.
- Full verification: `make check` -> `336 passed, 1 skipped`, coverage `82.81%`.
- Release verification: `make release-check` passed; `twine check` passed for wheel and sdist.

Suggested files:

- `src/zhivex_ai/__init__.py`
- `src/zhivex_ai/api_stability.py`
- `tests/test_api_stability.py`
- `STABILITY.md`
- `VERSIONING.md`
- `RELEASING.md`
- `pyproject.toml`

Exit criteria:

- [x] Every public export has a stability classification.
- [x] Public API drift is caught by tests.
- [x] Stable API removal requires an intentional test/documentation update.
- [x] Built packages include typing markers and documented public entrypoints.

Verification:

- `make check`
- `python -m build --no-isolation --outdir dist`
- `python -m twine check dist/*`
- Optional: install the built wheel in a temp venv and import the stable API set.

## Phase 2: Type Safety And Internal Quality Gates

Status: Complete.

Goal: expand confidence from "tests pass" to "core SDK internals are typed, maintainable, and harder to break."

Checklist:

- [x] `ruff` is part of `make check`.
- [x] `compileall` is part of `make check`.
- [x] `pytest-cov` enforces an 80% global floor.
- [x] `mypy` is part of `make check`.
- [x] Expand `mypy` from the current subset to additional public/core API modules.
- [x] Typecheck `types`, `messages`, `schema`, and provider base contracts.
- [x] Typecheck `agent_state`, `workflow`, and `safety`.
- [x] Typecheck tier-1 provider modules or record explicit tracked exclusions.
- [x] Typecheck `agent.py` or split it into smaller typed modules first if needed.
- [x] Typecheck `skillpacks.py` and `skills.py` or explicitly keep them beta with tracked debt.
- [x] Fix ResourceWarning noise from unclosed SQLite connections in tests.
- [x] Split tests into unit, contract, example, smoke, and release/install groups.
- [x] Add per-area coverage expectations for core runtime and tier-1 providers.

Evidence:

- `pyproject.toml` now typechecks `types.py`, `messages.py`, `schema.py`, `providers/base.py`, `agent_state.py`, `_serde.py`, `agent.py`, `workflow.py`, `safety.py`, `realtime.py`, `skills.py`, and `skillpacks.py` in addition to the prior gate.
- `agent_state.py` now deserializes tool calls/results and metadata with explicit narrowing so the durable run-state layer typechecks independently.
- `_serde.py`, `messages.py`, and `agent.py` now narrow dataclass instances, message parts, tool definitions, provider-managed approval payloads, and skill artifact metadata before serialization or event emission.
- `workflow.py`, `safety.py`, `skills.py`, `skillpacks.py`, and `realtime.py` now pass the expanded gate without broadening the stable public API.
- `Makefile` exposes `test-contract`, `test-core`, `test-providers`, `test-examples`, and `test-agents`.
- ResourceWarning validation: `.venv/bin/python -m pytest tests/test_skill_packages.py::SkillCLITests::test_cli_validate_and_run_commands -q -W error::ResourceWarning` -> `1 passed`.
- Focused verification: `.venv/bin/python -m mypy src/zhivex_ai/types.py src/zhivex_ai/messages.py src/zhivex_ai/schema.py src/zhivex_ai/providers/base.py src/zhivex_ai/agent_state.py` -> `Success`.
- Focused runtime verification: `.venv/bin/python -m mypy src/zhivex_ai/_serde.py src/zhivex_ai/realtime.py src/zhivex_ai/agent.py src/zhivex_ai/workflow.py src/zhivex_ai/safety.py src/zhivex_ai/skills.py src/zhivex_ai/skillpacks.py` -> `Success`.
- `make typecheck` -> `Success: no issues found in 21 source files`.
- Grouped tests: `make test-contract` -> `21 passed`; `make test-core` -> `52 passed`; `make test-providers` -> `177 passed`; `make test-examples` -> `9 passed`; `make test-agents` -> `77 passed, 1 skipped`.
- Full verification: `make check` -> `336 passed, 1 skipped`, coverage `82.72%`, no warning summary.

Tracked debt:

- Tier-1 provider adapters remain excluded from the main `mypy` gate through the explicit `zhivex_ai.providers.*` override until Phase 3 hardens adapter internals.
- Provider debt is measured with `.venv/bin/python -m mypy src/zhivex_ai/providers/openai.py src/zhivex_ai/providers/anthropic.py src/zhivex_ai/providers/azure_openai.py src/zhivex_ai/providers/gemini.py src/zhivex_ai/providers/vertex.py src/zhivex_ai/providers/vllm.py` -> `45 errors in 4 files`.

Per-area coverage expectations:

- Stable contract and core foundation tests should stay green through `make test-contract` and `make test-core`; global coverage remains release-gated at `80%`.
- Agent/workflow runtime changes should keep `make test-agents` green and should not reduce the global coverage floor below the release gate.
- Provider changes should keep `make test-providers` green offline; provider adapter typing, shared provider contract rows, and live smoke evidence move to Phase 3.
- Examples should remain offline-friendly and covered by `make test-examples` unless explicitly documented as live smoke.

Suggested files:

- `pyproject.toml`
- `Makefile`
- `tests/`
- `src/zhivex_ai/types.py`
- `src/zhivex_ai/messages.py`
- `src/zhivex_ai/providers/base.py`
- `src/zhivex_ai/_serde.py`
- `src/zhivex_ai/agent.py`
- `src/zhivex_ai/workflow.py`
- `src/zhivex_ai/safety.py`
- `src/zhivex_ai/realtime.py`
- `src/zhivex_ai/skills.py`
- `src/zhivex_ai/skillpacks.py`

Exit criteria:

- [x] Core public API modules are typechecked.
- [x] Tier-1 provider adapters are typechecked or have explicit tracked exclusions.
- [x] Test warnings are either fixed or documented as accepted.
- [x] `make check` remains the authoritative local gate.

Verification:

- `make check`
- Focused `mypy` runs when expanding module coverage
- Focused provider and workflow test runs

## Phase 3: Provider Tier-1 Contract

Status: Complete.

Goal: make provider support mean a tested contract, not only a README table.

Target tier-1 providers:

- [x] OpenAI
- [x] Anthropic
- [x] Azure OpenAI
- [x] Gemini
- [x] Vertex
- [x] vLLM
- [x] DeepSeek decision recorded: defer from Python GA.

Provider contract checklist:

- [x] Runtime capability metadata exists.
- [x] README support matrix is generated from metadata.
- [x] Add common portable contract tests for each tier-1 provider.
- [x] Add native-vs-portable boundary tests for each tier-1 provider.
- [x] Add consistent unsupported-feature checks through shared metadata and provider-specific tests.
- [x] Add focused examples for each tier-1 provider.
- [x] Add live smoke commands and env var docs for each tier-1 provider.
- [x] Add release-blocking checks for support-matrix drift.
- [x] Add provider auth/setup documentation with local, CI, and production notes.
- [x] Document which provider features are stable, beta, native-only, or model-dependent.

Capability areas to verify:

- [x] Text generation
- [x] Streaming
- [x] Structured output
- [x] Tool calling
- [x] Tool choice semantics
- [x] Embeddings where supported
- [x] Audio/transcription/speech where supported
- [x] Grounding/search/retrieval where supported
- [x] Hosted tools and remote MCP where supported
- [x] Realtime/live paths where supported
- [x] Provider-specific native escape hatches

Evidence:

- `tests/contracts/test_tier1_provider_contracts.py` adds a shared offline contract row for OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex, and vLLM.
- `make test-provider-contracts` runs the shared provider contract suite; `make test-contract` includes it.
- `pyproject.toml` now includes tier-1 provider adapters in the main `mypy` gate.
- `scripts/run_live_smoke.py` includes Azure OpenAI and documents tier-1 provider selection.
- `docs/providers/tier-1.md` documents auth/setup, optional live smoke env vars, stable/beta/native-only boundaries, and the deferred DeepSeek decision.
- `examples/text/tier1_providers.py` provides one focused portable example path across the tier-1 set.
- Focused tier-1 provider typecheck: `.venv/bin/python -m mypy src/zhivex_ai/providers/openai.py src/zhivex_ai/providers/anthropic.py src/zhivex_ai/providers/azure_openai.py src/zhivex_ai/providers/gemini.py src/zhivex_ai/providers/vertex.py src/zhivex_ai/providers/vllm.py` -> `Success`.
- Full verification: `make check` -> `360 passed, 1 skipped`, coverage `82.75%`.

Suggested files:

- `src/zhivex_ai/provider_support.py`
- `src/zhivex_ai/providers/`
- `tests/test_provider_support.py`
- `tests/contracts/`
- `scripts/generate_support_matrix.py`
- `scripts/run_live_smoke.py`
- `README.md`
- `docs/providers/`

Exit criteria:

- [x] Each tier-1 provider has a shared contract test row.
- [x] Each provider's README/support claim is backed by metadata and tests.
- [x] Live smoke paths are documented but do not block offline development.
- [x] Provider gaps are explicit instead of implied.

Verification:

- `make support-matrix-check`
- `make check`
- Focused provider contract tests
- Optional live smoke per configured provider

## Phase 4: Agent Runtime As A Mature Product Surface

Status: Complete.

Goal: make Python clearly strong for real agents: tools, approvals, persistence, replay, observability, and safe extension points.

Checklist:

- [x] `Agent`, `run_agent`, `stream_agent`, and `resume_agent` exist.
- [x] Handoffs and native subagent tools exist.
- [x] `AgentRuntime`, `AgentRegistry`, and `ToolRegistry` exist.
- [x] In-memory, SQLite, and Postgres memory/checkpoint/run stores exist.
- [x] Approval policies and guardrails exist.
- [x] Agent replay/evaluation helpers exist.
- [x] Hierarchical traces and cost helpers exist.
- [x] Define stable vs beta agent runtime APIs in the API stability manifest.
- [x] Add a production agent guide.
- [x] Add a durable agent state guide with SQLite and Postgres examples.
- [x] Add a human-in-the-loop approval guide.
- [x] Add a tool registry and permissions guide.
- [x] Add contract tests for resume/cancel/replay across store backends.
- [x] Add clearer semantics for partial runs, retries, timeouts, and pending approvals.
- [x] Add examples for multi-agent handoff, approval, replay, and trace inspection.
- [x] Add event contract docs for streaming and UI transport.

Evidence:

- `src/zhivex_ai/api_stability.py` classifies `AgentRuntime` and `AgentRegistry` as stable while keeping run stores, replay/evaluation, safety, traces, provider-managed approvals, SQLite/in-memory stores, packaged skills, and workflows beta.
- `docs/AGENTS.md` and `docs/PRODUCTION.md` document production agent semantics, failure modes, storage ownership, approval ownership, event ordering, and recovery.
- `docs/agents/durable-state.md`, `docs/agents/approvals.md`, and `docs/agents/tool-registries.md` document focused operational workflows.
- `tests/contracts/test_agent_runtime_contracts.py` covers run/stream/resume, tool approvals, handoffs, run-store idempotency, cancellation, replay, guardrail failures, and tool context across in-memory and SQLite paths.
- `examples/agents/multi_agent_handoff.py`, `examples/agents/human_approval.py`, `examples/agents/durable_state_resume.py`, and `examples/agents/replay_and_trace.py` provide offline examples.

Suggested files:

- `src/zhivex_ai/agent.py`
- `src/zhivex_ai/agent_state.py`
- `src/zhivex_ai/agent_evaluation.py`
- `src/zhivex_ai/safety.py`
- `tests/test_agent*.py`
- `examples/agents/`
- `docs/AGENTS.md`
- `docs/PRODUCTION.md`

Exit criteria:

- [x] A user can build a multi-step agent with tools, HITL, persistence, replay, and tracing using documented public APIs only.
- [x] Agent runtime failure modes are documented and tested.
- [x] Store-backed agent flows have focused tests.

Verification:

- `make check`
- Focused agent runtime tests
- Example execution tests

## Phase 5: Workflows And Reference Applications

Status: Complete.

Goal: show that the SDK can orchestrate serious backend workflows while application policy and storage remain outside the SDK.

Checklist:

- [x] `SequentialAgent`, `ParallelAgent`, and `LoopAgent` exist.
- [x] Workflow state, output keys, templated inputs, and error policies exist.
- [x] Offline small-business loan reference app exists.
- [x] Offline HR candidate selection reference app exists.
- [x] Offline examples have deterministic tests.
- [x] Decide whether workflow APIs are stable or beta in the manifest.
- [x] Add structured step output examples with Pydantic schemas.
- [x] Add workflow resume examples.
- [x] Add before/after step hooks if they are needed for ADK-style parity.
- [x] Add artifact/document pipeline examples.
- [x] Add workflow docs covering sequential, parallel, loop, error policy, resume, and evaluation.
- [x] Add a reference document-processing workflow.
- [x] Add a research/report workflow example.

Decision:

- Workflow APIs remain beta in `src/zhivex_ai/api_stability.py` and `STABILITY.md`.
- No new before/after step hooks were added. Existing beta primitives already cover the Phase 5 examples through `AgentSession.state`, `metadata_key`, `run_store`, workflow traces, replay, and application-owned wrappers.

Evidence:

- `docs/WORKFLOWS.md` documents `SequentialAgent`, `ParallelAgent`, `LoopAgent`, `WorkflowStep`, shared state, `output_key`, `input_template`, `metadata_key`, error policies, app-owned resume, replay, and evaluation.
- `examples/agents/structured_workflow_outputs.py` demonstrates Pydantic validation of step text in application code.
- `examples/agents/workflow_resume.py` demonstrates app-owned workflow resume without adding a public `resume_workflow(...)` API.
- `examples/agents/artifact_document_workflow.py` demonstrates document artifact creation while keeping file storage application-owned.
- `examples/agents/research_report_workflow.py` demonstrates parallel research, sequential report synthesis, replay, and trace artifact inspection.
- `tests/test_workflow_examples.py` covers the four focused offline workflow examples.
- `Makefile` includes `tests/test_workflow_examples.py` in `make test-examples`.
- Focused verification: `.venv/bin/python -m pytest tests/test_workflow.py tests/test_workflow_examples.py -q` -> `14 passed`.
- Example verification:
  - `.venv/bin/python examples/agents/structured_workflow_outputs.py`
  - `.venv/bin/python examples/agents/workflow_resume.py`
  - `.venv/bin/python examples/agents/artifact_document_workflow.py`
  - `.venv/bin/python examples/agents/research_report_workflow.py`

Suggested files:

- `src/zhivex_ai/workflow.py`
- `tests/test_workflow.py`
- `examples/agents/`
- `examples/README.md`
- `docs/WORKFLOWS.md`

Exit criteria:

- [x] A user can learn workflows from docs and examples without reading source code.
- [x] Reference apps stay offline and deterministic by default.
- [x] The SDK remains orchestration infrastructure, not a vertical business-rule product.

Verification:

- `make check`
- Focused example tests
- Manual review of examples for no required provider credentials

## Phase 6: Documentation, DX, And Onboarding

Status: Complete.

Goal: make the SDK adoptable by a new backend developer without conversational context or source spelunking.

Checklist:

- [x] README exists.
- [x] Examples README exists.
- [x] Stability/versioning/support docs exist.
- [x] Add a concise quickstart that works offline or with one provider.
- [x] Add provider setup docs.
- [x] Add agent runtime guide.
- [x] Add workflow guide.
- [x] Add production API guide with FastAPI examples.
- [x] Add gateway/routing guide.
- [x] Add observability guide.
- [x] Add troubleshooting guide.
- [x] Add `.env.example`.
- [x] Add or refresh `CONTRIBUTING.md`.
- [x] Add migration guide from TypeScript mental model to Python.
- [x] Add examples index with requirements, runtime mode, and verification command per example.

Evidence:

- `docs/QUICKSTART.md` provides the 30-minute path: `make dev`, `make check`, offline examples, and one scoped live provider smoke.
- `docs/PROVIDERS.md` centralizes portable/native provider setup and links to `docs/providers/tier-1.md`.
- `docs/GATEWAY.md`, `docs/OBSERVABILITY.md`, and `docs/TROUBLESHOOTING.md` cover gateway routing, correlation fields, optional extras, auth, smoke skips, and common local errors.
- `docs/PARITY_MATRIX.md` closes the Phase 0 parity/GA-boundary documentation debt with implemented, documented, offline-tested, live-smoked, and stability columns.
- `docs/MIGRATING_FROM_TYPESCRIPT.md` documents the TypeScript-to-Python mental model.
- `.env.example` covers environment variables used by `scripts/run_live_smoke.py`.
- `CONTRIBUTING.md` documents setup, public API discipline, provider changes, examples, and release checks.
- `examples/README.md` now includes a verification index with mode, requirements, command, and verification path per key example.
- `tests/test_docs_onboarding.py` protects docs existence, README links, smoke env coverage, example index coverage, parity columns, and local markdown links.
- `Makefile` exposes `make test-docs`.

Suggested files:

- `README.md`
- `examples/README.md`
- `docs/QUICKSTART.md`
- `docs/PROVIDERS.md`
- `docs/AGENTS.md`
- `docs/WORKFLOWS.md`
- `docs/PRODUCTION.md`
- `docs/GATEWAY.md`
- `docs/OBSERVABILITY.md`
- `docs/TROUBLESHOOTING.md`
- `.env.example`
- `CONTRIBUTING.md`

Exit criteria:

- [x] A new developer can install, run `make check`, execute an offline example, and configure one live provider in under 30 minutes.
- [x] Docs clearly state what is stable, beta, experimental, portable, native-only, and provider-specific.

Verification:

- `make test-docs`
- `make check`
- Optional package install smoke in a fresh venv remains Phase 7 release discipline.

## Phase 7: Release, Packaging, And GA Discipline

Status: Complete.

Goal: make releases boring, auditable, and safe for downstream integrators.

Suggested release ladder:

- [x] `0.6.x`: API stability manifest and contract hardening
- [x] `0.7.x`: provider tier-1 contract and smoke docs
- [x] `0.8.x`: agent/workflow docs and mature beta examples
- [x] `0.9.x`: release candidate path documented and ready for final evidence capture
- [ ] `1.0.0`: stable GA surface after RC artifact install and human review

Checklist:

- [x] Package metadata exists in `pyproject.toml`.
- [x] `make build` exists.
- [x] `make release-check` exists.
- [x] Changelog exists.
- [x] Add release checklist that requires API stability review.
- [x] Add release checklist that requires provider matrix review.
- [x] Add release checklist that requires migration notes for breaking changes.
- [x] Add wheel/sdist install smoke in a fresh venv.
- [x] Verify optional extras: `postgres`, `mcp`, `api`, `otel`, `docx`.
- [x] Verify Python 3.11, 3.12, 3.13, and 3.14 in CI.
- [x] Add TestPyPI publishing flow if missing.
- [x] Add PyPI publishing flow if missing or incomplete.
- [x] Add release evidence template for final human review.

Evidence:

- `scripts/verify_release_artifacts.py` verifies the built wheel and sdist in fresh temporary virtual environments.
- The artifact verifier checks top-level imports, public exports, `py.typed`, `zhivex-skills`, and an offline agent/workflow smoke from the installed package.
- The artifact verifier installs and import-checks the optional extras: `postgres`, `mcp`, `api`, `otel`, and `docx`.
- `Makefile` exposes `release-install-check` and includes it in `make release-check`.
- `.github/workflows/ci.yml` now includes Python 3.11, 3.12, 3.13, and 3.14 and runs artifact verification after build.
- `.github/workflows/publish-pypi.yml` and `.github/workflows/publish-testpypi.yml` run `twine check` and artifact verification before publish.
- `RELEASING.md` requires API stability review, provider matrix review, migration notes, release artifact checks, and release evidence.
- `docs/RELEASE_EVIDENCE.md` provides the human review template.
- `docs/RC_READINESS.md` documents the RC evidence gate, `1.0.0` gate, and beta areas that can remain beta after RC.
- `tests/test_release_artifacts.py` protects release tooling, CI matrix coverage, and publish workflow verification.

Suggested files:

- `RELEASING.md`
- `CHANGELOG.md`
- `pyproject.toml`
- `Makefile`
- `.github/workflows/`
- `scripts/`

Exit criteria:

- [x] A release can be built, installed, smoke-tested, and audited before upload.
- [x] Stable API changes cannot ship silently.
- [x] Optional extras are verified or explicitly excluded from the release gate.

Verification:

- `make check`
- `make release-check`
- Fresh venv install from built wheel
- Optional extras install smoke

## Phase 8: Production Operations, Security, And Observability

Status: Complete.

Goal: make production failures diagnosable and risky capabilities explicit.

Checklist:

- [x] Middleware exists for telemetry, caching, and circuit breaking.
- [x] Safety policies, redaction, budget guards, and approval policies exist.
- [x] Observability module exists.
- [x] Add OpenTelemetry documentation and examples.
- [x] Standardize request IDs, session IDs, run IDs, and gateway attempt IDs in docs.
- [x] Document retry/backoff/circuit-breaker patterns.
- [x] Document provider error normalization.
- [x] Document cost reporting and budget guard patterns.
- [x] Document concurrency and cancellation guidance.
- [x] Document serverless vs long-running worker guidance.
- [x] Add security guide for secrets, tools, MCP, remote code/tool risks, and data retention.
- [x] Add secure defaults review for tool execution.
- [x] Add threat model notes for hosted tools, remote MCP, file access, and shell-like capabilities.

Evidence:

- `docs/OBSERVABILITY.md` documents OpenTelemetry wiring, standard correlation fields, gateway attempt logging, retryability, retry-after metadata, and cost fields.
- `docs/OPERATIONS.md` documents request/session/run/gateway attempt IDs, retry/backoff, circuit breakers, provider error normalization, cost and budget guard patterns, concurrency, cancellation, and serverless vs worker guidance.
- `SECURITY.md` documents secrets, data retention, secure tool defaults, MCP, hosted tools, file access, shell-like/code-execution risks, and reporting guidance.
- `docs/THREAT_MODEL.md` records threat-model notes for hosted tools, remote MCP, file access, shell-like capabilities, observability exports, and retained run artifacts.
- `examples/integrations/operations_hardening.py` is an offline example covering telemetry, circuit breakers, redaction, budget guards, retryable provider errors, and correlation metadata.
- `tests/test_operations_docs.py` and `tests/test_operations_hardening_example.py` protect the new documentation and offline example.

Suggested files:

- `src/zhivex_ai/observability.py`
- `src/zhivex_ai/middleware.py`
- `src/zhivex_ai/safety.py`
- `src/zhivex_ai/agent.py`
- `docs/OBSERVABILITY.md`
- `docs/PRODUCTION.md`
- `SECURITY.md`
- `examples/integrations/`

Exit criteria:

- [x] A production integrator knows how to log, trace, retry, budget, redact, and restrict agent/tool behavior.
- [x] Risky capabilities are clearly opt-in or clearly documented.
- [x] Operational examples are tested or manually verifiable.

Verification:

- `make check`
- Focused middleware/safety/agent tests
- Manual docs review against production examples

## Recommended Implementation Order

Use this order unless a concrete release need changes priorities:

1. Phase 0: finish parity matrix and GA boundary decisions.
2. Phase 2: expand typing coverage across public/core modules.
3. Phase 3: harden provider tier-1 contract tests and smoke docs.
4. Phase 4: make agent runtime documentation and contract tests product-grade.
5. Phase 5: finish workflow docs and reference app coverage.
6. Phase 6: polish onboarding and production docs.
7. Phase 7: prepare RC/GA release discipline.
8. Phase 8: complete production operations and security posture.

## Definition Of Done For Mature Product

The SDK can be treated as a mature product when all of these are true:

- [x] Public API surface is classified and drift-protected.
- [x] Stable APIs have versioning and deprecation rules.
- [x] `make check` remains green.
- [x] Core public modules and tier-1 provider contracts are typechecked.
- [x] Tier-1 providers have shared contract tests.
- [x] Provider support matrix is generated and verified.
- [x] Live smoke paths exist for tier-1 providers.
- [x] Agent runtime supports documented persistence, approvals, replay, tracing, and failure semantics.
- [x] Workflow APIs are documented with deterministic examples.
- [x] Production API, worker, and gateway integration examples exist.
- [x] Observability and security guidance are first-class docs.
- [x] Release process verifies build, install, changelog, API stability, and migration notes.
- [x] The package can graduate from beta to RC without relying on hidden maintainer knowledge.
- [ ] The package can graduate from RC to `1.0.0` after final artifact install review and release-owner approval.
