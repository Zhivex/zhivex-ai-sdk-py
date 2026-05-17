# Changelog

All notable changes to Zhivex AI SDK will be documented in this file.

The format follows grouped release notes with these sections when relevant:

- Added
- Changed
- Fixed
- Deprecated
- Removed

Related documents:

- [README.md](./README.md)
- [STABILITY.md](./STABILITY.md)
- [SUPPORT.md](./SUPPORT.md)
- [VERSIONING.md](./VERSIONING.md)

## Unreleased

### Added

- Azure OpenAI now supports Microsoft Entra ID authentication via `entra_token` or `entra_token_provider` on `create_azure_openai(...)`, covering generation, lifecycle clients, and realtime bootstrap without adding an `azure-identity` dependency.

### Changed

- Azure OpenAI credential configuration now fails fast when API key and Entra ID authentication are both configured.

### Fixed

- Validated Postgres agent run-store table prefixes with the same SQL identifier rule used by the Postgres memory and checkpoint stores.
- Updated FastAPI integration examples so provider HTTP failures return sanitized client messages instead of upstream response-body snippets.

### Deprecated

- None.

### Removed

- None.

## 0.8.0

### Added

- Qwen and Kimi/Moonshot are now tier-1 portable providers for text generation, streaming, structured output, and callable tools through `provider("model-id")`.
- Qwen now exposes OpenAI-compatible native Files and Batch clients through `provider.files()` and `provider.batches()` while keeping File Search as a hosted Responses tool with `vector_store_ids`.
- Tier-1 examples, live smoke configuration, provider support metadata, and shared contract tests now include Qwen and Kimi.

### Changed

- Bumped the package version to `0.8.0`.
- Regenerated the provider support matrix for the expanded tier-1 set.
- Kept Qwen hosted tools, Qwen ASR/TTS, Kimi Files/Batch/token counting, and Kimi Formulas as native/provider-specific beta surfaces rather than portable guarantees.

### Fixed

- None.

### Deprecated

- None.

### Removed

- None.

## 0.7.0

### Added

- Azure OpenAI now exposes beta native lifecycle clients for `provider.responses()`, `provider.conversations()`, and `provider.file_search_stores()` through the existing OpenAI-compatible `/openai/v1` route and Azure `api-key` authentication.

### Changed

- Updated the generated provider support matrix so Azure OpenAI reports native File Search, Responses, and Conversations support while keeping OpenAI-only clients such as Containers and Skills out of the Azure bundle.
- Prepared release evidence for the beta line, including fresh local validation, artifact verification, support-matrix review, and documented live-smoke skips.
- Updated release instructions so the current package version and tag examples point at `0.7.0`.
- Bumped the package version to `0.7.0`.

### Fixed

- Corrected stale release-guide references that still described the old `0.5.0` publishing path.

### Deprecated

- None.

### Removed

- None.

## 0.6.0

### Added

- Machine-readable API stability manifest in `src/zhivex_ai/api_stability.py`, with drift tests covering every symbol exported by `zhivex_ai.__all__`.
- `py.typed` packaging marker so downstream type checkers can treat the package as typed.
- Grouped local test targets for contract, core, providers, examples, and agent/workflow checks.
- Tier-1 vLLM support with `create_vllm(...)`, covering the SDK's OpenAI-compatible text, streaming, structured output/tools, embeddings, transcription, and realtime ASR primitives.
- Provider-agnostic agent skills with `Agent(..., skills=...)`, `skill(...)`, `load_skill(...)`, and `discover_skills(...)`.
- `SKILL.md` loading with optional `agents/openai.yaml` metadata for display text, implicit-invocation policy, and MCP dependency discovery.
- Agent-skills example coverage in `examples/agents/skills.py`.
- Sticky skill persistence through agent sessions plus `AgentSkillActivatedEvent` and `AgentSkillSkippedEvent` observability hooks.
- Explicit session helpers `set_agent_session_skills(...)` and `clear_agent_session_skills(...)` for replacing or clearing sticky skills without editing metadata manually.
- Declarative production policies for agent skills: `priority`, `triggers`, `anti_triggers`, provider/model allowlists, non-sticky skills, dependency failure modes, and session introspection with `get_agent_session_skills(...)`.
- Beta packaged-skill support with `skill.yaml`, `load_skill_package(...)`, `validate_skill(...)`, `install_skill(...)`, `list_installed_skills(...)`, `run_skill(...)`, and `publish_skill(...)`.
- Project-level skills manifests and lockfiles in `.agents/skills.toml` and `.agents/skills.lock.toml`, plus cache installs under `~/.zhivex/skills/`.
- Static HTTP skill-registry publishing via `publish_skill(...)` and the `zhivex-skills` CLI.
- Packaged-skill runtime artifacts in `AgentRunResult.artifacts` plus beta observability events for package resolution, dependency checks, executions, and created artifacts.
- The first official packaged skill: `docx`, powered by `python-docx`.
- Beta provider agent capability metadata with `AgentCapabilities`, `AgentSupportTier`, `get_agent_capabilities(...)`, `get_agent_support_tier(...)`, and an agent-capabilities section in the generated support matrix.
- Beta first-class hosted tools with `HostedToolDefinition`, `HostedToolClass`, `hosted_tool(...)`, hosted-tool inspectors, `ProviderDataPart`, `provider_data_part(...)`, OpenAI/Azure MCP approval-response helpers, and native provider mapping for OpenAI, Azure OpenAI, Gemini, Vertex, and Anthropic.
- Beta hosted-tool phase 2 coverage with shared fail-fast validation, typed OpenAI/Azure provider-data payloads and parsers, `StreamProviderDataEvent`, and provider-managed MCP approval handling in `run_agent(...)` / `stream_agent(...)`.
- Beta response-reference ergonomics with `openai_response_reference(...)`, `azure_openai_response_reference(...)`, response-id extraction helpers, generic provider-data extraction helpers, and `UIMessageProviderDataChunk` support in UI streaming helpers.
- Conservative Anthropic hosted-tool compatibility updates: current MCP can be opted into with `anthropic_mcp_server(..., version="current")` or `provider_options={"anthropic_mcp_beta": "mcp-client-2025-11-20"}`, while current web search and newer code-execution tool versions remain explicit `tool_type` choices.
- Google native model coverage for Gemini and Vertex: multimodal `embed_content(...)` / `embed_content_many(...)`, Gemini/Nano Banana and Imagen images, Veo long-running videos, Lyria media generation, Gemini Batch jobs, and Gemini Interactions/Deep Research clients.
- Kimi/Moonshot native support for Chat Completions, Files, Batch, token estimation, and official Formulas tools through `provider.formulas()`, `KimiFormulaClient`, `kimi_formula_toolset(...)`, and `KIMI_OFFICIAL_TOOL_URIS`.
- Qwen native updates for Alibaba Cloud Model Studio's current surface: `DASHSCOPE_API_KEY` fallback, `qwen_mcp_tool(...)`, Qwen3-ASR transcription via `provider.native.transcription_model("qwen3-asr-flash")`, and catalog entries for current Qwen 3.6, Qwen3.5, Qwen3 Max, Coder, embedding, rerank, ASR, and TTS model IDs.
- Dedicated `examples/text/qwen_native.py` coverage for Qwen text, hosted web search, embeddings, optional Qwen3-ASR, and optional Qwen3-TTS.
- Refreshed `default_model_catalog` recommendations for the current reference-model set across OpenAI GPT-5.5/GPT-5.4, Claude Opus 4.7/Sonnet 4.6/Haiku 4.5, Gemini 3.1/3 Flash, Vertex Gemini, Bedrock Claude 4.x/Nova, Qwen, and Kimi.
- Beta agent platform parity helpers: durable `AgentRunState` stores, idempotent run reuse, run-tree cancellation, native subagent tools, `run_agent_group(...)`, replay/snapshot helpers, evaluation fixtures/reports, trace artifacts, redaction policies, and budget guards.
- Declarative beta workflow agents with `WorkflowStep`, `SequentialAgent`, `ParallelAgent`, `LoopAgent`, shared `AgentSession.state`, workflow trace events, and workflow expectation validation.
- Mock model/tool helpers for deterministic agent evaluation and local examples without provider credentials.
- Shared offline tier-1 provider contract tests covering OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex, and vLLM.
- Tier-1 provider setup documentation in `docs/providers/tier-1.md` plus a focused `examples/text/tier1_providers.py` portable text example.
- Shared offline agent-runtime contract tests covering run, stream, resume, approvals, handoffs, run-store idempotency, cancellation, replay, and failure persistence.
- Production agent runtime documentation in `docs/AGENTS.md`, `docs/PRODUCTION.md`, and focused guides under `docs/agents/`.
- Offline agent runtime examples for multi-agent handoff, human approval, durable resume, replay, and trace inspection.
- Workflow production guidance in `docs/WORKFLOWS.md`, with offline examples for structured step validation, app-owned resume, document artifacts, and research/report synthesis.
- Onboarding and DX documentation for quickstart, providers, gateway routing, observability, troubleshooting, TypeScript migration, parity/GA boundaries, `.env.example`, and contributor workflow.
- Release artifact verification with fresh venv wheel/sdist install smoke, optional extras checks, CI/publish workflow hardening, and a release evidence template.
- Production operations and security hardening guidance covering OpenTelemetry, request/session/run/gateway correlation IDs, retries, circuit breakers, provider error normalization, cost reporting, budget guards, concurrency, cancellation, serverless/worker deployment, MCP, hosted tools, file access, shell-like tools, and data retention.
- Offline `examples/integrations/operations_hardening.py` reference for telemetry, circuit breakers, redaction, budget guards, retryable provider errors, and correlation metadata without live provider credentials.

### Changed

- Expanded the `mypy` gate to cover additional public/core modules including provider base contracts, durable agent run state, serialization helpers, agent runtime, workflow/safety internals, realtime setup, and packaged-skill support.
- Recorded provider adapter typing as explicit Phase 3 debt while keeping the Phase 2 internal quality gate passing through the documented provider override.
- Expanded the main `mypy` gate to include tier-1 provider adapters and recorded DeepSeek as deferred for Python GA.
- Added Azure OpenAI to the live smoke runner so every tier-1 provider has documented optional smoke coverage.
- Promoted `AgentRuntime` and `AgentRegistry` into the stable manifest while keeping run stores, replay/evaluation, traces, safety helpers, and provider-managed approvals in beta.
- `create_qwen()` now supports `region="intl" | "us" | "cn"` for Alibaba Cloud Model Studio's documented OpenAI-compatible endpoints while retaining explicit `base_url` and `responses_base_url` overrides.
- `create_kimi()` now follows Moonshot's documented environment names and runtime surface: `MOONSHOT_API_KEY`, optional `MOONSHOT_BASE_URL`, Chat Completions for text generation, and `kimi-k2.6` as the catalog default.
- Kimi provider metadata now reports native Files and Batch support, removes the unsupported embeddings claim, and keeps Kimi in the compatibility tier without the portable badge.
- Clarified the documentation split between portable agent skills and the native OpenAI `provider.skills()` lifecycle client.
- Promoted the portable agent-skill runtime, session controls, and skill observability hooks into the documented stable surface.
- Clarified that the original runtime skills remain stable while the new packaged-skill layer is beta.
- Clarified that first-class hosted tools inside `tools={...}` are now the preferred beta-native path, while legacy raw `provider_options` tool payloads remain accepted where already supported for backward compatibility.
- Hosted tools now fail fast in shared foundation APIs when they target the wrong provider, require unsupported capability classes, or attempt unsupported `tool_choice` combinations such as named hosted-tool forcing.
- `openai_response_options(...)` can now derive `previous_response_id` directly from provider-data response references, assistant messages, or prior results through the new beta response-reference helpers.
- The README support matrix is now intended to be rewritten from runtime metadata via `scripts/generate_support_matrix.py --write-readme`, keeping provider docs aligned with the generated portable/native/agent capability tables.
- Clarified Azure OpenAI's hosted-tool helper scope separately from OpenAI-only lifecycle clients for vector-store/file-search administration, Responses, and Conversations.
- Gemini text generation now preserves non-text inline outputs as image or file parts instead of silently dropping provider media payloads.
- Promoted Qwen and Kimi agent-capability metadata to `tier-b` while keeping both providers in the compatibility tier for portable support.
- Added Bedrock Converse native tool-use mapping for callable SDK tools and promoted Bedrock agent-capability metadata to `tier-b` without changing its native-only portable tier.
- Agent runs now surface provider-managed tool result blocks, such as Anthropic MCP tool results, as `AgentToolResultEvent` events in both regular and streaming runs.
- Gemini and Vertex tool-loop messages now preserve incoming `thoughtSignature` / `thought_signature` values and resend them as the official `thoughtSignature` field.
- Added native Bedrock ConverseStream support for text deltas, incremental tool-use calls, finish reasons, and token usage.

### Fixed

- Moved Qwen provider coverage into dedicated tests, exposed the raw native Responses client, and hardened its native Responses contract for hosted tools, MCP, streaming, response continuation options, ASR, and TTS without promoting it beyond compatibility support.
- Aligned beta provider agent-capability metadata with the hosted-tool helpers already supported by adapters, including Anthropic code execution and Gemini/Vertex file-search plus computer-use hosted tools.
- Clarified across README, STABILITY, and SUPPORT that the generated `Agent Capabilities` matrix is beta guidance for hosted tools and provider-managed events, with provider-managed approval/runtime integration currently limited to OpenAI and Azure OpenAI.
- Fixed the local `make check` type gate by tightening protocol annotations, dataclass serialization narrowing, schema-adapter typing, HTTP response headers, and provider model-cache generics.
- Restored the backwards-compatible `EmbedOutput.embedding` convenience accessor while keeping `EmbedOutput.embeddings` as the canonical multi-result field.

### Deprecated

- None.

### Removed

- None.

## 0.5.0

### Added

- Stability, versioning, support, and changelog documentation for the documented public surface.
- Production API guidance with FastAPI integration examples for direct, streaming, and gateway-backed APIs.
- Observability guidance and examples for telemetry, request correlation, and gateway attempt hooks.
- Contract coverage for the stable surface, provider support matrix, tier-1 provider assertions, and public package status.
- Dedicated Ollama provider coverage for native text generation, streaming, structured output, tool calling, embeddings, and local smoke validation.

### Changed

- Promoted the package maturity signal from `Alpha` to `Beta`.
- Promoted Anthropic to the tier-1 portable provider set for text-generation API paths.
- Documented tier-1 providers for the stable production API story: OpenAI, Anthropic, Azure OpenAI, Gemini, and Vertex.
- Enforced CI quality gates for linting, type checking, coverage, build validation, and a minimum coverage floor of `80%`.
- Expanded `mypy` coverage over core API-facing modules including `generate_text`, `generate_object`, `middleware`, and `transport`.
- Documented the recommended local Ollama path with `provider.native.*`, the default compatibility token, and optional smoke-run configuration.
- Added async context manager support to `ToolRegistry` and updated MCP guidance to close registries cleanly after use.
- Promoted MCP helpers, MCP-backed registries, and Postgres-backed agent stores into the documented stable surface for production integrations.

### Fixed

- Fixed file-cache serialization so cached generate results round-trip correctly through the on-disk cache.
- Fixed SSE response serialization for dataclass-backed UI message chunks.
- Fixed request snapshotting in `generate_text()` so recorded step requests do not drift as later messages are appended.
- Fixed `stream_agent()` so output guardrails can block streamed assistant text before it is emitted, while preserving live non-text agent events.
- Fixed Postgres-backed agent stores to reject invalid `table_prefix` values early with a clear validation error.
- Fixed agent tool-callable inspection to fall back gracefully when `inspect.signature()` is unavailable.
- Fixed realtime/live voice adapters to distinguish turn completion from true session shutdown, align OpenAI and Azure browser bootstrap with `realtime/client_secrets`, support current OpenAI output-audio events, and accept Gemini ephemeral `access_token` connections.

### Deprecated

- None.

### Removed

- None.

## 0.4.0

### Added

- Initial public release line for the current package version.

### Changed

- No additional entries recorded.

### Fixed

- No additional entries recorded.

### Deprecated

- None.

### Removed

- None.
