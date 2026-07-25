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

- None.

### Changed

- None.

### Fixed

- None.

### Deprecated

- None.

### Removed

- None.

## 0.13.0

### Added

- Added stable `create_deepseek()` support for current DeepSeek V4 Chat Completions, including text generation, streaming, JSON structured output, callable tools, reasoning controls, replay-safe `reasoning_content`, provider usage metadata, and retry/error handling.
- Added DeepSeek tier-1 contracts, gateway routing metadata, model catalog entries, live and installed-wheel smoke coverage, release workflow configuration, and runnable examples.

### Changed

- Promoted DeepSeek to the tier-1 portable provider set while keeping unsupported vision, files, embeddings, audio, moderation, and hosted-tool surfaces explicit.
- Bumped the package version to `0.13.0` for the DeepSeek provider release.

### Fixed

- None.

### Deprecated

- None.

### Removed

- None.

## 0.12.1

### Added

- Added direct Anthropic Claude Opus 5 catalog guidance and request-contract coverage for `claude-opus-5`.
- Added Opus 5 coverage for adaptive effort, disabled thinking, forced tool choice, intermediate system sections, refusals, context limits, and installed-wheel live smoke configuration.

### Changed

- Bumped the package version to `0.12.1` while keeping the package in Beta.
- Anthropic Fast mode now adds its required beta header automatically.
- Anthropic live agent smoke omits incompatible temperature sampling and uses a larger output-token budget.

### Fixed

- Preserved interleaved Anthropic thinking and redacted-thinking blocks, including signatures and ordering, across non-streaming and SSE tool loops.
- Prevented inherited Anthropic tool-call metadata from replaying the original `tool_use` block inside the following user tool-result message.
- Validated the final merged Opus 5 thinking/effort configuration so disabled thinking cannot be combined with `xhigh` or `max`.
- Rejected unsupported Opus 5 assistant prefill, manual thinking budgets, non-default sampling, and server-side Web Fetch before dispatch, including token counting.
- Preserved Anthropic refusal details, normalized context-window exhaustion to `length`, and discarded collected partial streaming text when the final result is a refusal.
- Tightened mid-conversation system-section placement to require a preceding user message or completed server-tool result.

### Deprecated

- None.

### Removed

- None.

## 0.12.0

### Added

- Added current catalog guidance for OpenAI/Azure GPT-5.6 Sol/Terra/Luna, restricted-access Anthropic Claude Mythos 5, Gemini Omni Flash and Flash-Lite Image, multimodal Qwen3.7 Max, and Kimi K3.
- Added beta `openai_programmatic_tool_calling_tool()` plus callable-tool `output_schema` and `allowed_callers` mapping, including replay-safe `program`, nested `caller`, function-output, and `program_output` handling.
- Added beta `anthropic_web_fetch_tool()` with the current Anthropic web-fetch contract.
- Added a `0.12.0` release plan focused on current provider contracts and model guidance.
- Added a production-oriented agent onboarding guide, explicit stable exports for agent results/streams, tools, handoffs, Qwen, and Kimi, plus strict live-smoke and install-from-artifact release gates.
- Added an opt-in strict agent-first live smoke that executes `run_agent(...)`, validates a real local tool call/result loop, and redacts configured secrets from failure output.
- Added isolated Postgres integration coverage for complete agent persistence plus concurrent idempotency and approval claims.
- Added versioned, revisioned `AgentRunState` persistence and explicit stale approval-resume claim reconciliation with `fail_agent_run_resume_claim(...)`.
- Added `AgentRunCancelled`, raised when an operator cancellation atomically wins against an active worker.
- Added `AgentEventDeliveryError`, which identifies the run/event and whether durable terminal state was already committed when an application callback failed.
- Added `ToolExecutionOutcomeUnknown` and idempotency/deadline fields on `ToolExecutionContext` so timed-out external operations are explicit and reconcilable.

### Changed

- Bumped the package version to `0.12.0` while keeping the package in Beta.
- Updated Qwen Responses reasoning to send all seven supported `reasoning.effort` values instead of deprecated `enable_thinking`.
- Updated Qwen Responses to use Alibaba Cloud Model Studio's current `/compatible-mode/v1/responses` path, validate Web Extractor/tool-choice constraints before requests, and document regional Batch model limits.
- Updated Anthropic hosted-tool defaults to `web_search_20260318`, `web_fetch_20260318`, and GA `code_execution_20260521`; expanded adaptive-thinking and mid-conversation system-message rules for current model families.
- Updated Kimi K3 to its distinct always-on reasoning contract with `reasoning_effort=low|high|max`, multimodal inputs, strict structured output, sampling validation, and reasoning-content continuation.
- Updated recommended OpenAI examples to GPT-5.6 Terra while keeping older catalog entries available for compatibility.
- Agent safety policies now apply configured redaction, execution options, and step/tool budgets to the actual runtime instead of acting only as descriptive helpers.
- Remote HTTP and MCP tools now require approval by default; only an explicit HTTP opt-out or an exact MCP `trusted_tools` allowlist entry grants trust.
- Tool timeouts now stop agent execution instead of allowing the model to continue after a side effect whose outcome is uncertain.
- Workflow suspension is represented explicitly and `fail_fast` cancels cooperative sibling tasks instead of letting pending branches continue.
- Built-in run stores now enforce compare-and-swap updates, atomic cancellation, terminal-state immutability, and unique non-null idempotency keys.

### Fixed

- Corrected current Gemini/Veo, Qwen snapshot, retired Anthropic Sonnet, and deprecated Imagen catalog guidance.
- Fixed Qwen mixed text/image Responses inputs to emit `input_text` alongside `input_image`.
- Fixed Qwen TTS downloads by upgrading allowlisted official signed HTTP URLs to HTTPS before the existing SSRF/host validation, without permitting untrusted hosts.
- Fixed Qwen live agent smoke schemas to use closed Pydantic objects and accept the exact success token with or without its optional final period.
- Preserved provider metadata on tool execution results so nested programmatic calls keep their `caller` linkage through serialization and replay.
- Made idempotency-key acquisition atomic in built-in run stores and restored the original persisted session/run identity when a key is reused.
- Bound durable approvals to a fingerprint of the complete tool definition and executor state, and reject legacy or changed tools before side effects run.
- Prevented a late worker completion from overwriting a run that an operator already cancelled.
- Prevented event callback failures from rewriting an already committed terminal run, and persist an explicit failed run when delivery fails before the terminal commit.
- Preflighted approval-claim recovery capabilities before executing resumed tools and reconciled a parent claim when its resumed child is cancelled.
- Kept the base custom `AgentRunStore` protocol source-compatible; atomic idempotency, cancellation, approval, and reconciliation methods are capability contracts required only when those features are used.
- Preserved completed tool results when a later call in the same model response suspends for approval; approval-capable batches execute sequentially so later side effects cannot race ahead.
- Redacted remote/MCP credentials, sensitive URLs, provider options, and raw responses from persisted checkpoints and checkpoint events.
- Accumulated usage, steps, tool results, artifacts, provider/model identity, and wall-clock limits across direct handoffs.
- Propagated MCP `isError` results as tool failures, moved blocking synchronous tools off the event loop, and made absent approval-policy decisions fail closed.
- Hardened release workflows to verify exact-version wheel/sdist metadata, tag ancestry, optional extras, Postgres dependencies, strict smoke execution, Twine checks, and PyPI/TestPyPI pre-publish gates.
- Blocked PyPI/TestPyPI Trusted Publisher jobs on a protected, installed-wheel agent tool smoke against at least one configured real provider.
- Raised optional API/MCP and development `click` floors to exclude the affected 8.3.2 release reported by the release security gate.

### Deprecated

- None.

### Removed

- None.

## 0.11.0

### Added

- Added durable human-in-the-loop agent approvals with `ApprovalDecision.require_human(...)`, persisted `PendingApproval` records, `get_pending_agent_approvals(...)`, and `resume_agent_run(...)`.
- Added `UIMessageToolApprovalChunk` so `to_ui_message_stream(...)` can preserve agent approval requests for frontend/SSE consumers.
- Added current catalog guidance for OpenAI GPT Realtime 2.1, Azure OpenAI `gpt-chat-latest`, Anthropic Claude Sonnet 5, and stable Gemini/Vertex `gemini-3.1-flash-lite`.
- Added a `0.11.0` release plan focused on production agent apps.

### Changed

- Bumped the package version to `0.11.0` while keeping the package in Beta.
- Added `stream_agent(..., idempotency_key=...)` parity with `run_agent(...)` for retry-safe streaming agent APIs backed by run stores.
- Promoted the local-tool approval lifecycle types and helpers into the documented stable surface when backed by the production run-store contract.
- Clarified production agent docs around suspended runs, pending approvals, and the distinction between `resume_agent(...)` and `resume_agent_run(...)`.
- Updated Anthropic Sonnet 5 reasoning support to use the adaptive-thinking `ReasoningConfig(effort=...)` path.
- Clarified that GPT-5.6 is not promoted into default OpenAI catalog guidance while official availability remains limited preview.

### Fixed

- Preserved serialized `AgentRunStep.messages` when deserializing stored run state so suspended runs retain enough context for approval resume and audit.
- Covered durable approval denial/resume behavior so rejected pending approvals persist as denied tool results and clear the pending queue.
- Normalized all Gemini callable-tool schemas before sending `functionDeclarations`, so strict Pydantic schemas with `additionalProperties: false` work in live agent tool loops.
- Raised known-vulnerable optional/development and CI tooling floors, including `setuptools>=83.0.0`, and added a dependency-audit release gate.
- Split package building from trusted publishing so only protected publish jobs receive PyPI OIDC permission, with GitHub Actions pinned to immutable commits.
- Release evidence now records the source commit, working-tree state, tool versions, and SHA256 digests for built artifacts.
- Required local-tool approvals now fail closed when no `approval_policy` is configured.
- Approval resume now atomically claims pending work in the in-memory, SQLite, and Postgres run stores, preventing concurrent workers from executing the same approved tool twice.
- Packaged-skill registry installs now require explicit remote-code trust, HTTPS/same-origin artifacts, bounded downloads and extraction, safe archive paths, and an installed-content checksum; generated agent tools also require approval for code execution, while skill permissions are documented as guardrails rather than an OS sandbox.
- Provider-returned download URLs now reject legacy numeric/private hosts and Qwen audio downloads are constrained to the configured provider host or Alibaba Cloud domains.
- HTTP response and UI-message request limits are enforced incrementally while data is read instead of after an unbounded buffer allocation.
- The production FastAPI agent example now fails closed on authentication/tenant/model configuration and applies tenant-scoped storage, bounded inputs, sanitized gateway attempts, and rate limiting.
- Hardened beta skill packages with explicit remote-code trust, HTTPS-only remote registries outside loopback, bounded downloads/extraction, archive and package-path validation, lockfile content verification, and entrypoint imports covered by the declared network policy.

### Deprecated

- None.

### Removed

- None.

## 0.10.0

### Added

- Added a `0.10.0` release plan focused on GA-candidate groundwork while keeping the package in Beta.

### Changed

- Bumped the package version to `0.10.0` while keeping the package in Beta.
- Promoted the production run-state, Postgres run-store, run serialization, cancellation, replay, and run-snapshot helpers into the documented stable surface.
- Clarified that in-memory and SQLite run stores, workflow agents, packaged skills, provider-managed approvals, trace artifacts, evaluation reports, and safety/redaction/budget helpers remain beta.
- Updated production API guidance so the tier-1 provider set includes OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex, Qwen, Kimi/Moonshot, and vLLM.

### Fixed

- Corrected gateway production docs to reflect that `GatewayConfig.fallback_on_refusal` defaults to `False` and fallback-on-refusal must be explicitly enabled.

### Deprecated

- None.

### Removed

- None.

## 0.9.0

### Added

- Production examples now include a FastAPI agent API boundary and an offline worker resume/idempotency boundary.
- `scripts/collect_release_evidence.py` and `make release-evidence` now generate release gate evidence under `docs/releases/<version>-evidence.md`.
- Anthropic support now tracks Claude Opus 4.8 (`claude-opus-4-8`), including native `reasoning.effort` mapping, adaptive thinking for Opus 4.7/4.8, and Opus 4.8 mid-conversation system messages.
- Anthropic support now tracks Claude Fable 5 (`claude-fable-5`) with adaptive-thinking request validation, refusal finish-reason normalization, and model-catalog guidance.
- Model catalog support for current OpenAI/Azure OpenAI GPT Realtime 2, GPT Image 2, Azure GPT-5.5, and text-embedding-3 reference IDs.
- Model catalog support for current Gemini/Vertex Gemini 3.5 Flash, Gemini 3.1 live/image/TTS, Imagen 4, Veo 3.1, and Lyria 3 reference IDs.
- Gemini realtime support now tracks Gemini 3.5 Live Translate (`gemini-3.5-live-translate-preview`) with typed translation config, audio-only validation, browser-token constraints, and model-catalog guidance.
- Model catalog support for current Qwen3.7 Max/Plus reference IDs while retaining Qwen3.6 and Qwen3.5 aliases.

### Changed

- Bumped the package version to `0.9.0` while keeping the package in Beta.
- `GatewayConfig.fallback_on_refusal` now defaults to `False`; set `fallback_on_refusal=True` to retry provider refusals on fallback targets.
- Gateway routing now emits `on_attempt` payloads for skipped targets, including missing adapters, capability skips, vision skips, and cost-budget skips.
- Gateway routing can now fail fast on missing provider adapters with `GatewayConfig(fail_on_missing_adapter=True)`.
- Updated OpenAI, Gemini, and Vertex examples/tests to use current realtime, image, and media model IDs while keeping OpenAI/Azure Sora or video-generation clients out of scope.
- Updated Qwen docs/examples/tests to use Qwen3.7 Plus as the current balanced reference model.

### Fixed

- Skill entrypoint tools no longer rebase `project_root` from absolute `output_path` values, and generated skill tools now propagate filesystem/network permission metadata for approval policies.
- Gemini resumable uploads and Qwen speech audio downloads now validate provider-returned URLs before sending user bytes or fetching generated media.
- Provider HTTP errors now redact sensitive response-body fields before formatting exceptions, gateway attempt payloads, and log-friendly messages.
- HTTP, SSE, realtime, UI-message parsing, and OpenAI-compatible audio streaming paths now apply defensive timeout, size, history, or raw-event caps.
- Gemini Files API get/delete now normalize official `files/*` names correctly instead of constructing duplicate `/files/files/*` paths.

### Deprecated

- None.

### Removed

- None.

## 0.8.0

### Added

- Gemini now exposes beta explicit context caching through `create_gemini().caches()`, including create/get/list/update/delete helpers and top-level `CachedContent` types.
- Azure OpenAI now supports Microsoft Entra ID authentication via `entra_token` or `entra_token_provider` on `create_azure_openai(...)`, covering generation, lifecycle clients, and realtime bootstrap without adding an `azure-identity` dependency.
- Qwen and Kimi/Moonshot are now tier-1 portable providers for text generation, streaming, structured output, and callable tools through `provider("model-id")`.
- Qwen now exposes OpenAI-compatible native Files and Batch clients through `provider.files()` and `provider.batches()` while keeping File Search as a hosted Responses tool with `vector_store_ids`.
- Tier-1 examples, live smoke configuration, provider support metadata, and shared contract tests now include Qwen and Kimi.

### Changed

- Azure OpenAI credential configuration now fails fast when API key and Entra ID authentication are both configured.
- Bumped the package version to `0.8.0`.
- Regenerated the provider support matrix for the expanded tier-1 set.
- Kept Qwen hosted tools, Qwen ASR/TTS, Kimi Files/Batch/token counting, and Kimi Formulas as native/provider-specific beta surfaces rather than portable guarantees.

### Fixed

- Gemini function-calling now preserves official `functionCall.id` values and sends matching `functionResponse.id` values for Gemini 3 tool loops.
- Validated Postgres agent run-store table prefixes with the same SQL identifier rule used by the Postgres memory and checkpoint stores.
- Updated FastAPI integration examples so provider HTTP failures return sanitized client messages instead of upstream response-body snippets.

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
