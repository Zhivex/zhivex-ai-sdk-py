# Support Policy

Zhivex AI SDK is currently published as a `Beta` package.

Related documents:

- [README.md](./README.md)
- [STABILITY.md](./STABILITY.md)
- [VERSIONING.md](./VERSIONING.md)
- [CHANGELOG.md](./CHANGELOG.md)
- [docs/SCOPE.md](./docs/SCOPE.md)

## Product boundary

The supported core product is the portable foundation plus the agent runtime, durable agent execution, provider adapters, gateway, backend transport contracts, and Stable application-owned workflow orchestration. Evaluations, protocol hosting, the general CLI/playground, packaged-skill distribution, realtime, named external workflow-engine adapters, and broad provider-native resource lifecycles are focused extension areas. Their presence in this repository does not make them part of the minimum adoption or package-wide GA promise.

Existing top-level imports remain supported according to their current stability level. New Beta or Experimental APIs should use a focused namespace instead of enlarging the `zhivex_ai` root.

## Provider evidence levels

Provider capability and live certification are deliberately separate:

- **Contract-supported**: the provider participates in runtime metadata, deterministic shared contracts, provider-specific tests, documentation, and live-smoke configuration.
- **Release-certified**: a recorded live run passed for an exact provider, model, operation set, built artifact, and source revision.
- **Experimental/native-only**: the provider or feature does not carry the portable compatibility promise.

The current Tier-1 roster is a contract-supported classification. It must not be described as release-certified unless matching evidence exists for the exact release being discussed.

<!-- BEGIN GENERATED PROVIDER CERTIFICATION -->
## Current provider certification

This table is generated from the versioned certification policy and validated evidence records.
Contract tests, installed-wheel execution, and live certification are separate evidence layers.
A passed exact-artifact live record remains current for 30 days; older records are shown as `stale`.
Missing, blocked, failed, unsupported, local-only, or malformed evidence never produces `release-certified` status.
Meta Standard and Meta Contributor are independent targets; Contributor cannot certify the Stable Standard route.

| Provider | Target | Surface | Model | Source tests | Installed wheel | Live | Recorded at | Operations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openai | openai-standard | standard | gpt-5.6-luna | contract-supported | passed | integration-only | 2026-09-05T18:50:48.041672+00:00 | agent-tool=passed, generation=passed, streaming=passed, structured-output=passed |
| anthropic | anthropic-standard | standard | claude-fable-5-1 | contract-supported | passed | integration-only | 2026-09-05T18:51:08.221408+00:00 | agent-tool=passed, generation=passed, streaming=passed, structured-output=passed |
| azure-openai | azure-openai-standard | standard | unconfigured | contract-supported | passed | blocked | 2026-09-05T18:51:08.545930+00:00 | agent-tool=blocked, generation=blocked, streaming=blocked, structured-output=blocked |
| gemini | gemini-standard | standard | gemini-3.8-flash | contract-supported | passed | blocked | 2026-09-05T18:57:24.043231+00:00 | agent-tool=blocked, generation=blocked, streaming=blocked, structured-output=blocked |
| vertex | vertex-standard | standard | gemini-3.8-flash | contract-supported | passed | blocked | 2026-09-05T18:51:39.596587+00:00 | agent-tool=blocked, generation=blocked, streaming=blocked, structured-output=blocked |
| qwen | qwen-standard | standard | qwen3.8-max-0902 | contract-supported | passed | integration-only | 2026-09-05T18:51:47.345485+00:00 | agent-tool=passed, generation=passed, portable-retrieval=unsupported, streaming=passed, structured-output=passed |
| kimi | kimi-standard | standard | kimi-k3 | contract-supported | passed | blocked | 2026-09-05T18:51:47.623477+00:00 | agent-tool=blocked, generation=blocked, portable-retrieval=unsupported, streaming=blocked, structured-output=blocked |
| deepseek | deepseek-standard | standard | deepseek-v4-flash | contract-supported | passed | integration-only | 2026-09-05T18:51:52.053643+00:00 | agent-tool=passed, generation=passed, portable-retrieval=unsupported, streaming=passed, structured-output=passed |
| meta | meta-standard | standard | muse-spark-1.2 | contract-supported | passed | integration-only | 2026-09-05T18:52:04.774757+00:00 | agent-tool=passed, generation=passed, portable-retrieval=passed, streaming=passed, structured-output=passed |
| vllm | vllm-deployment | deployment | Qwen/Qwen2.5-1.5B-Instruct | contract-supported | passed | blocked | 2026-09-05T18:52:05.069155+00:00 | agent-tool=blocked, generation=blocked, portable-retrieval=blocked, streaming=blocked, structured-output=blocked |
| meta | meta-contributor | contributor | muse-spark-1.2-contributor | beta-contract | passed | certified | 2026-09-05T18:15:14.927784+00:00 | agent-tool=passed, generation=passed, portable-retrieval=passed, streaming=passed, structured-output=passed |
<!-- END GENERATED PROVIDER CERTIFICATION -->

## Support expectations

- The latest beta release line is the primary target for fixes, documentation updates, and examples.
- The previous beta minor release may receive critical fixes when the change is low risk and clearly scoped.
- Stable APIs are the main compatibility contract for production integrations.
- Stable agent integrations include the core runtime and typed result/stream handles, typed in-process dependencies, dynamic instructions, lifecycle hooks, run middleware, local tool definitions and execution contracts, handoffs, session helpers, portable agent skills, MCP helper path, MCP-backed registries, Postgres-backed memory/checkpoint stores, Postgres-backed run-state/replay helpers, and durable local-tool approval resume documented in [STABILITY.md](./STABILITY.md). Native subagent tools such as `create_subagent_tool(...)` remain beta.
- Agent dependencies are application-owned runtime objects. They are propagated in process but intentionally excluded from checkpoints, run state, traces, and serialized tool contexts; applications must supply them again when resuming a run.
- Agent production guidance is documented in [docs/AGENTS.md](./docs/AGENTS.md), [docs/PRODUCTION.md](./docs/PRODUCTION.md), [docs/OPERATIONS.md](./docs/OPERATIONS.md), and [PRODUCTION_APIS.md](./PRODUCTION_APIS.md).
- Onboarding, provider setup, gateway routing, observability, security, and troubleshooting guidance live under [docs/QUICKSTART.md](./docs/QUICKSTART.md), [docs/PROVIDERS.md](./docs/PROVIDERS.md), [docs/GATEWAY.md](./docs/GATEWAY.md), [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md), [SECURITY.md](./SECURITY.md), and [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md).
- Beta APIs are supported for early adoption, but they may still evolve between minor releases with changelog coverage.
- The Beta `create_file_generate_cache(...)` runtime is supported on application-owned local filesystems that provide atomic same-directory `os.replace(...)`. Linux is exercised in CI across the supported Python versions; Windows and network/distributed filesystems require deployment-specific validation. File data is flushed and synced before publication, POSIX directory metadata is synced when the platform supports it, corrupt or non-regular entries raise `ValidationError`, and cache-owned stale-temp cleanup is deliberately bounded.
- Beta provider capability and model-catalog metadata describes a dated, source-backed provider snapshot and can drive an explicitly configured gateway's fallback ranking and fail-closed lifecycle, surface, price, and capability checks. Recommendations never substitute for typed capabilities, and catalog metadata itself is not a stable behavioral guarantee or live certification.
- Beta hosted-tool definitions and `provider-data` control parts describe the preferred native tool-registration path, but provider-specific execution semantics may still evolve between minor releases.
- Beta provider-managed approval flows currently cover OpenAI and Azure OpenAI only, including typed `provider-data` payload parsing and agent-runtime approval-policy integration.
- Beta response-reference helpers, `provider-data` UI chunks, and `tool-approval` UI chunks are supported for OpenAI/Azure continuation workflows, approval UIs, and observability, but their exact ergonomics may still evolve between minor releases.
- Beta agent platform helpers cover in-memory and SQLite run stores, native subagent tools, cooperative in-process cancellation tokens, per-tool guardrails, controlled agent groups, evaluation reports, hierarchical trace artifacts, run-tree snapshots, redaction policies, and budget guards. The Postgres run store, run-state serialization, durable cancellation records, replay, run-snapshot helpers, and durable local-tool approval lifecycle are part of the stable agent surface.
- Beta evaluations support repeated trials, bounded concurrency, pass-rate confidence intervals, latency dispersion, token/application-cost metrics, redacted trajectories, schema-versioned JSON/JUnit artifacts, and application-curated trace datasets as documented in [docs/EVALUATIONS.md](./docs/EVALUATIONS.md). The default judge is deterministic and expectation-based; offline fixtures and model judges do not replace live quality evidence, domain review, or regulated review.
- Beta A2A v1, AG-UI, Responses-compatible hosting, CLI, and playground support is documented in [docs/PROTOCOLS.md](./docs/PROTOCOLS.md) and [docs/CLI.md](./docs/CLI.md). The supported A2A and AG-UI extras use the pinned upstream major/minor lines; the Responses host is a strict text/message subset with optional application-owned storage/replay. Authentication, tenant-scoped durable protocol/task state, rate limits, DLP, audit policy, and public deployment remain application responsibilities. Included in-memory stores are one-process references only.
- Stable workflow support covers declarative agents, validated DAGs, append-only checkpoints, explicit v1→v2 checkpoint migration, interrupts, resume/fork/cancel, step retry, typed operational failures, execution leases with atomic fenced checkpoint commits, and replay projections as documented in [docs/WORKFLOWS.md](./docs/WORKFLOWS.md). In-memory managers are process-local; SQLite targets one reviewed filesystem; optional Postgres checkpoint and lease managers use bounded/injectable pools, server time, namespaces, and checked schema metadata. Deployment-specific contention/integration validation remains required, and external effects still need destination idempotency or reconciliation.
- Beta workflow adapter factories for DBOS, Temporal, Prefect, and Restate provide dependency-free callback envelopes and conservative capability metadata only. They are not engine clients, managed schedulers, workers, or live-certified third-party integrations; applications own the actual engine binding and its end-to-end validation.
- Beta Google native clients cover Gemini/Vertex image, video, music/audio, batch, interaction, and Gemini explicit context-cache workflows where the official Google endpoints expose them. The catalog tracks Gemini 3.7 Flash, GA Omni 1.1 Flash, Gemini 3.5 Transcribe, Live Translate, Gemini 3.1 live/image/TTS, Veo, and Lyria reference IDs. Deprecated and retired previews remain separate lifecycle records rather than aliases. Preview Google models remain subject to Google availability, quota, and deprecation windows.
- Beta Kimi/Moonshot native support covers Chat Completions, Files, Batch, token estimation, and official Formulas tools according to the current Kimi Open Platform docs. Portable Kimi is tier-1 for text, streaming, structured output, and callable tools.
- DeepSeek is tier-1 for text generation, streaming, JSON structured output, callable tools, and reasoning through its official Chat Completions API. Provider-specific strict-tool/prefix beta routing is supported, while vision, files, embeddings, audio, moderation, and hosted tools are outside the current contract.
- vLLM is supported as a tier-1 provider for SDK primitives exposed by its OpenAI-compatible server. Embeddings, transcription, and realtime ASR support depends on the model/task served by vLLM; vLLM custom endpoints such as tokenize, rerank, classify, and score are outside the SDK support contract.
- Meta Model API is tier-1 for the Stable `create_meta()` factory and Standard `muse-spark-1.2` portable text, streaming, JSON Schema structured output, callable tools, agent tool loops, and application-supplied retrieval through `PortableRetrievalConfig`. That retrieval capability injects bounded `PortableDocument` text; it is not Meta Files, hosted search, or raw Responses. `tool_choice` is limited to `auto`. Contributor models, Responses continuation, hosted web/tool search, Files, raw Responses, hosted tools, and multimodal/native extras remain Beta. Embeddings, speech output, transcription, grounding, Realtime, image generation, and video generation are not claimed. Worktree or locally built wheel smokes are integration evidence only until a clean release candidate records the exact artifact and source revision.
- Tier-1 provider claims are contract-supported through generated support metadata, shared offline contract tests, provider-specific tests, and optional live smoke documentation in [docs/providers/tier-1.md](./docs/providers/tier-1.md). Tier-1 does not by itself assert current release certification across the complete set; live evidence is provider-, model-, operation-, artifact-, and SHA-specific.
- The README support matrix is generated from runtime metadata and reflects the current provider capability story, but its `Agent Capabilities` section should still be read as beta guidance rather than a stable behavioral guarantee.
- Experimental APIs are available for evaluation and feedback, but they do not carry support or compatibility guarantees.

## What qualifies for patch releases

Patch releases are intended for focused changes such as:

- bug fixes in the stable surface
- low-risk regressions in tier-1 providers
- documentation corrections that unblock adoption
- packaging, build, or release metadata fixes

Patch releases should not introduce silent breaking changes to the documented stable surface.

## Provider support scope

The current tier-1 providers for the stable production API story are:

- OpenAI
- Anthropic
- Azure OpenAI
- Gemini
- Vertex
- Qwen
- Kimi/Moonshot
- DeepSeek
- Meta Model API
- vLLM

Meta Model API is tier-1 only for `create_meta()` with Standard `muse-spark-1.2` portable text generation, streaming, structured output, callable tools, agent tool loops, and application-supplied retrieval through `PortableRetrievalConfig`. Portable retrieval uses application-owned `PortableDocument` text and does not promote Contributor models, `meta_hosted_tool()`, `meta_web_search_tool()`, `meta_tool_search_tool()`, Files, raw Responses, hosted tools, or multimodal/native extras. Tier-1 contract support is not release certification. See [docs/providers/meta.md](./docs/providers/meta.md).

Anthropic is tier-1 for the portable text-generation surface in this repository. The catalog tracks Claude Opus 5, Fable 5, Sonnet 5, restricted-access Mythos 5, and Opus 4.8. Opus 5 uses adaptive thinking by default, supports portable effort through `max`, and can disable thinking only through effort `high`; manual budgets and non-default sampling fail before dispatch. Opus 5, Fable 5, Mythos 5, and Opus 4.8 accept valid intermediate system sections. Opus 5 does not support assistant prefill, server-side Web Fetch, or Priority Tier through this integration. Native hosted helpers otherwise default to `web_search_20260318`, `web_fetch_20260318`, and GA `code_execution_20260521`. Anthropic refusals preserve provider details and normalize to `finish_reason="refusal"`; embeddings, transcription, and speech remain unavailable.

Azure OpenAI is tier-1 for the portable production surface and exposes beta native lifecycle clients for Responses, Conversations, Realtime, and Vector Store / File Search management through the versionless `/openai/v1` route. The catalog tracks GPT-5.6 Sol/Terra/Luna, `gpt-chat-latest`, GPT Realtime 2.1, GPT Image 2, and text-embedding-3 reference IDs where deployments are available. The Azure provider supports either API key authentication or Microsoft Entra ID token/provider authentication; configure only one credential mode at a time. OpenAI-only native clients such as Containers, Skills, Uploads, Moderations, Images, Batches, and Sora/video lifecycle clients are not exposed on the Azure provider bundle.

OpenAI is tier-1 for the portable production surface. The catalog tracks GA GPT-5.6 Sol/Terra/Luna, using `gpt-5.6-sol` as the flagship and `gpt-5.6` as its alias, while retaining older entries for compatibility and GPT Realtime 2.1 for realtime guidance. Responses is the recommended reasoning/tool route. The beta native surface includes explicit prompt-cache/reasoning options and Programmatic Tool Calling with replay-safe preservation of `program`, nested `caller`, function outputs, and `program_output` items.

Gemini and Vertex are tier-1 for the portable production surface. The latest catalog reference is `gemini-3.7-flash`; `gemini-3.6-flash` and `gemini-3.5-flash-lite` retain exact offline-contract evidence, and the adapter rejects unsupported custom sampling and assistant-prefill requests for those covered contracts before dispatch. Google-specific media generation, Batch API, Interactions API, Deep Research, explicit context caching, Live Translate, and Veo operation workflows are native clients rather than portable contracts. Gemini Developer API catalog guidance includes GA Interactions-only `gemini-omni-1.1-flash`, `gemini-3.5-transcribe`, and `gemini-3.1-flash-lite-image`; Omni is not claimed for Vertex. Deprecated/retired preview IDs are recorded separately with replacements and are not aliases. Region and quota availability still apply.

Qwen is tier-1 for portable text generation, streaming, structured output, callable tools, and embeddings. `create_qwen(region="intl" | "us" | "cn")` maps to Alibaba Cloud Model Studio's documented OpenAI-compatible regions and the current `/compatible-mode/v1/responses` path, while `base_url` and `responses_base_url` remain explicit overrides. The catalog tracks pay-as-you-go GA `qwen3.8-max` separately from the Token Plan's `qwen3.8-max-preview`. For the GA model, Responses covers text, streaming, current `input_text` / `input_image` vision input, all seven `ReasoningConfig` efforts, function tools, and the announced web/code/image-search built-ins. The adapter selects Chat Completions for native JSON Schema output, image/video `FilePart` input, or a reasoning token budget; structured output disables thinking and Chat reasoning state is preserved for replay. Web Extractor requires Web Search, and explicit reasoning cannot be combined with forced required/named tool choice. Hosted helpers, raw Responses, Files, Batch, ASR, and TTS remain beta native surfaces; Batch model availability is regional, with Singapore currently limited to the documented stable aliases.

Kimi/Moonshot is tier-1 for portable text generation, streaming, structured output, and callable tools through the official Chat Completions API. `kimi-k3` is the current catalog reference: it is multimodal, always reasons, accepts `reasoning_effort` values `low`, `high`, or `max`, and rejects incompatible K2 thinking/sampling choices. K2.6/K2.5 keep their legacy `thinking` contract. Files, Batch, token estimation, and Formulas remain beta native clients; portable embeddings, speech, and transcription are not claimed.

DeepSeek is tier-1 for portable text generation, streaming, JSON structured output, callable tools, and reasoning through Chat Completions. The catalog tracks current `deepseek-v4-flash` and `deepseek-v4-pro`; retired `deepseek-chat` and `deepseek-reasoner` IDs fail before dispatch. The adapter preserves `reasoning_content` for tool replay, maps portable reasoning effort, retains provider-specific usage metadata, and routes strict tools or prefix completion to DeepSeek's beta base URL when requested. Vision, files, embeddings, audio, moderation, and hosted tools are not claimed.

vLLM is tier-1 for portable text, streaming, structured output/tools, embeddings, and transcription through the vLLM OpenAI-compatible server. Realtime ASR is exposed through `provider.native.realtime_model(...)` and remains subject to the experimental realtime API stability level.

Other providers remain available, but they should be treated according to the support matrix and the stability level of the specific feature area.

## Upgrade expectations

- Every user-visible change should appear in [CHANGELOG.md](./CHANGELOG.md).
- Changes to stable APIs require migration guidance.
- Deprecations should be documented before removal.

## September 5, 2026 model refresh

The catalog and offline tests cover GPT-6 Astra on OpenAI/Azure Responses, Claude Fable 5.1 and restricted-access Mythos 5.1, Gemini 3.8 Flash on Gemini/Vertex, and Qwen3.8-Max-0902 with its documented dated alias. Older IDs stay distinct. DeepSeek's new `deepseek-v4-flash-vision-exp` adds an explicitly Experimental user-image path; the Stable DeepSeek text models continue to reject images. Files, Responses, and other DeepSeek endpoints remain outside this adapter's contract. See [the dated source review](./docs/MODEL_REFRESH_2026_09.md) for exact capabilities, limitations, prices, and unavailable evidence. Offline tests are not live certification.

## Dependency compatibility update

Development and CI use a reviewed uv lock with independent minimum/latest range tests. Realtime remains Experimental and its default websocket transport now requires `zhivex-ai-sdk[realtime]`; core/provider imports remain available without websockets. See [dependency compatibility](./docs/DEPENDENCY_COMPATIBILITY.md) for migration and update commands.

## Stable local storage and reviewed catalogs

Local agent run stores and memory/checkpoint factories have Stable contracts: InMemory for process-local tests/demos, SQLite for persistent files on a single host. Applications serialize session memory updates and reconcile external effects after failures. See [the storage guarantees](./docs/agents/durable-state.md#local-storage-guarantees).

Application-owned `ModelCatalog` construction, lookup and metadata types are Stable. Pin reviewed entries and effective pricing windows for production routing. The maintained `default_model_catalog` snapshot and provider capability discovery remain Beta; metadata does not establish live certification.
