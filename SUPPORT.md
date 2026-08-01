# Support Policy

Zhivex AI SDK is currently published as a `Beta` package.

Related documents:

- [README.md](./README.md)
- [STABILITY.md](./STABILITY.md)
- [VERSIONING.md](./VERSIONING.md)
- [CHANGELOG.md](./CHANGELOG.md)

## Support expectations

- The latest beta release line is the primary target for fixes, documentation updates, and examples.
- The previous beta minor release may receive critical fixes when the change is low risk and clearly scoped.
- Stable APIs are the main compatibility contract for production integrations.
- Stable agent integrations include the core runtime and typed result/stream handles, typed in-process dependencies, dynamic instructions, lifecycle hooks, run middleware, local tool definitions and execution contracts, handoffs, session helpers, portable agent skills, MCP helper path, MCP-backed registries, Postgres-backed memory/checkpoint stores, Postgres-backed run-state/replay helpers, and durable local-tool approval resume documented in [STABILITY.md](./STABILITY.md). Native subagent tools such as `create_subagent_tool(...)` remain beta.
- Agent dependencies are application-owned runtime objects. They are propagated in process but intentionally excluded from checkpoints, run state, traces, and serialized tool contexts; applications must supply them again when resuming a run.
- Agent production guidance is documented in [docs/AGENTS.md](./docs/AGENTS.md), [docs/PRODUCTION.md](./docs/PRODUCTION.md), [docs/OPERATIONS.md](./docs/OPERATIONS.md), and [PRODUCTION_APIS.md](./PRODUCTION_APIS.md).
- Onboarding, provider setup, gateway routing, observability, security, and troubleshooting guidance live under [docs/QUICKSTART.md](./docs/QUICKSTART.md), [docs/PROVIDERS.md](./docs/PROVIDERS.md), [docs/GATEWAY.md](./docs/GATEWAY.md), [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md), [SECURITY.md](./SECURITY.md), and [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md).
- Beta APIs are supported for early adoption, but they may still evolve between minor releases with changelog coverage.
- Beta provider capability metadata describes current provider agent ergonomics, but it is not a stable behavioral guarantee yet.
- Beta hosted-tool definitions and `provider-data` control parts describe the preferred native tool-registration path, but provider-specific execution semantics may still evolve between minor releases.
- Beta provider-managed approval flows currently cover OpenAI and Azure OpenAI only, including typed `provider-data` payload parsing and agent-runtime approval-policy integration.
- Beta response-reference helpers, `provider-data` UI chunks, and `tool-approval` UI chunks are supported for OpenAI/Azure continuation workflows, approval UIs, and observability, but their exact ergonomics may still evolve between minor releases.
- Beta agent platform helpers cover in-memory and SQLite run stores, native subagent tools, evaluation reports, hierarchical trace artifacts, run-tree snapshots, redaction policies, and budget guards. The Postgres run store, run-state serialization, cancellation, replay, run-snapshot helpers, and durable local-tool approval lifecycle are part of the stable agent surface.
- Beta workflow support covers declarative sequential, parallel, and loop agents plus validated DAGs, agent or functional graph steps, conditional routing, append-only checkpoints, explicit interrupts, `resume_workflow(...)`, `fork_workflow(...)`, step retry policy, and replay projections as documented in [docs/WORKFLOWS.md](./docs/WORKFLOWS.md). Functional executors receive ephemeral dependencies and must return finite JSON durable results. In-memory workflow checkpoints are process-local; SQLite persists on one filesystem; the optional Postgres workflow checkpoint store provides transactional shared-worker storage but remains beta and requires deployment-specific integration validation. CLI/UI/deploy automation is outside this surface.
- Beta workflow adapter factories for DBOS, Temporal, Prefect, and Restate provide dependency-free callback envelopes and conservative capability metadata only. They are not engine clients, managed schedulers, workers, or live-certified third-party integrations; applications own the actual engine binding and its end-to-end validation.
- Beta Google native clients cover Gemini/Vertex image, video, music/audio, batch, interaction, and Gemini explicit context-cache workflows where the official Google endpoints expose them. The catalog tracks current Gemini 3.5 Flash plus Gemini 3.5 Live Translate, Gemini 3.1 live, image, TTS, Veo, and Lyria reference IDs. Preview Google models remain subject to Google availability, quota, and deprecation windows.
- Beta Kimi/Moonshot native support covers Chat Completions, Files, Batch, token estimation, and official Formulas tools according to the current Kimi Open Platform docs. Portable Kimi is tier-1 for text, streaming, structured output, and callable tools.
- DeepSeek is tier-1 for text generation, streaming, JSON structured output, callable tools, and reasoning through its official Chat Completions API. Provider-specific strict-tool/prefix beta routing is supported, while vision, files, embeddings, audio, moderation, and hosted tools are outside the current contract.
- vLLM is supported as a tier-1 provider for SDK primitives exposed by its OpenAI-compatible server. Embeddings, transcription, and realtime ASR support depends on the model/task served by vLLM; vLLM custom endpoints such as tokenize, rerank, classify, and score are outside the SDK support contract.
- Tier-1 provider claims are backed by generated support metadata, shared offline contract tests, provider-specific tests, and optional live smoke documentation in [docs/providers/tier-1.md](./docs/providers/tier-1.md).
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
- vLLM

Anthropic is tier-1 for the portable text-generation surface in this repository. The catalog tracks Claude Opus 5, Fable 5, Sonnet 5, restricted-access Mythos 5, and Opus 4.8. Opus 5 uses adaptive thinking by default, supports portable effort through `max`, and can disable thinking only through effort `high`; manual budgets and non-default sampling fail before dispatch. Opus 5, Fable 5, Mythos 5, and Opus 4.8 accept valid intermediate system sections. Opus 5 does not support assistant prefill, server-side Web Fetch, or Priority Tier through this integration. Native hosted helpers otherwise default to `web_search_20260318`, `web_fetch_20260318`, and GA `code_execution_20260521`. Anthropic refusals preserve provider details and normalize to `finish_reason="refusal"`; embeddings, transcription, and speech remain unavailable.

Azure OpenAI is tier-1 for the portable production surface and exposes beta native lifecycle clients for Responses, Conversations, Realtime, and Vector Store / File Search management through the versionless `/openai/v1` route. The catalog tracks GPT-5.6 Sol/Terra/Luna, `gpt-chat-latest`, GPT Realtime 2.1, GPT Image 2, and text-embedding-3 reference IDs where deployments are available. The Azure provider supports either API key authentication or Microsoft Entra ID token/provider authentication; configure only one credential mode at a time. OpenAI-only native clients such as Containers, Skills, Uploads, Moderations, Images, Batches, and Sora/video lifecycle clients are not exposed on the Azure provider bundle.

OpenAI is tier-1 for the portable production surface. The catalog tracks GA GPT-5.6 Sol/Terra/Luna, using `gpt-5.6-sol` as the flagship and `gpt-5.6` as its alias, while retaining older entries for compatibility and GPT Realtime 2.1 for realtime guidance. Responses is the recommended reasoning/tool route. The beta native surface includes explicit prompt-cache/reasoning options and Programmatic Tool Calling with replay-safe preservation of `program`, nested `caller`, function outputs, and `program_output` items.

Gemini and Vertex are tier-1 for the portable production surface. Google-specific media generation, Batch API, Interactions API, Deep Research, explicit context caching, Live Translate, and Veo operation workflows are native clients rather than portable contracts. Gemini Developer API catalog guidance includes Interactions-only `gemini-omni-flash-preview` and `gemini-3.1-flash-lite-image`; Omni is not claimed for Vertex. Veo uses the current `veo-3.1-fast-generate-preview` ID, and deprecated Imagen 4 is no longer a catalog recommendation. Region and quota availability still apply.

Qwen is tier-1 for portable text generation, streaming, structured output, callable tools, and embeddings. `create_qwen(region="intl" | "us" | "cn")` maps to Alibaba Cloud Model Studio's documented OpenAI-compatible regions and the current `/compatible-mode/v1/responses` path, while `base_url` and `responses_base_url` remain explicit overrides. The Responses adapter preserves all seven `ReasoningConfig` efforts as `reasoning.effort`, maps mixed vision content to `input_text` / `input_image`, requires Web Search alongside Web Extractor, and rejects forced required/named tool choice when reasoning is enabled. The catalog distinguishes the multimodal `qwen3.7-max-2026-06-08` snapshot from text-only Qwen3.7 Max guidance. Hosted tools, Files, Batch, ASR, and TTS remain beta native surfaces; Batch model availability is regional, with Singapore currently limited to the documented stable aliases.

Kimi/Moonshot is tier-1 for portable text generation, streaming, structured output, and callable tools through the official Chat Completions API. `kimi-k3` is the current catalog reference: it is multimodal, always reasons, accepts `reasoning_effort` values `low`, `high`, or `max`, and rejects incompatible K2 thinking/sampling choices. K2.6/K2.5 keep their legacy `thinking` contract. Files, Batch, token estimation, and Formulas remain beta native clients; portable embeddings, speech, and transcription are not claimed.

DeepSeek is tier-1 for portable text generation, streaming, JSON structured output, callable tools, and reasoning through Chat Completions. The catalog tracks current `deepseek-v4-flash` and `deepseek-v4-pro`; retired `deepseek-chat` and `deepseek-reasoner` IDs fail before dispatch. The adapter preserves `reasoning_content` for tool replay, maps portable reasoning effort, retains provider-specific usage metadata, and routes strict tools or prefix completion to DeepSeek's beta base URL when requested. Vision, files, embeddings, audio, moderation, and hosted tools are not claimed.

vLLM is tier-1 for portable text, streaming, structured output/tools, embeddings, and transcription through the vLLM OpenAI-compatible server. Realtime ASR is exposed through `provider.native.realtime_model(...)` and remains subject to the experimental realtime API stability level.

Other providers remain available, but they should be treated according to the support matrix and the stability level of the specific feature area.

## Upgrade expectations

- Every user-visible change should appear in [CHANGELOG.md](./CHANGELOG.md).
- Changes to stable APIs require migration guidance.
- Deprecations should be documented before removal.
