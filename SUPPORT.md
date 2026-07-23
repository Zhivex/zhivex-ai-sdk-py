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
- Stable agent integrations include the core runtime and result/stream handles, local tool definitions and execution contracts, handoffs, session helpers, portable agent skills, MCP helper path, MCP-backed registries, Postgres-backed memory/checkpoint stores, Postgres-backed run-state/replay helpers, and durable local-tool approval resume documented in [STABILITY.md](./STABILITY.md). Native subagent tools such as `create_subagent_tool(...)` remain beta.
- Agent production guidance is documented in [docs/AGENTS.md](./docs/AGENTS.md), [docs/PRODUCTION.md](./docs/PRODUCTION.md), [docs/OPERATIONS.md](./docs/OPERATIONS.md), and [PRODUCTION_APIS.md](./PRODUCTION_APIS.md).
- Onboarding, provider setup, gateway routing, observability, security, and troubleshooting guidance live under [docs/QUICKSTART.md](./docs/QUICKSTART.md), [docs/PROVIDERS.md](./docs/PROVIDERS.md), [docs/GATEWAY.md](./docs/GATEWAY.md), [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md), [SECURITY.md](./SECURITY.md), and [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md).
- Beta APIs are supported for early adoption, but they may still evolve between minor releases with changelog coverage.
- Beta provider capability metadata describes current provider agent ergonomics, but it is not a stable behavioral guarantee yet.
- Beta hosted-tool definitions and `provider-data` control parts describe the preferred native tool-registration path, but provider-specific execution semantics may still evolve between minor releases.
- Beta provider-managed approval flows currently cover OpenAI and Azure OpenAI only, including typed `provider-data` payload parsing and agent-runtime approval-policy integration.
- Beta response-reference helpers, `provider-data` UI chunks, and `tool-approval` UI chunks are supported for OpenAI/Azure continuation workflows, approval UIs, and observability, but their exact ergonomics may still evolve between minor releases.
- Beta agent platform helpers cover in-memory and SQLite run stores, native subagent tools, evaluation reports, hierarchical trace artifacts, run-tree snapshots, redaction policies, and budget guards. The Postgres run store, run-state serialization, cancellation, replay, run-snapshot helpers, and durable local-tool approval lifecycle are part of the stable agent surface.
- Beta workflow agents cover declarative sequential, parallel, and loop orchestration with shared `session.state`, app-owned resume, structured output validation, document artifacts, replay, and evaluation as documented in [docs/WORKFLOWS.md](./docs/WORKFLOWS.md); CLI/UI/deploy automation is intentionally outside this beta surface.
- Beta Google native clients cover Gemini/Vertex image, video, music/audio, batch, interaction, and Gemini explicit context-cache workflows where the official Google endpoints expose them. The catalog tracks current Gemini 3.5 Flash plus Gemini 3.5 Live Translate, Gemini 3.1 live, image, TTS, Veo, and Lyria reference IDs. Preview Google models remain subject to Google availability, quota, and deprecation windows.
- Beta Kimi/Moonshot native support covers Chat Completions, Files, Batch, token estimation, and official Formulas tools according to the current Kimi Open Platform docs. Portable Kimi is tier-1 for text, streaming, structured output, and callable tools.
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
- vLLM

Anthropic is tier-1 for the portable text-generation surface in this repository. The catalog tracks Claude Fable 5, Sonnet 5, restricted-access Mythos 5, and Opus 4.8. Adaptive-thinking and mid-conversation system-message rules are enforced per family; Fable 5, Mythos 5, and Opus 4.8 accept valid intermediate system messages. Native hosted helpers default to `web_search_20260318`, `web_fetch_20260318`, and GA `code_execution_20260521`. Anthropic refusals normalize to `finish_reason="refusal"`; embeddings, transcription, and speech remain unavailable.

Azure OpenAI is tier-1 for the portable production surface and exposes beta native lifecycle clients for Responses, Conversations, Realtime, and Vector Store / File Search management through the versionless `/openai/v1` route. The catalog tracks GPT-5.6 Sol/Terra/Luna, `gpt-chat-latest`, GPT Realtime 2.1, GPT Image 2, and text-embedding-3 reference IDs where deployments are available. The Azure provider supports either API key authentication or Microsoft Entra ID token/provider authentication; configure only one credential mode at a time. OpenAI-only native clients such as Containers, Skills, Uploads, Moderations, Images, Batches, and Sora/video lifecycle clients are not exposed on the Azure provider bundle.

OpenAI is tier-1 for the portable production surface. The catalog tracks GA GPT-5.6 Sol/Terra/Luna, using `gpt-5.6-sol` as the flagship and `gpt-5.6` as its alias, while retaining older entries for compatibility and GPT Realtime 2.1 for realtime guidance. Responses is the recommended reasoning/tool route. The beta native surface includes explicit prompt-cache/reasoning options and Programmatic Tool Calling with replay-safe preservation of `program`, nested `caller`, function outputs, and `program_output` items.

Gemini and Vertex are tier-1 for the portable production surface. Google-specific media generation, Batch API, Interactions API, Deep Research, explicit context caching, Live Translate, and Veo operation workflows are native clients rather than portable contracts. Gemini Developer API catalog guidance includes Interactions-only `gemini-omni-flash-preview` and `gemini-3.1-flash-lite-image`; Omni is not claimed for Vertex. Veo uses the current `veo-3.1-fast-generate-preview` ID, and deprecated Imagen 4 is no longer a catalog recommendation. Region and quota availability still apply.

Qwen is tier-1 for portable text generation, streaming, structured output, callable tools, and embeddings. `create_qwen(region="intl" | "us" | "cn")` maps to Alibaba Cloud Model Studio's documented OpenAI-compatible regions, while `base_url` and `responses_base_url` remain explicit overrides. The Responses adapter preserves all seven `ReasoningConfig` efforts as `reasoning.effort` and no longer emits deprecated `enable_thinking`. The catalog distinguishes the multimodal `qwen3.7-max-2026-06-08` snapshot from text-only Qwen3.7 Max guidance. Hosted tools, Files, Batch, ASR, and TTS remain beta native surfaces.

Kimi/Moonshot is tier-1 for portable text generation, streaming, structured output, and callable tools through the official Chat Completions API. `kimi-k3` is the current catalog reference: it is multimodal, always reasons, accepts `reasoning_effort` values `low`, `high`, or `max`, and rejects incompatible K2 thinking/sampling choices. K2.6/K2.5 keep their legacy `thinking` contract. Files, Batch, token estimation, and Formulas remain beta native clients; portable embeddings, speech, and transcription are not claimed.

vLLM is tier-1 for portable text, streaming, structured output/tools, embeddings, and transcription through the vLLM OpenAI-compatible server. Realtime ASR is exposed through `provider.native.realtime_model(...)` and remains subject to the experimental realtime API stability level.

Other providers remain available, but they should be treated according to the support matrix and the stability level of the specific feature area.

DeepSeek is deferred and is not part of the tier-1 provider contract.

## Upgrade expectations

- Every user-visible change should appear in [CHANGELOG.md](./CHANGELOG.md).
- Changes to stable APIs require migration guidance.
- Deprecations should be documented before removal.
