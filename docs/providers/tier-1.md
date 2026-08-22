# Tier-1 Provider Contract

Tier-1 providers are the supported provider set for the portable SDK contract:

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

Tier-1 means the provider is **contract-supported**: it has runtime support metadata, generated README support-matrix coverage, shared offline contract tests, provider-specific tests, and documented live smoke configuration. It does not mean every provider has been live-smoked for every release. The generated evidence table therefore remains fail-closed at `contract-supported` unless a separate validator supplies matching evidence for the exact artifact. Any `release-certified` claim must identify the provider, model, operation set, built artifact, and release SHA that produced the evidence.

Meta Model API has a separate [provider guide](./meta.md) because its Stable tier-1 scope is deliberately narrower than its Beta native surface.

## Offline Contract

Run the shared tier-1 contract tests with:

```bash
make test-provider-contracts
```

The shared contract verifies that each tier-1 provider:

- appears in `TIER_1_PROVIDERS`
- reports `portable` tier metadata and the portable badge
- builds portable and native language models without mixing their boundaries
- supports portable text generation, streaming, structured output, and its documented tool-choice contract through fake transports; Meta accepts `auto` only, while the other tier-1 adapters support the named tool-choice contract asserted for them

Provider-specific tests remain responsible for deeper native behavior, provider-managed tools, hosted-tool variants, media, realtime, and edge-case request mapping.

## Live Smoke

Run configured provider smoke checks with:

```bash
make smoke
```

Scope checks with `ZHIVEX_SMOKE_PROVIDERS=openai,anthropic,azure-openai,gemini,vertex,qwen,kimi,deepseek,meta,vllm`. The smoke runner skips providers whose required credentials or model IDs are not configured.

For an agent release candidate, use the strict agent-first variant:

```bash
ZHIVEX_SMOKE_PROVIDERS=openai,anthropic make smoke-agents
```

For every provider explicitly selected, this runs `run_agent(...)`, requires one local nonce-validation tool call, validates the tool result, and verifies the model continued to a final answer. `make smoke-agents` sets `ZHIVEX_SMOKE_AGENTS=1` and `ZHIVEX_SMOKE_STRICT=1`; with `ZHIVEX_SMOKE_PROVIDERS`, strict mode fails if any selected provider or its agent loop is skipped. Without an explicit selector, strict mode keeps the local-development rule that at least one configured provider and agent loop must execute. Failure output redacts configured API-key, access-token, password, and secret values.

| Provider | Required environment |
| --- | --- |
| OpenAI | `OPENAI_API_KEY`, `ZHIVEX_SMOKE_OPENAI_MODEL` |
| Anthropic | `ANTHROPIC_API_KEY`, `ZHIVEX_SMOKE_ANTHROPIC_MODEL` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `ZHIVEX_SMOKE_AZURE_OPENAI_MODEL` |
| Gemini | `GEMINI_API_KEY` or `GOOGLE_GENERATIVE_AI_API_KEY` or `GOOGLE_API_KEY`, `ZHIVEX_SMOKE_GEMINI_MODEL` |
| Vertex | `VERTEX_ACCESS_TOKEN` or `GOOGLE_ACCESS_TOKEN`, `GOOGLE_CLOUD_PROJECT` or `GCLOUD_PROJECT`, `ZHIVEX_SMOKE_VERTEX_MODEL` |
| Qwen | `DASHSCOPE_API_KEY` or `QWEN_API_KEY`, `ZHIVEX_SMOKE_QWEN_MODEL`; optional `ZHIVEX_SMOKE_QWEN_REGION`, `ZHIVEX_SMOKE_QWEN_BASE_URL`, `ZHIVEX_SMOKE_QWEN_RESPONSES_BASE_URL` |
| Kimi/Moonshot | `MOONSHOT_API_KEY` or `KIMI_API_KEY`, `ZHIVEX_SMOKE_KIMI_MODEL`; optional `MOONSHOT_BASE_URL` or `ZHIVEX_SMOKE_KIMI_BASE_URL` |
| DeepSeek | `DEEPSEEK_API_KEY`, `ZHIVEX_SMOKE_DEEPSEEK_MODEL`; optional `DEEPSEEK_BASE_URL` or `ZHIVEX_SMOKE_DEEPSEEK_BASE_URL` |
| Meta Model API | `MODEL_API_KEY`, `ZHIVEX_SMOKE_META_MODEL=muse-spark-1.2`; optional `META_BASE_URL` or `ZHIVEX_SMOKE_META_BASE_URL` |
| vLLM | `ZHIVEX_SMOKE_VLLM_MODEL`; optional `ZHIVEX_SMOKE_VLLM_BASE_URL`, `ZHIVEX_SMOKE_VLLM_API_KEY` or `VLLM_API_KEY` |

Optional Google media smoke checks are gated behind `ZHIVEX_SMOKE_GOOGLE_MEDIA=1` plus the relevant Gemini or Vertex image/video/media model IDs, such as `gemini-3.1-flash-image`, `gemini-3.1-flash-lite-image`, `gemini-3.5-live-translate-preview`, `veo-3.1-generate-preview`, `veo-3.1-fast-generate-preview`, `lyria-3-pro-preview`, or `lyria-3-clip-preview`.

## Focused Example

`examples/text/tier1_providers.py` runs the same portable text-generation shape for whichever tier-1 providers are configured through `ZHIVEX_EXAMPLE_OPENAI_MODEL`, `ZHIVEX_EXAMPLE_ANTHROPIC_MODEL`, `ZHIVEX_EXAMPLE_AZURE_OPENAI_MODEL`, `ZHIVEX_EXAMPLE_GEMINI_MODEL`, `ZHIVEX_EXAMPLE_VERTEX_MODEL`, `ZHIVEX_EXAMPLE_QWEN_MODEL`, `ZHIVEX_EXAMPLE_KIMI_MODEL`, `ZHIVEX_EXAMPLE_DEEPSEEK_MODEL`, `ZHIVEX_EXAMPLE_META_MODEL`, or `ZHIVEX_EXAMPLE_VLLM_MODEL`.

## Capability Notes

- OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex, Qwen, Kimi, DeepSeek, Meta Model API, and vLLM expose their documented portable text, streaming, structured output, and tool paths.
- OpenAI catalog guidance tracks GA GPT-5.6 Sol/Terra/Luna and GPT Realtime 2.1. Responses is the recommended reasoning/tool route; the beta native path preserves Programmatic Tool Calling `program`, `caller`, and `program_output` items.
- Azure OpenAI additionally exposes beta native lifecycle clients for Responses, Conversations, Realtime, and Vector Store / File Search management through `provider.native` / bundle helper methods. The catalog tracks GPT-5.6 Sol/Terra/Luna, `gpt-chat-latest`, GPT Image 2, and GPT Realtime 2.1 subject to deployment region/quota.
- Anthropic is tier-1 for portable text-generation paths. Direct `claude-opus-5` uses adaptive thinking by default, supports effort through `max`, and can disable thinking only through `high`; manual budgets and non-default sampling are rejected. Opus 5, Fable 5, restricted-access Mythos 5, and Opus 4.8 accept valid mid-conversation system sections. Opus 5 does not support assistant prefill, server-side Web Fetch, or Priority Tier here, and the current Bedrock Converse adapter does not claim Opus 5. Other hosted helpers default to the 2026 web search, web fetch, and code execution types.
- Gemini and Vertex use `gemini-3.6-flash` and `gemini-3.5-flash-lite` as the current stable regular-generation catalog references. Both reject custom sampling and assistant prefill before dispatch. Google media, Batch, Interactions, Deep Research, Live Translate, and Veo-style workflows remain native clients rather than the stable portable contract. Gemini Developer API additionally tracks Interactions-only `gemini-omni-flash-preview` and `gemini-3.1-flash-lite-image`; Omni is not claimed for Vertex. Imagen 4 is no longer catalog guidance.
- Qwen catalog guidance starts with pay-as-you-go GA `qwen3.8-max`; the Token Plan's exact `qwen3.8-max-preview` ID remains separate. Responses uses the current `/compatible-mode/v1/responses` route for text, streaming, `input_text` / `input_image` vision input, all seven reasoning efforts, function tools, and hosted `web_search`, `web_extractor`, `code_interpreter`, `web_search_image`, and `image_search`. The adapter selects Chat Completions for native JSON Schema output, image/video `FilePart` input, or `ReasoningConfig.budget_tokens`; structured output disables thinking and Qwen reasoning state is preserved for multi-turn replay. Web Extractor requires Web Search, and explicit thinking cannot force required/named tool choice. Raw Responses, Files, Batch, ASR, and TTS remain native/provider-specific beta surfaces. Batch support varies by region; Singapore currently documents only the stable `qwen-max`, `qwen-plus`, `qwen-flash`, and `qwen-turbo` aliases.
- Kimi K3 is the current catalog reference with always-on reasoning, `reasoning_effort=low|high|max`, vision, tools, and strict structured output. K2.6/K2.5 retain their separate `thinking` contract. Portable Kimi does not claim embeddings, speech, or transcription.
- DeepSeek uses the official Chat Completions API with current `deepseek-v4-flash` and `deepseek-v4-pro` models. It supports text, streaming, JSON structured output, callable tools, and thinking; the adapter preserves `reasoning_content` during tool replay and automatically selects `/beta` for strict tools or prefix completion. Retired `deepseek-chat` / `deepseek-reasoner` IDs are rejected. Vision, files, embeddings, audio, moderation, and hosted tools are not claimed.
- Meta Model API is tier-1 only for `create_meta()` with Standard `muse-spark-1.2` portable text, streaming, structured output, callable tools, and the agent tool loop. Its tool choice remains `auto` only. Contributor models, `meta_hosted_tool()`, `meta_web_search_tool()`, `meta_tool_search_tool()`, Files, raw Responses/continuation, hosted tools, and multimodal/native extras remain Beta. Embeddings, speech output, transcription, grounding, Realtime, image generation, and video generation are not claimed. Contract support does not establish release certification.
- vLLM support depends on the tasks served by the local OpenAI-compatible server. Custom endpoints such as tokenize, rerank, classify, and score are outside the stable SDK contract.
