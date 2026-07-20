# Tier-1 Provider Contract

Tier-1 providers are the production provider set for the portable SDK surface:

- OpenAI
- Anthropic
- Azure OpenAI
- Gemini
- Vertex
- Qwen
- Kimi/Moonshot
- vLLM

Tier-1 means the provider has runtime support metadata, generated README support-matrix coverage, shared offline contract tests, provider-specific tests, and documented live smoke configuration. Live smoke is optional and credentials-driven; it must not block local offline development.

## Offline Contract

Run the shared tier-1 contract tests with:

```bash
make test-provider-contracts
```

The shared contract verifies that each tier-1 provider:

- appears in `TIER_1_PROVIDERS`
- reports `portable` tier metadata and the portable badge
- builds portable and native language models without mixing their boundaries
- supports portable text generation, streaming, structured output, and named tool choice through fake transports

Provider-specific tests remain responsible for deeper native behavior, provider-managed tools, hosted-tool variants, media, realtime, and edge-case request mapping.

## Live Smoke

Run configured provider smoke checks with:

```bash
make smoke
```

Scope checks with `ZHIVEX_SMOKE_PROVIDERS=openai,anthropic,azure-openai,gemini,vertex,qwen,kimi,vllm`. The smoke runner skips providers whose required credentials or model IDs are not configured.

| Provider | Required environment |
| --- | --- |
| OpenAI | `OPENAI_API_KEY`, `ZHIVEX_SMOKE_OPENAI_MODEL` |
| Anthropic | `ANTHROPIC_API_KEY`, `ZHIVEX_SMOKE_ANTHROPIC_MODEL` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `ZHIVEX_SMOKE_AZURE_OPENAI_MODEL` |
| Gemini | `GEMINI_API_KEY` or `GOOGLE_GENERATIVE_AI_API_KEY` or `GOOGLE_API_KEY`, `ZHIVEX_SMOKE_GEMINI_MODEL` |
| Vertex | `VERTEX_ACCESS_TOKEN` or `GOOGLE_ACCESS_TOKEN`, `GOOGLE_CLOUD_PROJECT` or `GCLOUD_PROJECT`, `ZHIVEX_SMOKE_VERTEX_MODEL` |
| Qwen | `DASHSCOPE_API_KEY` or `QWEN_API_KEY`, `ZHIVEX_SMOKE_QWEN_MODEL`; optional `ZHIVEX_SMOKE_QWEN_REGION`, `ZHIVEX_SMOKE_QWEN_BASE_URL`, `ZHIVEX_SMOKE_QWEN_RESPONSES_BASE_URL` |
| Kimi/Moonshot | `MOONSHOT_API_KEY` or `KIMI_API_KEY`, `ZHIVEX_SMOKE_KIMI_MODEL`; optional `MOONSHOT_BASE_URL` or `ZHIVEX_SMOKE_KIMI_BASE_URL` |
| vLLM | `ZHIVEX_SMOKE_VLLM_MODEL`; optional `ZHIVEX_SMOKE_VLLM_BASE_URL`, `ZHIVEX_SMOKE_VLLM_API_KEY` or `VLLM_API_KEY` |

Optional Google media smoke checks are gated behind `ZHIVEX_SMOKE_GOOGLE_MEDIA=1` plus the relevant Gemini or Vertex image/video/media model IDs, such as `gemini-3.1-flash-image`, `gemini-3.1-flash-lite-image`, `gemini-3.5-live-translate-preview`, `veo-3.1-generate-preview`, `veo-3.1-fast-generate-preview`, `lyria-3-pro-preview`, or `lyria-3-clip-preview`.

## Focused Example

`examples/text/tier1_providers.py` runs the same portable text-generation shape for whichever tier-1 providers are configured through `ZHIVEX_EXAMPLE_OPENAI_MODEL`, `ZHIVEX_EXAMPLE_ANTHROPIC_MODEL`, `ZHIVEX_EXAMPLE_AZURE_OPENAI_MODEL`, `ZHIVEX_EXAMPLE_GEMINI_MODEL`, `ZHIVEX_EXAMPLE_VERTEX_MODEL`, `ZHIVEX_EXAMPLE_QWEN_MODEL`, `ZHIVEX_EXAMPLE_KIMI_MODEL`, or `ZHIVEX_EXAMPLE_VLLM_MODEL`.

## Capability Notes

- OpenAI, Azure OpenAI, Gemini, Vertex, Qwen, Kimi, and vLLM expose portable text, streaming, structured output, and tool paths.
- OpenAI catalog guidance tracks GA GPT-5.6 Sol/Terra/Luna and GPT Realtime 2.1. Responses is the recommended reasoning/tool route; the beta native path preserves Programmatic Tool Calling `program`, `caller`, and `program_output` items.
- Azure OpenAI additionally exposes beta native lifecycle clients for Responses, Conversations, Realtime, and Vector Store / File Search management through `provider.native` / bundle helper methods. The catalog tracks GPT-5.6 Sol/Terra/Luna, `gpt-chat-latest`, GPT Image 2, and GPT Realtime 2.1 subject to deployment region/quota.
- Anthropic is tier-1 for portable text-generation paths. Fable 5, Sonnet 5, restricted-access Mythos 5, and Opus 4.8 use their current adaptive-thinking rules; Fable 5, Mythos 5, and Opus 4.8 accept valid mid-conversation system messages. Hosted helpers default to the 2026 web search, web fetch, and code execution types.
- Gemini and Vertex expose Google media, Batch, Interactions, Deep Research, Live Translate, and Veo-style workflows through native clients rather than the stable portable contract. Gemini Developer API additionally tracks Interactions-only `gemini-omni-flash-preview` and `gemini-3.1-flash-lite-image`; Omni is not claimed for Vertex. Imagen 4 is no longer catalog guidance.
- Qwen exposes hosted web/file/code/MCP tools, raw Responses, Files, Batch, ASR, and TTS as native/provider-specific beta surfaces. All seven supported reasoning efforts map to `reasoning.effort`; the catalog distinguishes multimodal `qwen3.7-max-2026-06-08` from the text-only snapshot.
- Kimi K3 is the current catalog reference with always-on reasoning, `reasoning_effort=low|high|max`, vision, tools, and strict structured output. K2.6/K2.5 retain their separate `thinking` contract. Portable Kimi does not claim embeddings, speech, or transcription.
- vLLM support depends on the tasks served by the local OpenAI-compatible server. Custom endpoints such as tokenize, rerank, classify, and score are outside the stable SDK contract.
- DeepSeek is deferred for Python GA and is not part of `TIER_1_PROVIDERS`.
