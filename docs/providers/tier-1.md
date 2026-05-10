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

Optional Google media smoke checks are gated behind `ZHIVEX_SMOKE_GOOGLE_MEDIA=1` plus the relevant Gemini or Vertex image/video/media model IDs.

## Focused Example

`examples/text/tier1_providers.py` runs the same portable text-generation shape for whichever tier-1 providers are configured through `ZHIVEX_EXAMPLE_OPENAI_MODEL`, `ZHIVEX_EXAMPLE_ANTHROPIC_MODEL`, `ZHIVEX_EXAMPLE_AZURE_OPENAI_MODEL`, `ZHIVEX_EXAMPLE_GEMINI_MODEL`, `ZHIVEX_EXAMPLE_VERTEX_MODEL`, `ZHIVEX_EXAMPLE_QWEN_MODEL`, `ZHIVEX_EXAMPLE_KIMI_MODEL`, or `ZHIVEX_EXAMPLE_VLLM_MODEL`.

## Capability Notes

- OpenAI, Azure OpenAI, Gemini, Vertex, Qwen, Kimi, and vLLM expose portable text, streaming, structured output, and tool paths.
- Azure OpenAI additionally exposes beta native lifecycle clients for Responses, Conversations, and Vector Store / File Search management through `provider.native` / bundle helper methods.
- Anthropic is tier-1 for portable text-generation paths; embeddings, transcription, and speech are outside the current Anthropic SDK surface.
- Gemini and Vertex expose Google media, Batch, Interactions, Deep Research, and Veo-style workflows through native clients rather than the stable portable contract.
- Qwen exposes hosted web/file/code/MCP tools, raw Responses, Files, Batch, ASR, and TTS as native/provider-specific beta surfaces. File Search remains a hosted Responses tool with `vector_store_ids`, not a lifecycle client.
- Kimi exposes Files, Batch, token counting, and Formulas as native/provider-specific beta surfaces. Portable Kimi does not claim embeddings, speech, or transcription.
- vLLM support depends on the tasks served by the local OpenAI-compatible server. Custom endpoints such as tokenize, rerank, classify, and score are outside the stable SDK contract.
- DeepSeek is deferred for Python GA and is not part of `TIER_1_PROVIDERS`.
