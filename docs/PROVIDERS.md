# Providers

Zhivex AI SDK exposes provider bundles with two paths:

- `provider("model-id")` or `provider.portable.language_model("model-id")` for the SDK-owned portable contract
- `provider.native.*` for provider-specific options, hosted tools, lifecycle clients, and escape hatches

Use portable paths for application contracts you want to keep stable across vendors. Use native paths when the feature is provider-specific by design.

## Tier-1 Providers

The current tier-1 set for production API work is:

- OpenAI
- Anthropic
- Azure OpenAI
- Gemini
- Vertex
- Qwen
- Kimi/Moonshot
- DeepSeek
- vLLM

Tier-1 setup details, env vars, and smoke commands live in [providers/tier-1.md](./providers/tier-1.md).

DeepSeek is tier-1 for current V4 text generation, streaming, JSON structured output, callable tools, and reasoning through its official Chat Completions API. Ollama, Bedrock, and OpenRouter remain available according to their support-matrix tier, but they are not part of the current tier-1 portable production promise.

## Beta Portable Providers

Meta Model API is available through `create_meta()` as a Beta portable provider and is not Tier-1. The direct adapter targets Muse Spark through Chat Completions and Responses; Muse Glimmer and Llama remain host/open-weight routes. Setup, capability, privacy, and evidence boundaries are documented in [providers/meta.md](./providers/meta.md).

Direct Anthropic Messages support includes the fixed `claude-opus-5` ID. This does not imply Opus 5 support through the current Bedrock Converse adapter; model-specific thinking, tool, and service-tier limits are documented in the tier-1 guide.

## Local Setup

For a single provider, fill only that provider's variables in `.env` and scope smoke:

```bash
ZHIVEX_SMOKE_PROVIDERS=openai make smoke
```

For local models:

```bash
ZHIVEX_SMOKE_PROVIDERS=vllm make smoke
ZHIVEX_SMOKE_PROVIDERS=ollama make smoke
```

For opt-in Meta validation:

```bash
MODEL_API_KEY=... ZHIVEX_SMOKE_META_MODEL=muse-spark-1.2 ZHIVEX_SMOKE_PROVIDERS=meta make smoke
```

The smoke runner intentionally skips missing providers so teams can validate only the providers configured in their environment.

## Capability Checks

The README support matrix is generated from runtime metadata. Do not edit it by hand.

```bash
.venv/bin/python scripts/generate_support_matrix.py --check-readme
```

Provider changes should update adapter metadata, tests, docs, and examples together.
