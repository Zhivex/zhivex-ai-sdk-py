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
- vLLM

Tier-1 setup details, env vars, and smoke commands live in [providers/tier-1.md](./providers/tier-1.md).

DeepSeek is deferred for Python GA. Qwen, Kimi, Ollama, Bedrock, and OpenRouter remain available according to their support-matrix tier, but they are not part of the current tier-1 portable production promise.

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

The smoke runner intentionally skips missing providers so teams can validate only the providers configured in their environment.

## Capability Checks

The README support matrix is generated from runtime metadata. Do not edit it by hand.

```bash
.venv/bin/python scripts/generate_support_matrix.py --check-readme
```

Provider changes should update adapter metadata, tests, docs, and examples together.
