# Providers

Zhivex AI SDK exposes provider bundles with two paths:

- `provider("model-id")` or `provider.portable.language_model("model-id")` for the SDK-owned portable contract
- `provider.native.*` for provider-specific options, hosted tools, lifecycle clients, and escape hatches

Use portable paths for application contracts you want to keep stable across vendors. Use native paths when the feature is provider-specific by design.

Provider support and release certification are separate claims:

- **Contract-supported** providers have runtime metadata, deterministic shared contracts, provider-specific tests, documentation, and live-smoke configuration.
- **Release-certified** providers have recorded live evidence for an exact model, operation set, built artifact, and source revision.
- **Experimental/native-only** providers do not carry the portable compatibility promise.

Tier-1 identifies the contract-supported portable roster. It does not certify every Tier-1 provider for the current release.

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
- Meta Model API
- vLLM

Tier-1 setup details, env vars, and smoke commands live in [providers/tier-1.md](./providers/tier-1.md).

DeepSeek is tier-1 for current V4 text generation, streaming, JSON structured output, callable tools, and reasoning through its official Chat Completions API. Meta Model API is tier-1 for Standard `muse-spark-1.2` portable text, streaming, structured output, callable tools, and agent tool loops. Ollama, Bedrock, and OpenRouter remain available according to their support-matrix tier, but they are not part of the current tier-1 portable production promise.

## Meta Model API Boundary

`create_meta()` is Stable for the Standard `muse-spark-1.2` tier-1 scope above. Contributor models, hosted-tool helpers, Files, raw Responses/continuation, hosted tools, and multimodal/native extras remain Beta. Embeddings, speech output, transcription, grounding, Realtime, image generation, and video generation are not claimed. Setup, privacy, and evidence boundaries are documented in [providers/meta.md](./providers/meta.md).

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

For strict Meta validation:

```bash
MODEL_API_KEY=... ZHIVEX_SMOKE_META_MODEL=muse-spark-1.2 ZHIVEX_SMOKE_PROVIDERS=meta make smoke-agents
```

The smoke runner intentionally skips missing providers so teams can validate only the providers configured in their environment.

## Capability Checks

The README support matrix is generated from runtime metadata. Do not edit it by hand.

```bash
.venv/bin/python scripts/generate_support_matrix.py --check-readme
```

Provider changes should update adapter metadata, tests, docs, and examples together.
