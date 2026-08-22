# Meta Model API

Meta Model API is upstream GA. In Zhivex AI SDK `0.19.0`, `create_meta()` joins the Tier-1 contract-supported roster with a narrow Stable boundary: Standard `muse-spark-1.2` portable text generation, streaming, structured output, callable tools, the resulting agent tool loop, and application-supplied retrieval through `PortableRetrievalConfig`. Tier-1 does not imply release certification; exact-artifact evidence remains a separate gate.

## Setup

Create a Meta Model API key and expose it as `MODEL_API_KEY`. The adapter defaults to `https://api.meta.ai/v1` and the recommended direct model is the Standard `muse-spark-1.2` variant.

```python
import asyncio

from zhivex_ai import create_meta, generate_text


async def main() -> None:
    meta = create_meta()
    result = await generate_text(
        model=meta("muse-spark-1.2"),
        prompt="Summarize this request in one sentence.",
    )
    print(result.text)


asyncio.run(main())
```

Use `base_url=` only for an explicitly trusted proxy. Portable calls reject `provider_options`; provider-specific routing and continuation settings belong on `provider.native.language_model(...)`.

## Stable Tier-1 surface

- `create_meta()` from the top-level `zhivex_ai` package
- Standard `muse-spark-1.2`
- portable text generation and streaming
- portable JSON Schema structured output
- portable callable tools and complete `run_agent(...)` tool loops
- portable retrieval over application-supplied `PortableDocument` values through `PortableRetrievalConfig`
- `tool_choice="auto"`; forced, disabled, and named tool choice are not part of the Meta contract

Portable retrieval is an SDK-owned prompt-context operation: it injects bounded document text before the user request and uses the normal Chat Completions route. It does not upload files or invoke Meta Files, hosted `web_search`, `tool_search`, or raw Responses. Those provider-native retrieval surfaces remain Beta.

## Beta native and model extensions

The following implemented surfaces remain Beta and do not inherit Tier-1 stability:

- Contributor models, including `muse-spark-1.2-contributor`
- `meta_hosted_tool()`, `meta_web_search_tool()`, and `meta_tool_search_tool()`
- raw Responses and `provider_options={"previous_response_id": ...}` continuation
- native Files and Responses clients through `provider.files()` and `provider.responses()`
- hosted tools and multimodal/native input extras
- other Muse Spark versions or model routes

Embeddings, speech output, transcription, grounding, Realtime, image generation, and video generation are not claimed. Muse Glimmer and Llama 4 remain open-weight/host routes rather than direct Meta Model API models and are outside this promotion.

## Model and privacy boundary

- Standard `muse-spark-1.2` is the only model in the Stable Tier-1 boundary.
- `muse-spark-1.2-contributor` is Beta, has different data-use terms, and must not be selected implicitly for sensitive workloads.
- `muse-spark-1.1` remains cataloged for projects where it is available.
- `meta-models/Muse-Glimmer-30B` is cataloged for vLLM; Ollama and OpenRouter routes are also cataloged. These host routes are not certification of the direct Meta API or of every host capability.
- Glimmer is modeled conservatively as one tool call per turn, without parallel tool calls and without an SDK guarantee for JSON Schema output.

Check current Meta account, region, retention, zero-data-retention, model availability, and pricing terms before production use. The Standard and Contributor tiers have materially different privacy terms.

## Contract and live evidence

Shared provider contracts and Meta-specific tests exercise requests, responses, SSE streaming, tools, structured output, continuation, hosted tools, validation, errors, retries, and URL safety with fake transports. This makes the Stable Standard surface contract-supported; it does not prove authenticated service behavior for a release artifact.

Authenticated source and locally built wheel smokes have exercised deterministic generation and a complete `run_agent(...)` → local tool → tool result → final answer loop with Standard `muse-spark-1.2`. A wheel built from a modified worktree is integration evidence, not formal release certification. The `0.19.0` release candidate remains uncertified until the exact clean artifact and source revision pass the recorded release smoke.

To run an opt-in live generation and agent-tool smoke with the Standard model:

```bash
MODEL_API_KEY=... \
ZHIVEX_SMOKE_META_MODEL=muse-spark-1.2 \
ZHIVEX_SMOKE_PROVIDERS=meta \
make smoke-agents
```

Any release-certified claim must name the exact provider, model, operation set, installed artifact, and source revision from the recorded run.

Official references:

- [Meta Model API documentation](https://dev.meta.ai/docs/)
- [Models](https://dev.meta.ai/docs/models)
- [Protocols](https://dev.meta.ai/docs/protocols)
- [Tool calling](https://dev.meta.ai/docs/tool-calling)
- [Structured output](https://dev.meta.ai/docs/structured-output)
- [Responses](https://dev.meta.ai/docs/protocols/responses)
- [Pricing, rate limits, and data-use tiers](https://dev.meta.ai/docs/pricing-rate-limits)
- [Muse Glimmer](https://dev.meta.ai/docs/muse-glimmer)
