# Meta Model API

`create_meta()` is a Beta portable provider for Meta Model API. It is not part of the Tier-1 stable provider set. A local-source authenticated smoke passed on 2026-08-14, but the dirty working tree was not a built wheel or immutable release SHA, so release certification remains pending.

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

## Supported surface

- Chat Completions and Responses text generation
- streaming text and buffered function-call arguments
- native JSON Schema structured output
- callable tools; Meta currently accepts only `tool_choice="auto"`
- image and PDF/file input; MP3/WAV input is implemented from Meta's protocol guidance but remains specifically pending authenticated certification because Meta's current capability table and surrounding prose are inconsistent about audio
- Responses continuation through native `provider_options={"previous_response_id": ...}`
- Meta hosted `web_search` and `tool_search` through `meta_web_search_tool()` and `meta_tool_search_tool()`
- native Files and Responses clients through `provider.files()` and `provider.responses()`

Embeddings, speech output, transcription, grounding, Realtime, image generation, and video generation are not claimed. The direct API catalog currently lists Muse Spark IDs; Muse Glimmer and Llama 4 remain open-weight/host routes rather than direct Meta Model API models.

## Model and privacy boundary

- `muse-spark-1.2` is the default recommendation for direct use.
- `muse-spark-1.2-contributor` has different data-use terms and must not be selected implicitly for sensitive workloads.
- `muse-spark-1.1` remains cataloged for projects where it is available.
- `meta-models/Muse-Glimmer-30B` is cataloged for vLLM; Ollama and OpenRouter routes are also cataloged. These host routes are not certification of the direct Meta API or of every host capability.
- Glimmer is modeled conservatively as one tool call per turn, without parallel tool calls and without an SDK guarantee for JSON Schema output.

Check current Meta account, region, retention, zero-data-retention, model availability, and pricing terms before production use. The Standard and Contributor tiers have materially different privacy terms.

## Offline and live evidence

Provider tests exercise requests, responses, SSE streaming, tools, structured output, continuation, hosted tools, validation, errors, retries, and URL safety with fake transports. They do not prove authenticated service behavior.

The 2026-08-14 authenticated local-source smoke passed deterministic generation and a complete `run_agent(...)` → local tool → tool result → final answer loop with Standard `muse-spark-1.2`. The same strict run also passed OpenAI `gpt-5.6-luna`, Gemini `gemini-3.6-flash`, and Qwen `qwen3.8-max`. Because the checkout had uncommitted changes and the smoke did not install a built artifact, treat this as integration evidence rather than release certification.

To run an opt-in live generation and agent-tool smoke with the Standard model:

```bash
MODEL_API_KEY=... \
ZHIVEX_SMOKE_META_MODEL=muse-spark-1.2 \
ZHIVEX_SMOKE_PROVIDERS=meta \
make smoke-agents
```

Any live claim must name the exact provider, model, operation, installed artifact, and Git SHA from the recorded run.

Official references:

- [Meta Model API documentation](https://dev.meta.ai/docs/)
- [Models](https://dev.meta.ai/docs/models)
- [Protocols](https://dev.meta.ai/docs/protocols)
- [Tool calling](https://dev.meta.ai/docs/tool-calling)
- [Structured output](https://dev.meta.ai/docs/structured-output)
- [Responses](https://dev.meta.ai/docs/protocols/responses)
- [Pricing, rate limits, and data-use tiers](https://dev.meta.ai/docs/pricing-rate-limits)
- [Muse Glimmer](https://dev.meta.ai/docs/muse-glimmer)
