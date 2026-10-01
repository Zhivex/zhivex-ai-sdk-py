# Model refresh — September 30, 2026

The Python SDK now includes the eight direct-provider model IDs identified in the
latest-release review, with request validation and native functionality where
the upstream API changed. Package maturity remains Beta. Azure and Bedrock
adapters, models and classifications remain unchanged.

| Provider | Added catalog IDs | Implementation |
| --- | --- | --- |
| OpenAI | `gpt-6-sol`, `gpt-6-luna`, `gpt-6.1-sol` | Responses generation/streaming, effort and sampling guards, native hosted multi-agent |
| Anthropic | `claude-opus-5-5`, `claude-sonnet-5-5` | Adaptive/between-tools reasoning, forced-tool guards, toolset/advisor metadata |
| Gemini Developer API | `gemini-3.8-flash-tts`, `gemini-3.8-flash-lite-tts` | Speech voice schema and native Voices lifecycle |
| Qwen | `qwen3.7-text-rerank` | Native text ranking, Beijing/workspace routing and response validation |

Entries contain primary-source links, `verified_at=2026-09-30` and
`support_evidence=offline-contract`. This date records metadata review;
mocked transport tests establish SDK request/response contracts. Neither
establishes account entitlement, live availability or protected release
certification. No credentialed calls were performed for this refresh.

## OpenAI

GPT-6 Sol/Luna accept `none`, `low`, `medium`, `high`, `xhigh`, `max` effort;
GPT-6.1 Sol/Astra reject `none` and `minimal`. Sampling options require a model
supporting effort `none`. GPT-6 `prompt_cache_retention` is rejected with a
migration hint to native `prompt_cache_options.ttl`.

Hosted multi-agent is native and available on GPT-6.1 Sol and GPT-5.6:

```python
from zhivex_ai import create_openai, generate_text, user

model = create_openai().native.language_model("gpt-6.1-sol")
options = {"multi_agent": {"enabled": True, "max_concurrent_subagents": 3}, "store": False}
result = await generate_text(model=model, prompt="Review two independent approaches.", provider_options=options)
followup = await generate_text(
    model=model, messages=[*result.messages, user("Explain the recommendation.")],
    provider_options=options,
)
```

The SDK sends the required beta header. Hosted collaboration and encrypted items
remain opaque provider-data parts and are replayed unchanged. Only `/root`
messages with `phase=final_answer` contribute normalized text. Commentary and
worker text stay in raw data. Missing attribution never becomes final text in
enabled multi-agent mode. Streams lacking attribution entirely fail explicitly.
Client function calls from any agent still use the SDK tool loop; provider-hosted
collaboration calls do not become local tools. Keep opaque history private.

`StreamTextDeltaEvent.provider_metadata` preserves agent/phase information through
collection. This field defaults to an independent empty dictionary and preserves
old constructor positions. Multi-agent rejects `reasoning.summary`,
`max_tool_calls` and Responses compaction. This refresh covers HTTP Responses;
it does not implement hosted multi-agent WebSocket steering or the Agents API.

Runnable example: [openai_multi_agent.py](../examples/text/openai_multi_agent.py).
Sources: [latest model guide](https://developers.openai.com/api/docs/guides/latest-model),
[multi-agent guide](https://developers.openai.com/api/docs/guides/responses-multi-agent),
[Sol](https://developers.openai.com/api/docs/models/gpt-6-sol),
[Luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
[6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol).

## Anthropic, Gemini and Qwen

- [Claude 5.5 guide](providers/anthropic-5-5.md): Opus always adaptive; Sonnet
  maps effort `none` to `between_tools`. Native metadata retains toolset names,
  advisor encrypted outputs and input transformations. Validation runs after
  merging native options. Provider signature checks remain upstream-owned.
- [Gemini Voices guide](providers/gemini-voices.md): focused Beta types in
  `zhivex_ai.experimental.gemini`, native design/replication/list/get/delete,
  explicit consent audio, stateless voice keys and pagination. Gemini 3.8 TTS
  uses `voiceConfig.voice`; older Gemini/Vertex voice behavior is unchanged.
- [Qwen rerank guide](providers/qwen-rerank.md): native 3.7 nested and earlier
  3 flat routes, 1–500 text documents, valid indices and finite scores, explicit
  regional restrictions. Rerank is not an embeddings capability or a portable
  retrieval implementation.

Claude 5.5 on Vertex is covered through the existing native
`provider.native.model_garden().anthropic_messages()` route, with endpoint/model
contract tests. See the [Vertex guide](providers/vertex.md). These models are not
added to the portable language catalog because that surface routes Gemini
GenerateContent. Gemini Developer API Voices/3.8 TTS are not claimed on Vertex.

## Pricing and freshness

OpenAI base input/output/cache-read USD per million tokens:
Sol 2/10/0.20, Luna 0.10/0.50/0.01, 6.1 Sol 2/10/0.10.
Above 272,000 input tokens the full request uses doubled input and 1.5x output
rates. New optional `ModelPricing.long_context_*` fields preserve the threshold
and rates; conservative cost routing uses the highest input/output tier.
This estimate does not calculate invoice totals, cache-write or hosted tool fees.
Anthropic Opus/Sonnet base input/output are 4/20 and 2/10 USD per million;
cache-read is 0.20 for each. Sources are the individual model references.

Gemini audio pricing and Qwen regional CNY pricing remain in source references;
they are not flattened into USD text-token estimates. Gemini Lite TTS has a
time-limited promotional rate, so callers must review the effective date in the
[Google pricing guide](https://ai.google.dev/gemini-api/docs/pricing).

[Catalog freshness checks](providers/catalog-refresh.md) add an opt-in metadata
age check and repeatable provider filters. They detect missing/future/old dates,
not new releases or API changes. Existing dates are not bulk refreshed to pass.

```bash
.venv/bin/python scripts/check_catalog_freshness.py --as-of 2026-09-30 \
  --metadata-max-age-days 30 \
  --provider openai --provider anthropic --provider gemini --provider qwen
```

## Verification

Focused regressions cover latest-model request guards, final-answer attribution,
encrypted replay, streaming client-function execution, Anthropic native merging,
voice lifecycle/malformed responses, regional reranking and metadata freshness.
The existing Azure/Bedrock contract suite is also run. The complete `make check`
and an isolated installed-wheel smoke establish local and packaged behavior;
they do not replace credentialed provider certification or a published release.

Verified locally on September 30:

- `UV_CACHE_DIR=/tmp/zhivex-uv-cache make check`: 1,285 tests and 472 subtests
  passed, 16 skipped; 85.70% coverage. Lock, compilation, lint, public stub,
  type checking, certification schema and generated support matrix passed.
- Built the current 0.27.0 wheel without changing the package version and
  installed it into an isolated temporary target. Latest-provider and
  Azure/Bedrock regressions against that installation: 61 tests and 112
  subtests passed, with SDK imports pinned to the installed artifact.
- Compared the 25 Azure/Bedrock model records with the initial snapshot:
  model metadata/rates unchanged (new optional pricing fields remain None).
  Both cloud adapter files have no diff.
