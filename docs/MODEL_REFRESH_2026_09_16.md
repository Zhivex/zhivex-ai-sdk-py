# Provider refresh — September 16, 2026

This update has deterministic, injected-transport contract tests and a [dated installed-wheel live report](releases/0.24.0-september-live-report.md). The live report records the exact passing operations and remaining blockers; it is not release certification or a publication. Upstream model availability, SDK stability, regional access, and exact-wheel evidence remain separate.

| Provider | Model or native API | Implemented boundary | SDK maturity |
| --- | --- | --- | --- |
| DeepSeek | `deepseek-flash` (V4.1) | Chat, streaming, JSON/tools, user-image URL/base64, current effort mapping; preserve legacy Flash IDs on the wire | Existing text core; vision Beta |
| Meta | `muse-spark-1.3` | Text, streaming, JSON Schema, max reasoning, tool replay through the existing direct adapter | Beta model extension; Standard 1.2 cohort unchanged |
| OpenAI | `gpt-image-2.5-sunburst`, `gpt-image-2.5-flare` | Native image generation/editing, including `xhigh`/`max` quality | Beta |
| OpenAI | `gpt-live-1` | Separate Live WebSocket startup/raw events/finalization; WebRTC creation/fork; hangup and recording download | Experimental |
| OpenAI | Agents API | Managed session create/list/retrieve/delete, saved items, submit/cancel events, SSE subscription | Beta |
| Anthropic | On-demand compaction | Raw Messages/compact client; preserve signed compaction blocks in normalized nonstreaming/streaming replay; beta header | Beta |
| Gemini | `gemini-3.8-live`, `gemini-3.8-live-extended-thinking` | Live API audio, text input, tools, thinking validation, combined audio/tool event parsing | Experimental |
| Gemini | `lyria-3.5` | Native GenerateContent music and audio normalization | Beta |

GPT-6 Astra, Claude Fable/Mythos 5.1, Gemini 3.8 Flash and Qwen3.8-Max-0902 retain the [September 5 contracts](MODEL_REFRESH_2026_09.md). These direct additions do not certify Azure, Vertex, Bedrock, OpenRouter or vLLM routes.

## DeepSeek migration

Prefer `create_deepseek()("deepseek-flash")`. DeepSeek now serves the legacy `deepseek-v4-flash` and `deepseek-v4-flash-vision-exp` IDs with V4.1 Flash and its billing. Both remain accepted, with distinct deprecated catalog records pointing to `deepseek-flash`. The SDK does not replace the ID supplied by the application. Historical evidence for the old model cannot establish current behavior behind the moving alias.

The three Flash routes accept `ImagePart` only in user messages. V4 Pro remains available and text-only. V4.1 maps `minimal`/`low` to `low`, `medium`/`high`/`xhigh` to `high`, and `max`/`ultra` to `max`; `none` disables thinking. Native `provider_options={"top_p": 0.98}` is effective with thinking. Values outside [0.95, 1] or nondefault top-p with thinking disabled fail locally instead of silently being clamped/ignored. Existing V4 Pro behavior is preserved.

Files, Responses-format and Anthropic-format endpoints are not added to the DeepSeek adapter. They are distinct native integrations, outside this Chat Completions update. Peak/off-peak pricing is not collapsed into a misleading universal token rate.

## Native images and music

```python
from zhivex_ai import create_openai, create_gemini

images = await create_openai().native.images().generate(
    model="gpt-image-2.5-sunburst", prompt="A simple botanical illustration", quality="max"
)
music = await create_gemini().native.media().generate_music(
    model="lyria-3.5", prompt="A short instrumental piano piece"
)
```

The music client returns provider audio MIME types/data without resampling. The existing image edit client accepts the new model IDs and qualities. No new default model is substituted into existing calls.

## Anthropic on-demand compaction

```python
from zhivex_ai import create_anthropic

messages = create_anthropic().native.messages()
history = [{"role": "user", "content": "Help me plan a small application."}]
request = {"model": "claude-fable-5-1", "max_tokens": 4096, "messages": history}
summary = await messages.compact(request)
blocks = summary.get("content", [])
if len(blocks) == 1 and blocks[0].get("type") == "compaction" and blocks[0].get("content") and blocks[0].get("signature"):
    # Replace exactly the summarized prefix; keep any newer turns after it.
    continuation = [{"role": "assistant", "content": blocks},
                    {"role": "user", "content": "Continue the plan."}]
    result = await messages.create({**request, "messages": continuation})
```

Both calls include `compact-2026-09-04`. Retain the original history if there is no usable signed block. Preserve system/tools and thinking settings as required by Anthropic; signatures are opaque and must not be edited. The SDK does not automatically rewrite Agent memory or run a background summary job. Do not combine on-demand compaction with threshold context management, stop sequences, forced tools or JSON format constraints.

For normalized native calls, pass `provider_options={"compaction": {"type": "summarize"}}` to `provider.native.language_model(...)`. Nonstreaming results preserve the block as `ProviderDataPart`; streaming emits a provider-data event with the complete block. Replaying that signed block adds the beta header automatically. The provider's `compaction` stop reason stays available in raw/provider finish metadata.

## Gemini 3.8 Live

Use `provider.native.realtime_model("gemini-3.8-live")`; the [existing example](../examples/realtime/gemini_realtime.py) now uses it. Audio is the default response modality for the new models. Standard Live rejects `thinkingLevel`; Extended Thinking accepts `low`, `medium`, or `high` and supports only NON_BLOCKING function declarations. Application code still owns tool execution and concurrent scheduling; no claim is made that `stream_live_agent` becomes a parallel scheduler.

`send_text` sends explicit user client content with `turnComplete=true`, which interrupts active generation. Extended Thinking can remain busy after a turn completes; when upstream reports `interaction_status=IN_PROGRESS`, the adapter does not emit an idle/completion signal. Simultaneous audio and tool events are both retained. These features do not promote realtime into Stable.

## GPT-Live

GPT-Live is a separate native protocol, cataloged as `api_surface="live"`; passing its model ID to the old Realtime client is not the supported route. Install `zhivex-ai-sdk[realtime]` for WebSockets.

```python
from zhivex_ai import create_openai

client = create_openai().native.live()
async with await client.connect({
    "model": "gpt-live-1",
    "audio": {"format": {"type": "audio/pcm", "rate": 24000},
              "output": {"voice": "marin"}},
    "delegation": {"type": "client"},
}) as session:
    # connect waits for session.started; session.started retains that event.
    # Feed paced base64 PCM through session.send and consume session.receive.
    # Handle delegated work and response.event payloads in your own application.
    final = await session.finish(timeout_ms=10_000)
```

The raw event API intentionally preserves all native events, including delegation and nested Responses events. One reader owns `receive()`; stop that reader before `finish()`, which drains events until `session.closed`. Complete/reconcile required backend work before finishing: `finish` drains finalization events without returning intermediate events. Its timeout/error paths still close the socket. `aclose()` and context exit release transport only and do not assert a successful provider finalization. Duration usage is cumulative seconds; token costs for the backend are separate.

For browser WebRTC, the trusted backend calls `client.create(session=..., transport={"type": "webrtc", "sdp": offer})` and returns the response's transport SDP answer to the browser. `fork`, `hangup`, and `download_recording` operate on explicit session IDs; recording requires provider-side storage. Microphone capture, playback, SIP, client delegation execution and application approvals remain application-owned. No project key belongs in browser code.

## Managed OpenAI Agents

```python
from contextlib import aclosing
from zhivex_ai import create_openai

sessions = create_openai().native.agent_sessions()
created = await sessions.create({
    "agent": {"model": "gpt-6-astra", "instructions": "Help with the requested task."},
    "environment": {"type": "openai_hosted"},
})
session_id = created["id"]
# Run this subscriber in an application-owned task before submitting input.
async with aclosing(sessions.events(session_id)) as events:
    async for event in events:
        # Dispatch native events to your durable application state and approval UI.
        pass
```

Submit input via `send_events(session_id, [{"type": "agent.session.input.message", "input": ...}])`; `cancel` submits `agent.session.input.cancel`. Retrieve session state and saved `items` after a stream interruption. Events do not replay; connect before starting a turn and explicitly close subscriptions. List/items return native pages; callers follow pagination. Resource IDs and cursors are URL-encoded. Creation and input submission do not automatically retry uncertain delivery.

This is a native API client, not a second implementation of the managed harness or a replacement for `Agent`/`run_agent`. Sandbox provisioning/configuration, multi-agent policy, approvals and durable business effects remain the application's/provider's responsibilities.

## Prompt cache diagnostics

`openai.native.responses().create({...,
"prompt_cache_options": {"comparison_response_id": previous_id}})` forwards the diagnostic request and retains `prompt_cache_diagnostics` in the native response. This comparison does not load history and is distinct from `previous_response_id`. The offline contract verifies that distinction.

## Evidence and sources

`tests/test_september_provider_updates.py` covers exact IDs, request shapes, validation, tool replay, signed blocks, streaming, HTTP errors, Live startup/finalization/timeouts and catalog boundaries. Existing certification artifacts remain immutable. Run the provider smokes against a newly built wheel and record exact model/operation/artifact/source identity before claiming live certification. No new scalar prices or inherited live evidence are fabricated.

Official sources reviewed September 16:

- [DeepSeek model identity and redirects](https://api-docs.deepseek.com/) and [thinking](https://api-docs.deepseek.com/guides/thinking_mode/).
- [Meta Spark 1.3 availability](https://research.meta.ai/blog/introducing-muse-spark-1-3).
- [OpenAI changelog](https://developers.openai.com/api/docs/changelog), [GPT-Live WebSockets](https://developers.openai.com/api/docs/guides/voice-websockets?api=live), [Live lifecycle](https://developers.openai.com/api/docs/guides/live-conversations), [Agents sessions](https://developers.openai.com/api/docs/guides/agents-api/sessions), [events](https://developers.openai.com/api/docs/guides/agents-api/sessions/events), and [cache diagnostics](https://developers.openai.com/api/docs/guides/prompt-caching/diagnostics).
- [Official Python Live API wire client](https://github.com/openai/openai-python/blob/main/src/openai/resources/live/live.py) and [Agents event wire client](https://github.com/openai/openai-python/blob/main/src/openai/resources/beta/agents/sessions/events.py).
- [Anthropic signed compaction](https://platform.claude.com/docs/en/build-with-claude/compaction#compact-on-demand-with-the-compaction-parameter).
- [Gemini Live capabilities](https://ai.google.dev/gemini-api/docs/live-guide), [release notes](https://ai.google.dev/gemini-api/docs/changelog), and [Lyria 3.5](https://ai.google.dev/gemini-api/docs/models/lyria-3.5).
