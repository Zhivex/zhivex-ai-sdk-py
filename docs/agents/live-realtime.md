# Stable live and realtime agents

The normalized SDK runtime is **Stable**. Import its contracts from `zhivex_ai.live`.
Existing imports from `zhivex_ai` and `zhivex_ai.experimental` remain aliases.
The distribution remains Beta. Stable describes the normalized API compatibility contract; release certification is a separate exact-artifact gate.

```python
from zhivex_ai import Agent, create_openai
from zhivex_ai.live import RealtimeSessionConfig, stream_live_agent

provider = create_openai()
agent = Agent(name="voice", model=provider.realtime_model("gpt-realtime-2.1"))

async def answer():
    async with stream_live_agent(
        agent=agent,
        prompt="Say hello briefly.",
        realtime_config=RealtimeSessionConfig(voice="alloy"),
        stream_buffer_size=4096,
    ) as stream:
        async for event in stream.event_stream():
            # Route audio frames to your player and approved text to your UI.
            pass
        return await stream.collect()
```

Install `zhivex-ai-sdk[realtime]` for the default websocket transport. Applications
can supply their own connection factory. The [runnable example](../../examples/realtime/live_agent_realtime.py)
adds a local tool.

## Contract

- A live run owns one connection and ends at the first completed response after
  tool execution. It is not an indefinitely running conversation service.
- `aclose()` and async context management cancel and join the producer. Built-in
  session shutdown cancels a blocked receiver and bounds close-payload and
  connection-close waits to five seconds each. Custom sessions own equivalent
  cancellation-safe cleanup. Sending to a closed built-in session raises `ValidationError`.
- The built-in transport retains 1,000 events. A consumer that falls behind fails
  explicitly instead of losing events silently. The agent result has a separate
  `stream_buffer_size`; use a finite value for request-owned streams.
- EOF or session closure before response completion, explicit provider errors,
  failed/incomplete/cancelled responses, and conflicting tool-call IDs fail the run.
  Gemini generation completion is not treated as turn completion. OpenAI tool-only
  response completion continues the loop. Identical repeated tool calls execute
  once per run; this is not durable exactly-once execution of external effects.
- Local tools use the agent's approval policy, guards, timeout handling and call
  limits. Unknown tools return an error result. Cancellation and wall-time limits
  are checked while connecting, waiting for events, and sending results.
- `max_steps` counts completed provider response turns, including tool-only turns.
  Gemini's intermediate `generation-complete` event does not consume another step.
  If another response is required after the last allowed step, the run fails and
  closes its connection. Already executed tools are not rolled back.
- `result.trace.events` retains agent events in publication order, including the
  terminal event. Realtime audio and wire metadata are not copied into this trace.
- Session updates retain the previous local configuration if validation or sending
  fails. Successful sending records the last sent configuration; this does not
  attest server acceptance. Concurrent updates are serialized. Gemini configuration
  is fixed for an open connection, so `update` raises `UnsupportedFeatureError`
  without sending another setup message; use a new connection for a new config.
- Gemini `connect()` waits for `setupComplete` before exposing the session. The
  handshake uses `RealtimeConnectOptions.timeout_ms` (10 seconds by default).
- For OpenAI manual audio turns, set `turn_detection={"type": "none"}`. This
  sends an explicit null detection config, so `AudioFrame.is_final` commits the
  audio once. Server VAD otherwise owns its own automatic commit boundary.
- Run stores support idempotency, durable cancellation and approval suspension.
  SQLite remains single-host; shared workers should use Postgres.
- Typed output uses prompted JSON followed by validation; native typed output is
  rejected. Text output guards buffer text before publication. They do not inspect
  or redact audio frames or arbitrary provider metadata.

## Approval recovery and limitations

Suspension closes the realtime connection and saves the pending tool approval.
`resume_agent_run` continues through the normal language-model runtime: reconstruct
an agent with the same tool definitions and a compatible language model with
`generate`, sharing the run store. Incompatible models are rejected before claiming
the approval or executing its tool. It does not reconnect or restore the provider's
voice session. Re-supply dependencies and reconcile external effects after crashes.
Use the identical tool implementation and bound configuration in both processes:
approval fingerprints include the executor, not just its name and input schema.
For Gemini, normalized Live tool calls retain their origin in provider metadata.
When replayed through GenerateContent, unsigned Live calls use Google's documented
history-migration marker; any genuine thought signature is preserved unchanged.
This transfers tool history, not the Live model's hidden reasoning or voice state.
A new realtime session is application-owned; automatic reconnect/replay and native
session resumption are outside this Stable guarantee.

Direct live handoff orchestration, native structured output, multi-turn voice
session recovery, audio safety filtering and provider billing reconciliation are
not claimed by this promotion. Runtime hooks and stored state do not certify the
native provider transport. GPT-Live `/live/sessions` has a [separate Stable WebSocket contract](gpt-live.md); model-specific Gemini Live
extensions, and non-portable Bedrock/vLLM realtime integrations retain their
separate classifications.

## Validation and evidence

`tests/test_realtime.py` covers transport normalization, bounded retention and
shutdown, finite completion, duplicate tools, approvals, idempotency, cancellation,
and unknown tool outcomes. Public-contract tests pin Stable classification and alias
identity. These deterministic tests do not establish live provider availability.
Run exact-model live checks only with configured credentials and record the wheel,
model, operation and result separately from API stability.

Protocol references: [OpenAI server events](https://platform.openai.com/docs/api-reference/realtime-server-events)
and [Gemini Live protocol](https://ai.google.dev/api/live).

## Local live verification

The [September 18 report](../releases/2026-09-18-live-runtime-readiness.md) records
10 passing real-provider operations on one installed candidate wheel, the issues
found and corrected, and the work remaining for Stable. This is local integration
evidence, not release certification or package-wide GA.

Every `RealtimeEvent` variant is importable from `zhivex_ai.live`, including
`RealtimeGoAwayEvent` and `RealtimeSessionResumptionEvent`. Applications may use
these notifications to manage connections; the runtime does not reconnect or
replay external effects automatically.

The [current Stable candidate status](../releases/2026-09-18-realtime-stable-candidate.md)
separates the unreleased compatibility promotion from local live validation and
outstanding protected release certification.
