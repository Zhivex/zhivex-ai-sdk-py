# GPT-Live voice and durable Agent delegation

GPT-Live uses `/live/sessions`, separately from OpenAI `/realtime`. The supported
WebSocket subset of `create_openai().native.live()` consists of `connect`, session
`send`/`receive`, `append_audio`, `append_context`, `run_delegation`, `finish` and
`aclose`. That subset is Stable; [protected evidence](../releases/2026-09-19-gpt-live-protected.md)
records the exact certified wheel and scenarios. WebRTC creation, forks, recording downloads, SIP,
Responses-managed delegation and automatic reconnect are not promoted by this work.

Install `zhivex-ai-sdk[realtime]`. Use the runnable
[Agent example](../../examples/realtime/gpt_live_agent.py) with a mono PCM16 24 kHz
WAV and an output WAV path. It receives audio without claiming local playback.

```python
session = await create_openai().native.live().connect({
    "model": "gpt-live-1",
    "instructions": "Be concise. Delegate requests needing tools to the backend.",
    "delegation": {"type": "client"},
    "audio": {"format": {"type": "audio/pcm", "rate": 24000},
              "output": {"voice": "marin"}},
})
await session.append_audio(pcm16_bytes)
```

Feed audio continuously at its natural rate, including silence while waiting.
`append_audio` rejects empty/partial PCM16 samples. The application must match the
negotiated sample rate and route returned `session.output_audio.delta` bytes to its
player. GPT-Live has no per-turn output-audio-done event. Transcript receipt and
command acknowledgment do not prove that a person heard the audio.

## Delegation ownership

Receive events with one reader. Preserve transcript delta text and timestamps;
`session.delegation.created` contains metadata, not the user's request. Select an
explicit context snapshot from conversation history and application state, then run
`session.run_delegation(event, agent=backend_agent, prompt=context, ...)` in an
application-owned task while the reader keeps handling voice events.

The backend must have a run store. Normal `run_agent` options, tools, limits,
guardrails and approvals apply. The bridge derives the idempotency key from the
Live session and delegation IDs, claims an ID before awaiting backend work, rejects
duplicates and caps each session at 256 delegation claims. Failed or cancelled
claims are not automatically retried: inspect the durable run before recovery.
Use SQLite or Postgres for process durability; an in-memory store is process-local.

Only a backend run whose persisted state is `completed` is announced using
`session.commentary.append` with the original delegation ID. Suspended approvals,
failures and cancellations are returned without a success announcement. Use the
normal approval/resume APIs, validate the recovered result, then explicitly call
`append_context` with the original delegation ID. Closing the voice connection
does not roll back completed tool effects.

`append_context` supports `commentary`, `thinking` and `instructions`. Its
500-byte UTF-8 ceiling is stricter than the provider's 500-token limit; overlong or
empty output fails locally without truncation. Keep backend results concise. If
publication fails after a completed backend action, retrieve the durable result;
do not repeat the action just to retry speech. Raw `send` preserves provider event
payloads and their upstream validation requirements.

An interruption changes the voice conversation; it does not implicitly cancel a
backend action. The application owns task cancellation, revisions and suppression
of outdated results. Stop/join sender and backend tasks, cancel/join the reader,
then call `finish`. It waits for `session.closed` and retains final usage, with an
explicit timeout. `aclose` releases the transport but does not establish provider
finalization. Send and transport cleanup are bounded to five seconds.

## Certification boundary

The exact-wheel runner checks two full rounds of audio roundtrip, an audio request
that interrupts active speech, and client delegation to `gpt-5.6-luna` with exactly
one real callable-tool execution, returned-result acknowledgment and spoken marker.
Every case requires confirmed finalization, usage, closed transport and actual
audio output. No microphone, recordings of people, or raw transcripts are retained.
Fixtures are synthetic. Local evidence and protected CI provenance remain distinct.

This matrix does not certify speaker playback, arbitrary conversations, every
language, long-session behavior, WebRTC, native Responses delegation or reconnect.
The normalized Realtime/OpenAI and Gemini matrices remain separate jobs on the same
wheel and must be rerun when the package source changes.

Sources: [GPT-Live WebSockets](https://developers.openai.com/api/docs/guides/voice-websockets?api=live),
[client delegation](https://developers.openai.com/api/docs/guides/live-delegation?delegation-mode=client),
[session lifecycle](https://developers.openai.com/api/docs/guides/live-conversations).
