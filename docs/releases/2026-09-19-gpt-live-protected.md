# GPT-Live WebSocket Stable certification

Workflow: https://github.com/Zhivex/zhivex-ai-sdk-py/actions/runs/35444028791

Revision: `6fbea4339355a7b9c085b2234b1286b79258f1d1` (`main`, merged PR #49).
Wheel SHA-256: `bd4142d1675f579cd1aae7ca4ade5cbda0d8b7bc46b2a6ac7d18c6e47e545e09`.

## Verified protected evidence

- **GPT-Live: 6/6 passed**, two full rounds of audio roundtrip, interruption during
  active speech and durable Agent delegation with exactly one callable-tool effect.
- **OpenAI Realtime: 20/20 passed** on the same wheel.
- Downloaded receipts independently passed the verifiers against the exact wheel,
  complete package source manifest, runner/fixture hashes, main revision and workflow
  identity. GPT-Live additionally verifies workflow attempt 1.

The documented GPT-Live WebSocket subset is promoted to Stable. The guarantee covers
connect/send/receive, PCM16 audio/context helpers, client delegation to a stored
Agent, single-reader enforcement and bounded finalization/cleanup. It does not
promote WebRTC, SIP, recording/fork APIs, Responses-managed delegation, automatic
reconnect, or local speaker playback. The distribution remains Beta; no package
release was published by this workflow.

## Gemini acceptance and separate recheck

Gemini's previous complete **20/20 protected certification** remains the accepted
baseline by explicit user decision. It belongs to revision `40606d7` and the previous
wheel; see [the prior protected report](2026-09-19-protected-realtime-certification.md).
It is not relabelled as new-artifact evidence.

The first recheck on the new wheel passed **19/20**: round 1 `cancel-active` finished
without output audio before cancellation could be exercised. No HTTP 429 was recorded,
so the evidence does not establish RPM as its cause. A full Gemini retry had already
been dispatched when the user accepted the prior certification. Its result is retained
separately; no individual cases from failed matrices are merged into passing evidence.

Evidence: [GPT-Live](2026-09-19-gpt-live-protected.json),
[OpenAI Realtime](2026-09-19-openai-realtime-protected.json),
[Gemini first recheck](2026-09-19-gemini-realtime-recheck.json).

## Already-dispatched Gemini retry

Attempt 2 passed 19/20. The combined workflow remains red because the Gemini
recheck failed; GPT-Live and OpenAI Realtime jobs passed. Per the user's explicit
acceptance, the previous complete Gemini certification is the accepted baseline and
this recheck does not block GPT-Live's scoped promotion. No further retry was started.

[Full retry evidence](2026-09-19-gemini-realtime-recheck-attempt2.json).
