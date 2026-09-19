# GPT-Live WebSocket candidate — 2026-09-19

The candidate adds PCM16 audio/context helpers and durable Agent client delegation
to the existing native GPT-Live WebSocket session. It does not promote WebRTC, SIP,
recording/fork APIs, Responses-managed delegation or automatic reconnect.

## Local exact-artifact evidence

- `gpt-live-1` with backend `gpt-5.6-luna`: **6/6 passed**, two full rounds of audio
  roundtrip, speech interruption and client delegation with one actual tool effect.
- Every case received output audio and the expected transcript marker, finalized
  with usage and closed its connection. Both delegated backend runs completed.
- The independent verifier accepted the complete matrix against the wheel, full
  package source manifest, runner and synthetic audio fixture digests.
- Wheel SHA-256: `ec8bac9807473f6794bd59b7623971a78c4fa6b4238719b39e96734fc9667e38`.
- Wheel and sdist passed isolated artifact installation with the realtime extra.
- `make check`: 1,119 passed, 264 subtests, 16 optional tests skipped, 84.84% coverage.
  A subsequent cancellation regression also passed (7 GPT-Live tests, 6 subtests).
- Contract/stability/verifier checks: 32 passed and 29 subtests.

The first two matrices each passed 5/6. The runner now uses scenario-specific
instructions and establishes a continuous audio timeline with one second of silence
before playing a fixture. Earlier results are retained separately; no individual
case was cherry-picked to construct this successful matrix.

## Remaining protected gate

This is local evidence from a working tree, not protected certification. Review and
merge the candidate, then run `Realtime exact-artifact certification` from `main`.
The added `gpt-live` job verifies both rounds against the same built wheel as the
OpenAI normalized realtime and Gemini jobs. Because package source changed, the
existing protected evidence for commit `40606d7` does not certify this new artifact.
Only promote the documented GPT-Live WebSocket subset after the protected job passes.
The distribution remains Beta, and this change does not publish a package release.

Evidence: [complete passing matrix](2026-09-19-gpt-live-candidate.json),
[first matrix](2026-09-19-gpt-live-first-matrix.json),
[second matrix](2026-09-19-gpt-live-second-matrix.json).
