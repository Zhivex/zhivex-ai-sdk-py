# Protected realtime certification — PASSED — 2026-09-19 UTC

Workflow: https://github.com/Zhivex/zhivex-ai-sdk-py/actions/runs/35413884198

## Final verified outcome

The workflow is successful at attempt 7 after the user replaced the Google release
key. Gemini passed its complete two-round matrix (20/20), and OpenAI retains its
complete passing matrix (20/20) from the earlier attempt. These are two complete
provider matrices; individual cases from failed attempts were not combined.
Both downloaded receipts independently passed `verify_realtime_certification.py`
with the expected workflow run ID, main revision, source/runner hashes and the same
wheel digest recorded below. Gemini artifact ID: `10584435783`, created at
2026-09-19T12:23:26Z; receipt created at 2026-09-19T12:22:29.840086+00:00.

The normalized realtime Stable cohort now has passing protected live certification
for OpenAI and Gemini direct. This does not certify Vertex, browser tokens/WebRTC,
long-session backpressure or reconnect behavior, and does not imply package-wide
GA or publication of a new package version.

Passing evidence: [OpenAI](2026-09-19-protected-realtime-openai.json),
[Gemini attempt 7](2026-09-19-protected-realtime-gemini-attempt7.json).

## Historical first attempt (superseded)


- Merged PR: #47; revision `40606d7b4660fe78132ad04a67ab5c8155c8f490` on `main`.
- Wheel: `zhivex_ai_sdk-0.25.0-py3-none-any.whl`.
- Wheel SHA-256: `9e8a15280cc832201135673615412503a62f645500355925de254ea58e01b163`.
- Build passed. Both providers tested the same uploaded wheel in isolated installations.
- OpenAI: **20/20 passed**; the protected workflow verifier passed, and downloaded
  evidence was independently verified against the downloaded wheel, source and runner
  hashes, workflow identity, run ID and revision.
- Gemini: **15/20 passed**; the aggregate certification correctly failed.

## Historical Gemini blocker

All five failed cases received HTTP 429 `RESOURCE_EXHAUSTED` from
`gemini-3.8-flash`, the text model used after recovering persisted approvals:
round 1 deny/concurrent, and round 2 allow/deny/concurrent. The first allow/restart
passed. Realtime response, audio output, audio roundtrip, tool loop, suspension and
both cancellation scenarios passed in both rounds. Connections were closed.

The quota diagnostic does not establish whether the exhausted limit is per-minute,
per-day or another project limit. No automatic retry was dispatched after the quota
failures. The key is usable; available quota for the recovery model remains the
external blocker. Concurrent claims still prevented duplicate tool execution, but
failed generations are not counted as completed recoveries.

## Gates recorded before the successful retry (now satisfied)

Restore sufficient quota for `gemini-3.8-flash` on the project behind the Google
release key (or configure a key with sufficient quota), then rerun the complete
Gemini job and retain its passing two-round evidence against this same wheel.
If source, runners or fixtures change, build and certify the new artifact instead.
Do not combine successful individual cases from failed runs into a passing matrix.
The normalized realtime contracts are merged as Stable; this combined live
certification is still incomplete. No package release was published by this run.

Evidence: [OpenAI](2026-09-19-protected-realtime-openai.json),
[Gemini](2026-09-19-protected-realtime-gemini.json).

## Billing follow-up: attempt 2

After billing was enabled by the user, the complete Gemini job was rerun against
the same wheel. All 20 cases failed before setup completed, with connection-close
errors (cancellation assertions also failed). No provider error payload was received.
Artifact ID `10574767650` is the new evidence; downloading by artifact name selected
the older attempt, so the new artifact was fetched explicitly by ID. A single local
handshake using the local credential also failed with WebSocket close code 1011;
this does not prove that the local and GitHub credentials are identical. The new
failure cannot be attributed to quota from the available diagnostics.

Evidence: [Gemini attempt 2](2026-09-19-protected-realtime-gemini-attempt2.json).

## Final bounded retry: attempt 3

The user confirmed billing applies to the same project and key already configured
in GitHub. One additional full Gemini retry again failed all 20 cases before any
wire event arrived. Artifact ID `10575252471` was downloaded explicitly and retained.
No further automatic retries were launched. The remaining blocker is immediate
Gemini connection closure; available evidence no longer establishes a quota error.
Billing activation alone did not resolve the connection in these attempts.

Evidence: [Gemini attempt 3](2026-09-19-protected-realtime-gemini-attempt3.json).
