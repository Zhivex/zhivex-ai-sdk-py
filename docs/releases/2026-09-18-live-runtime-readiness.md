# Live runtime verification and Stable readiness — 2026-09-18

The corrected candidate passed **10/10 local live operations** across two real
providers. This is installed-wheel integration evidence, not protected-workflow
release certification. The runtime and distribution remain Beta.

## Candidate identity

- Package: `zhivex-ai-sdk` 0.25.0, local uncommitted candidate.
- Wheel SHA-256: `a1f768e640b5f5b80879d9bb177682c15419c8abc8cb4b34dae04b18ff166803`.
- Every installed package Python file was compared with the wheel before testing;
  the wheel package manifest matched the source package at verification time.
  Subsequent Stable-readiness changes require a new wheel and new live evidence.
- Runner: `scripts/certify_live_runtime.py`. Receipts retain its SHA-256, the
  package manifest, HEAD, dirty-tree flag, time, requested model and operation.
- Model IDs are requested API IDs, not independently attested server snapshots.
- Each operation runs once, with a 40-second agent wall limit and 50-second outer
  deadline. Results do not establish reliability percentages or latency SLOs.
- Synthetic prompts and local tools only. No raw provider payloads, transcripts,
  keys or audio are stored in these receipts.

## Results

| Operation | OpenAI `gpt-realtime-2.1` | Gemini `gemini-3.8-live` |
| --- | --- | --- |
| Finite response | Passed: text | Passed: audio and transcript |
| Audio output | Passed: 86,400 bytes | Passed: 86,880 bytes |
| Tool loop | Passed: one execution and returned marker | Passed: one execution and returned marker |
| Human approval suspension | Passed: durable pending approval, zero executions | Passed: durable pending approval, zero executions |
| Cancel connected session | Passed: cancelled state and closed connection | Passed: cancelled state and closed connection |

Receipts: [OpenAI](2026-09-18-live-runtime-openai.json) and
[Gemini](2026-09-18-live-runtime-gemini.json). Cancellation here is after connection
creation, not an assertion about interrupting active audio generation. Approval
coverage ends at suspension; process-restart recovery and subsequent approval
execution were not live-tested in this matrix. Audio input, browser tokens,
WebRTC, long sessions and automatic reconnect were not tested.

## Local validation

- `make check` with temporary local Postgres: passed, including lint, typecheck,
  compilation and 1,101 tests plus 204 subtests; coverage 85.84%.
- Wheel and sdist independently installed and verified with `realtime,postgres`
  extras, including the installed live-runtime smoke and required Postgres
  workflow smoke. These artifact smoke tests are deterministic, separate from
  the real-provider operations above.
- Final receipts were checked against the current runner digest and every package
  Python source file. The runner exits nonzero on failed or blocked operations.

## Findings corrected before the passing run

1. OpenAI rejected `session.tools[0].strict` with `unknown_parameter`. Realtime
   had reused the Responses serializer. It now emits only realtime function
   declaration fields and does not impose the Responses strict-schema policy.
2. A Gemini tool turn could emit completion before the assistant's post-tool
   response. The runtime now waits for assistant output after executing a tool.
3. Gemini transcription can arrive as non-final fragments followed by a turn
   boundary. The runtime now collects those fragments into the result.
4. The first verification runner checked an incorrect normalized audio event name.
   Its audio-byte assertions were invalid. The runner was corrected and regression
   tested before both final matrices were rerun.

The `*-initial.json` receipts preserve the initial run, including the invalid
runner assertions; they must not be used to infer provider audio failure. The
OpenAI `*-diagnostic.json` receipt isolates the actual rejected field. Final
receipts use a different wheel hash and replace those attempts for this cohort.

## Work required for Stable

### Subsequent audio evidence

The expanded runner now tests synthetic spoken input (with no corresponding text
prompt), checks the spoken marker in the returned transcript, and requires output
audio. Separate receipts preserve each attempt:

- [OpenAI round trip](2026-09-18-live-runtime-openai-audio-v2.json): passed with
  explicit manual turn detection. Earlier attempts exposed automatic server VAD
  committing the buffer before the client's final commit.
- [Gemini round trip](2026-09-18-live-runtime-gemini-audio-vad.json): passed with
  explicit automatic activity detection and input transcription. Unconfigured
  attempts timed out, including after setup acknowledgement was added and at
  both 24 kHz and 16 kHz. Default audio-only behavior remains unverified.
- Active-generation cancellation passed for both providers in the `*-extended.json`
  receipts. Those receipts also retain failed audio attempts; their aggregate
  status is correctly failed.

These are intermediate-candidate receipts with their own wheel and runner digests,
not final Stable certification. The current tree additionally bounds initialization
cleanup. Repeat the complete matrix after the remaining runtime and contract work.

GitHub inspection found a branch-protected `release-smoke` environment; the old
provider workflow's `provider-certification` environment is not currently configured.
The realtime gate must use a verified protected environment and available credentials.
The environment currently exposes an OpenAI release credential but no Gemini
credential (secret names inspected only; no values retrieved).

### Subsequent recovery evidence

Recovery now runs in a fresh interpreter, reconstructing the tool from a shared
module and reopening SQLite. The parent retains no authority over the child run;
the child receives only database/effect-log paths, provider, text-model ID and the
approval decision. Two child processes compete in the concurrent operation.

- [OpenAI recovery](2026-09-18-live-runtime-openai-recovery-v3.json): approve,
  deny and concurrent recovery all passed using `gpt-5.6-luna` as the text bridge.
- [Gemini recovery](2026-09-18-live-runtime-gemini-recovery-v4.json): approve and
  concurrent recovery passed using `gemini-3.8-flash`; denial hit a provider 503.
  The [separate denial retry](2026-09-18-live-runtime-gemini-recovery-deny-retry.json)
  passed. The first attempt is retained and remains failed.
- Approved runs recorded exactly one local effect; denied runs recorded zero.
  Concurrent losers failed the durable claim before executing any effect.

The runs exposed and corrected OpenAI assistant-history content serialization,
Gemini slotted error serialization and migration of unsigned Live tool history to
GenerateContent. Real Gemini signatures remain unchanged; only Live-origin calls
without signatures receive the documented migration marker. The initial runner
also used different executors between processes; the fingerprint rejection was
correct and the harness now reconstructs identical definitions.

These receipts still cover intermediate artifacts, not the final Stable wheel.
Focused validation after these changes: 166 tests and 32 subtests passed, one
optional test skipped; lint and typecheck passed. Final full-matrix certification
and the transitive contract/policy gates remain open.

### 1. Freeze a precise, transitive public contract

The cohort currently includes types that depend on other Beta exports. Applying
the repository's stable-dependency audit finds 25 distinct Beta dependencies
outside `zhivex_ai.live`, including `AudioFrame`, `AgentCancellationToken`,
`ToolChoiceName`, `AnyToolDefinition`, and agent event types. Review and promote
only the required contracts, or isolate optional Beta branches explicitly. Do not
promote the entire SDK by changing one manifest entry.

Define whether Stable covers only finite turns, callable tools and the existing
text-runtime approval bridge. Reconnection, provider-native live APIs, direct live
handoffs and WebRTC can remain outside that contract; implementing all of them is
not a prerequisite for a smaller Stable cohort.

### 2. Resolve runtime semantics before promising compatibility

- Response-step enforcement and ordered `trace.events` were missing in the
  certified candidate. The subsequent working-tree changes implement both, with
  deterministic regression tests. They still need final-artifact live verification.
- Define ordering and meaning of tool-only completion, terminal completion,
  transcript fragments, duplicate call IDs and session errors in public tests.
  Pin configuration/update semantics and what can change after connection.
  The working tree now serializes configuration updates, preserves the last sent
  config on send/validation failure, and rejects Gemini updates on open connections.
- Keep audio guards and provider metadata outside text-redaction claims. Specify
  cleanup errors after durable completion and typed failures for truncated runs.

### 3. Expand live and failure evidence for the chosen cohort

Required evidence should cover real audio input/output round trips, approval
allow/deny after process restart using the documented text-model bridge, concurrent
resume claims, active-generation cancellation, timeout/stalled transport recovery,
slow consumers, bounded memory and repeated sessions. Combine deterministic fault
injection with bounded repeated live runs; a single passing session cannot establish
latency or reliability objectives. Include typed output and the supported Python
versions in the cohort matrix.

### 4. Bind evidence to a release artifact and protected workflow

Add a realtime-specific certification schema/policy and a protected workflow with
required operations per provider/model. Bind results to the exact wheel digest,
source identity, runner identity, workflow and expiry, with explicit blocked/failed
statuses. The local receipts deliberately set `release_certified=false`; they do
not upgrade the existing portable provider certification matrix. Once the gates
pass, update the stability manifest, compatibility policy and release notes for
that named cohort. This does not require package-wide GA.

## Primary protocol references

- [OpenAI GPT-Realtime-2.1](https://developers.openai.com/api/docs/models/gpt-realtime-2.1)
- [OpenAI realtime API](https://platform.openai.com/docs/api-reference/realtime)
- [Gemini 3.8 Live](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live)
- [Gemini Live protocol](https://ai.google.dev/api/live)

### Certification gate implementation

The working tree now contains `scripts/verify_realtime_certification.py`, a
versioned JSON schema, mutation tests and a dedicated `realtime-certification.yml`
workflow. The policy requires all ten operations twice for each named provider,
within seven days, on one exact wheel matching every package source file. It also
checks runner/helper digests, spoken-input fixture hashes, durable state, connection
closure and effect counts. Missing/blocked operations and duplicate samples fail.

The protected job additionally requires matching GitHub run/revision/workflow identity,
a clean main checkout and the `release-smoke` environment. Local verification does
not attest a GitHub run; release use still requires checking the actual successful
workflow and retrieving its retained artifacts. The workflow has not run yet.

The transitive export audit now identifies 49 contracts for promotion, including
the newly exposed `RealtimeGoAwayEvent` and `RealtimeSessionResumptionEvent` variants.
These two variants previously existed in the event union without public imports.
All remain Beta until the promotion and evidence gates are completed.

## Current candidate status

The subsequent promotion and complete-matrix results are recorded in the
[Stable candidate report](2026-09-18-realtime-stable-candidate.md). The working-tree
manifest now marks the full 49-contract cohort Stable; formal certification is
still pending. The earlier Beta descriptions above are historical candidate evidence.
