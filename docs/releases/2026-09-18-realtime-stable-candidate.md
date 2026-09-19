# Realtime Stable candidate — validation status

The working tree promotes 49 normalized realtime contracts and their supporting
data types to Stable. It is an unreleased candidate, not a completed protected
certification or a package-wide GA claim. Distribution version remains 0.25.0 Beta.

## Verified locally

- The transitive contract audit reports zero unstable dependencies for all 49
  contracts. Every normalized realtime event variant has a public import.
- `make check` with temporary Postgres: 1,124 tests and 250 subtests passed;
  coverage 85.90%. Lint, public stub and type checking passed.
- Wheel and sdist installations passed the artifact verifier, including realtime
  smoke and required Postgres workflow smoke.
- Python 3.11, 3.12 and 3.13 each passed 47 focused tests and 54 subtests plus the
  installed-wheel smoke. These minimal test environments emitted one harmless
  warning for the unused pytest-asyncio configuration. Full validation used 3.14.
- Wheel SHA-256: `9e8a15280cc832201135673615412503a62f645500355925de254ea58e01b163`.

## Complete-matrix live evidence

- [OpenAI](2026-09-18-realtime-stable-openai.json): 20/20 operations passed;
  the exact-artifact verifier accepted the local evidence.
- [Gemini initial](2026-09-18-realtime-stable-gemini.json): 19/20 passed. The first
  active-generation cancellation case received no audio before the wall deadline.
- [Gemini complete retry](2026-09-18-realtime-stable-gemini-retry.json): 17/20 passed.
  Recovery in the second round encountered a read timeout followed by two
  `429 RESOURCE_EXHAUSTED` responses from the text bridge. Earlier operations
  and the concurrent claim exclusion passed, but the aggregate gate correctly fails.

All runs used the same wheel. Results from separate failed matrices are not merged
into a passing report. Earlier isolated audio/recovery evidence is described in
[the investigation report](2026-09-18-live-runtime-readiness.md).

## Remaining release gates

1. Obtain sufficient Gemini quota and execute a complete passing two-round matrix.
2. Complete review and integration of [draft PR #47](https://github.com/Zhivex/zhivex-ai-sdk-py/pull/47).
   The OpenAI and Google release secret names are now configured in `release-smoke`;
   their presence does not establish credential validity or available quota.
   That environment permits only `main` and `v*` tags, so the draft branch cannot
   run the protected certification. GitHub currently requires review of the PR.
3. Integrate the reviewed workflow/code and execute the protected
   `Realtime exact-artifact certification` workflow from main. Verify both jobs
   and their retained evidence against the artifact built by that run.
4. Rebuild and recertify if package source, runner/helper code or audio fixtures
   change. Local evidence must not be re-labelled as protected certification.

The local receipts were collected before commit. A review PR carries the candidate;
merge, protected certification, tags and package publication remain separate steps.

## PR compatibility follow-up

The first PR CI run exposed a schema snapshot difference with the minimum supported
Pydantic version: it emits a redundant singleton `enum` alongside `const`. The
verifier now canonicalizes that equivalent representation without changing the
validation constraint. Local `minimum-core` validation passed 1,094 tests and 248
subtests (29 optional-dependency tests skipped); the focused regression suite passed
9 tests and 38 subtests. This tooling change does not alter package source or the
three live runner/helper files bound to the existing receipts.
