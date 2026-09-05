# HU14–17 implementation evidence

Date: 2026-09-05. Implementation commit: `77d48824f484a75786036dfa29c02bb9f569d5ed`.
Branch: `feat/py-adoption-evidence`. Evidence is local; push was rejected by the
automatic approval reviewer. No PR, remote CI, merge, portal deployment or new
package publication is claimed.

Candidate wheel version remains 0.23.0 and is **not** the published PyPI artifact.
SHA256: `3990a493cd778465fbadddce5fbca6679b936d53ca2fe9a5bfa45c2bedabd38e`.
It was built from the implementation checkout; the benchmark verifies installed
package bytes against the supplied wheel. Test consumers import the wheel from
fresh venvs outside the repository with isolated Python mode.

## Results

- `make check` with Postgres 16: **1045 passed**, **161 subtests**, **85.39% coverage**.
- Generated project's Ruff and Mypy: passed from the candidate wheel.
- [SQLite consumer](./evidence/sqlite.json): passed.
- [Postgres consumer](./evidence/postgres.json): passed.
- [OpenAI live/Postgres consumer](./evidence/live-openai.json): passed using
  `gpt-5.6-luna`, the model pinned by the existing release workflow. Provider keys
  were read without printing them. First response: 5.456 seconds; full automation:
  17.445 seconds. This is machine timing, not a new-user onboarding measurement.
- All consumer paths check real process restart, wrong approval, approval success
  with zero tool errors, duplicate approval rejection, denial and cancellation.
- [OTLP receipt](./evidence/otlp.json): the exact generated trace reached both
  Collector → Jaeger 2.20.0 and Tempo 2.8.2; eight spans correlated. Observer
  redaction and metric label bounds have in-memory regressions.
- [Performance/soak](./evidence/performance.json): 200 rounds, 16 contenders, seeded
  synthetic load, no duplicate winning leases, stale-owner fencing preserved.
  Memory is tracemalloc peak, not RSS; logical TTL fault injection does not claim
  arbitrary process-crash or external-side-effect recovery.

## Defect reproduced and fixed

Typed approval inputs were passed to JSON serialization as Pydantic instances,
so suspension failed. Resume also passed persisted dictionaries directly to typed
tool callbacks. The runtime now persists `model_dump(mode="json")` and validates
stored arguments through the tool schema before execution. A regression with a
date field, recreated SQLite store and duplicate approval checks both boundaries.
The golden path requires a closed schema for strict OpenAI tool calls and forces
one tool request in its initial step. Its smoke rejects tool-error completion.

## Closure boundaries

- **HU14 Testing:** technical offline/live path verified; external human timing
  and release/portal alignment remain pending. No participant result is invented.
- **HU15 Testing:** scaffold, safe destination handling, generated checks and both
  storage paths verified. Publication and upstream HU14 acceptance remain pending.
- **HU16 Testing:** local two-backend E2E and sanitization verified; remote CI and
  review pending. Resumed tool execution has no dedicated runtime span; cross-process
  trace-context persistence is application-owned and explicitly documented.
- **HU17 Testing:** reproducible local baseline and gate fixtures verified; remote
  CI/review and publication pending.
- **HU18–20:** not implemented in this block. HU18 depends on HU14 acceptance;
  HU19 also needs partners and HU20 needs the broader certification/adoption evidence.

The workflow is committed but has not run remotely. The next authorized publishing
step is pushing this branch, creating a PR and checking its actual CI results.
