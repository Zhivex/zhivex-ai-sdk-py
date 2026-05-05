# RC Readiness

This document is the handoff checkpoint for moving the Python SDK from beta maturity work into release-candidate review.

## Current Verdict

The repository is ready for RC review when the current working tree is intentionally committed and the release evidence template is filled with command output from the final commit.

The maturity phases in [MATURITY_PLAN.md](../MATURITY_PLAN.md) are complete. The remaining work is release governance, not another implementation phase.

## Required RC Evidence

Before tagging a `0.9.x` release candidate, record fresh output in [docs/RELEASE_EVIDENCE.md](./RELEASE_EVIDENCE.md) for:

- `make check`
- `make release-check`
- `git diff --check`
- artifact verifier output for wheel and sdist installs
- optional extras install checks for `postgres`, `mcp`, `api`, `otel`, and `docx`
- support matrix check
- API stability manifest review
- changelog and migration-note review

## RC Scope

The RC should include:

- stable public imports guarded by `src/zhivex_ai/api_stability.py`
- tier-1 provider contracts and optional live smoke paths
- production API, gateway, observability, security, and operations docs
- agent runtime docs and deterministic examples for persistence, approvals, replay, traces, and workflows
- release artifact verification from fresh virtual environments

The RC should not require:

- DeepSeek in Python GA
- live credentials for normal CI
- CLI/UI/deploy automation for workflow agents
- broad stability guarantees for beta packaged skills, workflow agents, replay/evaluation helpers, safety helpers, or provider-managed hosted-tool behavior

## 1.0.0 Gate

Promote to `1.0.0` only after at least one RC has been installed and exercised from the built artifacts, release notes are reviewed against the stable manifest, and any beta APIs that remain beta are explicitly called out in `STABILITY.md`, `VERSIONING.md`, and `SUPPORT.md`.

Blocking questions for GA:

- Is the stable surface in `src/zhivex_ai/api_stability.py` still the intended support contract?
- Are all stable API changes documented with migration guidance?
- Are tier-1 provider docs and support metadata aligned with current adapter behavior?
- Are release artifacts installable from a clean environment?
- Are live smoke skips documented with concrete credential/model reasons?

## Beta Areas After RC

These areas may remain beta through the RC:

- packaged skills and registry publishing
- declarative workflow agents
- replay/evaluation helpers and trace artifact helpers
- redaction, safety, and budget guard helpers
- provider-managed hosted-tool approvals
- provider-native media clients and compatibility providers

Production adopters can use these areas behind application-owned abstractions while the stable surface remains protected.
