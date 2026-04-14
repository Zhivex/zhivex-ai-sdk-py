# Versioning

Zhivex AI SDK follows a strict versioning policy for the documented stable surface even while the package remains in the `0.x` range.

Related documents:

- [README.md](./README.md)
- [STABILITY.md](./STABILITY.md)
- [SUPPORT.md](./SUPPORT.md)
- [CHANGELOG.md](./CHANGELOG.md)

## Policy

- Stable APIs must not change in silently breaking ways.
- Breaking changes to stable APIs require advance deprecation notice.
- Breaking changes to stable APIs require an entry in [CHANGELOG.md](./CHANGELOG.md).
- Breaking changes to stable APIs require migration guidance in release notes.
- Beta APIs may evolve between minor releases, but user-visible changes must still be documented in the changelog.
- Experimental APIs may change without strong compatibility guarantees, but they must remain clearly labeled in documentation.

## Stable surface rules

The stable surface is defined in [STABILITY.md](./STABILITY.md). For that surface:

- Prefer additive changes over behavioral changes.
- Do not rename, remove, or repurpose public stable exports without a deprecation path.
- Keep top-level imports from `zhivex_ai` as the primary public entrypoint.

## Deprecation workflow

When a stable API needs to change:

1. Mark the API as deprecated in docs and release notes.
2. Provide the replacement path.
3. Record the change in [CHANGELOG.md](./CHANGELOG.md).
4. Include migration guidance for downstream users.
5. Remove the deprecated API only in a planned breaking release after the warning period.

## Beta and experimental expectations

Beta APIs are intended for early adoption with documented change management.

The current beta-only areas are narrower than the full agent story. MCP helpers, MCP-backed registries, and Postgres-backed agent stores are now part of the documented stable surface and follow the stable-surface rules above.

Experimental APIs are intended for evaluation. They should be consumed behind an application-owned abstraction if production teams need to try them before they graduate.

## Current maturity target

This policy now governs the current `Beta` phase of the SDK. The goal of this phase is to keep the documented stable surface predictable while continuing to tighten provider contracts, support policy, and release discipline on the path toward a future stable release.
