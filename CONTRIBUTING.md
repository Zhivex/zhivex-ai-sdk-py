# Contributing

This repository is an async-first Python SDK. Keep changes aligned with the documented public contract and provider support story.

## Setup

```bash
make dev
make check
```

Useful focused targets:

```bash
make test-contract
make test-core
make test-providers
make test-examples
make test-agents
make typecheck
```

## Public API Discipline

Supported production imports come from `zhivex_ai`. If `zhivex_ai.__all__` changes, update:

- `src/zhivex_ai/api_stability.py`
- contract tests
- `STABILITY.md`
- `CHANGELOG.md`
- user-facing docs/examples when behavior changes

Stable APIs need migration guidance before breaking changes. Beta and experimental changes still need changelog coverage.

## Providers

Provider changes should keep portable and native behavior separate. Update adapter metadata, provider tests, support-matrix docs, examples, and live smoke docs together.

Do not hand-edit the generated support matrix in `README.md` unless you are using the generator.

## Examples And Docs

Examples are product surface. Prefer offline deterministic examples for docs and tests. Live provider examples should clearly list required credentials and model IDs.

When changing user-visible behavior, update the relevant docs in the same change.

## Release Checks

Before release-oriented changes:

```bash
make check
make release-evidence
make release-check
```

`make release-evidence` writes the local gate output to `docs/releases/<version>-evidence.md`; use it for publish candidates and attach live-smoke notes separately when provider credentials are available.

Optional live smoke runs require configured credentials:

```bash
ZHIVEX_SMOKE_PROVIDERS=openai make smoke
```

## Dependency compatibility update

Development and CI use a reviewed uv lock with independent minimum/latest range tests. Realtime remains Experimental and its default websocket transport now requires `zhivex-ai-sdk[realtime]`; core/provider imports remain available without websockets. See [dependency compatibility](./docs/DEPENDENCY_COMPATIBILITY.md) for migration and update commands.
