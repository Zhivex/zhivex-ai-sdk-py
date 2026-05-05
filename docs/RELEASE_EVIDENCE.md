# Release Evidence

Use this template before publishing a release.

## Release

- Version:
- Commit:
- Branch:
- Date:
- Reviewer:

## Required Commands

```bash
make check
make release-check
git diff --check
```

Record results:

- `make check`:
- `make release-check`:
- `git diff --check`:
- RC readiness reviewed:

## Artifact Verification

- Wheel:
- Source distribution:
- `twine check`:
- Fresh wheel install:
- Fresh sdist install:
- Extras install: `postgres`, `mcp`, `api`, `otel`, `docx`
- `py.typed` present:
- `zhivex-skills` entrypoint:
- Offline agent/workflow smoke:

## Contract Review

- `src/zhivex_ai/api_stability.py` reviewed:
- Stable API changes:
- Beta API changes:
- Experimental API changes:
- Migration notes required:
- Migration notes location:

## Provider Review

- Support matrix checked:
- Tier-1 provider metadata reviewed:
- Live smoke providers run:
- Live smoke skipped providers and reason:

## Documentation Review

- `README.md`:
- `STABILITY.md`:
- `VERSIONING.md`:
- `SUPPORT.md`:
- `CHANGELOG.md`:
- `docs/RC_READINESS.md`:
- Onboarding docs:

## Known Risks

- Packaging risks:
- Provider risks:
- Runtime risks:
- Docs gaps:

## Publication

- TestPyPI workflow:
- TestPyPI install verification:
- PyPI workflow:
- Tag:
- Final approval:
