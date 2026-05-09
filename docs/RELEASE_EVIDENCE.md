# Release Evidence

Use this template before publishing a release.

## Release

- Version: 0.6.1
- Commit: 3d72861 plus release-prep working-tree changes
- Branch: main
- Date: 2026-05-09
- Reviewer: Codex

## Required Commands

```bash
make check
make release-check
git diff --check
```

Record results:

- `make check`: Passed on 2026-05-09. Result: `390 passed, 2 skipped, 4 subtests passed`; coverage `82.96%`.
- `make release-check`: Passed on 2026-05-09 after rerun with network access for fresh pip dependency resolution. Built `zhivex_ai_sdk-0.6.1.tar.gz` and `zhivex_ai_sdk-0.6.1-py3-none-any.whl`; release artifacts verified.
- `git diff --check`: Passed on 2026-05-09 with no output.
- RC readiness reviewed: Not required for patch release; `docs/RC_READINESS.md` remains applicable to future `0.9.x` RC work.

## Artifact Verification

- Wheel: `dist/zhivex_ai_sdk-0.6.1-py3-none-any.whl` built and installed successfully in a fresh venv.
- Source distribution: `dist/zhivex_ai_sdk-0.6.1.tar.gz` built and installed successfully in a fresh venv.
- `twine check`: Passed for all artifacts currently in `dist/`, including `0.6.1` wheel and sdist.
- Fresh wheel install: Passed.
- Fresh sdist install: Passed.
- Extras install: `postgres`, `mcp`, `api`, `otel`, `docx`: Passed; each extra installed and import-checked in a fresh venv.
- `py.typed` present: Passed through artifact smoke assertion.
- `zhivex-skills` entrypoint: Passed; `zhivex-skills --help` ran from wheel and sdist installs.
- Offline agent/workflow smoke: Passed for wheel and sdist installs using `create_mock_language_model`, `generate_text`, `run_agent`, and `SequentialAgent`.

## Contract Review

- `src/zhivex_ai/api_stability.py` reviewed: Focused contract tests passed.
- Stable API changes: None intended.
- Beta API changes: None intended.
- Experimental API changes: None intended.
- Migration notes required: No.
- Migration notes location: Not applicable.

## Provider Review

- Support matrix checked: Passed via `make check` / `scripts/generate_support_matrix.py --check-readme`.
- Tier-1 provider metadata reviewed: Focused contract tests passed: `41 passed in 0.15s` for public contract, API stability, provider support, and tier-1 provider contracts.
- Live smoke providers run: None.
- Live smoke skipped providers and reason: OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex, and vLLM skipped intentionally for this patch-prep pass because no live credentials/models are being used; offline contract and artifact gates are the release gate for `0.6.1`.

## Documentation Review

- `README.md`: No user-facing surface changes.
- `STABILITY.md`: No stability-contract changes.
- `VERSIONING.md`: No versioning-policy changes.
- `SUPPORT.md`: No provider-support changes.
- `CHANGELOG.md`: Added `0.6.1` patch entry.
- `docs/RC_READINESS.md`: Reviewed as future RC guidance; not a blocker for `0.6.1`.
- Onboarding docs: No onboarding-flow changes.

## Known Risks

- Packaging risks: Low after artifact verification; note that fresh install verification requires network access to resolve dependencies such as `httpx`, `pydantic`, optional extras, and their transitive dependencies.
- Provider risks: Live provider smoke skipped by choice; provider behavior covered by offline contract tests for this patch.
- Runtime risks: No runtime code changes intended.
- Docs gaps: No known release-doc gaps for this patch-prep pass.

## Publication

- TestPyPI workflow: Not run in this preparation step.
- TestPyPI install verification: Not run in this preparation step.
- PyPI workflow: Not run in this preparation step.
- Tag: Not created in this preparation step; expected tag is `v0.6.1`.
- Final approval: Pending human review.
