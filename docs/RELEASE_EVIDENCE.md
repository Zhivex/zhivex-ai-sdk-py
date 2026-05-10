# Release Evidence

Use this template before publishing a release.

## Release

- Version: 0.7.0
- Commit: Pending
- Branch: next-version
- Date: 2026-05-09
- Reviewer: Codex

## Required Commands

```bash
make check
make release-check
git diff --check
```

Record results:

- Focused provider/public contract tests: Passed on 2026-05-09. Result: `21 passed` for public contract, provider support, and Azure OpenAI provider tests.
- `make check`: Passed on 2026-05-09. Result: `393 passed, 2 skipped, 4 subtests passed`; coverage `82.96%`.
- `make release-check`: Passed on 2026-05-09 after rerun with network access for fresh pip dependency resolution. Built `zhivex_ai_sdk-0.7.0.tar.gz` and `zhivex_ai_sdk-0.7.0-py3-none-any.whl`; release artifacts verified.
- `git diff --check`: Passed on 2026-05-09 with no output.

## Artifact Verification

- Wheel: `dist/zhivex_ai_sdk-0.7.0-py3-none-any.whl` built and installed successfully in a fresh venv.
- Source distribution: `dist/zhivex_ai_sdk-0.7.0.tar.gz` built and installed successfully in a fresh venv.
- `twine check`: Passed for all artifacts currently in `dist/`, including `0.7.0` wheel and sdist.
- Fresh wheel install: Passed.
- Fresh sdist install: Passed.
- Extras install: `postgres`, `mcp`, `api`, `otel`, `docx`: Passed; each extra installed and import-checked in a fresh venv.
- `py.typed` present: Passed through artifact smoke assertion.
- `zhivex-skills` entrypoint: Passed; `zhivex-skills --help` ran from wheel and sdist installs.
- Offline agent/workflow smoke: Passed for wheel and sdist installs using `create_mock_language_model`, `generate_text`, `run_agent`, and `SequentialAgent`.

## Contract Review

- `src/zhivex_ai/api_stability.py` reviewed: Focused contract tests and `make check` passed.
- Stable API changes: None intended.
- Beta API changes: Azure OpenAI now exposes native lifecycle clients for Responses, Conversations, and Vector Store / File Search management.
- Experimental API changes: None intended.
- Migration notes required: No.
- Migration notes location: Not applicable.

## Provider Review

- Support matrix checked: Passed via `make check` / `scripts/generate_support_matrix.py --check-readme`.
- Tier-1 provider metadata reviewed: Azure OpenAI now reports native File Search, Responses, and Conversations support; focused provider tests passed.
- Live smoke providers run: None.
- Live smoke skipped providers and reason: OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex, and vLLM are skipped by default for this offline release-prep pass unless credentials/models are explicitly configured.

## Documentation Review

- `README.md`: Updated support matrix and Azure OpenAI native lifecycle notes.
- `STABILITY.md`: Clarified Azure OpenAI beta native lifecycle scope.
- `VERSIONING.md`: No versioning-policy changes.
- `SUPPORT.md`: Clarified Azure OpenAI native lifecycle support and exclusions.
- `CHANGELOG.md`: Added `0.7.0` minor entry.
- `docs/providers/tier-1.md`: Updated Azure OpenAI capability notes.
- Onboarding docs: No onboarding-flow changes.

## Known Risks

- Packaging risks: Low after artifact verification; note that fresh install verification requires network access to resolve dependencies such as `httpx`, `pydantic`, optional extras, and their transitive dependencies.
- Provider risks: Azure OpenAI lifecycle clients are covered by offline endpoint/auth tests; live provider behavior is not verified unless smoke credentials are configured.
- Runtime risks: No agent runtime changes intended.
- Docs gaps: No known release-doc gaps before final validation.

## Publication

- TestPyPI workflow: Not run in this preparation step.
- TestPyPI install verification: Not run in this preparation step.
- PyPI workflow: Not run in this preparation step.
- Tag: Not created in this preparation step; expected tag is `v0.7.0`.
- Final approval: Pending human review.
