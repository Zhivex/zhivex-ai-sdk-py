# Release Evidence

Use this template before publishing a release.

## Release

- Version: 0.8.0
- Commit: Pending
- Branch: main
- Date: 2026-05-10
- Reviewer: Codex

## Required Commands

```bash
make check
make release-check
git diff --check
```

Record results:

- Focused provider/public contract tests: Passed on 2026-05-10. Result: `69 passed` for shared tier-1 contracts, provider support, Qwen, Kimi, and public contract tests.
- `make check`: Passed on 2026-05-10. Result: `410 passed, 2 skipped, 4 subtests passed`; coverage `82.98%`.
- `make release-check`: Passed on 2026-05-10 after rerun with network access for fresh pip dependency resolution. Built `zhivex_ai_sdk-0.8.0.tar.gz` and `zhivex_ai_sdk-0.8.0-py3-none-any.whl`; release artifacts verified.
- `git diff --check`: Passed on 2026-05-10 with no output.

## Artifact Verification

- Wheel: `dist/zhivex_ai_sdk-0.8.0-py3-none-any.whl` built and installed successfully in a fresh venv.
- Source distribution: `dist/zhivex_ai_sdk-0.8.0.tar.gz` built and installed successfully in a fresh venv.
- `twine check`: Passed for all artifacts currently in `dist/`, including `0.8.0` wheel and sdist.
- Fresh wheel install: Passed.
- Fresh sdist install: Passed.
- Extras install: `postgres`, `mcp`, `api`, `otel`, `docx`: Passed; each extra installed and import-checked in a fresh venv.
- `py.typed` present: Passed through artifact smoke assertion.
- `zhivex-skills` entrypoint: Passed; `zhivex-skills --help` ran from wheel and sdist installs.
- Offline agent/workflow smoke: Passed for wheel and sdist installs using `create_mock_language_model`, `generate_text`, `run_agent`, and `SequentialAgent`.

## Contract Review

- `src/zhivex_ai/api_stability.py` reviewed: Passed through public contract tests and `make check`.
- Stable API changes: Qwen and Kimi are promoted to tier-1 portable provider factories for text, streaming, structured output, and callable tools.
- Beta API changes: Qwen native Files and Batch clients are now exposed; Qwen hosted tools/Responses/ASR/TTS and Kimi Files/Batch/token/Formulas remain provider-specific beta surfaces.
- Experimental API changes: None intended.
- Migration notes required: No.
- Migration notes location: Not applicable.

## Provider Review

- Support matrix checked: Passed via `make check` / `scripts/generate_support_matrix.py --check-readme`.
- Tier-1 provider metadata reviewed: Qwen and Kimi added to `TIER_1_PROVIDERS` with portable badges; focused contract tests passed.
- Live smoke providers run: None.
- Live smoke skipped providers and reason: OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex, Qwen, Kimi, and vLLM are skipped by default for this offline release-prep pass unless credentials/models are explicitly configured.

## Documentation Review

- `README.md`: Regenerated support matrix and updated Qwen/Kimi tier-1 notes plus smoke guidance.
- `STABILITY.md`: Updated Qwen/Kimi native beta boundaries.
- `VERSIONING.md`: No versioning-policy changes intended.
- `SUPPORT.md`: Updated tier-1 provider scope and Qwen/Kimi boundaries.
- `CHANGELOG.md`: Added `0.8.0` minor entry.
- `docs/providers/tier-1.md`: Updated provider list, smoke envs, examples, and capability notes.
- Onboarding docs: `examples/README.md` and `.env.example` updated for Kimi smoke and Qwen/Kimi portable tier-1 notes.

## Known Risks

- Packaging risks: Low after artifact verification; note that fresh install verification requires network access to resolve dependencies such as `httpx`, `pydantic`, optional extras, and their transitive dependencies.
- Provider risks: Qwen and Kimi promotion is covered by offline contract tests; live provider behavior is credential-driven and must be recorded when run.
- Runtime risks: No agent runtime changes intended.
- Docs gaps: No known release-doc gaps before final validation.

## Publication

- TestPyPI workflow: Not run.
- TestPyPI install verification: Not run.
- PyPI workflow: Not run.
- Tag: Not created; expected tag is `v0.8.0`.
- Final approval: Pending human review.
