# Releasing

This repository is set up for Python package publishing with `hatchling`.

Related documents:

- [README.md](./README.md)
- [STABILITY.md](./STABILITY.md)
- [VERSIONING.md](./VERSIONING.md)
- [SUPPORT.md](./SUPPORT.md)
- [CHANGELOG.md](./CHANGELOG.md)
- [docs/RELEASE_EVIDENCE.md](./docs/RELEASE_EVIDENCE.md)
- [docs/RC_READINESS.md](./docs/RC_READINESS.md)

## 1. Prepare the release

- Update `version` in `pyproject.toml`
- Make sure `README.md` matches the published surface
- Update `CHANGELOG.md`
- Review `STABILITY.md` and `VERSIONING.md` if the public surface changed
- Review `src/zhivex_ai/api_stability.py` if `zhivex_ai.__all__` changed or an API stability level changed
- Review the generated provider matrix with `make support-matrix-check`
- Review onboarding docs when installation, provider setup, or examples changed
- Add migration notes when a release changes stable or beta behavior in a user-visible way
- Fill out [docs/RELEASE_EVIDENCE.md](./docs/RELEASE_EVIDENCE.md)
- For a release candidate, review [docs/RC_READINESS.md](./docs/RC_READINESS.md)
- Run local validation:

```bash
make dev
make check
make release-check
```

## 2. Build locally

Create and verify the distribution files:

```bash
make release-check
```

This uses the checked-in `.venv` toolchain for the initial build and `twine check`, then runs `scripts/verify_release_artifacts.py` against fresh temporary virtual environments.

The release artifact verifier checks:

- wheel install
- sdist install
- `zhivex_ai` top-level import
- public export availability
- `py.typed` inclusion
- `zhivex-skills` entrypoint
- offline agent and workflow smoke
- optional extras: `postgres`, `mcp`, `api`, `otel`, `docx`

The fresh venv install steps may need network access to resolve dependencies.

## 3. Publish to TestPyPI

Recommended for the first release:

```bash
python3 -m twine upload --repository testpypi dist/*
```

Then verify installation:

```bash
python3 -m venv .venv-test
. .venv-test/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple zhivex-ai-sdk
python -c "import zhivex_ai; print(zhivex_ai.__all__[:10])"
```

Also rerun the installed-package smoke from the release evidence template against the TestPyPI package before promoting to PyPI.

## 4. Publish to PyPI

Manual upload:

```bash
python3 -m twine upload dist/*
```

Or use the GitHub Actions workflow by pushing a version tag:

```bash
git tag v0.6.1
git push origin v0.6.1
```

## 5. Trusted Publishing

If you want passwordless publishing from GitHub Actions:

1. Create the package on PyPI or set up a pending publisher.
2. Add this GitHub repository as a trusted publisher in PyPI and optionally TestPyPI.
3. Keep the workflow files in `.github/workflows/`.

## Notes

- Current package version in `pyproject.toml`: `0.6.1`.
- PyPI currently has `0.6.0`; the next patch release should be `0.6.1`.
- The package name on PyPI must be available. Confirm it before publishing.
- Repository metadata and badges should point to `zhivex-ai-sdk-py`.
- Do not publish a release that changes the documented stable surface without updating `CHANGELOG.md` and the release notes with migration guidance.
- Do not publish a beta release without confirming that [SUPPORT.md](./SUPPORT.md) still matches the supported provider and API story.
- Do not publish without a passing `make release-check` result recorded in [docs/RELEASE_EVIDENCE.md](./docs/RELEASE_EVIDENCE.md).
