# Releasing

This repository is set up for Python package publishing with `hatchling`.

## 1. Prepare the release

- Update `version` in `pyproject.toml`
- Make sure `README.md` matches the published surface
- Run local validation:

```bash
make dev
make check
```

## 2. Build locally

Create the distribution files:

```bash
make release-check
```

This uses the checked-in `.venv` toolchain and disables build isolation, so it does not need to download build requirements again when the environment is already prepared with `make dev`.

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

## 4. Publish to PyPI

Manual upload:

```bash
python3 -m twine upload dist/*
```

Or use the GitHub Actions workflow by pushing a version tag:

```bash
git tag v0.3.0
git push origin v0.3.0
```

## 5. Trusted Publishing

If you want passwordless publishing from GitHub Actions:

1. Create the package on PyPI or set up a pending publisher.
2. Add this GitHub repository as a trusted publisher in PyPI and optionally TestPyPI.
3. Keep the workflow files in `.github/workflows/`.

## Notes

- Current recommended public version: `0.3.0`.
- The package name on PyPI must be available. Confirm it before publishing.
- Repository metadata and badges should point to `zhivex-ai-sdk-py`.
