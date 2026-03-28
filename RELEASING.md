# Releasing

This repository is set up for Python package publishing with `hatchling`.

## 1. Prepare the release

- Update `version` in `pyproject.toml`
- Make sure `README.md` matches the published surface
- Run local validation:

```bash
python3 -m compileall src tests examples
python3 -m unittest discover -s tests -v
```

## 2. Build locally

Install the build tooling if needed:

```bash
python3 -m pip install --upgrade build hatchling twine
```

Create the distribution files:

```bash
python3 -m build
python3 -m twine check dist/*
```

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
git tag v0.1.0a1
git push origin v0.1.0a1
```

## 5. Trusted Publishing

If you want passwordless publishing from GitHub Actions:

1. Create the package on PyPI or set up a pending publisher.
2. Add this GitHub repository as a trusted publisher in PyPI and optionally TestPyPI.
3. Keep the workflow files in `.github/workflows/`.

## Notes

- Current recommended first public version: `0.1.0a1`.
- The package name on PyPI must be available. Confirm it before publishing.
