PYTHON := .venv/bin/python

.PHONY: dev test compile check build release-check clean

dev:
	uv venv .venv
	uv pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest -q

compile:
	$(PYTHON) -m compileall src tests examples

check: compile test

build:
	$(PYTHON) -m build --no-isolation --outdir dist

release-check: build
	$(PYTHON) -m twine check dist/*

clean:
	rm -rf .venv dist build src/*.egg-info .pytest_cache
