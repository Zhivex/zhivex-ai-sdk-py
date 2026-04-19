PYTHON := .venv/bin/python

.PHONY: dev test test-cov lint typecheck smoke compile support-matrix-check check build release-check clean

dev:
	uv venv .venv
	uv pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest -q

test-cov:
	$(PYTHON) -m pytest --cov=src/zhivex_ai --cov-report=term-missing:skip-covered --cov-fail-under=80 -q

lint:
	$(PYTHON) -m ruff check src tests examples

typecheck:
	$(PYTHON) -m mypy

smoke:
	$(PYTHON) scripts/run_live_smoke.py

compile:
	$(PYTHON) -m compileall src tests examples

support-matrix-check:
	$(PYTHON) scripts/generate_support_matrix.py --check-readme

check: compile lint typecheck support-matrix-check test-cov

build:
	$(PYTHON) -m build --no-isolation --outdir dist

release-check: build
	$(PYTHON) -m twine check dist/*

clean:
	rm -rf .venv dist build src/*.egg-info .pytest_cache
