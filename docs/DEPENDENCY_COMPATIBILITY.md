# Dependency compatibility and reproducible development

Use uv 0.12.4 or newer. Hatchling remains the build backend; runtime dependencies retain ranges, while `uv.lock` pins public distributions and hashes for development/CI.

```bash
make dev
make check
```

`make dev` runs `uv sync --locked --all-extras`. CI uses the same lock on Python 3.11–3.14; `uv lock --check` detects metadata drift. To update dependencies, run `uv lock --upgrade`, review the diff, and rerun checks plus the compatibility matrix. Never add private indexes or tokens to the lock.

## Range validation

The separate CI matrix runs all tests from fresh Python 3.11 environments:

- `minimum-core`: pins each declared core lower bound exactly and resolves compatible development tools at their minima. Optional runtime extras are absent.
- `minimum-extras`: resolves the lowest compatible direct dependencies with every extra enabled. Transitive constraints can raise a floor; this is not a claim that mutually incompatible individual extra floors can coexist.
- `latest`: resolves the latest permitted dependencies independently of the committed development lock.

```bash
uv run --no-sync python scripts/check_dependency_compatibility.py --mode minimum-core
uv run --no-sync python scripts/check_dependency_compatibility.py --mode minimum-extras
uv run --no-sync python scripts/check_dependency_compatibility.py --mode latest
```

Each run produces a JSON result with exact package versions. CI retains these reports even on failure. The minimum Python remains 3.11. The pytest-asyncio tooling floor is 1.4.0 because the old 0.23 floor fails collection with pytest 9. FastAPI's current TestClient uses the development-only httpx2 dependency; the known httpx fallback warning is an error in tests. Other upstream deprecations are not silently suppressed.

See the official [uv locking](https://docs.astral.sh/uv/concepts/projects/sync/) and [resolution](https://docs.astral.sh/uv/concepts/resolution/) contracts and [Starlette TestClient](https://www.starlette.io/testclient/) guidance.

## Realtime dependency migration

Core installation no longer pulls in websockets. For the Experimental websocket transport use:

```bash
pip install "zhivex-ai-sdk[realtime]"
```

Existing realtime imports remain available; only opening the default websocket connection requires the extra. Custom connection factories do not require websockets. Artifact verification installs the core alone and checks that websockets is absent, then installs the realtime extra in another fresh environment and verifies its import. All extras remain covered by the dependency audit.

## Incremental provider lint boundary

`make lint` additionally checks openai_compat, _payload, and _url_security using import ordering (I), bugbear (B), async checks (ASYNC), and safe simplification rules SIM101/SIM103/SIM114. The first cohort is deliberately scoped to shared providers; global Ruff rules are unchanged. Imports are sorted without runtime behavior changes, no new ignores were introduced, and mypy already covers this boundary without skipped provider imports.
