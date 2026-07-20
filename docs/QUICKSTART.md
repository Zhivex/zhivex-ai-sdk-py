# Quickstart

This quickstart gets a new backend developer from a clean checkout to a verified local run.

## 1. Prepare The Repo

```bash
make dev
make check
```

`make dev` creates `.venv` and installs the SDK in editable mode with development dependencies. `make check` compiles sources, runs `ruff`, `mypy`, support-matrix drift checks, and the coverage suite.

## 2. Run An Offline Example

No provider credentials are required:

```bash
.venv/bin/python examples/agents/structured_workflow_outputs.py
.venv/bin/python examples/agents/workflow_resume.py
```

These examples use `create_mock_language_model(...)` and exercise the same workflow primitives used by production agents.

## 3. Try One Live Provider

Copy `.env.example` to `.env`, fill one provider, then scope smoke to that provider:

```bash
cp .env.example .env
set -a
. ./.env
set +a
ZHIVEX_SMOKE_PROVIDERS=openai make smoke
```

The smoke runner skips providers whose credentials or model IDs are missing. For provider-specific setup, see [PROVIDERS.md](./PROVIDERS.md).

## 4. Choose The Right Starting Point

- Foundation text or structured output: `examples/text/`
- Agent runtime: [AGENTS.md](./AGENTS.md)
- Declarative workflows: [WORKFLOWS.md](./WORKFLOWS.md)
- Production FastAPI patterns: [../PRODUCTION_APIS.md](../PRODUCTION_APIS.md)
- Gateway fallback routing: [GATEWAY.md](./GATEWAY.md)
- Observability: [OBSERVABILITY.md](./OBSERVABILITY.md)
- Troubleshooting: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

Production integrations should import from `zhivex_ai`, prefer stable APIs from [../STABILITY.md](../STABILITY.md), attach run stores for durable approvals/replay, and keep provider policy, business storage, authorization, and approval UI in application code.
