# Quickstart: One Portable Agent

This guide takes you from installation to one portable agent, then separates deterministic offline validation from an authenticated live smoke. Workflows, evaluations, protocol servers, packaged skills, and realtime are not required for this path.

## 1. Install The SDK

For an application:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install zhivex-ai-sdk
```

For a repository checkout:

```bash
make dev
```

The SDK requires Python 3.11 or newer.

## 2. Run A Portable Agent

Set one provider credential:

```bash
export OPENAI_API_KEY="your-api-key"
```

The canonical checkout example is [quickstart_agent.py](../examples/agents/quickstart_agent.py). It combines one portable agent with one application-owned, typed tool:

```python
import asyncio

from pydantic import BaseModel

from zhivex_ai import Agent, create_openai, run_agent, tool


class ProjectStatusInput(BaseModel):
    project: str


def lookup_project_status(input: ProjectStatusInput) -> dict[str, str]:
    return {"project": input.project, "status": "on track"}


async def main() -> None:
    provider = create_openai()
    agent = Agent(
        name="project-assistant",
        instructions="Use the project-status tool, then answer concisely.",
        model=provider("gpt-5.6-terra"),
        tools={
            "lookup_project_status": tool(
                name="lookup_project_status",
                description="Returns the current status for a project.",
                schema=ProjectStatusInput,
                execute=lookup_project_status,
            )
        },
    )

    result = await run_agent(agent=agent, prompt="What is the status of Apollo?")
    print(result.text)


asyncio.run(main())
```

Run it:

```bash
.venv/bin/python examples/agents/quickstart_agent.py
```

When using the installed package outside this checkout, save the same code as `agent.py` and run `.venv/bin/python agent.py`.

`provider("model-id")` selects the strict portable language-model contract. The tool and status data stay in application code. To change providers, construct another portable provider and change the model ID; `Agent`, `tool()`, and `run_agent()` stay the same. Use `provider.native.*` only when the application intentionally depends on provider-specific behavior.

Meta Model API follows the same agent path with `create_meta()` and Standard `muse-spark-1.2`. That Tier-1 Stable scope covers portable text, streaming, structured output, callable tools, agent tool loops with `tool_choice="auto"`, and application-supplied retrieval through `PortableRetrievalConfig`. Portable retrieval adds bounded `PortableDocument` text to Chat Completions; it is separate from the Meta Files and hosted-search extensions that remain Beta with Contributor models and other native behavior.

See [PROVIDERS.md](./PROVIDERS.md) for credentials, model IDs, and provider-specific setup.

## 3. Validate The Runtime Offline

Offline validation proves the normalized agent contract without spending tokens or depending on a provider. In a checkout, run:

```bash
make test-agent-contracts
```

For a small deterministic application test, inject the SDK mock model into the same `Agent` runtime:

```python
import asyncio

from zhivex_ai import Agent, run_agent
from zhivex_ai.evals import GenerateResult, create_mock_language_model


async def main() -> None:
    model = create_mock_language_model(
        responses=[
            GenerateResult(
                text="Authentication, limits, retries, telemetry, rollback.",
                finish_reason="stop",
            )
        ]
    )
    result = await run_agent(
        agent=Agent(name="assistant", model=model),
        prompt="Give me an API launch checklist.",
    )
    assert result.text == "Authentication, limits, retries, telemetry, rollback."


asyncio.run(main())
```

`create_mock_language_model()` is a development helper, not a production model. Use it to test application orchestration deterministically.

Contributors can validate the broader checkout with:

```bash
make check
```

## 4. Validate One Provider Live

Repository smoke tests are opt-in and evidence-scoped. Copy the environment template, fill only the provider you want to test, and run a strict smoke for that provider:

```bash
cp .env.example .env
set -a
. ./.env
set +a
ZHIVEX_SMOKE_PROVIDERS=openai \
ZHIVEX_SMOKE_AGENTS=1 \
ZHIVEX_SMOKE_STRICT=1 \
make smoke
```

This verifies real provider generation plus a `run_agent` tool loop. A passing offline suite establishes contract behavior; only a successful authenticated smoke establishes live evidence for the selected provider, model, checkout, and artifact under test.

## 5. Add Production Capabilities Deliberately

Keep the initial agent small, then add only the layers the application needs:

- Tools, streaming, handoffs, sessions, memory, approvals, and replay: [AGENTS.md](./AGENTS.md)
- Durable Postgres state and API/worker boundaries: [../PRODUCTION_APIS.md](../PRODUCTION_APIS.md)
- Provider fallback routing: [GATEWAY.md](./GATEWAY.md)
- Telemetry and correlation fields: [OBSERVABILITY.md](./OBSERVABILITY.md)
- Operational and security guidance: [OPERATIONS.md](./OPERATIONS.md) and [../SECURITY.md](../SECURITY.md)

Applications should own business policy, authorization, approval UI, durable vertical data, secrets, and provider-selection policy.

## Optional Extensions

These capabilities are available but are not part of the minimum agent journey:

- Beta workflow graphs and declarative orchestration: [WORKFLOWS.md](./WORKFLOWS.md)
- Beta evaluation experiments and CI gates: [EVALUATIONS.md](./EVALUATIONS.md)
- Beta A2A, AG-UI, and Responses-compatible hosting: [PROTOCOLS.md](./PROTOCOLS.md)
- Beta general CLI and loopback playground: [CLI.md](./CLI.md)
- Experimental realtime/live agents: see the stability classification in [../STABILITY.md](../STABILITY.md)

Use the focused imports `zhivex_ai.workflows`, `zhivex_ai.evals`, `zhivex_ai.integrations`, and `zhivex_ai.experimental` for these surfaces. Existing top-level imports remain compatible, but new extension code should make its dependency boundary explicit.

Keep Beta and Experimental dependencies behind an application-owned boundary so the core agent path remains easy to upgrade. The complete boundary and non-goals are documented in [SCOPE.md](./SCOPE.md).
