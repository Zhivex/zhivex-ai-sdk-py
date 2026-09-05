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

- Stable workflow graphs and declarative orchestration: [WORKFLOWS.md](./WORKFLOWS.md)
- Beta evaluation experiments and CI gates: [EVALUATIONS.md](./EVALUATIONS.md)
- Beta A2A, AG-UI, and Responses-compatible hosting: [PROTOCOLS.md](./PROTOCOLS.md)
- Beta general CLI and loopback playground: [CLI.md](./CLI.md)
- Experimental realtime/live agents: see the stability classification in [../STABILITY.md](../STABILITY.md)

Use the focused imports `zhivex_ai.workflows`, `zhivex_ai.evals`, `zhivex_ai.integrations`, and `zhivex_ai.experimental` for these surfaces. Existing top-level imports remain compatible, but new extension code should make its dependency boundary explicit.

Keep Beta and Experimental dependencies behind an application-owned boundary so the core agent path remains easy to upgrade. Stable workflows should still sit behind application authorization, storage, and side-effect controls. The complete boundary and non-goals are documented in [SCOPE.md](./SCOPE.md).

## Installed Durable Walkthrough (Candidate)

The next-release Beta CLI adds `zhivex init`. It is **not present in the published
0.23.0 wheel**. Use the candidate wheel from this change until a release includes it.
The SDK APIs used by the generated application already exist; no new Stable API is added.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install '/absolute/path/to/candidate.whl[postgres]'
.venv/bin/zhivex init durable-demo --backend postgres
cd durable-demo
```

Set `DATABASE_URL` to a dedicated Postgres database through your environment.
The generator writes placeholders only and never loads `.env` automatically.
The default `--backend sqlite` is a local Beta alternative, not the Postgres proof.
Keep this venv active, or use `../.venv/bin/python` below:

```bash
../.venv/bin/python app.py health
../.venv/bin/python app.py ready
../.venv/bin/python app.py first
../.venv/bin/python app.py start
```

`first` verifies generation; `start` asks a synthetic model for a read-only tool and
persists a pending approval before executing it. Copy the emitted `run_id` and
one `approval_ids` value. The command exits, providing a real process boundary.
After reviewing that pending request, continue in a new process:

```bash
../.venv/bin/python app.py approve --run-id RUN --approval-id APPROVAL
../.venv/bin/python app.py status --run-id RUN
```

The original run completes and a child run continues from durable state. Repeating
the approval fails. `deny` rejects the tool; `cancel --run-id RUN` cancels a pending
run and prevents approval. A cancellation cannot undo an external effect. The
example tool is a synthetic read; real writes require downstream idempotency.

For real provider calls, set `OPENAI_API_KEY` and an available `OPENAI_MODEL`, then
add `--live` to `first`, `start`, and `approve`. Keep the same model/mode throughout
a run. Each command has a 60-second bound; calls use 15-second timeouts, zero
retries and bounded output. A model that does not request approval fails the
walkthrough instead of producing false success. Only statuses, IDs and timings
are printed. Run/checkpoint storage still contains messages and requires access
controls, retention, backups and tenant isolation. The application owns approval
authorization and UI; possession of a run ID is not authorization.

### Reproduce the acceptance evidence

From a maintainer checkout, build a candidate and run the isolated consumer:

```bash
make build
.venv/bin/python scripts/verify_adoption.py dist/zhivex_ai_sdk-0.23.0-py3-none-any.whl --output /tmp/adoption-sqlite.json
.venv/bin/python scripts/verify_adoption.py dist/zhivex_ai_sdk-0.23.0-py3-none-any.whl --backend postgres --output /tmp/adoption-postgres.json
```

The Postgres command requires `DATABASE_URL`. Add `--live` only with the explicit
provider configuration above. The verifier installs the wheel in a fresh venv,
checks import origin, generates the project outside the checkout and checks
restart, wrong/duplicate approvals, denial and cancellation. It records wheel
SHA256 and machine timings. These timings **do not establish** first response
under five minutes or durable onboarding under fifteen minutes for a new user.
Use the [human timing protocol](./adoption/HUMAN_TIMING.md) for those criteria.
