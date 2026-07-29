# Agent Runtime Guide

The agent runtime is the production-oriented orchestration layer for typed, stateful assistants, tools, handoffs, approvals, persistence, replay, and traces.

Use the public imports from `zhivex_ai`:

```python
from zhivex_ai import Agent, AgentContext, AgentRunResult, AgentStreamResult, run_agent, stream_agent
```

## Stable Core

The stable core is:

- `Agent`, `AgentSession`, `AgentRuntime`, `AgentRegistry`, `AgentRunResult`, and `AgentStreamResult`
- generic `Agent[DepsT, OutputT]`, `AgentContext[DepsT]`, `ToolExecutionContext[DepsT]`, and typed `result.output`
- dynamic instructions, lifecycle `AgentHooks`, and `AgentMiddleware`
- local `tool(...)` definitions, tool execution contracts, `ToolRegistry`, and direct `handoff_to(...)` results
- `run_agent(...)`, `stream_agent(...)`, `resume_agent(...)`, and `resume_agent_run(...)`
- session helpers such as `create_agent_session(...)` and `load_agent_session(...)`
- portable skills and MCP discovery/registry helpers
- Postgres memory, checkpoint, and run stores
- run-state serialization, cancellation tree, replay, run-snapshot helpers, and durable pending approvals

The beta layer includes native subagent tools such as `create_subagent_tool(...)`, checkpoint events, evaluation reports, trace artifacts, safety policies, provider-managed approvals, in-memory/SQLite stores, packaged skills, workflow agents, and UI approval chunks. Live/realtime agent APIs are experimental.

## Minimal Tool-Using Agent

Install the package, configure one provider credential, and run an agent with a narrowly scoped local tool:

```bash
pip install zhivex-ai-sdk
export OPENAI_API_KEY="your-api-key"
```

```python
import asyncio

from pydantic import BaseModel, ConfigDict

from zhivex_ai import Agent, create_openai, run_agent, tool


class ProjectLookupInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str


async def main() -> None:
    provider = create_openai()
    agent = Agent(
        name="project_assistant",
        instructions="Use lookup_project before answering project-status questions.",
        model=provider("gpt-5.6-terra"),
        tools={
            "lookup_project": tool(
                name="lookup_project",
                description="Read the current status of a named project.",
                schema=ProjectLookupInput,
                execute=lambda input: {"project": input.project, "status": "on-track"},
            )
        },
    )

    result = await run_agent(agent=agent, prompt="What is the status of Apollo?")
    print(result.text)


asyncio.run(main())
```

Use an application-owned adapter instead of the inline callable when the tool reads a database or external service. Add `requires_approval=True` before allowing a tool to write data or trigger an external side effect.

## Typed Context And Outputs

`Agent[DepsT, OutputT]` connects application dependencies to dynamic instructions and local tools, and connects `output_type` to `AgentRunResult[OutputT]`:

```python
from dataclasses import dataclass
from pydantic import BaseModel

from zhivex_ai import Agent, AgentContext, ToolExecutionContext, run_agent, tool


@dataclass
class Deps:
    tenant_id: str
    repository: object


class Decision(BaseModel):
    approved: bool
    reason: str


async def instructions(context: AgentContext[Deps]) -> str:
    tenant_id = context.deps.tenant_id if context.deps else "unknown"
    return f"Review requests for tenant {tenant_id}."


async def lookup(input: dict[str, str], context: ToolExecutionContext[Deps]) -> dict[str, str]:
    if context.deps is None:
        raise RuntimeError("Missing dependencies.")
    return {"tenant_id": context.deps.tenant_id, "application_id": input["application_id"]}


agent: Agent[Deps, Decision] = Agent(
    name="reviewer",
    model=model,
    instructions=instructions,
    output_type=Decision,
    tools={
        "lookup": tool(
            name="lookup",
            description="Load one application.",
            schema={"type": "object", "properties": {"application_id": {"type": "string"}}},
            execute=lookup,
        )
    },
)
result = await run_agent(agent=agent, prompt="Review A-42.", deps=deps)
decision = result.output
```

Instructions may be a string or a sync/async callable accepting `AgentContext`; two-argument callables can also accept the current `Agent`. They resolve once per agent segment, after session memory and dependencies are available. The resolved system message is not retained in the session transcript.

`output_mode="auto"` selects native structured output when the current model advertises it and a prompted JSON Schema fallback otherwise. Set `"native"` to fail closed on models without native support, or `"prompted"` to force the fallback. Output guardrails run before parsing. Invalid JSON or schema values fail the run; suspended and stopped-on-handoff results have `output=None`. Raw `result.text` remains available.

The root agent owns the output contract for the full run. A terminal agent reached through direct handoff is instructed and validated against the root `output_type`. All direct-handoff agents share the same dependency type for that run. A native subagent invoked as a tool starts a child run whose own agent defines its output contract.

Dependencies are process-local capabilities and may contain clients or credentials. The runtime excludes them from reprs, serialized tool contexts, checkpoints, traces, metadata, and `AgentRunState`. Supply `deps=` again to `resume_agent(...)` or `resume_agent_run(...)`; do not use dependencies as durable state.

`stream_agent(...).collect()` returns the same typed `AgentRunResult`. Experimental realtime agents use prompted typed output and reject `output_mode="native"`.

## Lifecycle Hooks And Run Middleware

Subclass `AgentHooks` and override only the callbacks you need:

- `on_agent_start` / `on_agent_end`
- `on_model_start` / `on_model_end`
- `on_tool_start` / `on_tool_end` / `on_tool_error`
- `on_approval`
- `on_handoff`
- `on_error`

Attach hooks to `Agent(..., hooks=[...])`, `AgentRuntime(hooks=[...])`, or one call through `run_agent(..., hooks=[...])`. Entry callbacks run from runtime/call hooks to agent hooks; completion and error callbacks unwind in reverse. Model hooks cover each physical `LanguageModel.generate(...)` or completed `LanguageModel.stream(...)` call. Realtime connect/event traffic does not emit model hooks.

Hooks observe lifecycle decisions but do not authorize tools, replace `AgentEvent`, or create spans. Approval policy remains authoritative, `AgentEvent` remains the ordered stream/history contract, and `AgentObserver` remains the tracing boundary. Hook failures propagate through the surrounding lifecycle: agent/model hook failures fail the run, while tool hook failures use the normal tool-error path. Keep hooks bounded and idempotent.

Run middleware wraps the complete root run:

```python
from zhivex_ai import AgentRunRequest


async def tenant_boundary(request: AgentRunRequest, call_next):
    if request.deps is None:
        raise PermissionError("Missing tenant dependencies.")
    return await call_next(request)


result = await run_agent(agent=agent, prompt="Review A-42.", deps=deps, middleware=[tenant_boundary])
```

Runtime middleware is outermost, then call middleware, then agent middleware. Middleware may update the request and must call `call_next(request)` exactly once unless it intentionally returns a cached `AgentRunResult`. Agent middleware applies to a root run; lifecycle run-hooks propagate to direct handoffs and child subagent runs.

## Next Steps

- Human-in-the-loop suspension and resume: [agents/approvals.md](./agents/approvals.md)
- Memory, checkpoints, and production Postgres state: [agents/durable-state.md](./agents/durable-state.md)
- Local, remote, and MCP registries: [agents/tool-registries.md](./agents/tool-registries.md)
- Declarative sequential, parallel, and loop orchestration: [WORKFLOWS.md](./WORKFLOWS.md)
- Backend production defaults: [PRODUCTION.md](./PRODUCTION.md)
- Security boundaries for tools, MCP, and skills: [../SECURITY.md](../SECURITY.md)

## Runtime Shape

`run_agent(...)` executes a complete loop and returns `AgentRunResult`. `stream_agent(...)` returns `AgentStreamResult`, which exposes the same loop as ordered events and can be collected into the final result. `resume_agent(...)` loads persisted session memory by `session_id`, attaches the latest checkpoint metadata when available, and appends new user input. `resume_agent_run(...)` resumes a suspended run after a pending tool approval is approved or denied.

The app owns request identity, user identity, long-term storage policy, approval UI, audit retention, and compliance decisions. The SDK owns orchestration primitives and normalized runtime events.

## Tools And Handoffs

Use `tool(...)` for local tools and `ToolRegistry` for local, remote, or MCP-backed tool execution. Tools receive `ToolExecutionContext`, including `run_id` and `tool_call_id`, when their callable accepts a `context` parameter. The context also carries an operation `idempotency_key` and optional `deadline_ms`. Use the idempotency key for every downstream write. If a tool exceeds its configured timeout, the runtime raises `ToolExecutionOutcomeUnknown` and stops instead of letting the model treat an uncertain external action as a normal tool failure.

See [Tool Registries And Permissions](./agents/tool-registries.md) before combining local, remote, or MCP tools.

For multi-agent work, use `handoff_to(...)` from a tool result or `create_subagent_tool(...)` for native subagent tools. Handoffs update `AgentTrace.orchestration_path` and emit handoff events.

`handoff_to(...)` is part of the stable direct-handoff contract. `create_subagent_tool(...)` and the higher-level native subagent-tool helpers remain beta.

## Human Approval

Local tools can set `requires_approval=True` and optional `permissions=[...]`. Attach an `approval_policy` to the agent, such as `permission_allowlist_approval_policy(...)`, or provide an app-owned async policy. Required approvals fail closed when no policy is configured. Return `ApprovalDecision.require_human(...)` to persist `AgentRunState(status="suspended")` with a `PendingApproval`; call `get_pending_agent_approvals(...)` and `resume_agent_run(...)` when the user responds. Built-in run stores atomically claim a pending approval before resume, so concurrent workers cannot execute the same approved tool twice.

Pending approvals also store a fingerprint of the tool schema, permissions, metadata, remote configuration, callable code, and captured public state. Resume fails closed if the registered tool changed. When one model response requests several tools and any call can suspend, the runtime executes that batch in order, preserves earlier results, and does not start later side effects past the approval boundary.

See [Human-In-The-Loop Approvals](./agents/approvals.md) for the complete suspension and resume flow.

Provider-managed approvals are beta and currently integrated for OpenAI and Azure OpenAI remote MCP approval flows. The runtime emits `AgentToolApprovalEvent`, appends the provider-specific approval response, and continues the loop.

## Persistence

Attach memory, checkpoint, and run stores independently:

- memory stores preserve session transcript/summary
- checkpoint stores preserve per-step checkpoint records and expose the latest checkpoint metadata through `resume_agent(...)`
- run stores support idempotency, replay, snapshots, cancellation tree helpers, and pending approvals

Built-in run stores claim `idempotency_key` atomically. A concurrent duplicate receives the already-claimed run identity, including its original session id, and does not call the model again. Custom production stores used with idempotent runs must implement the same atomic `claim_idempotency_key(...)` contract.

Checkpoint persistence removes remote/MCP credentials, sensitive URL credentials/query values, provider options, and raw provider responses. Checkpoints can still contain prompts, generated text, and non-secret tool data; apply your retention and tenant-isolation policy accordingly.

SQLite and in-memory stores are excellent for local development and tests and remain beta. Use Postgres stores for production backend persistence.

See [Durable Agent State](./agents/durable-state.md) for concrete store setup.

For release-candidate verification against a real provider, select configured providers and run `make smoke-agents`. The strict gate verifies an actual agent tool loop rather than generation alone. CI and publish workflows also run the Postgres integration suite whenever `ZHIVEX_TEST_POSTGRES_DSN` is configured.

## Failure Semantics

`run_agent(...)` raises for unrecoverable runtime failures. If `run_store` is configured, failed runs persist `status="failed"` and `error` metadata. Runs that require explicit human approval persist `status="suspended"` and return an `AgentRunResult` whose `state.pending_approvals` can be shown to an approval UI.

Denied local tool approvals are recorded as tool-result errors and the loop can continue. Guardrail tripwires, missing handoff targets, model errors, provider-managed approvals without a policy, tool errors that escape the tool loop, and max-step/max-handoff failures are runtime failures. Cancellation is store-level: `cancel_agent_run_tree(...)` marks persisted states as cancelled but does not interrupt already-running Python tasks unless the app coordinates that.

Replay helpers analyze stored `AgentRunState`; they do not re-execute providers.

## Streaming Events

`stream_agent(...)` emits ordered agent events:

- `run-start`
- `delegation-start` / `delegation-finish`
- `tool-call`, `tool-approval`, `tool-result`
- `guardrail`
- `text-delta`
- `checkpoint`
- `handoff-*`
- `finish` or `error`

Output guardrails can buffer text until checks pass, but tool lifecycle and approval events remain live so UIs can show progress.

If an application-supplied event callback raises, the runtime raises `AgentEventDeliveryError`. Check `durable_state_committed`: `False` means the run was persisted as failed before model execution continued; `True` means the terminal run state already won and must not be rewritten merely because delivery failed. Reconcile the event sink by `run_id` instead of retrying the agent action blindly.
