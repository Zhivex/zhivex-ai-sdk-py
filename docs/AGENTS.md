# Agent Runtime Guide

The agent runtime is the production-oriented orchestration layer for stateful assistants, tools, handoffs, approvals, persistence, replay, and traces.

Use the public imports from `zhivex_ai`:

```python
from zhivex_ai import Agent, create_agent_session, run_agent, stream_agent, resume_agent
```

## Stable Core

The stable core is:

- `Agent`, `AgentSession`, `AgentRuntime`, `AgentRegistry`, and `ToolRegistry`
- `run_agent(...)`, `stream_agent(...)`, and `resume_agent(...)`
- session helpers such as `create_agent_session(...)` and `load_agent_session(...)`
- portable skills and MCP discovery/registry helpers
- Postgres memory and checkpoint stores

The beta layer includes run stores, checkpoint events, replay/evaluation, trace artifacts, safety policies, provider-managed approvals, in-memory/SQLite stores, packaged skills, and workflow agents. Live/realtime agent APIs are experimental.

## Runtime Shape

`run_agent(...)` executes a complete loop and returns `AgentRunResult`. `stream_agent(...)` exposes the same loop as ordered events and can be collected into the final result. `resume_agent(...)` loads memory and the latest checkpoint by `session_id`, then appends new user input.

The app owns request identity, user identity, long-term storage policy, approval UI, audit retention, and compliance decisions. The SDK owns orchestration primitives and normalized runtime events.

## Tools And Handoffs

Use `tool(...)` for local tools and `ToolRegistry` for local, remote, or MCP-backed tool execution. Tools receive `ToolExecutionContext`, including `run_id` and `tool_call_id`, when their callable accepts a `context` parameter.

For multi-agent work, use `handoff_to(...)` from a tool result or `create_subagent_tool(...)` for native subagent tools. Handoffs update `AgentTrace.orchestration_path` and emit handoff events.

## Human Approval

Local tools can set `requires_approval=True` and optional `permissions=[...]`. Attach an `approval_policy` to the agent, such as `permission_allowlist_approval_policy(...)`, or provide an app-owned async policy.

Provider-managed approvals are beta and currently integrated for OpenAI and Azure OpenAI remote MCP approval flows. The runtime emits `AgentToolApprovalEvent`, appends the provider-specific approval response, and continues the loop.

## Persistence

Attach memory, checkpoint, and run stores independently:

- memory stores preserve session transcript/summary
- checkpoint stores support `resume_agent(...)`
- run stores support idempotency, replay, snapshots, and cancellation tree helpers

SQLite and in-memory stores are excellent for local development and tests. Use Postgres stores for production backend persistence.

## Failure Semantics

`run_agent(...)` raises for unrecoverable runtime failures. If `run_store` is configured, failed runs persist `status="failed"` and `error` metadata.

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
