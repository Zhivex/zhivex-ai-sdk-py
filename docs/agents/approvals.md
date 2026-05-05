# Human-In-The-Loop Approvals

Tools can request approval:

```python
from zhivex_ai import permission_allowlist_approval_policy, tool

lookup = tool(
    name="lookup",
    schema=dict[str, str],
    execute=lambda input: {"ok": True},
    permissions=["project:read"],
    requires_approval=True,
)
```

Attach `approval_policy` to the agent. The policy can allow, deny, or return an approval decision with a reason.

The SDK emits `AgentToolApprovalEvent` and then either executes the tool or raises for denied approvals. Durable approval queues, user prompts, authorization, and audit retention are application-owned.

Provider-managed approvals are beta. They currently cover OpenAI and Azure OpenAI remote MCP approval payloads and reuse the same `approval_policy` hook.
