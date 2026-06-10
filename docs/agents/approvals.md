# Human-In-The-Loop Approvals

Tools can request approval:

```python
from zhivex_ai import ApprovalDecision, get_pending_agent_approvals, resume_agent_run, tool

lookup = tool(
    name="lookup",
    schema=dict[str, str],
    execute=lambda input: {"ok": True},
    permissions=["project:read"],
    requires_approval=True,
)
```

Attach `approval_policy` to the agent. The policy can allow, deny, or return an approval decision with a reason.

Return `ApprovalDecision.require_human(...)` when a tool call must wait for a person. With a run store configured, the SDK persists `AgentRunState(status="suspended")` and a `PendingApproval` entry before the tool executes:

```python
async def approval_policy(request):
    if "billing:write" in request.tool_permissions:
        return ApprovalDecision.require_human("Manager approval required.")
    return True
```

Use `get_pending_agent_approvals(store, run_id)` to load approval requests for a UI. When the user responds, call `resume_agent_run(agent=agent, run_id=run_id, approval_id=approval_id, approved=True)` to execute the pending tool and continue the model loop. Pass `approved=False` to resume with a denied tool-result error.

The SDK owns the durable run-state queue, event stream, and resume mechanics. Your application still owns user identity, authorization checks, notification delivery, approval UI, and audit retention.

Provider-managed approvals are beta. They currently cover OpenAI and Azure OpenAI remote MCP approval payloads and reuse the same `approval_policy` hook.
