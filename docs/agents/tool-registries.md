# Tool Registries And Permissions

`ToolRegistry` keeps tool definitions and runtimes together. Use it when tools come from several sources or when MCP sessions need lifecycle management.

Supported runtime sources:

- local callables
- remote HTTP tools
- MCP tools

Use permissions on tool definitions and enforce them through `approval_policy`:

```python
from zhivex_ai import ToolRegistry, permission_allowlist_approval_policy, tool

registry = ToolRegistry()
registry.register(
    tool(
        name="lookup",
        schema=dict[str, str],
        execute=lambda input: {"ok": True},
        permissions=["project:read"],
        requires_approval=True,
    )
)

agent.approval_policy = permission_allowlist_approval_policy("project:read")
```

Use `async with registry` or `await registry.aclose()` when MCP-backed tools are registered.
