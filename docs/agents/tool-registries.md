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

MCP annotations are untrusted hints. Every discovered tool requires approval by default, even when a server declares `readOnlyHint=true` and `destructiveHint=false`. An MCP call result with `isError=true` becomes a failed tool result instead of being returned as successful output.

After authenticating, pinning, and reviewing a server, an application can explicitly trust exact remote tool names. The names are matched before prefixing or `snake_case` conversion:

```python
registry = await create_mcp_tool_registry(
    server,
    trusted_tools={"read_file", "list_directory"},
)
```

`trusted_tools` bypasses per-call approval for only those names; it is not inferred from MCP annotations. Keep this allowlist narrow and owned by application configuration. Remote HTTP tools also require approval by default. Passing `requires_approval=False` to `remote_tool(...)` is an explicit application trust decision and should only be used for authenticated, reviewed, idempotent endpoints.
