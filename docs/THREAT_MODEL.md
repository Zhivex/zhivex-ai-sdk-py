# Threat Model Notes

This document summarizes the main risk boundaries for production applications using the Zhivex AI SDK.

## Assets

- provider credentials and cloud credentials
- tenant prompts, uploaded files, retrieved documents, and generated output
- run stores, checkpoints, approval records, and trace artifacts
- MCP credentials and server configuration
- tools that can write data, call networks, execute code, browse, or access files

## Trust Boundaries

- application user to backend API
- backend API to provider adapter
- agent runtime to local tools
- agent runtime to remote MCP servers
- provider-hosted tool execution outside the application process
- observability export to external logging and tracing systems

## Hosted Tools

Hosted web search, file search, remote MCP, computer use, code execution, and provider toolsets can access external systems or produce provider-managed approval events. Treat these as opt-in capabilities.

Controls:

- allowlist providers, tools, and hosted tool classes per tenant
- persist provider approval requests and decisions
- record provider response IDs when available
- disable hosted tools for regulated workflows until reviewed

## Remote MCP

Remote MCP extends the tool boundary to another server. Risks include credential leakage, prompt injection through tool descriptions, overbroad network access, and unreviewed side effects.

Controls:

- pin server URLs and auth scopes
- review tool schemas before registration
- require approval for write, network, file, and external-side-effect actions
- isolate credentials per tenant or project

## File Access

File tools and hosted file search can expose sensitive local or uploaded content. Treat file access as a scoped capability, not a general permission.

Controls:

- restrict read roots and write roots
- avoid passing provider credentials or `.env` files to file tools
- redact file excerpts before logging
- store uploaded files under app-owned retention policy

## Shell-Like Capabilities

Shell, code execution, terminal, deploy, HTTP request, and patch tools are high risk because they can mutate systems or exfiltrate data.

Controls:

- avoid exposing shell-like tools to untrusted prompts
- require human approval and role checks
- sandbox runtime filesystem and network access
- log command intent, arguments, request ID, run ID, and approval ID
- make destructive operations explicitly idempotent

## Data Retention

Trace artifacts, checkpoints, and logs can retain prompts, tool inputs, provider payloads, and generated output. Retain only what your support and compliance policy requires, and prefer redacted summaries when possible.

## Related Guides

- [../SECURITY.md](../SECURITY.md)
- [OPERATIONS.md](./OPERATIONS.md)
- [OBSERVABILITY.md](./OBSERVABILITY.md)
