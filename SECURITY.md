# Security

This guide describes the SDK security boundary for production integrators. The SDK provides agent orchestration, provider adapters, safety policies, redaction helpers, budget guards, approval hooks, MCP integration, and hosted-tool wiring. Your application owns authentication, authorization, tenant isolation, durable audit logs, and data-retention policy.

## Reporting Security Issues

Use the repository private advisory flow or the maintainer security channel used for your deployment. Do not open public issues with secrets, exploit payloads, tenant data, provider keys, MCP credentials, or prompt transcripts from real users.

## Secrets

- Store provider API keys, cloud credentials, MCP bearer tokens, database DSNs, and webhook secrets in your secret manager.
- Never put secrets in prompts, tool inputs, agent metadata, trace artifacts, approval records, or example fixtures.
- Use environment variables only as a process boundary, not as an audit store.
- Apply `create_redaction_policy(...)` before logging prompts, tool payloads, provider payloads, approval records, or trace summaries.
- Treat provider response bodies in errors as sensitive unless your app has already redacted and classified them.

## Data Retention

The SDK does not decide how long to retain prompts, tool inputs, generated text, uploaded files, run snapshots, checkpoints, traces, or approval records. Production services should define:

- which fields are persisted
- retention duration by data class
- delete/export behavior for user data
- access controls for run stores and trace artifacts
- encryption and backup policy for Postgres, object storage, and external observability sinks

Use summaries or redacted trace artifacts when full prompt retention is not required for support.

## Tool Execution

Local tools run inside your application process. A tool can read data, write data, call networks, delete records, or trigger external side effects if you give it that capability. Secure defaults for production:

- register only named tools needed by the workflow
- attach permission tags such as `read`, `write`, `filesystem`, `network`, `shell`, `code-execution`, or `external-side-effect`
- require approval for write, network, filesystem, shell-like, code-execution, deploy, and delete operations
- use `create_safety_policy(preset="review_sensitive")` or stricter for user-facing agents
- keep approval queues, role checks, escalation, and audit records in app-owned storage
- require human approval for tools that can mutate systems or disclose tenant data
- make destructive tools idempotent and auditable

Do not expose broad shell, filesystem, deployment, or HTTP request tools to untrusted prompts without an application sandbox and approval gate.

## MCP And Hosted Tools

Remote MCP servers and provider-hosted tools can move execution outside your process. Treat them as third-party integrations with their own trust boundary.

- Prefer allowlisted MCP servers and pinned server configuration.
- Scope MCP credentials to the minimum tenant, project, and action set.
- Review tool schemas before allowing provider-managed approvals.
- Persist provider approval requests and user decisions when a hosted tool can access remote data.
- Disable or gate remote MCP, computer use, code execution, file search, and toolsets for tenants that have not accepted those risks.
- Keep request IDs, session IDs, run IDs, and provider response IDs in audit records.

## Threat Model Notes

High-risk capability classes include:

- hosted tools that can browse, search files, call remote MCP, use computer control, or execute code
- local tools with filesystem, shell-like, network, deploy, delete, or external-side-effect permissions
- traces, checkpoints, and run stores that retain prompts or tool payloads
- provider error bodies that may include snippets of request data
- examples or smoke scripts that load live credentials from the environment

The SDK helps identify and gate these paths with approval policies, redaction policies, budget guards, run limits, and observability hooks. It does not replace application security review, tenant authorization, network egress controls, or runtime sandboxing.

## Related Guides

- [docs/PRODUCTION.md](./docs/PRODUCTION.md)
- [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md)
- [docs/OPERATIONS.md](./docs/OPERATIONS.md)
- [docs/THREAT_MODEL.md](./docs/THREAT_MODEL.md)
