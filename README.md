# Zhivex AI SDK for Python

Build provider-agnostic AI applications, durable agents, and production workflows with one typed Python API.

[![PyPI](https://img.shields.io/pypi/v/zhivex-ai-sdk)](https://pypi.org/project/zhivex-ai-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/zhivex-ai-sdk)](https://pypi.org/project/zhivex-ai-sdk/)
[![CI](https://github.com/Zhivex/zhivex-ai-sdk-py/actions/workflows/ci.yml/badge.svg)](https://github.com/Zhivex/zhivex-ai-sdk-py/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/Zhivex/zhivex-ai-sdk-py)](./LICENSE)

> This is a beta package. Stable APIs, experimental features, and compatibility commitments are documented explicitly in [STABILITY.md](./STABILITY.md).

## Why Zhivex

- One portable contract for OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex, Qwen, Kimi, DeepSeek, and vLLM.
- Durable agent runs with tools, approvals, resumable state, idempotency, and streaming.
- Typed structured output, embeddings, media, hosted tools, MCP, and provider-native extensions.
- Production controls for retries, budgets, observability, gateway routing, and secure tool execution.
- A matching [TypeScript SDK](https://github.com/Zhivex/zhivex-ai-sdk) for cross-language teams.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install zhivex-ai-sdk
```

```python
import asyncio
import os

from zhivex_ai import create_openai


async def main() -> None:
    provider = create_openai(api_key=os.environ["OPENAI_API_KEY"])
    result = await provider.generate_text(
        model="gpt-5-mini",
        prompt="Explain idempotency in one sentence.",
    )
    print(result.text)


asyncio.run(main())
```

Start with [docs/QUICKSTART.md](./docs/QUICKSTART.md), then choose a provider in [docs/PROVIDERS.md](./docs/PROVIDERS.md).

## Agents

```python
import asyncio
import os

from zhivex_ai import Agent, create_openai, run_agent, tool


@tool
def get_status(service: str) -> str:
    """Return a sample service status."""
    return f"{service}: healthy"


async def main() -> None:
    provider = create_openai(api_key=os.environ["OPENAI_API_KEY"])
    agent = Agent(
        provider=provider,
        model="gpt-5-mini",
        instructions="Be concise and use tools when helpful.",
        tools=[get_status],
    )
    result = await run_agent(agent, "Check the payments service.")
    print(result.output_text)


asyncio.run(main())
```

See [docs/AGENTS.md](./docs/AGENTS.md) for approvals and durable state, and [docs/WORKFLOWS.md](./docs/WORKFLOWS.md) for resumable orchestration.

## MCP and Packaged Skills

Filesystem and other sensitive tools should be scoped and approval-gated:

```python
from zhivex_ai import (
    MCPServerStdio,
    READ_ONLY_FILESYSTEM_TOOLS,
)

filesystem = MCPServerStdio(
    command="bunx",
    args=[
        "@modelcontextprotocol/server-filesystem@2026.7.10",
        ".",
    ],
    allowed_tools=READ_ONLY_FILESYSTEM_TOOLS,
    approval_policy="always",
)
```

The beta skill-package layer can install and validate reusable agent skills. Document generation is optional:

```bash
pip install "zhivex-ai-sdk[docx]"
```

Experimental realtime/live voice sessions plus `stream_live_agent()` are available behind the documented experimental surface.

## Provider Support

The generated tables below are sourced from runtime metadata and checked in CI.

<!-- BEGIN GENERATED SUPPORT MATRIX -->
### Tier-1 Providers

These providers back the stable surface for production API work in this SDK today:

- `openai`
- `anthropic`
- `azure-openai`
- `gemini`
- `vertex`
- `qwen`
- `kimi`
- `deepseek`
- `vllm`

### Portable Support

| Provider | Tier | Portable Badge | Text | Streaming | Structured Output | Tools | Embeddings | Grounding | Retrieval | Transcription | Speech |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic | portable | Yes | Yes | Yes | Yes | Yes | No | Yes | Yes | No | No |
| azure-openai | portable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| bedrock | native-only | No | Yes | No | No | No | No | No | Yes | No | No |
| deepseek | portable | Yes | Yes | Yes | Yes | Yes | No | No | No | No | No |
| gemini | portable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| kimi | portable | Yes | Yes | Yes | Yes | Yes | No | No | No | No | No |
| ollama | compatibility | No | Yes | Yes | Yes | Yes | Yes | No | Yes | No | No |
| openai | portable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| openrouter | native-only | No | Yes | Yes | Yes | Yes | Yes | No | Yes | No | Yes |
| qwen | portable | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | No |
| vertex | portable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| vllm | portable | Yes | Yes | Yes | Yes | Yes | Yes | No | Yes | Yes | No |

### Native Extras

| Provider | Text | Streaming | Structured Output | Tools | Files | File Search | Images | Uploads | Moderations | Batches | Videos | Media | Interactions | Containers | Skills | Realtime | Responses | Conversations | Caches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic | Yes | Yes | Yes | Yes | Yes | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| azure-openai | Yes | Yes | Yes | Yes | No | Yes | No | No | No | No | No | No | No | No | No | Yes | Yes | Yes | No |
| bedrock | Yes | Yes | No | Yes | No | No | No | No | No | No | No | No | No | No | No | Yes | No | No | No |
| deepseek | Yes | Yes | Yes | Yes | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| gemini | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | No | Yes | Yes | Yes | Yes | No | No | Yes | No | No | Yes |
| kimi | Yes | Yes | Yes | Yes | Yes | No | No | No | No | Yes | No | No | No | No | No | No | No | No | No |
| ollama | Yes | Yes | Yes | Yes | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| openai | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | Yes | Yes | Yes | Yes | Yes | No |
| openrouter | Yes | Yes | Yes | Yes | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| qwen | Yes | Yes | Yes | Yes | Yes | No | No | No | No | Yes | No | No | No | No | No | No | Yes | No | No |
| vertex | Yes | Yes | Yes | Yes | No | No | Yes | No | No | No | Yes | Yes | No | No | No | Yes | No | No | No |
| vllm | Yes | Yes | Yes | Yes | No | No | No | No | No | No | No | No | No | No | No | Yes | No | No | No |

### Agent Capabilities

| Provider | Support Tier | Tool Choice None | Approval Requests | Hosted Web Search | Hosted File Search | Remote MCP | Computer Use | Code Execution | Toolsets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic | tier-b | Yes | No | Yes | No | No | No | Yes | Yes |
| azure-openai | tier-a | Yes | Yes | Yes | Yes | Yes | Yes | No | No |
| bedrock | tier-b | Yes | No | No | No | No | No | No | No |
| deepseek | tier-b | Yes | No | No | No | No | No | No | No |
| gemini | tier-b | Yes | No | Yes | Yes | No | Yes | Yes | No |
| kimi | tier-b | Yes | No | No | No | No | No | No | Yes |
| ollama | tier-c | No | No | No | No | No | No | No | No |
| openai | tier-a | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No |
| openrouter | tier-c | Yes | No | Yes | No | No | No | No | No |
| qwen | tier-b | Yes | No | Yes | Yes | Yes | No | Yes | No |
| vertex | tier-b | Yes | No | Yes | Yes | No | Yes | Yes | No |
| vllm | tier-b | Yes | No | No | No | No | No | No | No |
<!-- END GENERATED SUPPORT MATRIX -->

For full stability and maturity details, see [docs/PARITY_MATRIX.md](./docs/PARITY_MATRIX.md) and [PRODUCTION_APIS.md](./PRODUCTION_APIS.md).

## Production Guides

| Need | Guide |
| --- | --- |
| Gateway routing and BYOK | [docs/GATEWAY.md](./docs/GATEWAY.md) |
| Observability and OpenTelemetry | [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md) |
| Operations and failure handling | [docs/OPERATIONS.md](./docs/OPERATIONS.md) |
| Production deployment | [docs/PRODUCTION.md](./docs/PRODUCTION.md) |
| Troubleshooting | [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md) |
| Security reporting | [SECURITY.md](./SECURITY.md) |

## Project Policy

- API stability: [STABILITY.md](./STABILITY.md)
- Versioning: [VERSIONING.md](./VERSIONING.md)
- Supported versions: [SUPPORT.md](./SUPPORT.md)
- Release history: [CHANGELOG.md](./CHANGELOG.md)
- Contributions: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Roadmap: [ROADMAP.md](./ROADMAP.md)

## Development

```bash
uv venv .venv
uv pip install -e ".[dev,postgres,mcp]"
make check
```

Security issues should be reported privately according to [SECURITY.md](./SECURITY.md), not in a public issue.

## License

[MIT](./LICENSE)
