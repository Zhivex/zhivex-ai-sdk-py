# Zhivex AI SDK for Python

[![CI](https://img.shields.io/github/actions/workflow/status/Zhivex/zhivex-ai-sdk-py/ci.yml?branch=main&label=CI)](https://github.com/Zhivex/zhivex-ai-sdk-py/actions)
[![PyPI](https://img.shields.io/pypi/v/zhivex-ai-sdk)](https://pypi.org/project/zhivex-ai-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/zhivex-ai-sdk)](https://pypi.org/project/zhivex-ai-sdk/)
[![License](https://img.shields.io/pypi/l/zhivex-ai-sdk)](./LICENSE)

Zhivex AI SDK for Python is an async-first SDK for building LLM applications against multiple providers with one shared contract.

It brings the same design goals as the TypeScript Zhivex AI SDK into Python:

- one normalized interface for text generation, streaming, tools, structured output, embeddings, and routing
- thin provider adapters instead of provider-specific app logic everywhere
- portable application code that can switch models and vendors with minimal changes

## Why Zhivex AI SDK

Modern AI apps usually start simple and then drift into provider lock-in:

- OpenAI requests look one way
- Anthropic uses a different message format
- Gemini and Vertex differ again
- local and routed setups add yet another layer

Zhivex AI SDK gives you a common language model contract so your application code can stay stable while providers change underneath.

## Highlights

- Unified `generate_text()` and `stream_text()` primitives
- Structured output with `generate_object()`
- Tool execution across multiple model steps
- Embeddings support where the provider supports it
- Provider factories for hosted and local models
- Gateway routing with fallback support
- Middleware for telemetry, caching, and circuit breaking
- Model catalog helpers for cost and recommendation metadata

## Supported Providers

| Provider | Text | Streaming | Tools | Structured Output | Embeddings |
| --- | --- | --- | --- | --- | --- |
| OpenAI | Yes | Yes | Yes | Yes | Yes |
| Azure OpenAI | Yes | Yes | Yes | Yes | Yes |
| Anthropic | Yes | Yes | Yes | Prompted fallback | No |
| Gemini | Yes | Yes | Yes | Yes | Yes |
| Vertex AI | Yes | Yes | Yes | Yes | Yes |
| Bedrock | Yes | No | No | No | No |
| OpenRouter | Yes | Yes | Yes | Yes | Yes |
| Qwen | Yes | Yes | Yes | Yes | Yes |
| Kimi | Yes | Yes | Yes | Yes | Yes |
| Ollama | Yes | Yes | Yes | Yes | Yes |

## Installation

For local development:

```bash
pip install -e .
```

When published to PyPI, installation will look like:

```bash
pip install zhivex-ai-sdk
```

## Quick Start

```python
import asyncio

from zhivex_ai import create_openai, generate_text


async def main() -> None:
    openai = create_openai()

    result = await generate_text(
        model=openai("gpt-4o-mini"),
        prompt="Describe Zhivex AI SDK in one sentence.",
    )

    print(result.text)
    print(result.usage)


asyncio.run(main())
```

## Core API

### Text generation

```python
import asyncio

from zhivex_ai import create_anthropic, generate_text


async def main() -> None:
    anthropic = create_anthropic()

    result = await generate_text(
        model=anthropic("claude-3-5-sonnet"),
        system="Be concise and technical.",
        prompt="What is a provider adapter?",
    )

    print(result.text)


asyncio.run(main())
```

### Structured output

```python
import asyncio

from pydantic import BaseModel

from zhivex_ai import create_openai, generate_object


class Recipe(BaseModel):
    title: str
    difficulty: str


async def main() -> None:
    openai = create_openai()

    result = await generate_object(
        model=openai("gpt-4o-mini"),
        prompt="Return a compact JSON recipe summary.",
        schema=Recipe,
    )

    print(result.object.model_dump())


asyncio.run(main())
```

### Streaming

```python
import asyncio

from zhivex_ai import create_openai, stream_text


async def main() -> None:
    openai = create_openai()
    result = stream_text(
        model=openai("gpt-4o-mini"),
        prompt="Reply in two short sentences.",
    )

    async for chunk in result.text_stream():
        print(chunk, end="")

    final = await result.collect()
    print("\n", final.finish_reason)


asyncio.run(main())
```

### Gateway fallback routing

```python
import asyncio

from zhivex_ai import (
    GatewayConfig,
    GatewayMessage,
    GatewayModelTarget,
    create_anthropic,
    create_gateway,
    create_openai,
)


async def main() -> None:
    gateway = create_gateway(
        GatewayConfig(
            adapters={
                "openai": create_openai(),
                "anthropic": create_anthropic(),
            }
        )
    )

    result = await gateway.generate(
        messages=[GatewayMessage(role="user", content="Say hello in one sentence.")],
        primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
        fallbacks=[GatewayModelTarget(provider="anthropic", model_id="claude-3-5-sonnet")],
    )

    print(result.text)
    print(result.provider_used, result.model_used)


asyncio.run(main())
```

## Provider Factories

The package currently exposes:

- `create_openai()`
- `create_azure_openai()`
- `create_anthropic()`
- `create_gemini()`
- `create_vertex()`
- `create_bedrock()`
- `create_openrouter()`
- `create_qwen()`
- `create_kimi()`
- `create_ollama()`

OpenAI-compatible providers such as OpenRouter, Qwen, Kimi, and Ollama reuse the same normalized adapter model.

## Why not use provider SDKs directly?

Using provider SDKs directly is totally reasonable when:

- you only target one provider
- you are comfortable rewriting message, tool, and streaming logic per vendor
- you do not need fallback routing or a shared abstraction layer

Zhivex AI SDK is a better fit when:

- you want one contract across multiple model vendors
- you expect to switch providers over time
- you want tools, structured output, caching, telemetry, and routing to live above the provider layer
- you want application code that reads the same whether the model is OpenAI, Anthropic, Gemini, or local

## Middleware

Zhivex AI SDK includes middleware helpers similar to the TypeScript SDK:

- `wrap_language_model(...)`
- `create_telemetry_middleware(...)`
- `create_cached_generate_middleware(...)`
- `create_in_memory_generate_cache()`
- `create_file_generate_cache(...)`
- `create_circuit_breaker_middleware(...)`

These let you keep cross-cutting concerns outside provider adapters and application prompts.

## Examples

- [openai_text.py](./examples/openai_text.py)
- [structured_output.py](./examples/structured_output.py)
- [gateway_fallback.py](./examples/gateway_fallback.py)

## Project Status

This project is usable today, but still early.

Current status:

- core generation and streaming primitives are implemented
- major provider adapters are in place
- gateway, catalog, and middleware helpers are included
- test coverage exists for the shared contract and key adapters

What to expect:

- API polish may continue as the Python port matures
- provider-specific behavior may still expand over time
- GitHub Copilot SDK integration is not included yet

## Roadmap

Near-term release goals:

- first TestPyPI release
- first public PyPI release
- more adapter coverage tests for Gemini, Vertex, and Bedrock
- API polish and docs cleanup

Potential next additions:

- GitHub Copilot SDK integration
- richer provider capability metadata
- more middleware and transport helpers
- higher-level chat/session utilities

## Development

Run local validation with:

```bash
python3 -m compileall src tests examples
python3 -m unittest discover -s tests -v
```

## Publishing

The repository already includes:

- CI workflow: [ci.yml](./.github/workflows/ci.yml)
- TestPyPI workflow: [publish-testpypi.yml](./.github/workflows/publish-testpypi.yml)
- PyPI workflow: [publish-pypi.yml](./.github/workflows/publish-pypi.yml)
- release guide: [RELEASING.md](./RELEASING.md)

Before the first public release, confirm:

- the final package name on PyPI
- the initial version strategy, for example `0.1.0` vs `0.1.0a1`
- Trusted Publishing configuration on PyPI and TestPyPI

## License

MIT. See [LICENSE](./LICENSE).
