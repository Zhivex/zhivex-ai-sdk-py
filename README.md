# Zhivex AI SDK for Python

[![CI](https://img.shields.io/github/actions/workflow/status/Zhivex/zhivex-ai-sdk-py/ci.yml?branch=main&label=CI)](https://github.com/Zhivex/zhivex-ai-sdk-py/actions)
[![PyPI](https://img.shields.io/pypi/v/zhivex-ai-sdk)](https://pypi.org/project/zhivex-ai-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/zhivex-ai-sdk)](https://pypi.org/project/zhivex-ai-sdk/)
[![License](https://img.shields.io/pypi/l/zhivex-ai-sdk)](./LICENSE)

Zhivex AI SDK for Python is an async-first, agent-first SDK for building orchestrated AI systems across multiple providers.

It brings the same design goals as the TypeScript Zhivex AI SDK into Python:

- one agent runtime with executable handoffs, shared sessions, memory summaries, approval policies, tool registries, and traces
- one normalized foundation layer for text generation, streaming, tools, structured output, embeddings, audio, grounded text, and routing
- thin provider adapters instead of provider-specific app logic everywhere
- portable application code that can switch models and vendors with minimal changes

## Why Zhivex AI SDK

Modern AI apps usually start simple and then drift into provider lock-in:

- OpenAI requests look one way
- Anthropic uses a different message format
- Gemini and Vertex differ again
- local and routed setups add yet another layer

Zhivex AI SDK gives you a common agent runtime and model contract so your application code can stay stable while providers change underneath.

## Highlights

- Agent runtime with executable handoffs, registry-based orchestration, transcript + summary memory, permission-aware tool execution, and traces
- `AgentRuntime`, `AgentRegistry`, and `ToolRegistry` as the primary orchestration layer
- Unified `generate_text()` and `stream_text()` foundation primitives
- Structured output with `generate_object()` and `stream_object()`
- Grounded text for providers with web search support
- Audio transcription and speech generation where the provider supports it
- Experimental realtime/live voice sessions plus `stream_live_agent()` for voice-first agents
- Embeddings support where the provider supports it
- Provider factories for hosted and local models
- Gateway routing with fallback support
- HTTP/UI helpers for SSE, plain text streams, and UI message transport
- Middleware for telemetry, caching, and circuit breaking
- Model catalog helpers for cost and recommendation metadata

## Supported Providers

| Provider | Text | Streaming | Tools | Structured Output | Embeddings | Audio In | Audio Out | Grounded Text | Realtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenAI | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Experimental |
| Azure OpenAI | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Experimental |
| Anthropic | Yes | Yes | Yes | Prompted fallback | No | No | No | No | No |
| Gemini | Yes | Yes | Yes | Yes | Yes | No | No | No | Experimental |
| Vertex AI | Yes | Yes | Yes | Yes | Yes | No | No | No | Experimental |
| Bedrock | Yes | No | No | No | No | No | No | No | Experimental |
| OpenRouter | Yes | Yes | Yes | Yes | Yes | No | No | No | No |
| Qwen | Yes | Yes | Yes | Yes | Yes | No | No | No | No |
| Kimi | Yes | Yes | Yes | Yes | Yes | No | No | No | No |
| Ollama | Yes | Yes | Yes | Yes | Yes | No | No | No | No |

### Tool Calling Notes

Tool support varies in practice even when the high-level SDK API is shared:

- OpenAI and Azure OpenAI currently have the most robust tool-calling path in this SDK, including MCP-oriented schema normalization for strict-mode function tools.
- Gemini and Vertex AI support tools through the Gemini function-calling path. The SDK preserves Gemini thought signatures across tool loops and normalizes MCP schemas to the subset Gemini accepts.
- Anthropic supports tools, but when using extended thinking (`reasoning.budget_tokens`) the SDK only allows `tool_choice="auto"` or `"none"` and preserves returned thinking blocks during tool loops.
- OpenRouter supports the shared Responses-style adapter, but this SDK does not allow `tool_choice="required"` there because the current OpenRouter Responses docs only document `auto`, `none`, or forcing a named function.
- Qwen is available for text generation through the OpenAI-compatible factory, but tool calling is not currently exposed through this SDK adapter because the implementation path here is Responses-based while the official Qwen docs for tools currently describe a different OpenAI-compatible flow.
- Kimi and Ollama use the shared OpenAI-compatible adapter in this SDK. Basic compatibility may work, but provider-specific tool behavior can still differ from OpenAI depending on the upstream compatibility layer.

## Installation

```bash
pip install zhivex-ai-sdk
```

Optional extras:

```bash
pip install "zhivex-ai-sdk[postgres]"
pip install "zhivex-ai-sdk[mcp]"
```

## Quick Start

```python
import asyncio

from zhivex_ai import Agent, create_in_memory_agent_memory_store, create_openai, run_agent


async def main() -> None:
    openai = create_openai()
    agent = Agent(
        name="assistant",
        instructions="Be concise and remember prior turns.",
        model=openai("gpt-4o-mini"),
        memory=create_in_memory_agent_memory_store(),
    )

    first = await run_agent(agent=agent, prompt="Remember that project Apollo is important.")
    second = await run_agent(agent=agent, session=first.session, prompt="What project did I mention?")

    print(second.text)


asyncio.run(main())
```

## Foundation APIs

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

### Structured output streaming

```python
import asyncio

from pydantic import BaseModel

from zhivex_ai import create_openai, stream_object


class Recipe(BaseModel):
    title: str
    servings: int


async def main() -> None:
    openai = create_openai()
    result = stream_object(
        model=openai("gpt-4o-mini"),
        prompt="Return a compact JSON recipe.",
        schema=Recipe,
    )

    async for partial in result.partial_object_stream():
        print(partial)

    final = await result.collect()
    print(final.object.model_dump())


asyncio.run(main())
```

### Grounded text

```python
import asyncio

from zhivex_ai import create_openai, generate_grounded_text


async def main() -> None:
    openai = create_openai()

    result = await generate_grounded_text(
        model=openai.grounded_language_model("gpt-4o-search-preview"),
        prompt="Find one recent fact about AI infrastructure.",
    )

    print(result.text)
    for source in result.sources:
        print(source.title, source.url)


asyncio.run(main())
```

Gemini search is explicit and opt-in:

```python
import asyncio

from zhivex_ai import create_gemini, generate_grounded_text


async def main() -> None:
    gemini = create_gemini()

    result = await generate_grounded_text(
        model=gemini.grounded_language_model("gemini-2.5-flash"),
        prompt="Find one recent fact about AI infrastructure.",
    )

    print(result.text)
    for source in result.sources:
        print(source.title, source.url)


asyncio.run(main())
```

### Audio

```python
import asyncio
from pathlib import Path

from zhivex_ai import AudioInput, create_openai, transcribe_audio


async def main() -> None:
    openai = create_openai()
    audio = AudioInput(
        data=Path("sample.wav").read_bytes(),
        media_type="audio/wav",
        filename="sample.wav",
    )

    result = await transcribe_audio(
        model=openai.transcription_model("gpt-4o-mini-transcribe"),
        audio=audio,
    )

    print(result.text)


asyncio.run(main())
```

### Agent runtime

```python
import asyncio
from pydantic import BaseModel, ConfigDict

from zhivex_ai import (
    Agent,
    create_openai,
    handoff_to,
    run_agent,
    tool,
)


class DelegateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str


async def main() -> None:
    openai = create_openai()
    researcher = Agent(
        name="researcher",
        instructions="Answer delegated research questions directly.",
        model=openai("gpt-4o-mini"),
    )
    triage = Agent(
        name="triage",
        instructions="Delegate research work to the researcher agent.",
        model=openai("gpt-4o-mini"),
        tools={
            "delegate": tool(
                name="delegate",
                schema=DelegateInput,
                execute=lambda input: handoff_to("researcher", input=input.task),
            )
        },
        subagents={"researcher": researcher},
    )

    result = await run_agent(agent=triage, prompt="Research the Apollo migration status.")

    print(result.text)
    print(result.orchestration_path)


asyncio.run(main())
```

If you want Gemini research with web search in an agent run, opt in explicitly:

```python
result = await run_agent(
    agent=triage,
    prompt="Research the Apollo migration status.",
    provider_options={"google_search": True},
)
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

Adapters may also expose optional factories such as:

- `provider.embedding_model("text-embedding-3-small")`
- `provider.transcription_model("gpt-4o-mini-transcribe")`
- `provider.speech_model("gpt-4o-mini-tts")`
- `provider.grounded_language_model("gpt-4o-search-preview")`
- `provider.realtime_model("gpt-realtime")`

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

## UI And Transport

The Python SDK now includes helpers for UI and transport-oriented flows:

- `to_ui_message(...)`, `to_ui_messages(...)`
- `from_ui_message(...)`, `from_ui_messages(...)`
- `serialize_ui_message(...)`, `deserialize_ui_message(...)`
- `parse_ui_message_request(...)`
- `create_ui_message_json_response(...)`
- `create_ui_message_lines_response(...)`
- `to_sse_stream(...)`, `to_sse_response(...)`
- `to_text_stream_response(...)`
- `to_ui_message_stream_response(...)`

These are useful when wiring the SDK into web servers, SSE endpoints, or custom chat frontends.

## Agents

The Python SDK now exposes an agent-first runtime on top of the core model contract:

- `Agent(...)`
- `AgentRuntime(...)`
- `AgentRegistry(...)`
- `ToolRegistry(...)`
- `AgentSession`
- `run_agent(...)`
- `resume_agent(...)`
- `stream_agent(...)`
- `create_in_memory_agent_memory_store()`
- `create_in_memory_checkpoint_store()`
- `create_sqlite_agent_memory_store(...)`
- `create_sqlite_checkpoint_store(...)`
- `create_postgres_agent_memory_store(...)`
- `create_postgres_checkpoint_store(...)`
- `create_otel_agent_observer()`
- `load_agent_session(...)`
- `ApprovalDecision`, `ToolApprovalRequest`
- `permission_allowlist_approval_policy(...)`
- `handoff_to(...)`
- `remote_tool(...)`
- `discover_mcp_tools(...)`
- `mcp_stdio_server(...)`
- `mcp_http_server(...)`
- `create_mcp_tool_registry(...)`

This layer is intended for stateful, tool-using, multi-agent assistants where you want executable handoffs, shared sessions, transcript + summary memory, approval hooks, and traces without rewriting the lower-level loop yourself.

For new MCP integrations, prefer the higher-level helpers:

```python
import asyncio

from zhivex_ai import Agent, create_mcp_tool_registry, create_openai, mcp_stdio_server, run_agent


async def main() -> None:
    tools = await create_mcp_tool_registry(
        mcp_stdio_server(
            name="fs",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "."],
        )
    )

    openai = create_openai()
    agent = Agent(
        name="assistant",
        instructions="Use the filesystem MCP tools when needed.",
        model=openai("gpt-4o-mini"),
        tools=tools,
    )

    result = await run_agent(agent=agent, prompt="List the Python files in the current directory.")
    print(result.text)


asyncio.run(main())
```

`discover_mcp_tools(...)` is still available when you want raw tool definitions or full control over prefixes and registry composition.

## Examples

See [examples/README.md](./examples/README.md) for the full list. Highlights:

- [openai_text.py](./examples/openai_text.py)
- [agent_basic.py](./examples/agent_basic.py)
- [stream_agent.py](./examples/stream_agent.py)
- [resume_agent.py](./examples/resume_agent.py)
- [remote_tool.py](./examples/remote_tool.py)
- [mcp_tools.py](./examples/mcp_tools.py)
- [stream_text.py](./examples/stream_text.py)
- [stream_object.py](./examples/stream_object.py)
- [messages_and_tools.py](./examples/messages_and_tools.py)
- [embeddings.py](./examples/embeddings.py)
- [grounded_text.py](./examples/grounded_text.py)
- [transcribe_audio.py](./examples/transcribe_audio.py)
- [generate_speech.py](./examples/generate_speech.py)
- [openai_realtime.py](./examples/openai_realtime.py)
- [azure_realtime.py](./examples/azure_realtime.py)
- [gemini_realtime.py](./examples/gemini_realtime.py)
- [bedrock_realtime.py](./examples/bedrock_realtime.py)
- [live_agent_realtime.py](./examples/live_agent_realtime.py)
- [ui_messages.py](./examples/ui_messages.py)
- [http_responses.py](./examples/http_responses.py)
- [middleware.py](./examples/middleware.py)
- [model_catalog.py](./examples/model_catalog.py)
- [gateway_fallback.py](./examples/gateway_fallback.py)

## License

MIT. See [LICENSE](./LICENSE).
