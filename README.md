# Zhivex AI SDK for Python

[![CI](https://img.shields.io/github/actions/workflow/status/Zhivex/zhivex-ai-sdk-py/ci.yml?branch=main&label=CI)](https://github.com/Zhivex/zhivex-ai-sdk-py/actions)
[![PyPI](https://img.shields.io/pypi/v/zhivex-ai-sdk)](https://pypi.org/project/zhivex-ai-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/zhivex-ai-sdk)](https://pypi.org/project/zhivex-ai-sdk/)
[![License](https://img.shields.io/pypi/l/zhivex-ai-sdk)](./LICENSE)

Zhivex AI SDK for Python is an async-first, agent-first SDK for building orchestrated AI systems across multiple providers.

Production maturity plan: see [MATURITY_PLAN.md](./MATURITY_PLAN.md).

It brings the same design goals as the TypeScript Zhivex AI SDK into Python:

- one agent runtime with executable handoffs, shared sessions, memory summaries, approval policies, tool registries, and traces
- one normalized foundation layer for text generation, streaming, tools, structured output, embeddings, audio, grounded text, and routing
- thin provider adapters instead of provider-specific app logic everywhere
- portable application code across the portable tier, with explicit native escape hatches for provider-specific features

## Why Zhivex AI SDK

Modern AI apps usually start simple and then drift into provider lock-in:

- OpenAI requests look one way
- Anthropic uses a different message format
- Gemini and Vertex differ again
- local and routed setups add yet another layer

Zhivex AI SDK gives you a common agent runtime and model contract so your application code can stay stable while providers change underneath.

## Stability And Support

Zhivex AI SDK is now published as a beta package with a documented stable surface, explicit stability levels, and versioning rules for downstream integrators.

Production integrations should import supported APIs from `zhivex_ai`, prefer the documented stable surface and tier-1 providers, and isolate beta or experimental areas behind an application-owned service layer.

See [STABILITY.md](./STABILITY.md), [VERSIONING.md](./VERSIONING.md), [SUPPORT.md](./SUPPORT.md), and [CHANGELOG.md](./CHANGELOG.md) for the contract that governs public API expectations, support scope, and release communication.

## Highlights

- Agent runtime with executable handoffs, input/output guardrails, registry-based orchestration, transcript + summary memory, permission-aware tool execution, and traces
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

Provider factories now return a `ProviderBundle` with two explicit namespaces:

- `provider("model-id")` or `provider.portable.language_model("model-id")` for the strict portable contract
- `provider.native.language_model("model-id")` for provider-specific options, hosted tools, and escape hatches

Portable construction fails fast for providers that do not satisfy the portable contract. Those providers remain available through `provider.native`.

For production API work, the current tier-1 provider story for the stable surface is OpenAI, Anthropic, Azure OpenAI, Gemini, and Vertex. Other providers remain available, but their supported feature set should be evaluated against the matrix below and the stability definitions in [STABILITY.md](./STABILITY.md).

This matrix is generated from runtime support metadata via `scripts/generate_support_matrix.py`.

### Tier-1 Providers

These providers back the stable surface for production API work in this SDK today:

- `openai`
- `anthropic`
- `azure-openai`
- `gemini`
- `vertex`

### Portable Support

| Provider | Tier | Portable Badge | Text | Streaming | Structured Output | Tools | Embeddings | Grounding | Retrieval | Transcription | Speech |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic | portable | Yes | Yes | Yes | Yes | Yes | No | Yes | Yes | No | No |
| azure-openai | portable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| bedrock | native-only | No | Yes | No | No | No | No | No | Yes | No | No |
| gemini | portable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| kimi | compatibility | No | Yes | Yes | Yes | Yes | Yes | No | Yes | No | No |
| ollama | compatibility | No | Yes | Yes | Yes | Yes | Yes | No | Yes | No | No |
| openai | portable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| openrouter | native-only | No | Yes | Yes | Yes | Yes | Yes | No | Yes | No | Yes |
| qwen | compatibility | No | Yes | Yes | Yes | No | Yes | No | Yes | No | Yes |
| vertex | portable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

### Native Extras

| Provider | Files | File Search | Images | Uploads | Moderations | Batches | Containers | Skills | Realtime | Responses | Conversations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic | Yes | No | No | No | No | No | No | No | No | No | No |
| azure-openai | No | No | No | No | No | No | No | No | Yes | No | No |
| bedrock | No | No | No | No | No | No | No | No | Yes | No | No |
| gemini | Yes | Yes | No | No | No | No | No | No | Yes | No | No |
| kimi | No | No | No | No | No | No | No | No | No | No | No |
| ollama | No | No | No | No | No | No | No | No | No | No | No |
| openai | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| openrouter | No | No | No | No | No | No | No | No | No | No | No |
| qwen | No | No | No | No | No | No | No | No | No | No | No |
| vertex | No | No | No | No | No | No | No | No | Yes | No | No |

### Tool Calling Notes

Tool support now follows the same rule everywhere:

- The portable layer accepts only the SDK-owned contract. It rejects `provider_options` and any provider-managed tool payloads.
- Provider-specific hosted tools, raw Responses settings, Gemini built-in tools, and similar knobs must go through `provider.native.*`.
- Tier-1 support means the provider participates in the stable surface story, production API examples, and contract-level support assertions in this repository.
- Anthropic is part of the tier-1 text-generation story in this SDK. Extended thinking still restricts `tool_choice` to `auto` or `none`, and embeddings, transcription, and speech remain unavailable on the Anthropic provider path here.
- OpenAI, Anthropic, and Azure OpenAI currently cover the broadest production text-generation API paths in this SDK.
- Gemini and Vertex are portable for the core contract, but Gemini built-in tools such as `google_search`, `google_maps`, `url_context`, `code_execution`, `file_search`, and `computer_use` are native-only entrypoints.
- Bedrock and OpenRouter remain available, but only through `provider.native` until they satisfy the portable contract end to end.
- Qwen, Kimi, and Ollama are compatibility providers in this refactor. They remain usable through native adapters, but they do not receive the portable badge.
- Ollama defaults to `base_url="http://localhost:11434/v1"` and `api_key="ollama"` for local compatibility setups. Use `provider.native.*` for Ollama examples and override `OLLAMA_API_KEY` only when you front it with a proxy or remote gateway that requires auth.

## Installation

```bash
pip install zhivex-ai-sdk
```

Optional extras:

```bash
pip install "zhivex-ai-sdk[postgres]"
pip install "zhivex-ai-sdk[mcp]"
pip install "zhivex-ai-sdk[api]"
pip install "zhivex-ai-sdk[otel]"
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
        model=openai("gpt-5.4-mini"),
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

from zhivex_ai import create_openai, generate_text


async def main() -> None:
    openai = create_openai()

    result = await generate_text(
        model=openai("gpt-5.4-mini"),
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
        model=openai("gpt-5.4-mini"),
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
        model=openai("gpt-5.4-mini"),
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
        model=openai("gpt-5.4-mini"),
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
        model=openai.grounded_language_model("gpt-5.4-mini"),
        prompt="Find one recent fact about AI infrastructure.",
    )

    print(result.text)
    for source in result.sources:
        print(source.title, source.url)


asyncio.run(main())
```

Anthropic grounding is also available on the portable tier:

```python
import asyncio

from zhivex_ai import create_anthropic, generate_grounded_text


async def main() -> None:
    anthropic = create_anthropic()

    result = await generate_grounded_text(
        model=anthropic.grounded_language_model("claude-sonnet-4-20250514"),
        prompt="Find one recent fact about AI infrastructure.",
    )

    print(result.text)
    for source in result.sources:
        print(source.title, source.url)


asyncio.run(main())
```

Gemini and Vertex grounding are also explicit and opt-in:

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

### Files and multimodal input

`FilePart` is no longer PDF-only. Gemini accepts inline or uploaded files for documents, audio, images, and video. Vertex accepts inline files plus URI-based file references such as `gs://...`.

Inline document input:

```python
import asyncio

from zhivex_ai import FilePart, ModelMessage, TextPart, create_openai, generate_text


async def main() -> None:
    openai = create_openai()
    result = await generate_text(
        model=openai("gpt-5.4-mini"),
        messages=[
            ModelMessage(
                role="user",
                parts=[
                    FilePart(
                        data="JVBERi0xLjQK",
                        media_type="application/pdf",
                        filename="statement.pdf",
                    ),
                    TextPart(text="Summarize this PDF in three bullets."),
                ],
            )
        ],
    )

    print(result.text)


asyncio.run(main())
```

Reusing a previously uploaded Anthropic file through the native file flow:

```python
import asyncio

from zhivex_ai import FilePart, ModelMessage, create_anthropic, generate_text


async def main() -> None:
    anthropic = create_anthropic()
    result = await generate_text(
        model=anthropic.native.language_model("claude-sonnet-4-20250514"),
        messages=[
            ModelMessage(
                role="user",
                parts=[FilePart(file_id="file_123"), FilePart(file_id="file_456")],
            )
        ],
    )

    print(result.text)


asyncio.run(main())
```

Using the Gemini Files API first, then passing the returned reference:

```python
import asyncio

from zhivex_ai import FilePart, ModelMessage, TextPart, create_gemini, generate_text


async def main() -> None:
    gemini = create_gemini()
    uploaded = await gemini.files().upload(
        data=b"%PDF-1.4...",
        filename="statement.pdf",
    )

    result = await generate_text(
        model=gemini("gemini-2.5-flash"),
        messages=[
            ModelMessage(
                role="user",
                parts=[
                    FilePart(file_uri=uploaded.file_uri),
                    TextPart(text="Extract the key numbers from this statement."),
                ],
            )
        ],
    )

    print(result.text)


asyncio.run(main())
```

Counting tokens before sending a request:

```python
import asyncio

from zhivex_ai import create_gemini


async def main() -> None:
    gemini = create_gemini()
    counts = await gemini.tokens().count(
        model_id="gemini-2.5-flash",
        prompt="Summarize this in one line.",
    )

    print(counts.total_tokens)


asyncio.run(main())
```

Managing Gemini File Search stores:

```python
import asyncio

from zhivex_ai import create_gemini


async def main() -> None:
    gemini = create_gemini()
    store = await gemini.file_search_stores().create(display_name="Docs")
    operation = await gemini.file_search_stores().upload(
        file_search_store_name=store.name,
        data=b"%PDF-1.4...",
        filename="manual.pdf",
        media_type="application/pdf",
    )

    await gemini.file_search_stores().wait_operation(operation.name)


asyncio.run(main())
```

Managing OpenAI Vector Stores / File Search stores:

```python
import asyncio

from zhivex_ai import create_openai


async def main() -> None:
    openai = create_openai()
    store = await openai.file_search_stores().create(display_name="Docs")
    operation = await openai.file_search_stores().upload(
        file_search_store_name=store.name,
        data=b"%PDF-1.4...",
        filename="manual.pdf",
        media_type="application/pdf",
        custom_metadata=[{"key": "lang", "value": "en"}],
    )

    await openai.file_search_stores().wait_operation(operation.name)


asyncio.run(main())
```

Using OpenAI uploads and images:

```python
import asyncio

from zhivex_ai import create_openai


async def main() -> None:
    openai = create_openai()
    file = await openai.uploads().upload_bytes(
        data=b'{"messages":[{"role":"user","content":"hello"}]}\n',
        filename="batch.jsonl",
        mime_type="application/jsonl",
        purpose="batch",
    )
    image = await openai.images().generate(prompt="A paper sketch of a transit map", model="gpt-image-1")

    print(file.id)
    print(image.images[0].b64_json is not None)


asyncio.run(main())
```

Passing inline audio to Gemini text generation:

```python
import asyncio

from zhivex_ai import FilePart, ModelMessage, TextPart, create_gemini, generate_text


async def main() -> None:
    gemini = create_gemini()
    result = await generate_text(
        model=gemini("gemini-2.5-flash"),
        messages=[
            ModelMessage(
                role="user",
                parts=[
                    FilePart(
                        data="SUQzBAAAAAAA...",
                        media_type="audio/mpeg",
                        filename="call.mp3",
                    ),
                    TextPart(text="Summarize the call in five bullets."),
                ],
            )
        ],
    )

    print(result.text)


asyncio.run(main())
```

Passing a Vertex-hosted file reference:

```python
import asyncio

from zhivex_ai import FilePart, ModelMessage, TextPart, create_vertex, generate_text


async def main() -> None:
    vertex = create_vertex()
    result = await generate_text(
        model=vertex("gemini-2.5-flash"),
        messages=[
            ModelMessage(
                role="user",
                parts=[
                    FilePart(file_uri="gs://my-bucket/meeting.mp4", media_type="video/mp4"),
                    TextPart(text="Extract the main decisions from this meeting."),
                ],
            )
        ],
    )

    print(result.text)


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
        model=openai.transcription_model("gpt-4o-transcribe"),
        audio=audio,
    )

    print(result.text)


asyncio.run(main())
```

```python
import asyncio
import wave
from pathlib import Path

from zhivex_ai import create_gemini, generate_speech


def save_wave(path: Path, pcm: bytes, *, channels: int = 1, rate: int = 24_000, sample_width: int = 2) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm)


async def main() -> None:
    gemini = create_gemini()
    result = await generate_speech(
        model=gemini.speech_model("gemini-2.5-flash-preview-tts"),
        input="Zhivex AI SDK makes provider switching easier.",
        voice="Kore",
    )
    save_wave(Path("speech.wav"), result.audio)


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
        model=openai("gpt-5.4-mini"),
    )
    triage = Agent(
        name="triage",
        instructions="Delegate research work to the researcher agent.",
        model=openai("gpt-5.4-mini"),
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

If you want Gemini research with provider-native web search in an agent run, put the agent on a native Gemini model first:

```python
from zhivex_ai import create_gemini

gemini = create_gemini()
triage.model = gemini.native.language_model("gemini-3-flash-preview")
result = await run_agent(
    agent=triage,
    prompt="Research the Apollo migration status.",
    provider_options={"google_search": True},
)
```

Built-in Gemini tools can also be configured directly:

```python
result = await generate_text(
    model=gemini.native.language_model("gemini-3-flash-preview"),
    prompt="Research this page and show your work.",
    provider_options={
        "google_search": {"excludeDomains": ["example.com"]},
        "url_context": {},
        "code_execution": True,
    },
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
        primary=GatewayModelTarget(provider="openai", model_id="gpt-5.4-mini"),
        fallbacks=[GatewayModelTarget(provider="anthropic", model_id="claude-sonnet-4-20250514")],
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

Every factory now returns a `ProviderBundle`.

Portable usage:

- `provider("model-id")`
- `provider.language_model("model-id")`
- `provider.embedding_model(...)`
- `provider.transcription_model(...)`
- `provider.speech_model(...)`
- `provider.grounded_language_model(...)`

Native usage:

- `provider.native.language_model("model-id")`
- `provider.native.embedding_model(...)`
- `provider.native.transcription_model(...)`
- `provider.native.speech_model(...)`
- `provider.native.grounded_language_model(...)`
- `provider.native.realtime_model(...)`

Portable model construction fails fast when the provider does not hold the portable badge. That is intentional: the default path is the portability promise, and `provider.native` is the explicit escape hatch.

OpenAI-compatible providers such as OpenRouter, Qwen, Kimi, and Ollama still reuse the same normalized adapter model internally, but they no longer imply portable parity.

Local Ollama usage follows the same native escape hatch:

```python
import asyncio

from zhivex_ai import create_ollama, generate_text


async def main() -> None:
    provider = create_ollama(base_url="http://localhost:11434/v1")
    result = await generate_text(
        model=provider.native.language_model("llama3.2"),
        prompt="Explain Zhivex AI SDK in one sentence.",
    )
    print(result.text)


asyncio.run(main())
```

For local Ollama installs, `api_key="ollama"` is the default compatibility token and is usually sufficient. Set `OLLAMA_API_KEY` only when your Ollama endpoint is behind a proxy or hosted gateway that expects authentication.

Adapters may also expose optional factories such as:

- `provider.native.realtime_model("gpt-realtime")`
- `provider.files()`
- `provider.images()`
- `provider.uploads()`
- `provider.tokens()`
- `provider.file_search_stores()`
- `provider.responses()`
- `provider.conversations()`

OpenAI providers may additionally expose low-level lifecycle clients:

- `provider.responses()` for raw Responses API operations such as `create`, `create_background`, `retrieve`, `wait`, `cancel`, `compact`, and `list_input_items`
- `provider.conversations()` for raw Conversations API operations such as `create`, `retrieve`, `update`, `create_item`, and `list_items`
- `provider.file_search_stores()` for Vector Store / File Search lifecycle operations such as `create`, `update`, `search`, `upload`, `list_documents`, `get_operation`, and `wait_operation`
- `provider.images()` for standalone image generation and edit flows
- `provider.uploads()` for multi-part uploads that complete into reusable OpenAI Files
- `provider.moderations()` for raw Moderations API requests
- `provider.batches()` for raw Batch API lifecycle operations such as `create`, `retrieve`, `list`, and `cancel`
- `provider.containers()` for container lifecycle and container-file management
- `provider.skills()` for skill lifecycle and skill-version management

OpenAI helper builders cover the modern hosted-tool surface, including file search filters, code-interpreter containers, shell environments, MCP servers, inline skills, skill references, custom tools, namespaces, and tool search.

### Capability Matrix

The canonical matrix now lives in runtime metadata:

- `provider.portable_support`
- `provider.native_support`
- `provider.tier`

To regenerate the markdown tables used above:

```bash
uv run python scripts/generate_support_matrix.py
```

Notes:

- `provider("model-id")` is shorthand for the portable namespace.
- `provider.native.*` is the only place where provider-specific request shapes belong.
- Some providers support a capability only for specific model IDs even when the adapter exposes the factory.
- `create_gemini().files()` exposes the Gemini Files API. `create_vertex()` does not expose a hosted files client; on Vertex, pass `FilePart(file_uri="gs://...")` or inline media instead.
- `create_anthropic().tokens()` exposes Anthropic message token counting.
- `anthropic_web_search_tool(...)`, `anthropic_mcp_server(...)`, and `anthropic_code_execution_tool(...)` build the current hosted-tool payloads for Claude-native runs.
- `create_gemini().tokens()` and `create_vertex().tokens()` expose token counting clients.
- `create_gemini().file_search_stores()` exposes Gemini File Search store management.
- `create_openai().file_search_stores()` exposes OpenAI Vector Store / File Search management.
- `create_vertex()` now exports native grounding helpers such as `vertex_google_search_tool(...)`, `vertex_google_maps_tool(...)`, `vertex_vertex_ai_search_tool(...)`, and `vertex_external_search_tool(...)`.
- `create_vertex().native.language_model(...)` and `create_vertex().native.grounded_language_model(...)` also accept `provider_options={"vertex_ai_search": {...}}` and `provider_options={"external_search": {...}}`, which are normalized into the Vertex tool payloads automatically.
- `create_openai().images()` and `create_openai().uploads()` expose standalone OpenAI Images and Uploads APIs.
- `create_openai().containers()` and `create_openai().skills()` expose the raw OpenAI Containers and Skills APIs.
- `create_vertex().realtime_model(...).create_browser_token()` is intentionally unsupported. Vertex realtime sessions use server-side authentication instead of OpenAI/Gemini-style ephemeral browser tokens in this SDK.
- OpenAI and Azure OpenAI browser bootstrap now follows the official `realtime/client_secrets` flow, while Gemini browser tokens still come from `v1alpha/authTokens`.
- Realtime sessions emit `realtime-response-complete` when a model turn finishes and reserve `realtime-end` for actual session shutdown or transport closure.
- `Gemini` and `Vertex` speech generation return PCM audio in the current examples, so the demo writes a `.wav` container around the bytes.

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

For production logging, request correlation, gateway attempt tracing, and OpenTelemetry guidance, see [OBSERVABILITY.md](./OBSERVABILITY.md).

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

For production-style FastAPI integration patterns, see [PRODUCTION_APIS.md](./PRODUCTION_APIS.md) and the reference apps in [`examples/integrations/`](./examples/integrations/).

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
- `GuardrailResult`, `InputGuardrailRequest`, `OutputGuardrailRequest`
- `permission_allowlist_approval_policy(...)`
- input and output guardrails on `Agent(...)`
- `handoff_to(...)`
- `remote_tool(...)`
- `discover_mcp_tools(...)`
- `mcp_stdio_server(...)`
- `mcp_http_server(...)`
- `create_mcp_tool_registry(...)`

This layer is intended for stateful, tool-using, multi-agent assistants where you want executable handoffs, shared sessions, transcript + summary memory, approval hooks, traces, durable Postgres-backed state, and MCP-backed tool registries without rewriting the lower-level loop yourself.

For new MCP integrations, prefer the higher-level helpers:

```python
import asyncio

from zhivex_ai import Agent, create_mcp_tool_registry, create_openai, mcp_stdio_server, run_agent


async def main() -> None:
    async with await create_mcp_tool_registry(
        mcp_stdio_server(
            name="fs",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "."],
        )
    ) as tools:
        openai = create_openai()
        agent = Agent(
            name="assistant",
            instructions="Use the filesystem MCP tools when needed.",
            model=openai("gpt-5.4-mini"),
            tools=tools,
        )

        result = await run_agent(agent=agent, prompt="List the Python files in the current directory.")
        print(result.text)


asyncio.run(main())
```

`discover_mcp_tools(...)` is still available when you want raw tool definitions or full control over prefixes and registry composition. `ToolRegistry` also supports `async with` so MCP-backed runtimes can be closed cleanly after use.

## Examples

See [examples/README.md](./examples/README.md) for the full list. Highlights:

- Text: [openai_text.py](./examples/text/openai_text.py), [stream_text.py](./examples/text/stream_text.py), [structured_output.py](./examples/text/structured_output.py)
- Local Ollama: [ollama_text.py](./examples/text/ollama_text.py)
- Agents: [agent_basic.py](./examples/agents/agent_basic.py), [stream_agent.py](./examples/agents/stream_agent.py), [mcp_tools.py](./examples/agents/mcp_tools.py)
- Realtime: [openai_realtime.py](./examples/realtime/openai_realtime.py), [gemini_realtime.py](./examples/realtime/gemini_realtime.py), [live_agent_realtime.py](./examples/realtime/live_agent_realtime.py)
- Audio: [transcribe_audio.py](./examples/audio/transcribe_audio.py), [generate_speech.py](./examples/audio/generate_speech.py)
- Integrations: [ui_messages.py](./examples/integrations/ui_messages.py), [http_responses.py](./examples/integrations/http_responses.py), [gateway_fallback.py](./examples/integrations/gateway_fallback.py)

For real provider validation, the repo also includes a live smoke runner:

```bash
export ZHIVEX_SMOKE_OPENAI_MODEL=your-openai-model
export ZHIVEX_SMOKE_GEMINI_MODEL=your-gemini-model
export ZHIVEX_SMOKE_ANTHROPIC_MODEL=your-anthropic-model
export ZHIVEX_SMOKE_VERTEX_MODEL=your-vertex-model
export ZHIVEX_SMOKE_OLLAMA_MODEL=your-local-ollama-model
make smoke
```

It only runs providers that have the required credentials and model IDs configured, and you can scope it with `ZHIVEX_SMOKE_PROVIDERS=openai,gemini,ollama`. Ollama smoke runs default to `http://localhost:11434/v1` and can be redirected with `ZHIVEX_SMOKE_OLLAMA_BASE_URL`.

If realtime examples fail on macOS with `ssl.SSLCertVerificationError: CERTIFICATE_VERIFY_FAILED`, the issue is usually the local Python certificate bundle rather than the SDK. Two practical fixes are:

```bash
SSL_CERT_FILE="$(".venv/bin/python" -c 'import certifi; print(certifi.where())')" \
GOOGLE_API_KEY=... \
.venv/bin/python examples/realtime/gemini_realtime.py
```

or, for a permanent fix with the official python.org installer:

```bash
"/Applications/Python 3.14/Install Certificates.command"
```

## License

MIT. See [LICENSE](./LICENSE).
