# Zhivex AI SDK for Python

[![CI](https://img.shields.io/github/actions/workflow/status/Zhivex/zhivex-ai-sdk-py/ci.yml?branch=main&label=CI)](https://github.com/Zhivex/zhivex-ai-sdk-py/actions)
[![PyPI](https://img.shields.io/pypi/v/zhivex-ai-sdk)](https://pypi.org/project/zhivex-ai-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/zhivex-ai-sdk)](https://pypi.org/project/zhivex-ai-sdk/)
[![License](https://img.shields.io/pypi/l/zhivex-ai-sdk)](./LICENSE)

Zhivex AI SDK for Python is an async-first, agent-first SDK for building orchestrated AI systems across multiple providers.

Production maturity plan: see [MATURITY_PLAN.md](./MATURITY_PLAN.md).

It brings the same design goals as the TypeScript Zhivex AI SDK into Python:

- one agent runtime with executable handoffs, native subagent tools, shared sessions, durable run state, evaluation helpers, safety policies, tool registries, and traces
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

- Agent runtime with executable handoffs, native subagent tools, input/output guardrails, registry-based orchestration, durable run state, transcript + summary memory, permission-aware tool execution, and traces
- Declarative workflow agents with `SequentialAgent`, `ParallelAgent`, `LoopAgent`, shared `session.state`, `output_key`, and templated step inputs
- `AgentRuntime`, `AgentRegistry`, and `ToolRegistry` as the primary orchestration layer
- Unified `generate_text()` and `stream_text()` foundation primitives
- Structured output with `generate_object()` and `stream_object()`
- Grounded text for providers with web search support
- Audio transcription and speech generation where the provider supports it
- Experimental realtime/live voice sessions plus `stream_live_agent()` for voice-first agents
- Text and multimodal embeddings support where the provider supports it
- Google native media clients for Gemini/Vertex image, video, music, batch, and interaction workflows
- Provider factories for hosted and local models
- Beta provider agent capability metadata for support tiers and hosted-agent features
- Beta first-class hosted tool definitions for provider-native tools in `tools={...}`
- Beta typed provider-data payloads and approval flows for OpenAI/Azure remote MCP
- Beta response-reference helpers and provider-data UI chunks for OpenAI/Azure continuation workflows
- Gateway routing with fallback support
- HTTP/UI helpers for SSE, plain text streams, and UI message transport
- Middleware for telemetry, caching, and circuit breaking
- Model catalog helpers for cost and recommendation metadata
- Platform helpers for replay, evaluation reports, hierarchical run traces, budget guards, redaction, and run cancellation
- Offline reference apps for business workflows, including small-business loan and HR candidate selection flows that demonstrate repair/resume, human approval, fairness checks, trace replay, and app-owned storage without requiring provider credentials

## Supported Providers

Provider factories now return a `ProviderBundle` with two explicit namespaces:

- `provider("model-id")` or `provider.portable.language_model("model-id")` for the strict portable contract
- `provider.native.language_model("model-id")` for provider-specific options, hosted tools, and escape hatches

Portable construction fails fast for providers that do not satisfy the portable contract. Those providers remain available through `provider.native`.

For production API work, the current tier-1 provider story for the stable surface is OpenAI, Anthropic, Azure OpenAI, Gemini, and Vertex. Other providers remain available, but their supported feature set should be evaluated against the matrix below and the stability definitions in [STABILITY.md](./STABILITY.md).

This matrix is generated from runtime support metadata via `scripts/generate_support_matrix.py`.
Regenerate the README block with `python3 scripts/generate_support_matrix.py --write-readme`.
It includes beta provider agent capability metadata alongside portable support and native extras, so the docs stay aligned with the runtime support model instead of drifting by hand.

<!-- BEGIN GENERATED SUPPORT MATRIX -->
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
| kimi | compatibility | No | Yes | Yes | Yes | Yes | No | No | Yes | No | No |
| ollama | compatibility | No | Yes | Yes | Yes | Yes | Yes | No | Yes | No | No |
| openai | portable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| openrouter | native-only | No | Yes | Yes | Yes | Yes | Yes | No | Yes | No | Yes |
| qwen | compatibility | No | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| vertex | portable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

### Native Extras

| Provider | Files | File Search | Images | Uploads | Moderations | Batches | Videos | Media | Interactions | Containers | Skills | Realtime | Responses | Conversations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic | Yes | No | No | No | No | No | No | No | No | No | No | No | No | No |
| azure-openai | No | No | No | No | No | No | No | No | No | No | No | Yes | No | No |
| bedrock | No | No | No | No | No | No | No | No | No | No | No | Yes | No | No |
| gemini | Yes | Yes | Yes | No | No | Yes | Yes | Yes | Yes | No | No | Yes | No | No |
| kimi | Yes | No | No | No | No | Yes | No | No | No | No | No | No | No | No |
| ollama | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| openai | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | Yes | Yes | Yes | Yes | Yes |
| openrouter | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| qwen | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| vertex | No | No | Yes | No | No | No | Yes | Yes | No | No | No | Yes | No | No |

### Agent Capabilities

| Provider | Support Tier | Tool Choice None | Approval Requests | Hosted Web Search | Hosted File Search | Remote MCP | Computer Use | Code Execution | Toolsets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic | tier-b | Yes | No | Yes | No | No | No | Yes | Yes |
| azure-openai | tier-a | Yes | Yes | Yes | Yes | Yes | Yes | No | No |
| bedrock | tier-c | No | No | No | No | No | No | No | No |
| gemini | tier-b | Yes | No | Yes | Yes | No | Yes | Yes | No |
| kimi | tier-c | Yes | No | No | No | No | No | No | No |
| ollama | tier-c | No | No | No | No | No | No | No | No |
| openai | tier-a | Yes | Yes | Yes | Yes | Yes | Yes | No | No |
| openrouter | tier-c | Yes | No | Yes | No | No | No | No | No |
| qwen | tier-c | Yes | No | Yes | Yes | Yes | No | Yes | No |
| vertex | tier-b | Yes | No | Yes | Yes | No | Yes | Yes | No |
<!-- END GENERATED SUPPORT MATRIX -->

### Tool Calling Notes

Tool support now follows the same rule everywhere:

- The portable layer accepts only the SDK-owned contract. It rejects `provider_options` and any provider-managed tool payloads.
- Provider-specific hosted tools, raw Responses settings, Gemini built-in tools, and similar knobs must go through `provider.native.*`.
- First-class hosted tool definitions are the preferred native path for OpenAI, Azure OpenAI, Gemini, Vertex, and Anthropic. Legacy raw `provider_options` payloads remain accepted where already supported for backward compatibility.
- Hosted tools now fail fast in the shared foundation layer when they target the wrong provider, request an unsupported hosted-tool class, or use hosted-only combinations such as unsupported `tool_choice="none"` / `ToolChoiceName(...)`.
- The `Agent Capabilities` table above is beta metadata. It documents the current hosted-tool and provider-managed approval story, but it is not a stable promise that every provider will keep identical semantics release to release.
- Tier-1 support means the provider participates in the stable surface story, production API examples, and contract-level support assertions in this repository.
- Anthropic is part of the tier-1 text-generation story in this SDK. Extended thinking still restricts `tool_choice` to `auto` or `none`, and embeddings, transcription, and speech remain unavailable on the Anthropic provider path here.
- Anthropic hosted-tool defaults remain backward-compatible: `anthropic_web_search_tool()` emits `web_search_20250305`, `anthropic_code_execution_tool()` emits `code_execution_20250825`, and `anthropic_mcp_server()` uses `mcp-client-2025-04-04`. Current Anthropic MCP can be opted into with `anthropic_mcp_server(..., version="current")` or `provider_options={"anthropic_mcp_beta": "mcp-client-2025-11-20"}`. Current web search can be opted into with `anthropic_web_search_tool(tool_type="web_search_20260209")`; newer code-execution tool versions are model-dependent and should be passed explicitly when needed.
- OpenAI, Anthropic, and Azure OpenAI currently cover the broadest production text-generation API paths in this SDK.
- Azure OpenAI hosted-tool helpers map OpenAI-style tool payloads for native model calls, but this SDK does not currently mirror OpenAI's native lifecycle clients for vector-store/file-search administration, Responses, or Conversations on the Azure provider bundle.
- Gemini and Vertex are portable for the core contract, but Gemini built-in tools such as `google_search`, `google_maps`, `url_context`, `code_execution`, `file_search`, and `computer_use` are native-only entrypoints.
- Bedrock and OpenRouter remain available, but only through `provider.native` until they satisfy the portable contract end to end.
- Qwen, Kimi, and Ollama are compatibility providers in this refactor. They remain usable through native adapters, but they do not receive the portable badge.
- Qwen follows Alibaba Cloud Model Studio's current OpenAI-compatible split: text, tools, built-in web/code/file/MCP tools, embeddings, and Qwen3-ASR run through compatible endpoints; Qwen speech synthesis uses DashScope's multimodal generation endpoint. `create_qwen()` accepts either `QWEN_API_KEY` or the official `DASHSCOPE_API_KEY`.
- Kimi/Moonshot uses the official Chat Completions route for native text generation. `create_kimi()` reads `MOONSHOT_API_KEY` first, then `KIMI_API_KEY`, and defaults to `https://api.moonshot.ai/v1`.
- Kimi native support includes Moonshot Files, Batch, token estimation, K2.6/K2.5 `thinking`, image/video chat inputs, and the beta `provider.formulas()` client for official tools such as `moonshot/web-search:latest`.
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
pip install "zhivex-ai-sdk[docx]"
```

The beta skill-package layer also exposes a CLI:

```bash
zhivex-skills validate path/to/skill
zhivex-skills install path/to/skill
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

Embedding multimodal Gemini content:

```python
import asyncio

from zhivex_ai import FilePart, TextPart, create_gemini, embed_content


async def main() -> None:
    gemini = create_gemini()
    result = await embed_content(
        model=gemini.native.embedding_model("gemini-embedding-2"),
        value=[
            TextPart(text="Find visually similar documents."),
            FilePart(data="JVBERi0xLjQK", media_type="application/pdf", filename="brief.pdf"),
        ],
    )

    print(len(result.embedding or []))


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

Using Google native media and long-running clients:

```python
import asyncio

from zhivex_ai import create_gemini


async def main() -> None:
    gemini = create_gemini()

    image = await gemini.images().generate(
        model="gemini-3.1-flash-image-preview",
        prompt="A clean product diagram of a solar microgrid.",
    )
    operation = await gemini.videos().generate(
        model="veo-3.1-generate-preview",
        prompt="A slow cinematic flyover of that microgrid at sunrise.",
    )
    music = await gemini.media().generate_music(
        model="lyria-3-clip-preview",
        prompt="A 30-second optimistic ambient technology track.",
    )

    print(image.images[0].b64_json is not None)
    print(operation.name)
    print(music.media[0].media_type)


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
triage.model = gemini.native.language_model("gemini-3.1-flash-preview")
result = await run_agent(
    agent=triage,
    prompt="Research the Apollo migration status.",
    provider_options={"google_search": True},
)
```

Built-in Gemini tools can also be configured directly:

```python
result = await generate_text(
    model=gemini.native.language_model("gemini-3.1-flash-preview"),
    prompt="Research this page and show your work.",
    provider_options={
        "google_search": {"excludeDomains": ["example.com"]},
        "url_context": {},
        "code_execution": True,
    },
)
```

Hosted tools can now live directly inside `tools={...}` on native models, alongside callable local tools:

```python
import asyncio

from pydantic import BaseModel, ConfigDict

from zhivex_ai import create_openai, generate_text, openai_web_search_tool, tool


class WeatherInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str


async def main() -> None:
    openai = create_openai()

    result = await generate_text(
        model=openai.native.language_model("gpt-5.4-mini"),
        prompt="Compare today's weather in Buenos Aires with what is happening in the news.",
        tools={
            "weather": tool(
                name="weather",
                schema=WeatherInput,
                execute=lambda input: {"city": input.city, "forecast": "18C and cloudy"},
            ),
            "search": openai_web_search_tool(search_context_size="high"),
        },
    )

    print(result.text)


asyncio.run(main())
```

Remote MCP approval responses also round-trip through assistant messages with a `provider-data` part:

```python
from zhivex_ai import assistant, openai_mcp_approval_response, user

messages = [
    assistant([openai_mcp_approval_response(approval_request_id="apr_123", approve=True)]),
    user("Continue with the approved MCP call."),
]
```

OpenAI and Azure OpenAI also expose beta typed provider-data payloads and parse helpers:

```python
from zhivex_ai import (
    OpenAIMcpApprovalRequest,
    parse_openai_provider_data_part,
    provider_data_part,
)

part = provider_data_part(
    "openai",
    OpenAIMcpApprovalRequest(
        id="apr_123",
        arguments='{"query":"apollo"}',
        name="docs_search",
        server_label="Docs",
    ),
)

parsed = parse_openai_provider_data_part(part)
assert parsed is not None
assert parsed.type == "mcp_approval_request"
```

You can also continue OpenAI Responses API workflows without manually threading raw IDs:

```python
from zhivex_ai import (
    get_openai_response_id,
    openai_response_options,
)

first = await generate_text(
    model=openai.native.language_model("gpt-5.4-mini"),
    prompt="Start a multi-turn responses workflow.",
)

follow_up = openai_response_options(previous_response=first)
assert get_openai_response_id(first) is not None
```

The UI helpers now preserve provider-managed control traffic as `provider-data` chunks as well, so `to_ui_message_stream(...)` can surface MCP approval requests, response references, and other typed provider-data events to frontend consumers.

When these approval requests appear inside `run_agent(...)` or `stream_agent(...)`, the runtime reuses `approval_policy` to emit `AgentToolApprovalEvent` events, append the provider-specific approval response, and continue the loop. This provider-managed approval path is currently beta and limited to OpenAI and Azure OpenAI; other providers may expose hosted-tool metadata in the support matrix before they share this same approval/runtime integration.

See [examples/text/native_hosted_tools.py](./examples/text/native_hosted_tools.py) for a compact mixed local + hosted tool example, and [examples/agents/provider_managed_approvals.py](./examples/agents/provider_managed_approvals.py) for the matching OpenAI/Azure remote MCP approval flow with `stream_agent(...)`.

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

OpenAI-compatible providers such as OpenRouter, Qwen, and Ollama still reuse normalized adapter paths internally, but they no longer imply portable parity. Kimi/Moonshot uses its own native Chat Completions adapter because the official Kimi API documents `/v1/chat/completions`, Files, Batch, token estimation, and Formulas as the current production surfaces.

Kimi/Moonshot native usage:

```python
import asyncio

from zhivex_ai import ImagePart, ReasoningConfig, create_kimi, generate_text, user


async def main() -> None:
    kimi = create_kimi()  # MOONSHOT_API_KEY, then KIMI_API_KEY

    result = await generate_text(
        model=kimi.native.language_model("kimi-k2.6"),
        messages=[
            user(
                [
                    ImagePart(image="data:image/png;base64,..."),
                ]
            )
        ],
        reasoning=ReasoningConfig(effort="none"),
    )
    print(result.text)


asyncio.run(main())
```

Kimi files, batch jobs, token estimation, and official tools are native-only:

```python
kimi = create_kimi()

uploaded = await kimi.files().upload(
    data=b'{"custom_id":"1","method":"POST","url":"/v1/chat/completions","body":{"model":"kimi-k2.6","messages":[{"role":"user","content":"hi"}]}}\n',
    filename="batch.jsonl",
    media_type="application/jsonl",
    purpose="batch",
)

batch = await kimi.batches().create(
    {
        "input_file_id": uploaded.id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }
)

tokens = await kimi.tokens().count(model_id="kimi-k2.6", prompt="hello")
tools = await kimi.formulas().toolset(["moonshot/web-search:latest"])
```

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
- `provider.videos()`
- `provider.media()`
- `provider.uploads()`
- `provider.tokens()`
- `provider.file_search_stores()`
- `provider.batches()`
- `provider.interactions()`
- `provider.formulas()`
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
.venv/bin/python scripts/generate_support_matrix.py
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
- `embed_content(...)` and `embed_content_many(...)` accept text plus `TextPart`, `ImagePart`, and `FilePart` values for Gemini Embedding 2 style multimodal embeddings; `embed(...)` and `embed_many(...)` remain text-compatible.
- `create_gemini().images()` covers Gemini/Nano Banana image models through `generateContent` and Imagen 4 models through `predict`.
- `create_vertex().images()` mirrors Google image routes through Vertex publisher model endpoints.
- `create_gemini().videos()` and `create_vertex().videos()` expose Veo long-running operation creation, polling, and download helpers.
- `create_gemini().media()` and `create_vertex().media()` expose Lyria-style native audio/music generation where the Google model route supports it.
- `create_gemini().batches()` exposes Gemini Batch API generation and embedding jobs.
- `create_gemini().interactions()` exposes Gemini Interactions and Deep Research polling/streaming helpers. Deep Research payloads default to background storage.
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
- `create_in_memory_agent_run_store()`
- `create_sqlite_agent_memory_store(...)`
- `create_sqlite_checkpoint_store(...)`
- `create_sqlite_agent_run_store(...)`
- `create_postgres_agent_memory_store(...)`
- `create_postgres_checkpoint_store(...)`
- `create_postgres_agent_run_store(...)`
- `create_otel_agent_observer()`
- `create_agent_trace_artifact(...)`
- `summarize_agent_trace(...)`
- `create_agent_run_snapshot(...)`
- `replay_agent_run(...)`
- `run_agent_evaluation(...)`
- `create_agent_evaluation_report(...)`
- `create_safety_policy(...)`
- `apply_safety_policy_to_agent(...)`
- `load_agent_session(...)`
- `ApprovalDecision`, `ToolApprovalRequest`
- `GuardrailResult`, `InputGuardrailRequest`, `OutputGuardrailRequest`
- `permission_allowlist_approval_policy(...)`
- input and output guardrails on `Agent(...)`
- `handoff_to(...)`
- `create_subagent_tool(...)`
- `prepare_subagents_for_agent(...)`
- `run_agent_group(...)`
- `remote_tool(...)`
- `skill(...)`
- `load_skill(...)`
- `discover_skills(...)`
- `discover_mcp_tools(...)`
- `mcp_stdio_server(...)`
- `mcp_http_server(...)`
- `create_mcp_tool_registry(...)`

This layer is intended for stateful, tool-using, multi-agent assistants where you want executable handoffs, native subagent tools, shared sessions, transcript + summary memory, approval hooks, replay/evaluation, traces, durable run state, and MCP-backed tool registries without rewriting the lower-level loop yourself.

Declarative workflows are available when the coordination pattern is known ahead of time:

```python
from zhivex_ai import SequentialAgent, WorkflowStep

pipeline = SequentialAgent(
    name="loan_pipeline",
    steps=[
        WorkflowStep("extract", extractor, prompt="Extract the application", output_key="application"),
        WorkflowStep("validate", validator, input_template="Validate {application}", output_key="validation"),
        WorkflowStep("decide", decider, input_template="Decide with {application} and {validation}"),
    ],
)

result = await pipeline.run()
```

Use `ParallelAgent` for fan-out research and `LoopAgent` for bounded refinement loops. Workflow steps share `session.state`; `output_key` writes the step text into state, and `input_template` reads state keys with Python format placeholders.

For fuller business-workflow references, see [`examples/agents/small_business_loan_agent.py`](./examples/agents/small_business_loan_agent.py) and [`examples/agents/hr_candidate_selection_agent.py`](./examples/agents/hr_candidate_selection_agent.py). They model regulated financial review and HR candidate selection with extraction/intake, scoring, human review, validation, fairness/compliance checks, repair/resume, and trace replay. The SDK owns orchestration primitives; the examples keep credit policy, hiring policy, pricing/scoring, persistence, approval UI, ATS integrations, and compliance systems application-owned behind replaceable interfaces.

Durable run state can be attached directly to an agent:

```python
from zhivex_ai import Agent, create_in_memory_agent_run_store, run_agent

store = create_in_memory_agent_run_store()
agent = Agent(name="assistant", model=model, run_store=store)
result = await run_agent(agent=agent, prompt="Draft a reply", idempotency_key="reply-1")
state = await store.load(result.run_id)
```

Safety policies compose approval, redaction, and budget defaults without mutating the original agent:

```python
from zhivex_ai import apply_safety_policy_to_agent, create_safety_policy

safe_agent = apply_safety_policy_to_agent(agent, create_safety_policy(preset="review_sensitive"))
```

Evaluation and trace helpers work from persisted state:

```python
from zhivex_ai import create_agent_trace_artifact, replay_agent_run, summarize_agent_trace

trace = create_agent_trace_artifact(state)
timeline = replay_agent_run(state)
summary = summarize_agent_trace(state)
```

Agent skills are also available across the agent runtime. These are provider-agnostic workflow packs that inject task-specific instructions and optional tool dependencies before a run starts. They are distinct from the raw OpenAI Skills API:

- `Agent(..., skills=...)` activates provider-agnostic runtime skills
- `skill(...)`, `load_skill(...)`, and `discover_skills(...)` help define or load skills from `SKILL.md`
- `set_agent_session_skills(...)`, `get_agent_session_skills(...)`, and `clear_agent_session_skills(...)` manage sticky session skills explicitly
- `provider.skills()` remains the native OpenAI lifecycle client for hosted OpenAI skills
- activated skills stick to the agent session through `session.metadata["sticky_skills"]`
- the runtime emits `AgentSkillActivatedEvent` and `AgentSkillSkippedEvent` for observability

Runtime skills follow the Codex-style `SKILL.md` layout: frontmatter with `name` and `description`, instruction body, and optional `agents/openai.yaml` metadata for display text, implicit-invocation policy, and MCP tool dependencies.

The SDK now also includes a beta skill-package layer for Anthropic-style packaged skills. This adds `skill.yaml`, installable skill packages, a static HTTP registry flow, direct `run_skill(...)`, artifacts, and the `zhivex-skills` CLI. The packaged-skill APIs are:

- `load_skill_package(...)`
- `validate_skill(...)`
- `install_skill(...)`
- `list_installed_skills(...)`
- `run_skill(...)`
- `publish_skill(...)`

Packaged skills can declare:

- versioned entrypoints
- local Python or binary dependencies
- produced artifacts
- explicit read/write/network permissions

The first official packaged skill is the beta `docx` skill under `zhivex_ai/official_skills/docx`, designed around `python-docx`.

The runtime also supports production-oriented policy metadata:

- `priority` to resolve competing implicit skills
- `triggers` and `anti_triggers` for deterministic activation rules
- `allowed_providers` and `allowed_models` to constrain where a skill can run
- `persist_to_session` to opt out of sticky reuse
- `dependency_failure_mode` set to `"skip"` or `"fail"`

```python
import asyncio

from zhivex_ai import (
    Agent,
    clear_agent_session_skills,
    create_agent_session,
    create_openai,
    get_agent_session_skills,
    run_agent,
    set_agent_session_skills,
    skill,
)


async def main() -> None:
    openai = create_openai()
    release_notes = skill(
        name="release-notes",
        description="Use when a user asks for changelog summaries or release notes.",
        instructions="Write highlights, breaking changes, and migration notes when needed.",
    )
    agent = Agent(
        name="assistant",
        instructions="You are a careful SDK assistant.",
        model=openai("gpt-5.4-mini"),
        skills={"release-notes": release_notes},
    )
    session = create_agent_session()

    set_agent_session_skills(session, "release-notes")
    result = await run_agent(agent=agent, session=session, prompt="Summarize the latest SDK updates.")
    print(result.text)
    print(result.session.metadata["active_skills"])
    print(get_agent_session_skills(session))
    clear_agent_session_skills(session)


asyncio.run(main())
```

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
- Agent skills: [skills.py](./examples/agents/skills.py)
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

It only runs providers that have the required credentials and model IDs configured, and you can scope it with `ZHIVEX_SMOKE_PROVIDERS=openai,gemini,ollama`. Optional Google media smoke checks are gated behind `ZHIVEX_SMOKE_GOOGLE_MEDIA=1` and model IDs such as `ZHIVEX_SMOKE_GEMINI_IMAGE_MODEL`, `ZHIVEX_SMOKE_GEMINI_VIDEO_MODEL`, `ZHIVEX_SMOKE_GEMINI_MEDIA_MODEL`, `ZHIVEX_SMOKE_VERTEX_IMAGE_MODEL`, `ZHIVEX_SMOKE_VERTEX_VIDEO_MODEL`, and `ZHIVEX_SMOKE_VERTEX_MEDIA_MODEL`. Ollama smoke runs default to `http://localhost:11434/v1` and can be redirected with `ZHIVEX_SMOKE_OLLAMA_BASE_URL`.

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
