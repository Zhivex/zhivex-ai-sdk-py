# Zhivex AI SDK for Python

[![CI](https://img.shields.io/github/actions/workflow/status/Zhivex/zhivex-ai-sdk-py/ci.yml?branch=main&label=CI)](https://github.com/Zhivex/zhivex-ai-sdk-py/actions)
[![PyPI](https://img.shields.io/pypi/v/zhivex-ai-sdk)](https://pypi.org/project/zhivex-ai-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/zhivex-ai-sdk)](https://pypi.org/project/zhivex-ai-sdk/)
[![License](https://img.shields.io/pypi/l/zhivex-ai-sdk)](./LICENSE)

Zhivex AI SDK for Python is an async-first runtime for building reliable agents across multiple AI providers.

The core product is deliberately small:

- define an `Agent` once and run it with `run_agent()` or `stream_agent()`
- use one portable contract for messages, tools, structured output, embeddings, and model calls
- add sessions, durable state, human approval, replay, gateway fallback, and observability when the application needs them
- keep provider-specific behavior behind explicit `provider.native.*` escape hatches

Stable workflow orchestration, evaluation pipelines, protocol hosting, packaged skills, the general CLI/playground, and realtime voice are available, but they are optional extensions rather than prerequisites for building an agent. Named external-engine adapters remain Beta contracts owned by the application.

## Why Zhivex AI SDK

Modern AI apps usually start simple and then drift into provider lock-in:

- OpenAI requests look one way
- Anthropic uses a different message format
- Gemini and Vertex differ again
- local and routed setups add yet another layer

Zhivex AI SDK gives you a common agent runtime and model contract so your application code can stay stable while providers change underneath. Start with one agent and one provider; adopt the rest only when the product requires it.

## Build One Portable Agent

```python
import asyncio

from pydantic import BaseModel

from zhivex_ai import Agent, create_openai, run_agent, tool


class ProjectStatusInput(BaseModel):
    project: str


def lookup_project_status(input: ProjectStatusInput) -> dict[str, str]:
    return {"project": input.project, "status": "on track"}


async def main() -> None:
    provider = create_openai()
    agent = Agent(
        name="project-assistant",
        instructions="Use the project-status tool, then answer concisely.",
        model=provider("gpt-5.6-terra"),
        tools={
            "lookup_project_status": tool(
                name="lookup_project_status",
                description="Returns the current status for a project.",
                schema=ProjectStatusInput,
                execute=lookup_project_status,
            )
        },
    )

    result = await run_agent(agent=agent, prompt="What is the status of Apollo?")
    print(result.text)


asyncio.run(main())
```

The agent code uses the portable model contract while the application owns the tool and its data. Run the complete example with `.venv/bin/python examples/agents/quickstart_agent.py`. Switching to another portable provider changes provider construction and the model ID, not the agent runtime. Follow the [quickstart](./docs/QUICKSTART.md) for installation plus offline and live verification.

## Product Boundary

| Layer | What belongs here | Adoption guidance |
| --- | --- | --- |
| Agent core | `Agent`, tools, streaming, handoffs, sessions, approvals, durable run state, replay | Default starting point |
| Foundation | Text, structured output, embeddings, grounding, normalized messages | Use directly or through agents |
| Providers and gateway | Portable adapters, native escape hatches, fallback routing | Choose only the providers your app needs |
| Stable orchestration | `zhivex_ai.workflows` core, stores, leases, migration, resume/fork/cancel | Adopt when the application needs durable coordination |
| Optional extensions | `zhivex_ai.evals`, `zhivex_ai.integrations`, named workflow-engine adapters | Beta; isolate behind app-owned boundaries |
| Incubating capabilities | `zhivex_ai.experimental`, including realtime/live agents | Experimental; expect contract changes |

## Stability And Support

Zhivex AI SDK is now published as a beta package with a documented stable surface, explicit stability levels, and versioning rules for downstream integrators.

Production integrations should import supported APIs from `zhivex_ai`, prefer the documented stable surface and tier-1 providers, and isolate beta or experimental areas behind an application-owned service layer.

For agent applications, the stable slice includes `Agent`, `AgentRunResult`, `AgentStreamResult`, local `tool(...)` definitions and execution types, `handoff_to(...)`, sessions, durable Postgres state, replay, approval resume, and the workflow orchestration core under `zhivex_ai.workflows`. Named DBOS/Temporal/Prefect/Restate adapter factories, native subagent tools such as `create_subagent_tool(...)`, evaluation trials/experiments, A2A/AG-UI/Responses hosting, the general CLI/playground, packaged skills, trace artifacts, safety-policy helpers, and local agent run stores remain beta.

See [docs/SCOPE.md](./docs/SCOPE.md), [STABILITY.md](./STABILITY.md), [VERSIONING.md](./VERSIONING.md), [SUPPORT.md](./SUPPORT.md), and [CHANGELOG.md](./CHANGELOG.md) for the product boundary, public API expectations, support scope, and release communication.

## Start Here

Core path:

- Understand the product boundary and non-goals: [docs/SCOPE.md](./docs/SCOPE.md)
- Install and verify one portable agent: [docs/QUICKSTART.md](./docs/QUICKSTART.md)
- Configure a provider and run targeted smoke checks: [docs/PROVIDERS.md](./docs/PROVIDERS.md)
- Add tools, streaming, handoffs, sessions, and approvals: [docs/AGENTS.md](./docs/AGENTS.md)
- Apply production API patterns: [PRODUCTION_APIS.md](./PRODUCTION_APIS.md)
- Add gateway fallback: [docs/GATEWAY.md](./docs/GATEWAY.md)
- Operate safely: [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md), [docs/OPERATIONS.md](./docs/OPERATIONS.md), [SECURITY.md](./SECURITY.md), and [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)

Optional extensions:

- Stable durable workflow orchestration: [docs/WORKFLOWS.md](./docs/WORKFLOWS.md)
- Beta evaluations and CI gates: [docs/EVALUATIONS.md](./docs/EVALUATIONS.md)
- Beta protocols and hosting: [docs/PROTOCOLS.md](./docs/PROTOCOLS.md)
- Beta CLI and local playground: [docs/CLI.md](./docs/CLI.md)
- Maturity and GA boundary: [docs/PARITY_MATRIX.md](./docs/PARITY_MATRIX.md)

Project setup:

- Contribution workflow: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Local environment template: [.env.example](./.env.example)

## Core Capabilities

- `Agent`, `run_agent()`, and `stream_agent()` with typed dependencies and outputs
- Local tools, executable handoffs, sessions, memory, approval/resume, durable run state, and replay
- `generate_text()`, `stream_text()`, `generate_object()`, `stream_object()`, embeddings, and grounded text
- Portable provider factories plus explicit native escape hatches
- Gateway fallback, transport helpers, middleware, model catalog helpers, and observability hooks
- Production-style API and worker examples for durable agents, idempotency, gateway attempts, and release evidence

## Optional And Incubating Capabilities

- Stable durable workflow orchestration under `zhivex_ai.workflows`, with DAG validation, persisted checkpoints, explicit schema migration, execution leases, resume/fork/cancel, retries, and a generic callback adapter contract
- Beta named DBOS, Temporal, Prefect, and Restate callback-adapter factories; applications own and certify the actual engine integration
- Beta repeated evaluation trials under `zhivex_ai.evals`, with concurrency, confidence intervals, cost/latency metrics, JSON/JUnit artifacts, variants, and CI gates
- Beta A2A v1, AG-UI, and Responses-compatible hosting under `zhivex_ai.integrations`, plus packaged skills and a loopback-only CLI playground
- Beta provider-native hosted tools, provider-data payloads, remote MCP approvals, media clients, and lifecycle clients
- Experimental realtime/live voice sessions plus `stream_live_agent()` under `zhivex_ai.experimental`, with durable approval suspension, idempotency, middleware, tool timeouts, and cancellation for voice-first agents
- Offline business reference apps that demonstrate repair/resume, human approval, fairness checks, trace replay, and app-owned storage

## Supported Providers

Provider factories now return a `ProviderBundle` with two explicit namespaces:

- `provider("model-id")` or `provider.portable.language_model("model-id")` for the strict portable contract
- `provider.native.language_model("model-id")` for provider-specific options, hosted tools, and escape hatches

Portable construction fails fast for providers that do not satisfy the portable contract. Those providers remain available through `provider.native`.

For production API work, the current ten-provider tier-1 story for the stable surface is OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex, Qwen, Kimi/Moonshot, DeepSeek, Meta Model API, and vLLM. Other providers remain available, but their supported feature set should be evaluated against the matrix below and the stability definitions in [STABILITY.md](./STABILITY.md).

Meta Model API is tier-1 only for the Stable `create_meta()` factory and Standard `muse-spark-1.2` portable text, streaming, structured output, callable tools, agent tool loops, and application-supplied retrieval through `PortableRetrievalConfig`. Portable retrieval injects bounded `PortableDocument` text into the request; it does not call Meta Files, hosted search, or raw Responses. Contributor models, hosted-tool helpers, Files, raw Responses, hosted tools, and multimodal/native extras remain Beta. Tier-1 means contract-supported, not release-certified. See [docs/providers/meta.md](./docs/providers/meta.md).

This matrix is generated from runtime support metadata via `scripts/generate_support_matrix.py`.
Regenerate the README block with `python3 scripts/generate_support_matrix.py --write-readme`.
It includes beta provider agent capability metadata alongside portable support and native extras, so the docs stay aligned with the runtime support model instead of drifting by hand.

<!-- BEGIN GENERATED SUPPORT MATRIX -->
### Tier-1 Providers

These providers back the stable portable contract for production API work in this SDK today:

- `openai`
- `anthropic`
- `azure-openai`
- `gemini`
- `vertex`
- `qwen`
- `kimi`
- `deepseek`
- `meta`
- `vllm`

Tier-1 identifies shared contract coverage; it does not establish live release certification.
Provider evidence is fail-closed: without separately validated evidence for the exact release artifact,
portable providers remain `contract-supported`. Certification does not change portability, API stability, or Tier-1 membership.

### Provider Evidence

| Provider | Evidence Status |
| --- | --- |
| anthropic | contract-supported |
| azure-openai | contract-supported |
| bedrock | experimental/native-only |
| deepseek | contract-supported |
| gemini | contract-supported |
| kimi | contract-supported |
| meta | contract-supported |
| ollama | experimental/native-only |
| openai | contract-supported |
| openrouter | experimental/native-only |
| qwen | contract-supported |
| vertex | contract-supported |
| vllm | contract-supported |

### Portable Support

| Provider | Tier | Portable Badge | Text | Streaming | Structured Output | Tools | Embeddings | Grounding | Retrieval | Transcription | Speech |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic | portable | Yes | Yes | Yes | Yes | Yes | No | Yes | Yes | No | No |
| azure-openai | portable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| bedrock | native-only | No | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| deepseek | portable | Yes | Yes | Yes | Yes | Yes | No | No | No | No | No |
| gemini | portable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| kimi | portable | Yes | Yes | Yes | Yes | Yes | No | No | No | No | No |
| meta | portable | Yes | Yes | Yes | Yes | Yes | No | No | Yes | No | No |
| ollama | compatibility | No | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| openai | portable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| openrouter | native-only | No | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| qwen | portable | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | No |
| vertex | portable | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| vllm | portable | Yes | Yes | Yes | Yes | Yes | Yes | No | Yes | Yes | No |

### Native Extras

| Provider | Text | Streaming | Structured Output | Tools | Embeddings | Grounding | Transcription | Speech | Files | File Search | Images | Uploads | Moderations | Batches | Videos | Media | Interactions | Containers | Skills | Realtime | Responses | Conversations | Caches | Token Count | Formulas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic | Yes | Yes | Yes | Yes | No | Yes | No | No | Yes | No | No | No | No | No | No | No | No | No | No | No | No | No | No | Yes | No |
| azure-openai | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | Yes | No | No | No | No | No | No | No | No | No | Yes | Yes | Yes | No | No | No |
| bedrock | Yes | Yes | No | Yes | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | Yes | No | No | No | No | No |
| deepseek | Yes | Yes | Yes | Yes | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| gemini | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | No | Yes | Yes | Yes | Yes | No | No | Yes | No | No | Yes | Yes | No |
| kimi | Yes | Yes | Yes | Yes | No | No | No | No | Yes | No | No | No | No | Yes | No | No | No | No | No | No | No | No | No | Yes | Yes |
| meta | Yes | Yes | Yes | Yes | No | No | No | No | Yes | No | No | No | No | No | No | No | No | No | No | No | Yes | No | No | No | No |
| ollama | Yes | Yes | Yes | Yes | Yes | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| openai | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | Yes | Yes | Yes | Yes | Yes | No | No | No |
| openrouter | Yes | Yes | Yes | Yes | Yes | No | No | Yes | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No | No |
| qwen | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | No | No | No | Yes | No | No | No | No | No | No | Yes | No | No | No | No |
| vertex | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | No | Yes | No | No | No | Yes | Yes | No | No | No | Yes | No | No | No | Yes | No |
| vllm | Yes | Yes | Yes | Yes | Yes | No | Yes | No | No | No | No | No | No | No | No | No | No | No | No | Yes | No | No | No | No | No |

### Agent Capabilities

| Provider | Support Tier | Tool Choice None | Approval Requests | Hosted Web Search | Hosted File Search | Remote MCP | Computer Use | Code Execution | Toolsets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anthropic | tier-b | Yes | No | Yes | No | No | No | Yes | Yes |
| azure-openai | tier-a | Yes | Yes | Yes | Yes | Yes | Yes | No | No |
| bedrock | tier-b | Yes | No | No | No | No | No | No | No |
| deepseek | tier-b | Yes | No | No | No | No | No | No | No |
| gemini | tier-b | Yes | No | Yes | Yes | No | Yes | Yes | No |
| kimi | tier-b | Yes | No | No | No | No | No | No | Yes |
| meta | tier-b | No | No | Yes | No | No | No | No | Yes |
| ollama | tier-c | No | No | No | No | No | No | No | No |
| openai | tier-a | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No |
| openrouter | tier-c | Yes | No | Yes | No | No | No | No | No |
| qwen | tier-b | Yes | No | Yes | Yes | Yes | No | Yes | No |
| vertex | tier-b | Yes | No | Yes | Yes | No | Yes | Yes | No |
| vllm | tier-b | Yes | No | No | No | No | No | No | No |
<!-- END GENERATED SUPPORT MATRIX -->

### Tool Calling Notes

Tool support now follows the same rule everywhere:

- The portable layer accepts only the SDK-owned contract. It rejects `provider_options` and any provider-managed tool payloads.
- `N/A` in the portable table means model construction through `provider(...)` is disabled for that provider; native capability values belong to the separate `Native Extras` table.
- Provider-specific hosted tools, raw Responses settings, Gemini built-in tools, and similar knobs must go through `provider.native.*`.
- First-class hosted tool definitions are the preferred native path for OpenAI, Azure OpenAI, Gemini, Vertex, and Anthropic. Legacy raw `provider_options` payloads remain accepted where already supported for backward compatibility.
- Hosted tools now fail fast in the shared foundation layer when they target the wrong provider, request an unsupported hosted-tool class, or use hosted-only combinations such as unsupported `tool_choice="none"` / `ToolChoiceName(...)`.
- The `Agent Capabilities` table above is beta metadata. It documents the current hosted-tool and provider-managed approval story, but it is not a stable promise that every provider will keep identical semantics release to release.
- Tier-1 support means the provider participates in the stable surface story, production API examples, and contract-level support assertions in this repository.
- Tier-1 is primarily an offline contract/support classification. It does not certify that every provider, model, or operation was live-smoked for the current release SHA; live claims apply only to the exact configured smoke evidence recorded for that artifact and SHA.
- Anthropic is part of the tier-1 text-generation story in this SDK. Claude Opus 5, Fable 5, Sonnet 5, restricted-access Mythos 5, and Opus 4.8 are cataloged. Opus 5 uses adaptive thinking by default; `ReasoningConfig(effort="low" | "medium" | "high" | "xhigh" | "max")` selects effort, while `ReasoningConfig(effort="none")` disables thinking only through `high`. Manual thinking budgets and non-default sampling are rejected for Opus 5. Forced tool choice is supported with adaptive thinking; the `auto`/`none` restriction applies only to manual extended thinking on older models.
- Claude Opus 5, Fable 5, restricted-access Mythos 5, and Opus 4.8 accept mid-conversation `ModelMessage(role="system", ...)` sections on the Anthropic native path when they follow Anthropic's placement rules. Opus 5 does not support assistant prefill or server-side Web Fetch. Fast mode remains a native escape hatch through `provider_options={"speed": "fast"}` and the adapter adds its required beta header.
- Anthropic `stop_reason="refusal"` is normalized as `finish_reason="refusal"` while preserving `provider_finish_reason` and `stop_details`; collected partial streaming text is discarded after a refusal. Gateway routes use `fallback_on_refusal=False` by default so a primary refusal is not re-sent to fallbacks unless the route explicitly opts in with `fallback_on_refusal=True`. Anthropic server-side fallback remains a beta raw `provider_options` escape hatch.
- Anthropic hosted-tool helpers now default to `web_search_20260318`, `web_fetch_20260318`, and GA `code_execution_20260521`. `anthropic_mcp_server()` keeps its compatibility default; opt into current MCP with `version="current"` when the target account/model supports it.
- OpenAI, Anthropic, Azure OpenAI, Gemini, Vertex, Qwen, Kimi, DeepSeek, Meta Model API, and vLLM now participate in the tier-1 portable text-generation contract. Meta's Stable scope is Standard `muse-spark-1.2` text, streaming, structured output, callable tools, and agent tool loops with `tool_choice="auto"`; its native/hosted extensions remain Beta.
- vLLM is tier-1 for the SDK primitives backed by its OpenAI-compatible server. Embeddings, transcription, and realtime ASR depend on serving compatible model tasks in vLLM; vLLM custom endpoints such as tokenize, rerank, classify, and score are not SDK APIs yet.
- OpenAI catalog guidance tracks the GA GPT-5.6 family: `gpt-5.6-sol` (alias `gpt-5.6`) for flagship work, `gpt-5.6-terra` for balanced workloads, and `gpt-5.6-luna` for high-volume paths. Responses is the recommended route for reasoning and tools; `ReasoningConfig` supports efforts through `max`.
- Azure OpenAI hosted-tool helpers map OpenAI-style tool payloads for native model calls, and the Azure provider bundle mirrors the beta native lifecycle clients for vector-store/file-search administration, Responses, and Conversations through `/openai/v1`. The catalog tracks the GPT-5.6 family, `gpt-chat-latest`, and `gpt-realtime-2.1`; actual deployments remain region/quota dependent.
- Gemini and Vertex are portable for the core contract, with `gemini-3.6-flash` as the current stable Flash reference and `gemini-3.5-flash-lite` as the stable low-latency Flash-Lite reference. Those two current models reject custom sampling and prefilled assistant turns before dispatch. Gemini Developer API guidance also tracks Interactions-only `gemini-omni-flash-preview` and `gemini-3.1-flash-lite-image`; Omni is not claimed for Vertex. Gemini built-in tools remain native-only entrypoints.
- Gemini function-calling preserves Google `functionCall.id` / `functionResponse.id` for Gemini 3 tool loops, while continuing to preserve `thoughtSignature` for reasoning-aware tool handoffs.
- Gateway routing emits `on_attempt` payloads for skipped targets as well as executed attempts, including a machine-readable `reason` for policy skips. Cost ceilings are fail-closed: `model_costs_per_1k_tokens` and configured catalog entries take precedence over the deprecated provider-wide fallback, and a target with unknown pricing is not invoked when `max_cost_per_1k_tokens` is set. Without a cost ceiling, unknown-price targets remain eligible. Use `GatewayConfig(fail_on_missing_adapter=True)` for production routes where a missing provider adapter should fail fast instead of falling through to a fallback. See [docs/GATEWAY.md](./docs/GATEWAY.md) for units, precedence, and migration guidance.
- Bedrock, OpenRouter, and Ollama remain available, but only through `provider.native` until they satisfy the portable contract end to end.
- Qwen catalog guidance now starts with GA `qwen3.8-max` for pay-as-you-go Singapore. Text, streaming, image understanding, function tools, all seven portable reasoning efforts, and the five announced built-ins use the current `/compatible-mode/v1/responses` route. Mixed vision input follows Qwen's current Responses `input_text` / `input_image` content contract; the legacy `/api/v2/apps/protocols/compatible-mode/v1` route is not used.
- `qwen3.8-max` transparently selects Chat Completions for native JSON Schema output, `FilePart` image/video inputs, or `ReasoningConfig.budget_tokens`, because those operations are not part of Qwen Responses. Structured output disables thinking, video uses `video_url`, and Chat reasoning state is preserved as Qwen `provider-data` for replay. Qwen hosted helpers cover `web_search`, `web_extractor`, `code_interpreter`, `web_search_image`, and `image_search`; Web Extractor still requires Web Search. The Token Plan continues to list the separate `qwen3.8-max-preview` ID, so it is not treated as a GA alias.
- Qwen native support also includes raw `provider.responses()`, Files, Batch, Qwen3-ASR, and DashScope TTS. File Search is exposed as a hosted Responses tool with `vector_store_ids`, not as a lifecycle client. Explicit reasoning-enabled requests must leave forced tool choice on `auto`; when no reasoning mode is selected, the `qwen3.8-max` adapter disables its default thinking only as needed to honor portable forced tool choice. Batch model availability is regional: Singapore currently documents the stable `qwen-max`, `qwen-plus`, `qwen-flash`, and `qwen-turbo` aliases.
- Kimi/Moonshot uses the official Chat Completions route for portable text generation, streaming, structured output, and callable tools. `create_kimi()` reads `MOONSHOT_API_KEY` first, then `KIMI_API_KEY`, and defaults to `https://api.moonshot.ai/v1`.
- Kimi native support includes K3 (`kimi-k3`) with always-on reasoning, `reasoning_effort` values `low`/`high`/`max`, native vision, callable tools and strict structured output. K2.6/K2.5 retain their separate `thinking` contract. Moonshot Files, Batch, token estimation and the beta `provider.formulas()` client remain available; embeddings, speech, and transcription are not claimed.
- DeepSeek uses its official Chat Completions API for portable text generation, streaming, JSON structured output, callable tools, and V4 thinking. `create_deepseek()` reads `DEEPSEEK_API_KEY`, optionally `DEEPSEEK_BASE_URL`, and otherwise targets `https://api.deepseek.com`.
- The current DeepSeek catalog entries are `deepseek-v4-flash` and `deepseek-v4-pro`. The adapter rejects retired `deepseek-chat` / `deepseek-reasoner` IDs, preserves `reasoning_content` across tool loops, maps portable reasoning effort, and fails fast on combinations that DeepSeek thinking does not accept. Vision, files, embeddings, audio, moderation, and hosted tools are not claimed.
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
pip install "zhivex-ai-sdk[a2a]"
pip install "zhivex-ai-sdk[ag-ui]"
pip install "zhivex-ai-sdk[otel]"
pip install "zhivex-ai-sdk[docx]"
```

The beta skill-package layer also exposes a CLI:

```bash
zhivex-skills validate path/to/skill
zhivex-skills install path/to/skill
# Remote packages contain executable Python and require explicit trust:
zhivex-skills install writer@1.0.0 --registry-url https://skills.example.com/index.json --trust-remote-code
```

The beta general CLI runs, evaluates, serves, and opens a local playground for a trusted `module:attribute` agent:

```bash
zhivex run my_app.agents:support_agent --prompt "Draft a reply"
zhivex eval my_app.agents:support_agent --dataset evals/support.json \
  --repetitions 5 --max-concurrency 4 \
  --output-json artifacts/eval.json --output-junit artifacts/eval.xml
zhivex playground my_app.agents:support_agent
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
        model=openai("gpt-5.6-terra"),
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
        model=openai("gpt-5.6-terra"),
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
        model=openai("gpt-5.6-terra"),
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
        model=openai("gpt-5.6-terra"),
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
        model=openai("gpt-5.6-terra"),
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
        model=openai.grounded_language_model("gpt-5.6-terra"),
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
        model=anthropic.grounded_language_model("claude-sonnet-5"),
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
        model=openai("gpt-5.6-terra"),
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
        model=anthropic.native.language_model("claude-sonnet-5"),
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

Using Gemini explicit context caching:

```python
import asyncio

from zhivex_ai import create_gemini, generate_text


async def main() -> None:
    gemini = create_gemini()
    cache = await gemini.caches().create(
        {
            "model": "models/gemini-2.5-flash",
            "displayName": "Product docs",
            "contents": [{"role": "user", "parts": [{"text": "Long reusable context..."}]}],
            "ttl": "3600s",
        }
    )
    result = await generate_text(
        model=gemini.native.language_model("gemini-2.5-flash"),
        prompt="Summarize the cached docs in three bullets.",
        provider_options={"cached_content": cache.name},
    )

    print(result.text)


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
    image = await openai.images().generate(prompt="A paper sketch of a transit map", model="gpt-image-2")

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
        model="gemini-3.1-flash-image",
        prompt="A clean product diagram of a solar microgrid.",
    )
    operation = await gemini.videos().generate(
        model="veo-3.1-generate-preview",
        prompt="A slow cinematic flyover of that microgrid at sunrise.",
    )
    music = await gemini.media().generate_music(
        model="lyria-3-pro-preview",
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
        model=openai("gpt-5.6-terra"),
    )
    triage = Agent(
        name="triage",
        instructions="Delegate research work to the researcher agent.",
        model=openai("gpt-5.6-terra"),
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

Typed agent applications can keep request-scoped services out of prompts and durable state while validating the terminal result:

```python
from dataclasses import dataclass
from pydantic import BaseModel

from zhivex_ai import Agent, AgentContext, create_openai, run_agent


@dataclass
class Deps:
    tenant_id: str


class Decision(BaseModel):
    approved: bool
    reason: str


def instructions(context: AgentContext[Deps]) -> str:
    tenant = context.deps.tenant_id if context.deps else "unknown"
    return f"Evaluate the request for tenant {tenant}. Return a concise decision."


openai = create_openai()
agent: Agent[Deps, Decision] = Agent(
    name="reviewer",
    model=openai("gpt-5.6-terra"),
    instructions=instructions,
    output_type=Decision,
)

result = await run_agent(agent=agent, prompt="Review application A-42.", deps=Deps("bank-ar"))
print(result.output.approved if result.output else None)
print(result.text)  # Raw text remains available for compatibility.
```

`output_mode="auto"` uses native structured output when the active model supports it and a schema-guided prompted fallback otherwise. `stream_agent(...).collect()` returns the same typed result. The root agent's `output_type` governs the final response after direct handoffs.

If you want Gemini research with provider-native web search in an agent run, put the agent on a native Gemini model first:

```python
from zhivex_ai import create_gemini

gemini = create_gemini()
triage.model = gemini.native.language_model("gemini-3.5-flash")
result = await run_agent(
    agent=triage,
    prompt="Research the Apollo migration status.",
    provider_options={"google_search": True},
)
```

Built-in Gemini tools can also be configured directly:

```python
result = await generate_text(
    model=gemini.native.language_model("gemini-3.5-flash"),
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
        model=openai.native.language_model("gpt-5.6-terra"),
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

OpenAI Programmatic Tool Calling is available as a beta native path for bounded, read-only tool workflows. Eligible functions declare `allowed_callers=["programmatic"]` and an optional `output_schema`; the SDK preserves `program`, nested `caller`, function results, and `program_output` items across multi-step and `store=False` replay:

```python
from pydantic import BaseModel, ConfigDict

from zhivex_ai import create_openai, generate_text, openai_programmatic_tool_calling_tool, tool


class WeatherInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str


class WeatherOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature_c: float


openai = create_openai()
result = await generate_text(
    model=openai.native.language_model("gpt-5.6-sol"),
    prompt="Reduce the weather records to one compact result.",
    tools={
        "programmatic": openai_programmatic_tool_calling_tool(),
        "weather": tool(
            name="weather",
            schema=WeatherInput,
            output_schema=WeatherOutput,
            allowed_callers=["programmatic"],
            execute=lambda input: {"temperature_c": 21},
        ),
    },
    max_steps=4,
)
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
    model=openai.native.language_model("gpt-5.6-terra"),
    prompt="Start a multi-turn responses workflow.",
)

follow_up = openai_response_options(previous_response=first)
assert get_openai_response_id(first) is not None
```

The UI helpers now preserve provider-managed control traffic as `provider-data` chunks and agent approval requests as `tool-approval` chunks, so `to_ui_message_stream(...)` can surface MCP approval requests, response references, pending local approvals, and other typed events to frontend consumers.

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
            },
            # Illustrative application-owned rates; validate them for your account and effective date.
            model_costs_per_1k_tokens={
                "openai": {"gpt-5.6-terra": 1.0},
                "anthropic": {"claude-sonnet-5": 3.0},
            },
        )
    )

    result = await gateway.generate(
        messages=[GatewayMessage(role="user", content="Say hello in one sentence.")],
        primary=GatewayModelTarget(provider="openai", model_id="gpt-5.6-terra"),
        fallbacks=[GatewayModelTarget(provider="anthropic", model_id="claude-sonnet-5")],
        max_cost_per_1k_tokens=3.0,
    )

    print(result.text)
    print(result.provider_used, result.model_used)


asyncio.run(main())
```

## Provider Factories

The core package exposes portable and Tier-1 factories from `zhivex_ai`:

- `create_openai()`
- `create_azure_openai()`
- `create_anthropic()`
- `create_gemini()`
- `create_vertex()`
- `create_vllm()`
- `create_qwen()`
- `create_kimi()`
- `create_meta()`
- `create_deepseek()`

Native-only or compatibility providers are explicit Experimental imports:

- `from zhivex_ai.experimental import create_bedrock`
- `from zhivex_ai.experimental import create_openrouter`
- `from zhivex_ai.experimental import create_ollama`

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

OpenAI-compatible providers such as OpenRouter, Qwen, Ollama, and vLLM reuse normalized adapter paths internally. Qwen and vLLM participate in the tier-1 portable story; Ollama and OpenRouter remain outside the tier-1 portable contract. Kimi/Moonshot and DeepSeek use dedicated Chat Completions adapters so their reasoning, tool replay, streaming, and error semantics remain faithful to their official APIs.

Meta uses a dedicated Model API adapter rather than the generic OpenAI-compatible helper. `create_meta()` with Standard `muse-spark-1.2` is Stable and tier-1 for portable text, streaming, structured output, callable tools, agent tool loops, and application-supplied retrieval through `PortableRetrievalConfig`; `tool_choice` is `auto` only. Portable retrieval is SDK-owned prompt context, not Meta Files or hosted search. Responses routing for hosted tools, file inputs, native `previous_response_id` continuation, Contributor models, and other native extras remains Beta.

Azure OpenAI supports API key authentication and Microsoft Entra ID authentication through the versionless `/openai/v1` route. API key usage reads `AZURE_OPENAI_API_KEY` plus `AZURE_OPENAI_ENDPOINT`; Entra ID usage passes a token or token provider explicitly and is mutually exclusive with API keys.

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from zhivex_ai import create_azure_openai

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://ai.azure.com/.default",
)

provider = create_azure_openai(
    endpoint="https://YOUR-RESOURCE-NAME.openai.azure.com",
    entra_token_provider=token_provider,
)
```

The SDK does not depend on `azure-identity`; install it in your application if you want to use `DefaultAzureCredential`.

Qwen/Alibaba Cloud Model Studio native usage:

```python
import asyncio

from zhivex_ai import ReasoningConfig, create_qwen, generate_text, qwen_web_search_tool


async def main() -> None:
    provider = create_qwen(region="intl")  # qwen3.8-max GA pay-as-you-go
    result = await generate_text(
        model=provider.native.language_model("qwen3.8-max"),
        prompt="Summarize the latest Qwen hosted tool surface.",
        tools={"search": qwen_web_search_tool()},
        reasoning=ReasoningConfig(effort="medium"),
    )
    print(result.text)


asyncio.run(main())
```

`create_qwen()` reads `QWEN_API_KEY` or the official `DASHSCOPE_API_KEY`. By default it targets Alibaba Cloud Model Studio's Singapore-compatible endpoint with `region="intl"` and uses the current `/compatible-mode/v1/responses` path; use `region="us"` for US Virginia or `region="cn"` for China Beijing. Existing DashScope domains remain supported, while a workspace-specific domain can be supplied with `base_url=...`; reserve `responses_base_url=...` for a gateway whose Responses root differs. GA `qwen3.8-max` uses Responses for text, streaming, images, reasoning, functions, and hosted tools. The adapter selects `/chat/completions` only for native JSON Schema output, `FilePart(url=..., media_type="video/mp4")` input, or a reasoning token budget; structured output is always sent with thinking disabled. The Token Plan's exact `qwen3.8-max-preview` ID remains separate from the GA pay-as-you-go model. Web Extractor must be registered together with Web Search, and explicit thinking cannot be combined with forced required/named tool choice. Singapore Batch currently supports the stable `qwen-max`, `qwen-plus`, `qwen-flash`, and `qwen-turbo` aliases, so check regional model availability before submitting a batch.

See `examples/text/qwen_native.py` for a fuller provider-specific example covering Qwen3.8 text/reasoning, JSON Schema output, optional video, hosted web search, embeddings, optional Qwen3-ASR, and optional Qwen3-TTS.

Meta Model API usage:

```python
import asyncio

from zhivex_ai import create_meta, generate_text, meta_web_search_tool


async def main() -> None:
    meta = create_meta()  # MODEL_API_KEY
    result = await generate_text(
        model=meta.native.language_model("muse-spark-1.2"),
        prompt="Summarize the latest relevant public information.",
        tools={"web_search": meta_web_search_tool()},
    )
    print(result.text)


asyncio.run(main())
```

Use `meta("muse-spark-1.2")` for the portable text/tool/structured-output path and `PortableRetrievalConfig` when the application already owns the retrieved document text. That retrieval path only adds bounded context to the portable request. Hosted `web_search` and `tool_search`, raw Responses continuation, and Files are Beta native features. The Contributor variant has different data-use terms and is never selected implicitly. Muse Glimmer and Llama 4 are open-weight/host routes rather than direct Meta Model API IDs; their exact structured-output and tool behavior depends on the runtime. See [docs/providers/meta.md](./docs/providers/meta.md) and [meta_text.py](./examples/text/meta_text.py).

vLLM usage targets its OpenAI-compatible server:

```python
import asyncio

from zhivex_ai import create_vllm, generate_text


async def main() -> None:
    provider = create_vllm(base_url="http://localhost:8000/v1")
    result = await generate_text(
        model=provider("NousResearch/Meta-Llama-3-8B-Instruct"),
        prompt="Explain vLLM in one sentence.",
    )
    print(result.text)


asyncio.run(main())
```

`create_vllm()` reads `VLLM_API_KEY` and `VLLM_BASE_URL`; local development can omit the API key and use the default compatibility token `vllm`. The tier-1 guarantee covers the SDK primitives vLLM exposes through OpenAI-compatible routes: text generation, streaming, structured output/tools, embeddings, transcription, and realtime ASR. Model-specific tasks still matter: embeddings require an embedding model, transcription/realtime require ASR-capable vLLM setup, and vLLM custom endpoints such as tokenize, rerank, classify, and score are intentionally outside the SDK surface for now.

Kimi/Moonshot native usage:

```python
import asyncio

from zhivex_ai import ImagePart, ReasoningConfig, create_kimi, generate_text, user


async def main() -> None:
    kimi = create_kimi()  # MOONSHOT_API_KEY, then KIMI_API_KEY

    result = await generate_text(
        model=kimi.native.language_model("kimi-k3"),
        messages=[
            user(
                [
                    ImagePart(image="data:image/png;base64,..."),
                ]
            )
        ],
        reasoning=ReasoningConfig(effort="high"),
    )
    print(result.text)


asyncio.run(main())
```

Kimi files, batch jobs, token estimation, and official tools are native-only:

```python
kimi = create_kimi()

uploaded = await kimi.files().upload(
    data=b'{"custom_id":"1","method":"POST","url":"/v1/chat/completions","body":{"model":"kimi-k3","messages":[{"role":"user","content":"hi"}]}}\n',
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

tokens = await kimi.tokens().count(model_id="kimi-k3", prompt="hello")
tools = await kimi.formulas().toolset(["moonshot/web-search:latest"])
```

DeepSeek V4 usage:

```python
import asyncio

from zhivex_ai import ReasoningConfig, create_deepseek, generate_text


async def main() -> None:
    deepseek = create_deepseek()  # DEEPSEEK_API_KEY
    result = await generate_text(
        model=deepseek("deepseek-v4-flash"),
        prompt="Explain the portable DeepSeek integration in one sentence.",
        reasoning=ReasoningConfig(effort="high"),
    )
    print(result.text)


asyncio.run(main())
```

DeepSeek thinking is enabled by default for V4. Portable `temperature`, `top_p`, or forced `tool_choice` requests automatically select non-thinking mode unless reasoning was explicitly requested; explicitly combining thinking with an incompatible option fails before network dispatch. Strict callable tools and assistant prefix completion use DeepSeek's beta route automatically. See `examples/text/deepseek_native.py` for text and structured-output usage.

Local Ollama usage follows the same native escape hatch:

```python
import asyncio

from zhivex_ai import generate_text
from zhivex_ai.experimental import create_ollama


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

- `provider.native.realtime_model("gpt-realtime-2.1")`
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

OpenAI and Azure OpenAI providers may additionally expose low-level lifecycle clients:

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
- `default_model_catalog` keeps recommendation metadata for current reference models such as OpenAI/Azure GPT-5.6 Sol/Terra/Luna, GPT Image 2 and GPT Realtime 2.1, Claude Opus/Fable/Sonnet/Mythos 5 and Opus 4.8, Gemini 3.5 Flash plus current Gemini 3.1 image/live and Omni guidance, Vertex Gemini, Bedrock Claude/Nova, GA Qwen3.8 Max plus retained Qwen3.7 guidance, Kimi K3, Meta Muse Spark, and hosted Muse Glimmer routes. It is guidance for model selection, not a separate execution path or live certification.

To regenerate the markdown tables used above:

```bash
.venv/bin/python scripts/generate_support_matrix.py
```

Notes:

- `provider("model-id")` is shorthand for the portable namespace.
- `provider.native.*` is the only place where provider-specific request shapes belong.
- Meta Model API is Stable and tier-1 only for `create_meta()` with Standard `muse-spark-1.2` portable text, streaming, structured output, callable tools, agent tool loops, and application-supplied retrieval through `PortableRetrievalConfig`. This retrieval capability injects bounded `PortableDocument` text and does not use Meta Files, hosted search, or raw Responses. `meta_hosted_tool()`, `meta_web_search_tool()`, and `meta_tool_search_tool()` remain Beta; forced, disabled, and named tool choice are rejected because the current API accepts only `auto`.
- Some providers support a capability only for specific model IDs even when the adapter exposes the factory.
- `create_gemini().files()` exposes the Gemini Files API. `create_vertex()` does not expose a hosted files client; on Vertex, pass `FilePart(file_uri="gs://...")` or inline media instead.
- `create_anthropic().tokens()` exposes Anthropic message token counting.
- `anthropic_web_search_tool(...)`, `anthropic_web_fetch_tool(...)`, `anthropic_mcp_server(...)`, and `anthropic_code_execution_tool(...)` build the current hosted-tool payloads for Claude-native runs.
- `create_gemini().tokens()` and `create_vertex().tokens()` expose token counting clients.
- `create_gemini().caches()` exposes Gemini explicit context caching through `cachedContents`; pass the returned cache name with `provider_options={"cached_content": cache.name}` or `provider_options={"cachedContent": cache.name}`.
- `create_gemini().file_search_stores()` exposes Gemini File Search store management.
- `embed_content(...)` and `embed_content_many(...)` accept text plus `TextPart`, `ImagePart`, and `FilePart` values for Gemini Embedding 2 style multimodal embeddings; `embed(...)` and `embed_many(...)` remain text-compatible.
- `create_gemini().images()` covers current Gemini/Nano Banana image models such as `gemini-3.1-flash-image`, `gemini-3.1-flash-lite-image`, and `gemini-3-pro-image` through `generateContent`. The legacy Imagen `predict` transport remains implemented for compatibility, but Imagen 4 is deprecated and no longer catalog guidance.
- `create_vertex().images()` mirrors Google image routes through Vertex publisher model endpoints.
- `create_gemini().videos()` and `create_vertex().videos()` expose Veo long-running operation creation, polling, and download helpers, including the current `veo-3.1-*` model IDs where available.
- `create_gemini().media()` and `create_vertex().media()` expose Lyria-style native audio/music generation where the Google model route supports it, including `lyria-3-pro-preview` and `lyria-3-clip-preview`.
- `create_gemini().realtime_model("gemini-3.5-live-translate-preview")` exposes Gemini Live Translate with typed `RealtimeSessionConfig(translation_target_language_code="es", translation_echo_target_language=True, input_audio_media_type="audio/pcm;rate=16000", output_audio_media_type="audio/pcm")` setup. Live Translate is audio-only; text input, tools, and instructions fail fast for that model. The catalog also tracks stable `gemini-3.6-flash` and stable `gemini-3.5-flash-lite` for regular generation paths.
- `create_gemini().batches()` exposes Gemini Batch API generation and embedding jobs.
- `create_gemini().interactions()` exposes Gemini Interactions and Deep Research polling/streaming helpers as a raw beta client, including `gemini-omni-flash-preview` payloads on the Gemini Developer API. Deep Research payloads default to background storage; Omni is not claimed for Vertex.
- `create_openai().file_search_stores()` exposes OpenAI Vector Store / File Search management.
- `create_azure_openai().file_search_stores()` exposes Azure OpenAI Vector Store / File Search management through the versionless `/openai/v1` endpoint and works with either API key or Entra ID authentication.
- `create_vertex()` now exports native grounding helpers such as `vertex_google_search_tool(...)`, `vertex_google_maps_tool(...)`, `vertex_vertex_ai_search_tool(...)`, and `vertex_external_search_tool(...)`.
- `create_vertex().native.language_model(...)` and `create_vertex().native.grounded_language_model(...)` also accept `provider_options={"vertex_ai_search": {...}}` and `provider_options={"external_search": {...}}`, which are normalized into the Vertex tool payloads automatically.
- `create_openai().images()` and `create_openai().uploads()` expose standalone OpenAI Images and Uploads APIs, including GPT Image 2 via `model="gpt-image-2"`.
- `create_openai().containers()` and `create_openai().skills()` expose the raw OpenAI Containers and Skills APIs.
- OpenAI/Azure Sora or video-generation lifecycle clients are intentionally not exposed in this SDK release.
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

For production logging, request correlation, gateway attempt tracing, and OpenTelemetry guidance, see [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md).

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

The stable agent slice covers the run/result/stream lifecycle, local tools, direct handoffs, sessions, Postgres persistence, replay, and durable approvals. Stable workflow orchestration is documented separately below. Items explicitly described as beta—including native subagent tools, named external workflow-engine factories, local agent run stores, evaluation helpers, and trace artifacts—may still evolve between minor releases.

- `Agent(...)`
- `AgentCancellationToken`
- `AgentContext`, `AgentHooks`, `AgentMiddleware`, `AgentRunRequest`
- `AgentRuntime(...)`
- `AgentRegistry(...)`
- `ToolRegistry(...)`
- `AgentSession`
- `run_agent(...)`
- `resume_agent(...)`
- `resume_agent_run(...)`
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
- `run_agent_evaluation_experiment(...)`
- `create_agent_evaluation_report(...)`
- `create_safety_policy(...)`
- `apply_safety_policy_to_agent(...)`
- `load_agent_session(...)`
- `ApprovalDecision`, `ToolApprovalRequest`
- `PendingApproval`, `get_pending_agent_approvals(...)`
- `GuardrailResult`, `InputGuardrailRequest`, `OutputGuardrailRequest`
- `ToolGuardrailResult`, `ToolInputGuardrailRequest`, `ToolOutputGuardrailRequest`
- `permission_allowlist_approval_policy(...)`
- input and output guardrails on `Agent(...)`
- annotation-derived `@tool` schemas and input/output guardrails on each tool
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

This layer is intended for stateful, tool-using, multi-agent assistants where you want typed dependencies and outputs, dynamic instructions, lifecycle hooks, run middleware, executable handoffs, native subagent tools, shared sessions, transcript + summary memory, durable pending approvals, cooperative cancellation, replay/evaluation, traces, durable run state, and MCP-backed tool registries without rewriting the lower-level loop yourself.

For production semantics, persistence, approvals, tool registries, event ordering, and recovery guidance, see [docs/AGENTS.md](./docs/AGENTS.md) and [docs/PRODUCTION.md](./docs/PRODUCTION.md).

Durable workflow graphs are available when coordination is known ahead of time and execution needs branching, per-transition checkpoints, interruption, resume, or fork:

```python
from zhivex_ai.workflows import (
    WorkflowBuilder,
    WorkflowStep,
    create_sqlite_workflow_checkpoint_store,
    resume_workflow,
)

store = create_sqlite_workflow_checkpoint_store("./workflow-checkpoints.sqlite3")
pipeline = (
    WorkflowBuilder("loan_pipeline", definition_version="1")
    .add_step(
        WorkflowStep("extract", extractor, prompt="Extract the application", output_key="application"),
        entrypoint=True,
    )
    .add_step(WorkflowStep("review", reviewer, input_template="Review {application}", output_key="review"))
    .add_step(WorkflowStep("decide", decider, input_template="Decide with {review}", output_key="decision"))
    .add_edge("extract", "review")
    .add_edge("review", "decide")
    .interrupt_before("review", reason="Human review")
    .build(checkpoint_store=store)
)

suspended = await pipeline.run(idempotency_key="loan-123")
pending = suspended.checkpoint.pending_interrupt
result = await resume_workflow(
    pipeline,
    suspended.run_id,
    interrupt_id=pending.interrupt_id,
    resume_value={"approved": True},
)
```

`WorkflowGraph` validates an acyclic definition, runs ready nodes in bounded parallel waves, and persists routing decisions before downstream dispatch. `WorkflowCheckpoint` is the canonical durable record; `WorkflowRunResult.state_snapshot` remains an agent-run projection for replay compatibility. Use step/edge `definition_revision` values for application configuration that callable source inspection cannot capture. Optional in-memory, SQLite, and Postgres lease managers add TTL, heartbeat, monotonic fencing, and atomic stale-owner rejection when paired with the matching checkpoint backend. Postgres also supports bounded or application-owned pools, namespaces, server-clock lease decisions, and checked schema metadata; validate it against the actual deployment database.

`fork_workflow(...)` creates a new run with explicit source lineage, while `cancel_workflow(...)` appends cooperative cancellation. `WorkflowRetryPolicy` retries a complete logical step separately from the existing model/provider `max_retries`. Applications must still deduplicate external writes and supply runtime dependencies again after resume. In `0.20.0` these workflow graph, declarative agent, checkpoint/store/lease, resume/fork/cancel, migration, and generic callback-envelope APIs are Stable.

Checkpoint schema v2 adds auditable migration history. Use `migrate_workflow_checkpoint(...)` for an in-memory or payload migration and `migrate_workflow_run_checkpoint(...)` to append a migrated latest checkpoint with compare-and-swap protection. Published schema-v1 checkpoints remain readable and canonically serializable; pause active workers before an operational migration. Terminal v1 runs remain readable and are not rewritten.

Re-entering a still-running idempotent workflow fails closed. With a lease manager, `recover_running=True` can take over only after expiry and increments the fence; without one, it remains an operator-reconciled operation. Recovery cannot make unknown external effects safe without destination idempotency or reconciliation.

Graph steps may use an `Agent` or a sync/async functional `executor`. Functional steps receive `WorkflowFunctionContext` with ephemeral dependencies and a stable idempotency key, then return a finite JSON value or `WorkflowFunctionResult` with an output, state patch, and metadata. They are graph-only and do not change the existing declarative agents.

The DBOS, Temporal, Prefect, and Restate adapter factories remain Beta. They create versioned callback contracts and conservative capability metadata, but do not install or operate those engines and are not certified integrations. See [docs/WORKFLOWS.md](./docs/WORKFLOWS.md) for graph validation, branching, persistence, security, migration, retry, resume/fork, and adapter boundaries.

For a runnable offline durable-graph flow, see [`examples/agents/durable_graph_workflow.py`](./examples/agents/durable_graph_workflow.py). The broader references in [`examples/agents/small_business_loan_agent.py`](./examples/agents/small_business_loan_agent.py), [`examples/agents/hr_candidate_selection_agent.py`](./examples/agents/hr_candidate_selection_agent.py), and the focused workflow examples model regulated review, structured validation, document artifacts, and research reports. The SDK owns orchestration primitives; applications keep credit/hiring policy, authorization, persistence, approval UI, external systems, artifact storage, and compliance controls behind replaceable interfaces.

Durable run state can be attached directly to an agent:

```python
from zhivex_ai import Agent, create_in_memory_agent_run_store, run_agent

store = create_in_memory_agent_run_store()
agent = Agent(name="assistant", model=model, run_store=store)
result = await run_agent(agent=agent, prompt="Draft a reply", idempotency_key="reply-1")
state = await store.load(result.run_id)
```

Built-in stores use optimistic revisions and atomic idempotency/cancellation claims. Pair `AgentCancellationToken` with `cancel_agent_run_tree(..., cancellation_token=token)` when the same process should also interrupt an active provider wait and signal cooperative tools. If an operator cancellation wins while a worker is still active, the durable state remains `cancelled` and the worker raises `AgentRunCancelled` instead of publishing or persisting a late success.

Tools can suspend a run for human approval by returning `ApprovalDecision.require_human(...)` from `approval_policy`. A tool with `requires_approval=True` fails closed if the agent has no approval policy. Load the pending request with `get_pending_agent_approvals(...)`, then call `resume_agent_run(...)` after the user approves or denies the tool call. Built-in run stores atomically claim the approval before executing it, so concurrent resume attempts cannot invoke the same tool twice.

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

The beta evaluation layer also supports repeated, bounded-concurrency trials with strict JSON/JUnit artifacts, finite pass-rate confidence intervals, latency/token/application-cost metrics, redacted trajectories, and baseline-aware CI gates:

```python
from zhivex_ai.evals import AgentEvaluationGate, run_agent_evaluation_experiment

experiment = await run_agent_evaluation_experiment(
    variants={"baseline": baseline_agent, "candidate": candidate_agent},
    baseline="baseline",
    dataset=dataset,
    gates=[AgentEvaluationGate("pass_rate", minimum=0.95, max_regression=0.01)],
    repetitions=5,
    max_concurrency=4,
)
```

Protocol adapters remain beta. A2A delegates wire semantics to the official SDK and accepts application task/context/queue infrastructure; AG-UI carries trusted run options and safe errors; Responses strictly validates the documented text subset and can inject application-owned result/event replay. Protocol IDs never authorize access. `create_agent_playground_app(...)` is development-only and `zhivex playground` refuses non-loopback binds. See [docs/PROTOCOLS.md](./docs/PROTOCOLS.md), [docs/EVALUATIONS.md](./docs/EVALUATIONS.md), and [docs/CLI.md](./docs/CLI.md).

Agent skills are also available across the agent runtime. These are provider-agnostic workflow packs that inject task-specific instructions and optional tool dependencies before a run starts. They are distinct from the raw OpenAI Skills API:

- `Agent(..., skills=...)` activates provider-agnostic runtime skills
- `skill(...)`, `load_skill(...)`, and `discover_skills(...)` help define or load skills from `SKILL.md`
- `set_agent_session_skills(...)`, `get_agent_session_skills(...)`, and `clear_agent_session_skills(...)` manage sticky session skills explicitly
- `provider.skills()` remains the native OpenAI lifecycle client for hosted OpenAI skills
- activated skills stick to the agent session through `session.metadata["sticky_skills"]`
- the runtime emits `AgentSkillActivatedEvent` and `AgentSkillSkippedEvent` for observability

Runtime skills follow the Codex-style `SKILL.md` layout: frontmatter with `name` and `description`, instruction body, and optional `agents/openai.yaml` metadata for display text, implicit-invocation policy, and MCP tool dependencies.

The SDK now also includes a beta skill-package layer for Anthropic-style packaged skills. This adds `skill.yaml`, installable skill packages, a static HTTPS registry flow, direct `run_skill(...)`, artifacts, and the `zhivex-skills` CLI. The packaged-skill APIs are:

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

Skill entrypoints are executable Python loaded into your application process. Agent tools generated from package entrypoints carry `code-execution` permission and require an approval policy; direct `run_skill(...)` is an explicit trusted-code call. Permission declarations validate declared path inputs and the network policy provides a runtime guard, but neither is an OS sandbox. Registry installs therefore fail closed until `trust_remote_code=True` (or CLI `--trust-remote-code`) is supplied. Use that opt-in only after reviewing and trusting the registry/package, and isolate untrusted code in an app-owned container or VM. Registry transport, same-origin artifact URLs, checksums, archive paths/links, download and extraction sizes, and installed content hashes are validated before execution. Legacy remote-package locks without `content_checksum` must be reviewed and reinstalled; legacy local-package locks remain compatible.

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
        model=openai("gpt-5.6-terra"),
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
from pathlib import Path

from zhivex_ai import (
    Agent,
    ApprovalDecision,
    ToolApprovalRequest,
    create_mcp_tool_registry,
    create_openai,
    mcp_stdio_server,
    run_agent,
)


READ_ONLY_FILESYSTEM_TOOLS = {
    "directory_tree",
    "get_file_info",
    "list_allowed_directories",
    "list_directory",
    "list_directory_with_sizes",
    "read_file",
    "read_media_file",
    "read_multiple_files",
    "read_text_file",
    "search_files",
}


async def approve_read_only_filesystem(request: ToolApprovalRequest) -> ApprovalDecision:
    remote_name = str(request.tool_metadata.get("mcp_tool_name") or "")
    if request.tool_source == "mcp" and remote_name in READ_ONLY_FILESYSTEM_TOOLS:
        return ApprovalDecision(approved=True)
    return ApprovalDecision(approved=False, reason="Only read-only filesystem MCP tools are allowed.")


async def main() -> None:
    allowed_root = (Path.cwd() / "examples").resolve()
    async with await create_mcp_tool_registry(
        mcp_stdio_server(
            name="fs",
            command="bunx",
            args=["@modelcontextprotocol/server-filesystem@2026.7.10", str(allowed_root)],
        )
    ) as tools:
        openai = create_openai()
        agent = Agent(
            name="assistant",
            instructions="Use the filesystem MCP tools when needed.",
            model=openai("gpt-5.6-terra"),
            tools=tools,
            approval_policy=approve_read_only_filesystem,
        )

        result = await run_agent(agent=agent, prompt="List the Python files in the allowed examples directory.")
        print(result.text)


asyncio.run(main())
```

The example pins the MCP server package, limits its filesystem root, and denies every tool outside an explicit read-only allowlist. Review and pin MCP servers before use; broader filesystem, network, write, or delete capabilities require an application-owned sandbox and approval policy. `discover_mcp_tools(...)` remains available when you want raw tool definitions or full control over prefixes and registry composition. `ToolRegistry` supports `async with` so MCP-backed runtimes can be closed cleanly after use.

## Examples

See [examples/README.md](./examples/README.md) for the full list. Highlights:

- Text: [openai_text.py](./examples/text/openai_text.py), [meta_text.py](./examples/text/meta_text.py), [stream_text.py](./examples/text/stream_text.py), [structured_output.py](./examples/text/structured_output.py), [deepseek_native.py](./examples/text/deepseek_native.py)
- Local Ollama: [ollama_text.py](./examples/text/ollama_text.py)
- Local vLLM: [vllm_text.py](./examples/text/vllm_text.py)
- Agents: [quickstart_agent.py](./examples/agents/quickstart_agent.py), [agent_basic.py](./examples/agents/agent_basic.py), [stream_agent.py](./examples/agents/stream_agent.py), [mcp_tools.py](./examples/agents/mcp_tools.py)
- Agent skills: [skills.py](./examples/agents/skills.py)
- Realtime: [openai_realtime.py](./examples/realtime/openai_realtime.py), [gemini_realtime.py](./examples/realtime/gemini_realtime.py), [live_agent_realtime.py](./examples/realtime/live_agent_realtime.py)
- Audio: [transcribe_audio.py](./examples/audio/transcribe_audio.py), [generate_speech.py](./examples/audio/generate_speech.py)
- Integrations: [ui_messages.py](./examples/integrations/ui_messages.py), [http_responses.py](./examples/integrations/http_responses.py), [gateway_fallback.py](./examples/integrations/gateway_fallback.py)
- Production: [fastapi_agent_api.py](./examples/production/fastapi_agent_api.py), [worker_resume.py](./examples/production/worker_resume.py)

For real provider validation, the repo also includes a live smoke runner:

```bash
export ZHIVEX_SMOKE_OPENAI_MODEL=your-openai-model
export ZHIVEX_SMOKE_GEMINI_MODEL=your-gemini-model
export ZHIVEX_SMOKE_ANTHROPIC_MODEL=claude-opus-5
export ZHIVEX_SMOKE_AZURE_OPENAI_MODEL=your-azure-openai-deployment
export ZHIVEX_SMOKE_VERTEX_MODEL=your-vertex-model
export ZHIVEX_SMOKE_QWEN_MODEL=qwen3.8-max
export ZHIVEX_SMOKE_KIMI_MODEL=your-kimi-model
export ZHIVEX_SMOKE_DEEPSEEK_MODEL=deepseek-v4-flash
export ZHIVEX_SMOKE_META_MODEL=muse-spark-1.2
export ZHIVEX_SMOKE_VLLM_MODEL=your-vllm-model
export ZHIVEX_SMOKE_OLLAMA_MODEL=your-local-ollama-model
export ZHIVEX_SMOKE_QWEN_REGION=intl
make smoke
```

It only runs providers that have the required credentials and model IDs configured, and you can scope it with `ZHIVEX_SMOKE_PROVIDERS=openai,anthropic,azure-openai,gemini,vertex,qwen,kimi,deepseek,meta,vllm`. Run `ZHIVEX_SMOKE_PROVIDERS=openai make smoke-agents` for the strict agent-first gate: every explicitly selected provider must execute its generation smoke and a real `run_agent(...)` loop that calls a local nonce-validation tool exactly once, consumes its result, and finishes successfully. Without a selector, strict mode keeps the local-development rule that at least one configured provider must execute. Secret values, authenticated URLs, paths, and query strings are redacted from reported failures. PyPI and TestPyPI publication additionally require the protected `release-smoke` GitHub environment; its configured provider subset runs from the exact verified wheel before the Trusted Publisher job can start. The 0.21.0 policy pins `gpt-5.6-luna` plus Beta `muse-spark-1.2-contributor` for non-sensitive synthetic canaries; that evidence certifies only those exact targets and does not certify Meta Standard. Tier-1 setup details live in [docs/providers/tier-1.md](./docs/providers/tier-1.md). Optional Google media smoke checks are gated behind `ZHIVEX_SMOKE_GOOGLE_MEDIA=1` and model IDs such as `ZHIVEX_SMOKE_GEMINI_IMAGE_MODEL`, `ZHIVEX_SMOKE_GEMINI_VIDEO_MODEL`, `ZHIVEX_SMOKE_GEMINI_MEDIA_MODEL`, `ZHIVEX_SMOKE_VERTEX_IMAGE_MODEL`, `ZHIVEX_SMOKE_VERTEX_VIDEO_MODEL`, and `ZHIVEX_SMOKE_VERTEX_MEDIA_MODEL`. Ollama smoke runs default to `http://localhost:11434/v1` and can be redirected with `ZHIVEX_SMOKE_OLLAMA_BASE_URL`. Qwen smoke uses `DASHSCOPE_API_KEY` or `QWEN_API_KEY`, supports `ZHIVEX_SMOKE_QWEN_BASE_URL` and `ZHIVEX_SMOKE_QWEN_RESPONSES_BASE_URL` overrides, and can optionally validate embeddings, ASR, and TTS with `ZHIVEX_SMOKE_QWEN_EMBEDDING_MODEL`, `ZHIVEX_SMOKE_QWEN_ASR_MODEL` plus `ZHIVEX_SMOKE_QWEN_ASR_AUDIO_PATH`, and `ZHIVEX_SMOKE_QWEN_TTS_MODEL`. Kimi smoke uses `MOONSHOT_API_KEY` or `KIMI_API_KEY`, with optional `MOONSHOT_BASE_URL` or `ZHIVEX_SMOKE_KIMI_BASE_URL`. DeepSeek smoke uses `DEEPSEEK_API_KEY` and `ZHIVEX_SMOKE_DEEPSEEK_MODEL`, with optional `DEEPSEEK_BASE_URL` or `ZHIVEX_SMOKE_DEEPSEEK_BASE_URL`. Meta smoke uses `MODEL_API_KEY` and an explicit `ZHIVEX_SMOKE_META_MODEL`; use `muse-spark-1.2` Standard for sensitive smoke data and do not infer live certification until that exact run is recorded.

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
