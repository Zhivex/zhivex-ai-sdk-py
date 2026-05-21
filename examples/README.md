# Examples

This folder contains runnable Python examples for the main public surfaces of the Zhivex AI SDK.

## Layout

- `text/`: basic text generation, streaming, structured output, embeddings, and grounded responses.
- `agents/`: agent orchestration, subagents, approvals, durable state, replay, traces, memory, remote tools, and MCP integration.
- `realtime/`: provider realtime sessions plus a live agent example.
- `audio/`: transcription and speech generation.
- `integrations/`: UI message helpers, HTTP responses, middleware, gateway fallback, model catalog helpers, FastAPI reference APIs, and observability patterns.
- `dev/`: provider-specific smoke tests used while iterating locally.
- Tier-1 providers: `text/tier1_providers.py` runs the portable text path for whichever tier-1 provider credentials and `ZHIVEX_EXAMPLE_*_MODEL` values are configured.

## Live Smoke

There is also a repository-level smoke runner for real provider validation:

```bash
export ZHIVEX_SMOKE_OPENAI_MODEL=your-openai-model
export ZHIVEX_SMOKE_GEMINI_MODEL=your-gemini-model
export ZHIVEX_SMOKE_ANTHROPIC_MODEL=your-anthropic-model
export ZHIVEX_SMOKE_AZURE_OPENAI_MODEL=your-azure-openai-deployment
export ZHIVEX_SMOKE_VERTEX_MODEL=your-vertex-model
export ZHIVEX_SMOKE_QWEN_MODEL=your-qwen-model
export ZHIVEX_SMOKE_KIMI_MODEL=your-kimi-model
export ZHIVEX_SMOKE_OLLAMA_MODEL=your-local-ollama-model
export ZHIVEX_SMOKE_VLLM_MODEL=your-local-vllm-model
export ZHIVEX_SMOKE_QWEN_REGION=intl
make smoke
```

It only runs providers that have the required credentials and model IDs in the environment. You can restrict the run with `ZHIVEX_SMOKE_PROVIDERS=openai,anthropic,azure-openai,gemini,vertex,qwen,kimi,vllm`.
Optional Gemini/Vertex media smoke checks are gated behind `ZHIVEX_SMOKE_GOOGLE_MEDIA=1` plus the matching image, video, or media model ID environment variable.
Ollama uses `http://localhost:11434/v1` by default for smoke runs and can be pointed elsewhere with `ZHIVEX_SMOKE_OLLAMA_BASE_URL`.
vLLM uses `http://localhost:8000/v1` by default and can be pointed elsewhere with `ZHIVEX_SMOKE_VLLM_BASE_URL` and `ZHIVEX_SMOKE_VLLM_API_KEY`.
Qwen uses `DASHSCOPE_API_KEY` or `QWEN_API_KEY`; optional checks are enabled by `ZHIVEX_SMOKE_QWEN_EMBEDDING_MODEL`, `ZHIVEX_SMOKE_QWEN_ASR_MODEL` plus `ZHIVEX_SMOKE_QWEN_ASR_AUDIO_PATH`, and `ZHIVEX_SMOKE_QWEN_TTS_MODEL`.

If a realtime example fails on macOS with `ssl.SSLCertVerificationError: CERTIFICATE_VERIFY_FAILED`, that usually means the local Python install is missing CA roots. You can work around it per-command with:

```bash
SSL_CERT_FILE="$(".venv/bin/python" -c 'import certifi; print(certifi.where())')" \
GOOGLE_API_KEY=... \
.venv/bin/python examples/realtime/gemini_realtime.py
```

For a permanent fix on python.org installs, run:

```bash
"/Applications/Python 3.14/Install Certificates.command"
```

## Run

From the repository root:

```bash
make dev
.venv/bin/python examples/text/openai_text.py
```

Most examples require provider credentials in environment variables. The files show which provider setup they expect.

The FastAPI examples also require:

```bash
pip install "zhivex-ai-sdk[api]"
```

Suggested order if you are new to the SDK:

```bash
.venv/bin/python examples/text/openai_text.py
.venv/bin/python examples/text/tier1_providers.py
.venv/bin/python examples/text/kimi_native.py
.venv/bin/python examples/text/stream_text.py
.venv/bin/python examples/text/structured_output.py
.venv/bin/python examples/text/native_hosted_tools.py
.venv/bin/python examples/agents/agent_basic.py
.venv/bin/python examples/agents/stream_agent.py
.venv/bin/python examples/agents/platform_parity.py
.venv/bin/python examples/agents/multi_agent_handoff.py
.venv/bin/python examples/agents/human_approval.py
.venv/bin/python examples/agents/durable_state_resume.py
.venv/bin/python examples/agents/replay_and_trace.py
.venv/bin/python examples/agents/small_business_loan_agent.py
.venv/bin/python examples/agents/hr_candidate_selection_agent.py
.venv/bin/python examples/agents/sequential_workflow.py
.venv/bin/python examples/agents/parallel_workflow.py
.venv/bin/python examples/agents/loop_workflow.py
.venv/bin/python examples/agents/structured_workflow_outputs.py
.venv/bin/python examples/agents/workflow_resume.py
.venv/bin/python examples/agents/artifact_document_workflow.py
.venv/bin/python examples/agents/research_report_workflow.py
.venv/bin/python examples/agents/provider_managed_approvals.py
.venv/bin/python examples/agents/kimi_official_tools.py
.venv/bin/python examples/agents/mcp_tools.py
.venv/bin/python examples/realtime/openai_realtime.py
```

## By Category

### Text

```bash
.venv/bin/python examples/text/openai_text.py
.venv/bin/python examples/text/tier1_providers.py
.venv/bin/python examples/text/kimi_native.py
.venv/bin/python examples/text/qwen_native.py
.venv/bin/python examples/text/ollama_text.py
.venv/bin/python examples/text/stream_text.py
.venv/bin/python examples/text/stream_object.py
.venv/bin/python examples/text/structured_output.py
.venv/bin/python examples/text/embeddings.py
.venv/bin/python examples/text/grounded_text.py
.venv/bin/python examples/text/native_hosted_tools.py
```

### Agents

```bash
.venv/bin/python examples/agents/agent_basic.py
.venv/bin/python examples/agents/stream_agent.py
.venv/bin/python examples/agents/resume_agent.py
.venv/bin/python examples/agents/messages_and_tools.py
.venv/bin/python examples/agents/remote_tool.py
.venv/bin/python examples/agents/multi_agent_handoff.py
.venv/bin/python examples/agents/human_approval.py
.venv/bin/python examples/agents/durable_state_resume.py
.venv/bin/python examples/agents/replay_and_trace.py
.venv/bin/python examples/agents/small_business_loan_agent.py
.venv/bin/python examples/agents/hr_candidate_selection_agent.py
.venv/bin/python examples/agents/sequential_workflow.py
.venv/bin/python examples/agents/parallel_workflow.py
.venv/bin/python examples/agents/loop_workflow.py
.venv/bin/python examples/agents/structured_workflow_outputs.py
.venv/bin/python examples/agents/workflow_resume.py
.venv/bin/python examples/agents/artifact_document_workflow.py
.venv/bin/python examples/agents/research_report_workflow.py
.venv/bin/python examples/agents/provider_managed_approvals.py
.venv/bin/python examples/agents/kimi_official_tools.py
.venv/bin/python examples/agents/mcp_tools.py
.venv/bin/python examples/agents/skills.py
```

### Realtime

```bash
.venv/bin/python examples/realtime/openai_realtime.py
.venv/bin/python examples/realtime/azure_realtime.py
.venv/bin/python examples/realtime/gemini_realtime.py
.venv/bin/python examples/realtime/bedrock_realtime.py
.venv/bin/python examples/realtime/live_agent_realtime.py
```

### Audio

```bash
.venv/bin/python examples/audio/transcribe_audio.py
.venv/bin/python examples/audio/generate_speech.py
```

### Integrations

```bash
.venv/bin/python examples/integrations/ui_messages.py
.venv/bin/python examples/integrations/http_responses.py
.venv/bin/python examples/integrations/middleware.py
.venv/bin/python examples/integrations/gateway_fallback.py
.venv/bin/python examples/integrations/model_catalog.py
.venv/bin/python examples/integrations/observability.py
.venv/bin/python examples/integrations/operations_hardening.py
```

FastAPI reference apps:

```bash
uvicorn examples.integrations.fastapi_chat_api:app --reload
uvicorn examples.integrations.fastapi_streaming_api:app --reload
uvicorn examples.integrations.fastapi_gateway_api:app --reload
```

### Dev

```bash
.venv/bin/python examples/dev/dev_gemini_grounded_search.py
.venv/bin/python examples/dev/dev_agent_gemini_search_tool.py
```

## Verification Index

| Example | Mode | Requirements | Command | Verification |
| --- | --- | --- | --- | --- |
| `agents/structured_workflow_outputs.py` | offline | dev env | `.venv/bin/python examples/agents/structured_workflow_outputs.py` | `make test-examples` |
| `agents/workflow_resume.py` | offline | dev env | `.venv/bin/python examples/agents/workflow_resume.py` | `make test-examples` |
| `agents/artifact_document_workflow.py` | offline | dev env | `.venv/bin/python examples/agents/artifact_document_workflow.py` | `make test-examples` |
| `agents/research_report_workflow.py` | offline | dev env | `.venv/bin/python examples/agents/research_report_workflow.py` | `make test-examples` |
| `agents/small_business_loan_agent.py` | offline | dev env | `.venv/bin/python examples/agents/small_business_loan_agent.py` | `make test-examples` |
| `agents/hr_candidate_selection_agent.py` | offline | dev env | `.venv/bin/python examples/agents/hr_candidate_selection_agent.py` | `make test-examples` |
| `text/tier1_providers.py` | live optional | one tier-1 provider credential and model env | `.venv/bin/python examples/text/tier1_providers.py` | `ZHIVEX_SMOKE_PROVIDERS=openai make smoke` |
| `integrations/fastapi_chat_api.py` | live optional | `zhivex-ai-sdk[api]` and provider env | `uvicorn examples.integrations.fastapi_chat_api:app --reload` | manual HTTP smoke |
| `integrations/fastapi_streaming_api.py` | live optional | `zhivex-ai-sdk[api]` and provider env | `uvicorn examples.integrations.fastapi_streaming_api:app --reload` | manual streaming smoke |
| `integrations/fastapi_gateway_api.py` | live optional | `zhivex-ai-sdk[api]` and provider env | `uvicorn examples.integrations.fastapi_gateway_api:app --reload` | manual HTTP smoke |
| `agents/mcp_tools.py` | optional extra | `zhivex-ai-sdk[mcp]` and MCP server | `.venv/bin/python examples/agents/mcp_tools.py` | manual MCP smoke |
| `integrations/observability.py` | live optional | provider env, optional `zhivex-ai-sdk[otel]` | `.venv/bin/python examples/integrations/observability.py` | manual log review |
| `integrations/operations_hardening.py` | offline | dev env | `.venv/bin/python examples/integrations/operations_hardening.py` | `make test-examples` |

## Notes

- OpenAI and Azure OpenAI currently expose the richest Python feature surface for grounded text and realtime session bootstrap.
- Speech generation is currently available through OpenAI, Azure OpenAI, Gemini, Vertex, OpenRouter, and Qwen adapters in this repo. Qwen also exposes native Qwen3-ASR transcription through `provider.native.transcription_model("qwen3-asr-flash")`; vLLM exposes transcription and realtime ASR when served with compatible ASR models.
- `ollama_text.py` shows the recommended local path for Ollama: `create_ollama(...)` plus `provider.native.language_model(...)`.
- `vllm_text.py` shows the recommended local path for vLLM's OpenAI-compatible server: `create_vllm(...)` plus the portable `provider("model-id")` path.
- `kimi_native.py` shows the native Kimi/Moonshot Chat Completions path plus Files, Batch, token estimation, and image/video input examples. Kimi also participates in the portable tier-1 text/tool contract through `provider("model-id")`; it expects `MOONSHOT_API_KEY` or `KIMI_API_KEY`.
- `qwen_native.py` shows the native Qwen/Alibaba Cloud Model Studio path for hosted web search, embeddings, optional Qwen3-ASR, and optional Qwen3-TTS. Qwen also participates in the portable tier-1 text/tool contract through `provider("model-id")`; it expects `DASHSCOPE_API_KEY` or `QWEN_API_KEY`.
- The new agent runtime is provider-agnostic, but it works best with models that support tools and streaming.
- `small_business_loan_agent.py` is an offline reference app for regulated, multi-step workflows: the SDK handles orchestration, repair/resume, approvals, traces, and replay, while the example keeps credit rules, pricing, persistence, and approval UI as application-owned components behind replaceable interfaces.
- `hr_candidate_selection_agent.py` is an offline reference app for human-centered HR workflows: the SDK handles resume intake orchestration, interview steps, recruiter review, fairness checks, traces, and replay, while ATS integrations, hiring policy, and compliance systems stay application-owned.
- `structured_workflow_outputs.py`, `workflow_resume.py`, `artifact_document_workflow.py`, and `research_report_workflow.py` are focused offline workflow examples for Pydantic validation, app-owned resume, document artifacts, parallel research, report synthesis, and replay.
- `skills.py` shows the provider-agnostic agent-skill runtime, which is separate from the native OpenAI Skills API.
- `native_hosted_tools.py` is the compact production-style example for mixing local callable tools with provider-managed hosted tools on OpenAI or Azure OpenAI native models.
- `provider_managed_approvals.py` is the compact production-style example for OpenAI/Azure remote MCP approvals with `stream_agent(...)` and `approval_policy`.
- `kimi_official_tools.py` loads Moonshot official Formulas tools through `provider.formulas().toolset(...)` and runs them inside the normal local tool loop.
- The realtime API is experimental. OpenAI, Azure OpenAI, Gemini, Vertex, Bedrock, and vLLM now expose `provider.realtime_model(...)`.
- The FastAPI examples are the recommended reference starting point for production-style API wiring in this repository.
- `observability.py` is the recommended starting point for request IDs, telemetry middleware, and gateway attempt logging.
- Realtime examples need the runtime dependencies installed in the environment you use to run them. If you see a missing `websockets` error, run `make dev` or `pip install -e .` first.
- The Bedrock realtime example requires an injected AWS-signed websocket connection factory.
- `resume_agent.py` and `mcp_tools.py` require optional extras if you want to run them against real backends.
- `mcp_tools.py` shows the recommended high-level MCP flow with `mcp_stdio_server(...)` plus `create_mcp_tool_registry(...)`.
- Some providers do not support every capability. The examples follow the actual adapter capabilities in this repo.
- Structured output examples use `pydantic`.
- Examples that read `.env` files use `python-dotenv` when available, but they still work if you export environment variables manually.
- `transcribe_audio.py` expects a WAV file at `examples/audio/sample.wav`.
- `dev_gemini_grounded_search.py` and `dev_agent_gemini_search_tool.py` are handy local smoke tests when iterating on Gemini search support without publishing a package.
- `make smoke` runs a stricter live pass against OpenAI, Gemini, Anthropic, Vertex, Qwen, and optional local Ollama/vLLM when the corresponding model IDs are configured.
