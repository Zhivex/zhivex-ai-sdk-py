# Examples

This folder contains runnable Python examples for the main public surfaces of the Zhivex AI SDK.

## Layout

- `text/`: basic text generation, streaming, structured output, embeddings, and grounded responses.
- `agents/`: agent orchestration, subagents, memory, remote tools, and MCP integration.
- `realtime/`: provider realtime sessions plus a live agent example.
- `audio/`: transcription and speech generation.
- `integrations/`: UI message helpers, HTTP responses, middleware, gateway fallback, and model catalog helpers.
- `integrations/`: UI message helpers, HTTP responses, middleware, gateway fallback, model catalog helpers, and FastAPI reference APIs.
- `integrations/`: UI message helpers, HTTP responses, middleware, gateway fallback, model catalog helpers, FastAPI reference APIs, and observability patterns.
- `dev/`: provider-specific smoke tests used while iterating locally.

## Live Smoke

There is also a repository-level smoke runner for real provider validation:

```bash
export ZHIVEX_SMOKE_OPENAI_MODEL=your-openai-model
export ZHIVEX_SMOKE_GEMINI_MODEL=your-gemini-model
export ZHIVEX_SMOKE_ANTHROPIC_MODEL=your-anthropic-model
export ZHIVEX_SMOKE_VERTEX_MODEL=your-vertex-model
export ZHIVEX_SMOKE_OLLAMA_MODEL=your-local-ollama-model
make smoke
```

It only runs providers that have the required credentials and model IDs in the environment. You can restrict the run with `ZHIVEX_SMOKE_PROVIDERS=openai,gemini,ollama`.
Ollama uses `http://localhost:11434/v1` by default for smoke runs and can be pointed elsewhere with `ZHIVEX_SMOKE_OLLAMA_BASE_URL`.

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
.venv/bin/python examples/text/stream_text.py
.venv/bin/python examples/text/structured_output.py
.venv/bin/python examples/agents/agent_basic.py
.venv/bin/python examples/agents/stream_agent.py
.venv/bin/python examples/agents/mcp_tools.py
.venv/bin/python examples/realtime/openai_realtime.py
```

## By Category

### Text

```bash
.venv/bin/python examples/text/openai_text.py
.venv/bin/python examples/text/ollama_text.py
.venv/bin/python examples/text/stream_text.py
.venv/bin/python examples/text/stream_object.py
.venv/bin/python examples/text/structured_output.py
.venv/bin/python examples/text/embeddings.py
.venv/bin/python examples/text/grounded_text.py
```

### Agents

```bash
.venv/bin/python examples/agents/agent_basic.py
.venv/bin/python examples/agents/stream_agent.py
.venv/bin/python examples/agents/resume_agent.py
.venv/bin/python examples/agents/messages_and_tools.py
.venv/bin/python examples/agents/remote_tool.py
.venv/bin/python examples/agents/mcp_tools.py
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

## Notes

- OpenAI and Azure OpenAI currently expose the richest Python feature surface for grounded text and realtime session bootstrap.
- Speech generation is currently available through OpenAI, Azure OpenAI, Gemini, Vertex, OpenRouter, and Qwen adapters in this repo.
- `ollama_text.py` shows the recommended local path for Ollama: `create_ollama(...)` plus `provider.native.language_model(...)`.
- The new agent runtime is provider-agnostic, but it works best with models that support tools and streaming.
- The realtime API is experimental. OpenAI, Azure OpenAI, Gemini, Vertex, and Bedrock now expose `provider.realtime_model(...)`.
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
- `make smoke` runs a stricter live pass against OpenAI, Gemini, Anthropic, Vertex, and optional local Ollama when the corresponding model IDs are configured.
