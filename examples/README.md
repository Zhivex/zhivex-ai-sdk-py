# Examples

This folder contains runnable Python examples for the main public surfaces of the Zhivex AI SDK.

## Run

From the repository root:

```bash
.venv/bin/python examples/openai_text.py
```

Most examples require provider credentials in environment variables. The files show which provider setup they expect.

Useful starting points:

```bash
.venv/bin/python examples/openai_text.py
.venv/bin/python examples/agent_basic.py
.venv/bin/python examples/stream_agent.py
.venv/bin/python examples/resume_agent.py
.venv/bin/python examples/remote_tool.py
.venv/bin/python examples/mcp_tools.py
.venv/bin/python examples/stream_text.py
.venv/bin/python examples/stream_object.py
.venv/bin/python examples/messages_and_tools.py
.venv/bin/python examples/embeddings.py
.venv/bin/python examples/grounded_text.py
.venv/bin/python examples/transcribe_audio.py
.venv/bin/python examples/generate_speech.py
.venv/bin/python examples/openai_realtime.py
.venv/bin/python examples/azure_realtime.py
.venv/bin/python examples/gemini_realtime.py
.venv/bin/python examples/bedrock_realtime.py
.venv/bin/python examples/live_agent_realtime.py
.venv/bin/python examples/ui_messages.py
.venv/bin/python examples/http_responses.py
.venv/bin/python examples/gateway_fallback.py
.venv/bin/python examples/dev_gemini_grounded_search.py
.venv/bin/python examples/dev_agent_gemini_search_tool.py
```

## Notes

- OpenAI and Azure OpenAI currently expose the richest Python feature surface for grounded text and realtime session bootstrap.
- Speech generation is currently available through OpenAI, Azure OpenAI, Gemini, Vertex, OpenRouter, and Qwen adapters in this repo.
- The new agent runtime is provider-agnostic, but it works best with models that support tools and streaming.
- The realtime API is experimental. OpenAI, Azure OpenAI, Gemini, Vertex, and Bedrock now expose `provider.realtime_model(...)`.
- The Bedrock realtime example requires an injected AWS-signed websocket connection factory.
- `resume_agent.py` and `mcp_tools.py` require optional extras if you want to run them against real backends.
- `mcp_tools.py` shows the recommended high-level MCP flow with `mcp_stdio_server(...)` plus `create_mcp_tool_registry(...)`.
- Some providers do not support every capability. The examples follow the actual adapter capabilities in this repo.
- Structured output examples use `pydantic`.
- `dev_gemini_grounded_search.py` and `dev_agent_gemini_search_tool.py` are handy local smoke tests when iterating on Gemini search support without publishing a package.
