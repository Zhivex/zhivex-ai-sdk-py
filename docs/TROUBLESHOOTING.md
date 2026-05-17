# Troubleshooting

## Import Errors From A Checkout

Run examples from a prepared environment:

```bash
make dev
.venv/bin/python examples/agents/structured_workflow_outputs.py
```

Most examples also add `src/` to `sys.path` for checkout execution, but editable install is still the recommended setup.

## Missing Optional Dependencies

Install the matching extra:

```bash
pip install "zhivex-ai-sdk[api]"
pip install "zhivex-ai-sdk[mcp]"
pip install "zhivex-ai-sdk[postgres]"
pip install "zhivex-ai-sdk[otel]"
pip install "zhivex-ai-sdk[docx]"
```

Inside this repo, `make dev` installs development dependencies used by tests and docs examples.

## Smoke Skips

`make smoke` skips providers without credentials or model IDs. Scope the run while configuring one provider:

```bash
ZHIVEX_SMOKE_PROVIDERS=openai make smoke
```

Use `.env.example` as the variable checklist.

## Provider Auth

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Azure OpenAI: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and deployment model ID
- Gemini: `GEMINI_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`, or `GOOGLE_API_KEY`
- Vertex: `VERTEX_ACCESS_TOKEN` or `GOOGLE_ACCESS_TOKEN`, plus `GOOGLE_CLOUD_PROJECT`
- Qwen: `DASHSCOPE_API_KEY` or `QWEN_API_KEY`, plus model ID
- Kimi/Moonshot: `MOONSHOT_API_KEY` or `KIMI_API_KEY`, plus model ID
- vLLM: local server URL and served model ID

See [PROVIDERS.md](./PROVIDERS.md) for setup links.

## macOS Certificate Errors

If realtime or HTTPS examples fail with `ssl.SSLCertVerificationError`, refresh the Python certificate bundle for the interpreter you are using, then rerun the example. This is usually local Python setup rather than SDK behavior.

## Realtime Dependencies

Realtime examples require `websockets`, which is installed by the base package. If you use an unusual environment, reinstall the package or run `make dev`.

## Support Matrix Drift

If `make check` fails at `support-matrix-check`, regenerate the README matrix only after confirming provider metadata is intentional:

```bash
.venv/bin/python scripts/generate_support_matrix.py --write-readme
```
