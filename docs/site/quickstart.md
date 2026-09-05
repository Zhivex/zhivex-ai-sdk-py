# First agent

Use Python 3.11 or newer. Create a virtual environment and install the version
shown by this documentation:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install "zhivex-ai-sdk=={{VERSION}}"
```

## Verify the runtime without a provider

Save this as `agent.py` and run `.venv/bin/python agent.py`. The mock model is a
**Beta evaluation helper**; `Agent` and `run_agent` are Stable runtime APIs.
This example makes no network calls and is executed against the installed wheel
when the site is built.

<!-- snippet: snippets/offline_agent.py -->

## Connect a provider

Set `OPENAI_API_KEY` and `ZHIVEX_MODEL` in your server environment. Choose a model
available to your account. Save the following as `live_agent.py` and run it with
the same Python environment. This example performs a billable model call; the
documentation build checks its syntax and public imports without executing it.

<!-- snippet: snippets/live_agent.py -->

Use environment configuration or a secret manager. Keep credentials out of code
and browser bundles. For tools, persistence, approvals and limits, continue to
[Agents](agents.md) and [Production](production.md).
