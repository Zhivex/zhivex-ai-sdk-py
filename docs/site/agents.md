# Agents, tools and durable state

`Agent`, `run_agent`, `stream_agent` and `resume_agent` provide the Stable runtime.
Tools belong to the application: define narrow inputs, validate them, and return
bounded results. Use an approval before a side effect such as changing a record.

```python
from zhivex_ai import Agent, run_agent, stream_agent, resume_agent, tool
```

An in-memory session is useful during development. For durable production runs,
use the documented Postgres memory, checkpoint and run-store factories. Application
code owns the database, tenant partitioning, authorization and recovery policy.
In-memory and SQLite agent stores remain Beta; do not infer their stability from
the Stable agent runtime.

The [API reference](reference/root.md) lists the exact store and approval contracts.
The existing [agent guide]({{SOURCE}}/docs/AGENTS.md) includes tools, typed context,
checkpoints and resume. The timed onboarding journey is separate work (PY-HU-14);
this page does not claim a measured onboarding time.
