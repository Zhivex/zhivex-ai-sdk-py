# Durable Agent State

Agent state is split by purpose:

- memory stores preserve session transcript and summary
- checkpoint stores support `resume_agent(...)`
- run stores support idempotency, replay, snapshots, and cancellation records

SQLite is useful for local development:

```python
from zhivex_ai import Agent, create_sqlite_agent_memory_store, create_sqlite_checkpoint_store

agent = Agent(
    name="assistant",
    model=model,
    memory=create_sqlite_agent_memory_store("agent.sqlite3"),
    checkpoint_store=create_sqlite_checkpoint_store("agent.sqlite3"),
)
```

Postgres is the production default:

```python
from zhivex_ai import create_postgres_agent_memory_store, create_postgres_agent_run_store, create_postgres_checkpoint_store

memory = create_postgres_agent_memory_store(dsn)
checkpoints = create_postgres_checkpoint_store(dsn)
runs = create_postgres_agent_run_store(dsn)
```

Postgres run stores and the matching run-state serialization, replay, run-snapshot, and cancellation helpers are stable. In-memory and SQLite run stores remain beta/local-development choices.
