# Durable Agent State

Agent state is split by purpose:

- memory stores preserve session transcript and summary
- checkpoint stores support `resume_agent(...)`
- run stores support idempotency, replay, snapshots, cancellation records, and pending approvals

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

Postgres run stores and the matching run-state serialization, replay, run-snapshot, cancellation, pending-approval, and approval-resume helpers are stable. In-memory and SQLite run stores are also Stable within the local storage guarantees below.

Built-in stores atomically claim `idempotency_key` before model execution. Concurrent retries reuse the original run and session identity instead of starting a second model loop. Custom stores used for idempotent execution must implement atomic `claim_idempotency_key(state)`; a lookup followed by a separate save is not sufficient.

Every `AgentRunState` carries `schema_version` and an optimistic `revision`. Built-in stores compare the candidate revision with the stored revision inside the same lock or database transaction, then increment it on update. Terminal states cannot be overwritten. In particular, a worker holding revision 0 cannot persist a late `completed` result after an operator has atomically changed the run to `cancelled` at revision 1.

When that race occurs through `run_agent(...)`, the runtime raises `AgentRunCancelled` with the durable `run_id` and cancellation reason. It persists no later `failed` or `completed` state and emits no successful finish event after cancellation.

`cancel_agent_run(...)` and `cancel_agent_run_tree(...)` require the store's atomic `cancel_run(...)` contract. Custom run stores must implement the same compare-and-swap semantics in `save(...)`, an atomic `cancel_run(...)`, atomic idempotency claims, and atomic pending-approval claims. Future `schema_version` values are rejected rather than interpreted as an older format.

`cancel_agent_run_tree(...)` applies that atomic transition to each run discovered during traversal; it is not one database transaction over the entire tree. A child created after its parent lookup can require another reconciliation pass. Production schedulers should stop child dispatch first, propagate cooperative cancellation, then run tree cancellation again after active workers settle.

The base `AgentRunStore` shape remains compatible with stores that only load, query, and save runs; `save(...)` may return the persisted state or `None`. Advanced features fail closed unless the custom store also exposes their named atomic methods. A production store should return the persisted state so callers receive its authoritative revision.

Approval-resume claims deliberately have no automatic retry. If a worker dies after claiming an approval, a lease monitor can read the persisted `resume_claim.claim_token` and call `fail_agent_run_resume_claim(...)` with that exact token. The store atomically moves only that claimed run to `failed` and preserves the approval plus failure metadata for operator reconciliation. A wrong or already-resolved token is a no-op; applications must reconcile an external side effect before starting another run.

SQLite and Postgres enforce uniqueness for non-null idempotency keys. When opening an older development database, the SDK adds the revision/schema columns and unique index. If the existing data contains duplicate idempotency keys, initialization fails with a migration error so an operator can reconcile the duplicates explicitly.

Checkpoint records and checkpoint events omit remote/MCP credentials, sensitive URL credentials/query parameters, provider options, and raw provider responses. They still retain application content needed for diagnostics and continuation, so production deployments must apply tenant isolation, access control, and retention limits.

## Local storage guarantees

The InMemory/SQLite run stores and local memory/checkpoint factories are Stable. Construct them through `zhivex_ai`; implementation-module imports are unsupported.

- InMemory stores are intended for tests and one-process demos. State is lost on restart. Share one run-store instance within one event loop for atomic claims; memory/checkpoint objects may share nested mutable values with callers.
- SQLite requires a persistent local database file on a single host. Create its parent directory before using memory/checkpoint factories. Each operation commits independently; the SDK does not transact together memory, checkpoints, run state, and external tool effects. SQLite lock timeouts and filesystem failures propagate; applications own backups and recovery. Network filesystems and distributed workers are outside this guarantee.
- Run stores enforce optimistic revisions and atomic idempotency/approval claims. A stale save fails with `ValidationError`; a claimed approval cannot be executed through a second successful claim. Namespaces isolate SQLite run IDs and idempotency keys. Claiming an idempotency key is the deduplication operation; ordinary `save` is not a substitute.
- Session memory saves replace the stored snapshot. Applications must serialize read/modify/write for the same session; there is no memory-store compare-and-swap. Checkpoints append records and select the latest by saved time and step index. Serialize writers and avoid identical ordering keys if deterministic latest selection matters.
- Existing supported SQLite rows remain readable, including the pre-extraction 0.22.0 memory/checkpoint format. Incompatible run revisions/schema metadata fail closed. A crash after a tool effect may require reconciliation; a durable claim is not an exactly-once external-effect guarantee. Use application idempotency keys or an outbox for those effects.

Use Postgres for shared workers and validate contention and recovery against the actual deployment database.
