# Migrating From The TypeScript Mental Model

The Python SDK follows the same product boundaries as the TypeScript SDK, but it is Pythonic and async-first.

## Imports

Use top-level imports:

```python
from zhivex_ai import Agent, create_openai, generate_text
```

Deep imports are not part of the stable contract unless documented.

## Providers

Provider bundles expose portable and native paths:

- Portable: `provider("model-id")`
- Explicit portable: `provider.portable.language_model("model-id")`
- Native: `provider.native.*`

Portable code should only use SDK-owned cross-provider options. Provider-specific hosted tools and lifecycle APIs belong in native paths.

## Async Runtime

The SDK is async-first. Use `await generate_text(...)`, `await run_agent(...)`, and `await workflow.run(...)` inside your application event loop.

## Agents And Workflows

Use agents for dynamic tool-using assistants. Use workflows when the orchestration graph is known ahead of time.

The SDK owns orchestration primitives. Your application owns:

- business policy
- persistence of vertical records
- human approval UI and queues
- provider routing policy
- compliance and audit systems

## Stability

Read [../STABILITY.md](../STABILITY.md) before depending on a surface. Stable APIs have stronger compatibility expectations. Beta APIs require changelog coverage but may still evolve. Experimental realtime/live paths should be isolated behind your own abstraction.
