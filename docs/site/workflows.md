# Workflows

Use the focused `zhivex_ai.workflows` namespace for declarative composition,
durable graphs, checkpoints, leases and resume. The core is Stable; named external
engine adapters retain their own Beta classification.

```python
from zhivex_ai.workflows import SequentialAgent, ParallelAgent, WorkflowBuilder
```

Use a sequential workflow when steps depend on previous results, parallel
composition for independent steps, and a durable graph when the run needs
checkpoints or explicit recovery. Keep retry and side-effect ownership explicit.

The [workflow reference](reference/workflows.md) is generated from the installed
package. The [workflow guide]({{SOURCE}}/docs/WORKFLOWS.md) covers application-owned
state and recovery. External engine support is distinct from the generic callback contract.
