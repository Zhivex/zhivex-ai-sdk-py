# Workflow Agents

Workflow agents are beta orchestration primitives for backend workflows whose shape is known before execution. They are useful when an application needs sequential pipelines, parallel fan-out, bounded refinement loops, replayable traces, and deterministic state handoff between steps.

The SDK owns orchestration. The application owns policy, vertical data models, durable business records, approval UI, artifact storage, and external integrations.

## Stability

The workflow surface is beta:

- `SequentialAgent`
- `ParallelAgent`
- `LoopAgent`
- `WorkflowStep`
- `WorkflowRunResult`
- `WorkflowStepResult`
- `WorkflowTraceEvent`
- `run_workflow`
- `workflow_step`
- `validate_workflow_expectations`

Use top-level imports from `zhivex_ai`. Avoid deep imports.

## Core Model

A workflow is a group of `WorkflowStep` objects. Each step wraps an `Agent` and can read or write `AgentSession.state`.

```python
from zhivex_ai import Agent, SequentialAgent, WorkflowStep

workflow = SequentialAgent(
    name="intake",
    steps=[
        WorkflowStep("extract", extractor, prompt="Extract the request", output_key="request"),
        WorkflowStep("review", reviewer, input_template="Review {request}", output_key="review"),
    ],
)

result = await workflow.run()
```

Important fields:

- `prompt`: fixed prompt for the step
- `input_template`: Python format template rendered from `session.state`
- `output_key`: stores the step text in `session.state`
- `metadata_key`: stores step metadata such as run id, agent name, text, status, and error
- `error_policy`: `fail_fast`, `continue`, or `capture`
- `timeout_ms` and `max_retries`: forwarded to `run_agent(...)`

## Sequential Workflows

Use `SequentialAgent` when every step depends on the previous state. Steps run in order, and `input_template` can read keys written by earlier steps.

Missing template keys fail the step with `ValidationError`. With the default `fail_fast` policy the workflow stops and returns a failed result. With `continue`, later steps still run. With `capture`, the error is stored under `output_key` as an application-readable error object.

If an agent step suspends for human approval, the step and workflow return `status="suspended"`; the step is not marked complete and later sequential/loop steps do not start. `output_key` is written only after completion, while `metadata_key` retains the suspended run id for the application-owned resume flow.

## Parallel Workflows

Use `ParallelAgent` for fan-out work such as independent research, policy review, or risk analysis. Each step starts from the same base session state. Only explicit `output_key` and `metadata_key` values are merged back into the shared session.

With `fail_fast`, a failed branch cooperatively cancels sibling tasks that have not completed. This prevents further cancel-aware work but cannot undo a side effect already started by a tool or stop synchronous code that ignores cancellation. `capture` preserves each isolated branch result before merging its declared keys.

Parallel `output_key` values must be unique. Duplicate output keys raise `ValidationError` during workflow construction.

## Loop Workflows

Use `LoopAgent` for bounded refinement. It requires `max_iterations > 0` and can stop early with `stop_condition`.

```python
workflow = LoopAgent(
    name="refine",
    steps=[WorkflowStep("draft", writer, prompt="Improve the draft", output_key="draft")],
    max_iterations=3,
    stop_condition=lambda result: result.state.get("draft") == "done",
)
```

The stop condition can be sync or async. It receives the current `WorkflowRunResult`.

## Structured Outputs

Workflow steps store text. For structured workflows, keep schema validation in application code:

```python
from pydantic import BaseModel

class Intake(BaseModel):
    company: str
    amount: int

result = await workflow.run()
intake = Intake.model_validate_json(result.state["intake"])
```

This keeps workflow orchestration portable while allowing each application to own its domain schemas and validation policy.

## Resume Pattern

There is no stable `resume_workflow(...)` API. Resume is app-owned:

1. Persist a business record with completed steps and the relevant `session.state`.
2. Recreate an `AgentSession` with that state.
3. Build a workflow with only the remaining steps.
4. Attach an `AgentRunStore` when replay or audit evidence is needed.

This avoids hidden business-policy decisions in the SDK and keeps retries idempotent at the application boundary.

## Replay And Evaluation

Attach `run_store` to `SequentialAgent`, `ParallelAgent`, or `LoopAgent` to persist the workflow state snapshot. Use `replay_agent_run(result.state_snapshot)` to inspect the run without calling providers again.

Use `validate_workflow_expectations(...)` with `AgentEvaluationExpectations` for deterministic checks:

```python
from zhivex_ai import AgentEvaluationExpectations, validate_workflow_expectations

failures = validate_workflow_expectations(
    result,
    AgentEvaluationExpectations(
        workflow_steps=["extract", "review"],
        state_contains=["request", "review"],
    ),
)
```

## Reference Examples

Offline examples:

- `examples/agents/sequential_workflow.py`
- `examples/agents/parallel_workflow.py`
- `examples/agents/loop_workflow.py`
- `examples/agents/structured_workflow_outputs.py`
- `examples/agents/workflow_resume.py`
- `examples/agents/artifact_document_workflow.py`
- `examples/agents/research_report_workflow.py`
- `examples/agents/small_business_loan_agent.py`
- `examples/agents/hr_candidate_selection_agent.py`

All examples use mock models or app-owned deterministic logic by default. Live providers and external systems should be added behind application adapters, not embedded in the workflow primitives.
