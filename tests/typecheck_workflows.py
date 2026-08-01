from __future__ import annotations

from zhivex_ai import (
    Agent,
    WorkflowBuilder,
    WorkflowFunctionContext,
    WorkflowFunctionResult,
    WorkflowGraph,
    WorkflowRetryPolicy,
    WorkflowStep,
    create_in_memory_workflow_checkpoint_store,
    fork_workflow,
    resume_workflow,
)


async def calculate(context: WorkflowFunctionContext) -> WorkflowFunctionResult:
    return WorkflowFunctionResult(
        output={"attempt": context.attempt},
        state_patch={"calculated": True},
    )


def build_graph(agent: Agent) -> WorkflowGraph:
    return (
        WorkflowBuilder("typed-workflow", definition_version="1")
        .add_step(
            WorkflowStep(
                "agent-step",
                agent,
                output_key="agent_output",
                retry_policy=WorkflowRetryPolicy(max_attempts=2),
            ),
            entrypoint=True,
        )
        .add_step(
            WorkflowStep("function-step", executor=calculate, output_key="function_output")
        )
        .add_edge("agent-step", "function-step")
        .build(checkpoint_store=create_in_memory_workflow_checkpoint_store())
    )


async def continue_graph(graph: WorkflowGraph, run_id: str, interrupt_id: str) -> None:
    await resume_workflow(graph, run_id, interrupt_id=interrupt_id, resume_value={"approved": True})
    await fork_workflow(graph, run_id, state_updates={"scenario": "audit"})
