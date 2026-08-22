from __future__ import annotations

import asyncio

from zhivex_ai import (
    Agent,
    apply_safety_policy_to_agent,
    create_agent_trace_artifact,
    create_in_memory_agent_run_store,
    create_safety_policy,
    run_agent,
    summarize_agent_trace,
)
from zhivex_ai.evals import (
    AgentEvaluationCase,
    AgentEvaluationExpectations,
    create_agent_evaluation_report,
    create_mock_language_model,
    run_agent_evaluation,
)


async def main() -> None:
    store = create_in_memory_agent_run_store()
    policy = create_safety_policy(preset="review_sensitive")
    agent = apply_safety_policy_to_agent(
        Agent(name="assistant", model=create_mock_language_model(), run_store=store),
        policy,
    )

    result = await run_agent(agent=agent, prompt="Say hello", idempotency_key="demo")
    state = await store.load(result.run_id)
    if state is None:
        raise RuntimeError("Expected persisted run state.")

    trace = create_agent_trace_artifact(state)
    summary = summarize_agent_trace(state)
    evaluation_agent = apply_safety_policy_to_agent(
        Agent(name="assistant", model=create_mock_language_model()),
        policy,
    )
    evaluation = await run_agent_evaluation(
        agent=evaluation_agent,
        dataset=[
            AgentEvaluationCase(
                name="smoke",
                prompt="Say hello",
                expectations=AgentEvaluationExpectations(output_contains="ok"),
            )
        ],
    )
    report = create_agent_evaluation_report(evaluation)

    print({"text": result.text, "trace": trace.output_preview, "steps": summary.steps, "pass_rate": report.pass_rate})


if __name__ == "__main__":
    asyncio.run(main())
