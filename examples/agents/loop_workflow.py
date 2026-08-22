from __future__ import annotations

import asyncio

from zhivex_ai import Agent
from zhivex_ai.evals import GenerateResult, create_mock_language_model
from zhivex_ai.workflows import LoopAgent, WorkflowStep


async def main() -> None:
    writer = Agent(
        name="writer",
        model=create_mock_language_model(
            responses=[
                GenerateResult(text="draft", finish_reason="stop"),
                GenerateResult(text="done", finish_reason="stop"),
            ]
        ),
    )
    workflow = LoopAgent(
        name="refine",
        steps=[WorkflowStep("draft", writer, prompt="Refine the draft", output_key="draft")],
        max_iterations=3,
        stop_condition=lambda result: result.state.get("draft") == "done",
    )

    result = await workflow.run()
    print({"iterations": len(result.step_results), "draft": result.state["draft"]})


if __name__ == "__main__":
    asyncio.run(main())
