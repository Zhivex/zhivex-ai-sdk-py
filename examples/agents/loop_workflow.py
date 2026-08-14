from __future__ import annotations

import asyncio

from zhivex_ai import Agent, GenerateResult, LoopAgent, WorkflowStep, create_mock_language_model


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
