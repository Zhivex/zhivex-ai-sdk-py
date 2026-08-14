from __future__ import annotations

import asyncio

from zhivex_ai import Agent, GenerateResult, ParallelAgent, WorkflowStep, create_mock_language_model


def mock_agent(name: str, text: str) -> Agent:
    return Agent(name=name, model=create_mock_language_model(responses=[GenerateResult(text=text, finish_reason="stop")]))


async def main() -> None:
    workflow = ParallelAgent(
        name="research",
        steps=[
            WorkflowStep("policy", mock_agent("policy", "policy notes"), prompt="Research policy", output_key="policy"),
            WorkflowStep("risk", mock_agent("risk", "risk notes"), prompt="Research risk", output_key="risk"),
        ],
    )

    result = await workflow.run()
    print(result.state)


if __name__ == "__main__":
    asyncio.run(main())
