from __future__ import annotations

import asyncio

from zhivex_ai import Agent, GenerateResult, SequentialAgent, WorkflowStep, create_mock_language_model


def mock_agent(name: str, text: str) -> Agent:
    return Agent(name=name, model=create_mock_language_model(responses=[GenerateResult(text=text, finish_reason="stop")]))


async def main() -> None:
    workflow = SequentialAgent(
        name="loan_pipeline",
        steps=[
            WorkflowStep("extract", mock_agent("extractor", "application"), prompt="Extract application", output_key="application"),
            WorkflowStep(
                "validate",
                mock_agent("validator", "valid"),
                input_template="Validate {application}",
                output_key="validation",
            ),
            WorkflowStep(
                "decide",
                mock_agent("decider", "approved"),
                input_template="Decide with {application} and {validation}",
                output_key="decision",
            ),
        ],
    )

    result = await workflow.run()
    print(result.state)


if __name__ == "__main__":
    asyncio.run(main())
