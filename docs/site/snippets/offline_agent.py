import asyncio

from zhivex_ai import Agent, run_agent
from zhivex_ai.evals import GenerateResult, create_mock_language_model


async def main() -> None:
    model = create_mock_language_model(
        responses=[GenerateResult(text="Ready", finish_reason="stop")]
    )
    result = await run_agent(agent=Agent(name="assistant", model=model), prompt="Check readiness.")
    assert result.text == "Ready"


asyncio.run(main())
