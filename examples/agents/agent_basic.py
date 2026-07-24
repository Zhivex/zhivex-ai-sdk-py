import asyncio
from pydantic import BaseModel, ConfigDict

from zhivex_ai import (
    Agent,
    create_openai,
    handoff_to,
    run_agent,
    tool,
)


class DelegateResearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str


async def main() -> None:
    provider = create_openai()
    researcher = Agent(
        name="researcher",
        instructions="You are a concise research assistant. Answer delegated tasks directly.",
        model=provider("gpt-5.6-terra"),
    )
    triage = Agent(
        name="triage",
        instructions="Delegate background research work to the researcher agent.",
        model=provider("gpt-5.6-terra"),
        tools={
            "delegate_research": tool(
                name="delegate_research",
                description="Delegates the current task to the researcher agent.",
                schema=DelegateResearchInput,
                execute=lambda input: handoff_to("researcher", input=input.task),
            )
        },
        subagents={"researcher": researcher},
    )

    result = await run_agent(agent=triage, prompt="Research the Apollo project status.")
    print(result.text)
    print(result.orchestration_path)


if __name__ == "__main__":
    asyncio.run(main())
