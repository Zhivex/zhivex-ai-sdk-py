from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict

from zhivex_ai import Agent, create_openai, run_agent, tool


class ProjectStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str


def lookup_project_status(input: ProjectStatusInput) -> dict[str, str]:
    """Example application-owned tool with deterministic local data."""

    return {"project": input.project, "status": "on track"}


async def main() -> None:
    openai = create_openai()
    agent = Agent(
        name="project-assistant",
        instructions="Use the project-status tool, then answer in one concise sentence.",
        model=openai("gpt-5.6-terra"),
        tools={
            "lookup_project_status": tool(
                name="lookup_project_status",
                description="Returns the current status for a project.",
                schema=ProjectStatusInput,
                execute=lookup_project_status,
            )
        },
    )

    result = await run_agent(agent=agent, prompt="What is the status of Apollo?")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
