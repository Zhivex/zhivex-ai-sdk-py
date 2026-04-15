import asyncio

from zhivex_ai import Agent, create_openai, run_agent, skill


async def main() -> None:
    provider = create_openai()
    release_notes = skill(
        name="release-notes",
        description="Use when a user asks for changelog summaries or release notes.",
        instructions="""
Write concise release notes with:
- Highlights
- Breaking changes
- Migration notes when needed
""".strip(),
    )
    agent = Agent(
        name="assistant",
        instructions="You are a careful SDK assistant.",
        model=provider("gpt-5.4-mini"),
        skills={"release-notes": release_notes},
    )

    result = await run_agent(agent=agent, prompt="$release-notes summarize the latest SDK updates.")
    print(result.text)
    print(result.session.metadata.get("active_skills"))


if __name__ == "__main__":
    asyncio.run(main())
