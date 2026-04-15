import asyncio

from pydantic import BaseModel, ConfigDict

from zhivex_ai import Agent, create_gemini, create_openai, generate_grounded_text, run_agent, tool


class SearchWebInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str


async def main() -> None:
    gemini = create_gemini()
    openai = create_openai()

    async def search_web(input: SearchWebInput) -> dict[str, object]:
        result = await generate_grounded_text(
            model=gemini.grounded_language_model("gemini-2.5-flash"),
            prompt=input.task,
        )
        return {
            "answer": result.text,
            "sources": [
                {
                    "title": source.title,
                    "url": source.url,
                }
                for source in result.sources
            ],
        }

    agent = Agent(
        name="triage",
        instructions=(
            "Use the search_web tool for any request that needs fresh information. "
            "After using the tool, answer concisely and include the returned sources when helpful."
        ),
        model=openai("gpt-5.4-nano"),
        tools={
            "search_web": tool(
                name="search_web",
                description="Runs a grounded web search with Gemini and returns an answer plus sources.",
                schema=SearchWebInput,
                execute=search_web,
            )
        },
    )

    result = await run_agent(
        agent=agent,
        prompt="Research the latest Apollo migration status and summarize it in 3 bullet points.",
        max_steps=3,
    )

    print(result.text)
    print(result.tool_results)


if __name__ == "__main__":
    asyncio.run(main())
