import asyncio

from zhivex_ai import Agent, create_openai, stream_agent


async def main() -> None:
    openai = create_openai()
    agent = Agent(
        name="assistant",
        instructions="Be concise and narrate your reasoning as plain text.",
        model=openai("gpt-5.6-terra"),
    )

    stream = stream_agent(agent=agent, prompt="Explain what an agent handoff is in two sentences.")

    async for chunk in stream.text_stream():
        print(chunk, end="")

    final = await stream.collect()
    print("\nfinish:", final.finish_reason)


if __name__ == "__main__":
    asyncio.run(main())
