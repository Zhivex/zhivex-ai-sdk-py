import asyncio
import os

from zhivex_ai import Agent, aclose_default_clients, create_openai, run_agent


async def main() -> None:
    try:
        provider = create_openai()
        result = await run_agent(
            agent=Agent(name="assistant", model=provider(os.environ["ZHIVEX_MODEL"])),
            prompt="Give me a short API launch checklist.",
        )
        print(result.text)
    finally:
        await aclose_default_clients()


asyncio.run(main())
