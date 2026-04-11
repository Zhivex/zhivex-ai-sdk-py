import asyncio
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

from zhivex_ai import Agent, create_mcp_tool_registry, create_openai, create_gemini, mcp_stdio_server, run_agent


async def main() -> None:
    mcp_tools = await create_mcp_tool_registry(
        mcp_stdio_server(
            name="fs",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "."],
        )
    )
    try:
        openai = create_openai(api_key=os.getenv("OPENAI_API_KEY"))
        gemini = create_gemini(api_key=os.getenv("GOOGLE_API_KEY"))
        agent = Agent(
            name="assistant",
            instructions="Use filesystem tools when needed.",
            model=openai("gpt-5.4-nano"),
            tools=mcp_tools,
        )

        result = await run_agent(agent=agent, prompt="List the Python files in the current directory.")
        print(result.text)
    finally:
        await mcp_tools.aclose()


if __name__ == "__main__":
    asyncio.run(main())
