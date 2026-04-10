import asyncio

from zhivex_ai import Agent, MCPServerConfig, create_openai, discover_mcp_tools, run_agent


async def main() -> None:
    server = MCPServerConfig(
        transport="stdio",
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
    )
    mcp_tools = await discover_mcp_tools(server, prefix="fs_")

    openai = create_openai()
    agent = Agent(
        name="assistant",
        instructions="Use filesystem tools when needed.",
        model=openai("gpt-4o-mini"),
        tools=mcp_tools,
    )

    result = await run_agent(agent=agent, prompt="List the Python files in the current directory.")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
