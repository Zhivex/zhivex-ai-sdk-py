import asyncio
import os
from typing import Callable

from _bootstrap import load_dotenv_if_available

load_dotenv_if_available()

from zhivex_ai import Agent, create_mcp_tool_registry, create_openai, create_gemini, mcp_stdio_server, run_agent


def _resolve_model_factory() -> tuple[Callable[[str], object], str, str]:
    provider = os.getenv("PROVIDER", "openai").strip().lower()
    model = os.getenv("MODEL")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY to use PROVIDER=openai.")
        factory = create_openai(api_key=api_key)
        return factory, provider, model or "gpt-5.4-nano"

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY to use PROVIDER=gemini.")
        factory = create_gemini(api_key=api_key)
        return factory, provider, model or "gemini-2.5-flash"

    raise RuntimeError(
        f'Unsupported PROVIDER="{provider}" for this example. '
        'Use PROVIDER=openai or PROVIDER=gemini, and optionally override MODEL.'
    )


async def main() -> None:
    mcp_tools = await create_mcp_tool_registry(
        mcp_stdio_server(
            name="fs",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "."],
        )
    )
    try:
        model_factory, provider_name, model_name = _resolve_model_factory()
        agent = Agent(
            name="assistant",
            instructions=(
                "Use filesystem tools when needed. "
                "If the user asks to list files in the project, search recursively when helpful."
            ),
            model=model_factory(model_name),
            tools=mcp_tools,
        )

        print(f"Using provider={provider_name} model={model_name}")
        result = await run_agent(agent=agent, prompt="List the Python files in the current directory.")
        print(result.text)
    finally:
        await mcp_tools.aclose()


if __name__ == "__main__":
    asyncio.run(main())
