from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Callable

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from _bootstrap import load_dotenv_if_available

load_dotenv_if_available()

from zhivex_ai import (
    Agent,
    ApprovalDecision,
    ToolApprovalRequest,
    create_gemini,
    create_mcp_tool_registry,
    create_openai,
    mcp_stdio_server,
    run_agent,
)


READ_ONLY_FILESYSTEM_TOOLS = frozenset(
    {
        "directory_tree",
        "get_file_info",
        "list_allowed_directories",
        "list_directory",
        "list_directory_with_sizes",
        "read_file",
        "read_media_file",
        "read_multiple_files",
        "read_text_file",
        "search_files",
    }
)


async def _approve_read_only_filesystem(request: ToolApprovalRequest) -> ApprovalDecision:
    remote_name = str(request.tool_metadata.get("mcp_tool_name") or "")
    if request.tool_source == "mcp" and remote_name in READ_ONLY_FILESYSTEM_TOOLS:
        return ApprovalDecision(approved=True)
    return ApprovalDecision(approved=False, reason="This example only allows read-only filesystem MCP tools.")


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
            command="bunx",
            args=["@modelcontextprotocol/server-filesystem@2026.7.10", str(EXAMPLES_ROOT)],
        )
    )
    try:
        model_factory, provider_name, model_name = _resolve_model_factory()
        agent = Agent(
            name="assistant",
            instructions=(
                "Use only read-only filesystem tools when needed. "
                "The configured MCP root is the repository examples directory."
            ),
            model=model_factory(model_name),
            tools=mcp_tools,
            approval_policy=_approve_read_only_filesystem,
        )

        print(f"Using provider={provider_name} model={model_name}")
        result = await run_agent(agent=agent, prompt="List the Python files in the allowed examples directory.")
        print(result.text)
    finally:
        await mcp_tools.aclose()


if __name__ == "__main__":
    asyncio.run(main())
