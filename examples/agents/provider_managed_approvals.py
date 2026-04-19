from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Callable

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from _bootstrap import load_dotenv_if_available

load_dotenv_if_available()

from zhivex_ai import (  # noqa: E402
    Agent,
    AgentTextDeltaEvent,
    AgentToolApprovalEvent,
    AgentToolCallEvent,
    ApprovalDecision,
    create_azure_openai,
    create_openai,
    get_azure_openai_response_id,
    get_openai_response_id,
    openai_mcp_tool,
    azure_openai_mcp_tool,
    stream_agent,
)


ProviderConfig = tuple[str, Any, str, Any, Callable[[Any], str | None]]


def _resolve_provider() -> ProviderConfig:
    provider_name = os.getenv("PROVIDER", "openai").strip().lower()
    model_name = os.getenv("MODEL", "gpt-5.4-mini")
    server_url = os.getenv("MCP_SERVER_URL")
    server_label = os.getenv("MCP_SERVER_LABEL", "Docs")
    allowed_tools = [name.strip() for name in os.getenv("MCP_ALLOWED_TOOLS", "").split(",") if name.strip()]
    require_approval = os.getenv("MCP_REQUIRE_APPROVAL", "always")

    if not server_url:
        raise RuntimeError(
            "Set MCP_SERVER_URL to a remote MCP endpoint before running this example. "
            "Optional: MCP_SERVER_LABEL, MCP_ALLOWED_TOOLS, MCP_SERVER_BEARER_TOKEN."
        )

    headers: dict[str, str] | None = None
    bearer_token = os.getenv("MCP_SERVER_BEARER_TOKEN")
    if bearer_token:
        headers = {"Authorization": f"Bearer {bearer_token}"}

    if provider_name == "openai":
        provider = create_openai()
        hosted_tool = openai_mcp_tool(
            server_url=server_url,
            server_label=server_label,
            headers=headers,
            allowed_tools=allowed_tools or None,
            require_approval=require_approval,
        )
        return provider_name, provider, model_name, hosted_tool, get_openai_response_id

    if provider_name in {"azure", "azure-openai"}:
        provider = create_azure_openai()
        hosted_tool = azure_openai_mcp_tool(
            server_url=server_url,
            server_label=server_label,
            headers=headers,
            allowed_tools=allowed_tools or None,
            require_approval=require_approval,
        )
        return "azure-openai", provider, model_name, hosted_tool, get_azure_openai_response_id

    raise RuntimeError(
        f'Unsupported PROVIDER="{provider_name}" for this example. '
        'Use PROVIDER=openai or PROVIDER=azure-openai.'
    )


async def main() -> None:
    provider_name, provider, model_name, hosted_tool, get_response_id = _resolve_provider()

    async def approval_policy(request) -> ApprovalDecision:
        server_label = request.tool_metadata.get("server_label")
        print(
            f"\n[approval] provider={request.tool_metadata.get('provider')} "
            f"server={server_label} tool={request.tool_name}"
        )
        if request.tool_source != "hosted":
            return ApprovalDecision(approved=False, reason="This example only approves hosted MCP requests.")
        if request.tool_metadata.get("hosted_tool_class") != "remote-mcp":
            return ApprovalDecision(approved=False, reason="Unexpected hosted tool class.")
        return ApprovalDecision(approved=True)

    agent = Agent(
        name="research-assistant",
        instructions=(
            "Use the remote MCP server when it helps answer the question. "
            "Keep the final answer concise and cite what you found."
        ),
        model=provider.native.language_model(model_name),
        tools={"docs_mcp": hosted_tool},
        approval_policy=approval_policy,
    )

    stream = stream_agent(
        agent=agent,
        prompt="Use the MCP server to find one relevant fact about the Apollo program and summarize it in Spanish.",
        max_steps=6,
    )

    print(f"provider={provider_name} model={model_name}")
    async for event in stream.event_stream():
        if isinstance(event, AgentToolApprovalEvent):
            print(
                f"[tool-approval] approved={event.approved} "
                f"provider_managed={event.provider_managed} "
                f"request_id={event.approval_request_id}"
            )
            continue
        if isinstance(event, AgentToolCallEvent) and event.tool_call.provider_metadata.get("provider_managed"):
            print(f"[provider-tool] {event.tool_call.name} input={event.tool_call.input}")
            continue
        if isinstance(event, AgentTextDeltaEvent):
            print(event.text_delta, end="", flush=True)

    final = await stream.collect()
    response_id = get_response_id(final)

    print()
    print(f"finish={final.finish_reason}")
    if response_id is not None:
        print(f"response_id={response_id}")


if __name__ == "__main__":
    asyncio.run(main())
