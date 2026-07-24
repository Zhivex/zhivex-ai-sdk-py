import asyncio

from zhivex_ai import Agent, ApprovalDecision, ToolApprovalRequest, create_openai, remote_tool, run_agent


async def approve_project_status(request: ToolApprovalRequest) -> ApprovalDecision:
    if request.tool_source == "remote" and request.tool_name == "project_status":
        return ApprovalDecision(approved=True)
    return ApprovalDecision(approved=False, reason="Only the reviewed project-status endpoint is allowed.")


async def main() -> None:
    openai = create_openai()
    agent = Agent(
        name="assistant",
        instructions="Use the remote tool when the user asks for project status.",
        model=openai("gpt-5.6-terra"),
        tools={
            "project_status": remote_tool(
                name="project_status",
                description="Fetches project status from a remote HTTP tool endpoint.",
                url="https://example.com/tools/project-status",
                schema=dict[str, str],
                headers={"x-api-key": "replace-me"},
                timeout_ms=5_000,
            )
        },
        approval_policy=approve_project_status,
    )

    result = await run_agent(agent=agent, prompt="Check the status for project Apollo.")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
