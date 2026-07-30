from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from zhivex_ai import (
    Agent,
    AgentContext,
    AgentHooks,
    AgentRunRequest,
    create_openai,
    run_agent,
)


@dataclass
class Dependencies:
    tenant_id: str


class Review(BaseModel):
    approved: bool
    reason: str


def dynamic_instructions(context: AgentContext[Dependencies]) -> str:
    tenant_id = context.deps.tenant_id if context.deps else "unknown"
    return f"Review the request for tenant {tenant_id}. Return a concise structured decision."


class ConsoleHooks(AgentHooks):
    async def on_agent_start(self, context: AgentContext[Any], agent: Agent[Any, Any]) -> None:
        print(f"starting agent={agent.name} run={context.run_id}")

    async def on_agent_end(self, context: AgentContext[Any], agent: Agent[Any, Any], result: Any) -> None:
        print(f"finished agent={agent.name} run={context.run_id}")


async def require_dependencies(request: AgentRunRequest[Any, Any], call_next: Any) -> Any:
    if request.deps is None:
        raise RuntimeError("This agent requires tenant dependencies.")
    return await call_next(request)


async def main() -> None:
    openai = create_openai()
    agent: Agent[Dependencies, Review] = Agent(
        name="reviewer",
        model=openai("gpt-5.6-terra"),
        instructions=dynamic_instructions,
        output_type=Review,
        hooks=[ConsoleHooks()],
    )
    result = await run_agent(
        agent=agent,
        prompt="Approve a low-risk documentation-only change.",
        deps=Dependencies(tenant_id="example-tenant"),
        middleware=[require_dependencies],
    )
    print(result.output)
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
