from __future__ import annotations

from dataclasses import dataclass
from typing import assert_type, cast

from pydantic import BaseModel

from zhivex_ai import (
    Agent,
    AgentContext,
    AgentRunResult,
    AgentStreamResult,
    LanguageModel,
    ToolDefinition,
    ToolExecutionContext,
    run_agent,
    stream_agent,
    tool,
)


@dataclass
class Dependencies:
    tenant: str


class Decision(BaseModel):
    approved: bool


@tool
def decorated_lookup(query: str, limit: int = 5) -> list[str]:
    return [query] * limit


assert_type(decorated_lookup, ToolDefinition)


async def configured_lookup(query: str, context: ToolExecutionContext[Dependencies]) -> str:
    return f"{context.tool_name}:{query}"


configured_tool = tool(name="configured_lookup")(configured_lookup)
assert_type(configured_tool, ToolDefinition)
explicit_tool = tool(
    name="explicit_lookup",
    schema={"type": "object"},
    execute=lambda input: input,
)
assert_type(explicit_tool, ToolDefinition)


def dynamic_instructions(context: AgentContext[Dependencies]) -> str:
    assert_type(context.deps, Dependencies | None)
    return f"Tenant: {context.deps.tenant if context.deps else 'unknown'}"


agent = Agent(
    name="typed",
    model=cast(LanguageModel, object()),
    instructions=dynamic_instructions,
    output_type=Decision,
)
assert_type(agent, Agent[Dependencies, Decision])


async def check_public_typing_contract() -> None:
    result = await run_agent(agent=agent, prompt="Decide", deps=Dependencies("bank-ar"))
    assert_type(result, AgentRunResult[Decision])
    assert_type(result.output, Decision | None)

    streamed = stream_agent(agent=agent, prompt="Decide", deps=Dependencies("bank-ar"))
    assert_type(streamed, AgentStreamResult[Decision])
    collected = await streamed.collect()
    assert_type(collected, AgentRunResult[Decision])

    tool_context: ToolExecutionContext[Dependencies] = ToolExecutionContext(
        tool_name="lookup",
        deps=Dependencies("bank-ar"),
    )
    assert_type(tool_context.deps, Dependencies | None)
