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
    ToolExecutionContext,
    run_agent,
    stream_agent,
)


@dataclass
class Dependencies:
    tenant: str


class Decision(BaseModel):
    approved: bool


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
