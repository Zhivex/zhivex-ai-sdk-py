from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    Agent,
    ModelMessage,
    ToolCall,
    create_text_message,
    permission_allowlist_approval_policy,
    run_agent,
    tool,
    tool_call_part,
)
from zhivex_ai.evals import GenerateResult, create_mock_language_model  # noqa: E402


async def main() -> None:
    model = create_mock_language_model(
        responses=[
            GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[tool_call_part(ToolCall(id="call_1", name="lookup", input={"item": "apollo"}))],
                    )
                ],
                finish_reason="tool-calls",
            ),
            GenerateResult(text="Approved lookup completed.", messages=[create_text_message("assistant", "Approved lookup completed.")]),
        ]
    )
    agent = Agent(
        name="assistant",
        model=model,
        approval_policy=permission_allowlist_approval_policy("project:read"),
        tools={
            "lookup": tool(
                name="lookup",
                schema=dict[str, str],
                execute=lambda input: {"item": input["item"], "status": "ok"},
                permissions=["project:read"],
                requires_approval=True,
            )
        },
    )

    result = await run_agent(agent=agent, prompt="Look up Apollo.")
    print(result.text)
    print([event.type for event in result.trace.events])


if __name__ == "__main__":
    asyncio.run(main())
