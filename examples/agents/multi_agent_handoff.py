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
    GenerateResult,
    ModelMessage,
    ToolCall,
    create_mock_language_model,
    create_text_message,
    handoff_to,
    run_agent,
    tool,
    tool_call_part,
)


async def main() -> None:
    researcher = Agent(
        name="researcher",
        model=create_mock_language_model(responses=[GenerateResult(text="Apollo migration is green.", messages=[create_text_message("assistant", "Apollo migration is green.")])]),
    )
    triage = Agent(
        name="triage",
        model=create_mock_language_model(
            responses=[
                GenerateResult(
                    messages=[
                        ModelMessage(
                            role="assistant",
                            parts=[tool_call_part(ToolCall(id="call_1", name="delegate", input={"task": "Check Apollo"}))],
                        )
                    ],
                    finish_reason="tool-calls",
                ),
                GenerateResult(text="Delegating.", messages=[create_text_message("assistant", "Delegating.")]),
            ]
        ),
        subagents={"researcher": researcher},
        tools={"delegate": tool(name="delegate", schema=dict[str, str], execute=lambda input: handoff_to("researcher", input=input["task"]))},
    )

    result = await run_agent(agent=triage, prompt="Route this research task.")
    print(result.orchestration_path)
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
