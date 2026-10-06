"""Offline OpenAI GA computer protocol demo; no network or desktop effects."""
import asyncio
import json

from zhivex_ai import Agent, create_openai
from zhivex_ai.experimental import (
    ComputerApproval, ComputerScreenshot, openai_computer_tool, run_computer_use,
)

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a4uoAAAAASUVORK5CYII="


class MockResponse:
    status_code = 200
    headers = {}

    def __init__(self, output):
        self.output = output

    async def json(self):
        return {"status": "completed", "output": self.output}

    async def text(self):
        return json.dumps(await self.json())


async def main():
    observed = []

    async def fetch(url, **kwargs):
        if not any(item["type"] == "computer_call_output" for item in kwargs["json_body"]["input"]):
            return MockResponse([{"type": "computer_call", "id": "item-1", "call_id": "call-1", "actions": [{"type": "screenshot"}], "pending_safety_checks": []}])
        return MockResponse([{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Observed the synthetic fixture."}]}])

    async def authorize(batch, context):
        # Only this mock screenshot is authorized. Real applications must bind
        # permission to the actual session, observation, actions and warning IDs.
        return ComputerApproval(batch["actions"] == [{"type": "screenshot"}])

    async def execute(batch, context):
        observed.append(context.tool_call_id)
        return ComputerScreenshot(PNG)

    async def is_complete(result):
        return observed == ["call-1"] and len(result.tool_results) == 1

    agent = Agent(name="computer-demo", model=create_openai(api_key="offline", fetch=fetch).native.language_model("gpt-5.4"), tools={
        "computer_use": openai_computer_tool(authorize=authorize, execute=execute),
    })
    result = await run_computer_use(agent=agent, prompt="Observe the fixture", is_complete=is_complete, max_steps=3)
    print({"verified": result.verified, "calls": observed})


if __name__ == "__main__":
    asyncio.run(main())
