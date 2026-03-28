from __future__ import annotations

import json
from collections.abc import AsyncIterable
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import create_anthropic, generate_text, tool


@dataclass
class FakeResponse:
    status_code: int
    payload: Any = None
    body_text: str = ""

    async def json(self) -> Any:
        return self.payload

    async def text(self) -> str:
        return self.body_text or json.dumps(self.payload)

    async def iter_lines(self) -> AsyncIterable[str]:
        for line in self.body_text.splitlines():
            yield line


class AnthropicProviderTests(IsolatedAsyncioTestCase):
    async def test_anthropic_tool_call_roundtrip(self) -> None:
        calls = 0

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            nonlocal calls
            calls += 1
            if calls == 1:
                return FakeResponse(
                    status_code=200,
                    payload={
                        "content": [{"type": "tool_use", "id": "tool-1", "name": "math", "input": {"value": 2}}],
                        "stop_reason": "tool_use",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                )
            return FakeResponse(
                status_code=200,
                payload={
                    "content": [{"type": "text", "text": "result is 4"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        provider = create_anthropic(api_key="test", fetch=fetch)
        result = await generate_text(
            model=provider("claude-3-5-sonnet"),
            prompt="double 2",
            max_steps=2,
            tools={
                "math": tool(
                    name="math",
                    schema=dict[str, int],
                    execute=lambda input: {"result": input["value"] * 2},
                )
            },
        )
        self.assertEqual(result.text, "result is 4")
        self.assertEqual(result.tool_results[0].tool_name, "math")
