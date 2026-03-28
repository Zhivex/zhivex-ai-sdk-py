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

from zhivex_ai import create_openai, generate_text, stream_text


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


class OpenAIProviderTests(IsolatedAsyncioTestCase):
    async def test_openai_maps_chat_completion(self) -> None:
        requests: list[dict[str, Any]] = []

        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            requests.append({"url": url, "headers": headers, "json": json_body, "stream": stream})
            return FakeResponse(
                status_code=200,
                payload={
                    "choices": [{"finish_reason": "stop", "message": {"content": "hello from openai"}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
                },
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = await generate_text(model=provider("gpt-4o-mini"), prompt="hello")
        self.assertEqual(result.text, "hello from openai")
        self.assertEqual(result.usage.total_tokens, 7)
        self.assertEqual(requests[0]["json"]["model"], "gpt-4o-mini")

    async def test_openai_stream_collects_text(self) -> None:
        async def fetch(
            url: str, *, headers: dict[str, str], json_body: dict[str, Any], timeout_ms: int | None, stream: bool = False
        ):
            return FakeResponse(
                status_code=200,
                body_text=(
                    'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                    'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )

        provider = create_openai(api_key="test", fetch=fetch)
        result = stream_text(model=provider("gpt-4o-mini"), prompt="hello")
        final = await result.collect()
        self.assertEqual(final.text, "hello world")
