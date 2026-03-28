from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class ResponseLike(Protocol):
    status_code: int

    async def json(self) -> Any: ...

    async def text(self) -> str: ...

    def iter_lines(self) -> AsyncIterable[str]: ...


class Fetcher(Protocol):
    async def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json_body: dict[str, Any],
        timeout_ms: int | None,
        stream: bool = False,
    ) -> ResponseLike: ...


@dataclass
class BufferedResponse:
    status_code: int
    body_text: str

    async def json(self) -> Any:
        return httpx.Response(200, text=self.body_text).json()

    async def text(self) -> str:
        return self.body_text

    async def iter_lines(self) -> AsyncIterable[str]:
        for line in self.body_text.splitlines():
            yield line


async def default_fetch(
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any],
    timeout_ms: int | None,
    stream: bool = False,
) -> ResponseLike:
    timeout = None if timeout_ms is None else timeout_ms / 1000
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=json_body)
        return BufferedResponse(status_code=response.status_code, body_text=response.text)
