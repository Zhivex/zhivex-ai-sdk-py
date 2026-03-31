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


class StreamingResponse:
    def __init__(self, response: httpx.Response, client: httpx.AsyncClient) -> None:
        self.status_code = response.status_code
        self._response = response
        self._client = client
        self._closed = False
        self._body_text: str | None = None

    async def _read_body(self) -> str:
        if self._body_text is not None:
            return self._body_text
        try:
            await self._response.aread()
            self._body_text = self._response.text
            return self._body_text
        finally:
            await self._close()

    async def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._response.aclose()
        await self._client.aclose()

    async def json(self) -> Any:
        body_text = await self._read_body()
        return httpx.Response(200, text=body_text).json()

    async def text(self) -> str:
        return await self._read_body()

    async def iter_lines(self) -> AsyncIterable[str]:
        if self._body_text is not None:
            for line in self._body_text.splitlines():
                yield line
            return

        lines: list[str] = []
        try:
            async for line in self._response.aiter_lines():
                lines.append(line)
                yield line
        finally:
            self._body_text = "\n".join(lines)
            await self._close()


async def default_fetch(
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any],
    timeout_ms: int | None,
    stream: bool = False,
) -> ResponseLike:
    timeout = None if timeout_ms is None else timeout_ms / 1000
    client = httpx.AsyncClient(timeout=timeout)
    request = client.build_request("POST", url, headers=headers, json=json_body)
    response = await client.send(request, stream=stream)
    if stream:
        return StreamingResponse(response=response, client=client)
    try:
        await response.aread()
        return BufferedResponse(status_code=response.status_code, body_text=response.text)
    finally:
        await response.aclose()
        await client.aclose()
