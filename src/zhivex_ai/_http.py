from __future__ import annotations

from collections.abc import AsyncIterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

_DEFAULT_CLIENTS: dict[float | None, httpx.AsyncClient] = {}


class ResponseLike(Protocol):
    status_code: int
    headers: Mapping[str, str]

    async def json(self) -> Any: ...

    async def text(self) -> str: ...

    async def read(self) -> bytes: ...

    def iter_lines(self) -> AsyncIterable[str]: ...


class Fetcher(Protocol):
    async def __call__(
        self,
        url: str,
        *,
        method: str = "POST",
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        body: Any = None,
        timeout_ms: int | None,
        stream: bool = False,
    ) -> ResponseLike: ...


@dataclass
class BufferedResponse:
    status_code: int
    body_bytes: bytes
    headers: Mapping[str, str] = field(default_factory=dict)

    async def json(self) -> Any:
        return httpx.Response(200, content=self.body_bytes).json()

    async def text(self) -> str:
        return self.body_bytes.decode("utf-8", errors="replace")

    async def read(self) -> bytes:
        return self.body_bytes

    async def iter_lines(self) -> AsyncIterable[str]:
        for line in self.body_bytes.decode("utf-8", errors="replace").splitlines():
            yield line


class StreamingResponse:
    def __init__(self, response: httpx.Response) -> None:
        self.status_code = response.status_code
        self.headers = dict(getattr(response, "headers", {}) or {})
        self._response = response
        self._closed = False
        self._body_bytes: bytes | None = None

    async def _read_body(self) -> bytes:
        if self._body_bytes is not None:
            return self._body_bytes
        try:
            self._body_bytes = await self._response.aread()
            return self._body_bytes
        finally:
            await self._close()

    async def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._response.aclose()

    async def json(self) -> Any:
        return httpx.Response(200, content=await self._read_body()).json()

    async def text(self) -> str:
        return (await self._read_body()).decode("utf-8", errors="replace")

    async def read(self) -> bytes:
        return await self._read_body()

    async def iter_lines(self) -> AsyncIterable[str]:
        if self._body_bytes is not None:
            for line in self._body_bytes.decode("utf-8", errors="replace").splitlines():
                yield line
            return

        chunks: list[str] = []
        try:
            async for line in self._response.aiter_lines():
                chunks.append(line)
                yield line
        finally:
            self._body_bytes = "\n".join(chunks).encode("utf-8")
            await self._close()


def _build_request_kwargs(json_body: dict[str, Any] | None, body: Any) -> dict[str, Any]:
    if body is None:
        return {"json": json_body}

    if isinstance(body, dict) and ("data" in body or "files" in body):
        return {
            "data": body.get("data"),
            "files": body.get("files"),
        }

    if isinstance(body, (bytes, bytearray, memoryview, str)):
        return {"content": body}

    return {"content": body}


def _shared_client(timeout: float | None) -> httpx.AsyncClient:
    client = _DEFAULT_CLIENTS.get(timeout)
    if client is not None and not getattr(client, "is_closed", False):
        return client
    client = httpx.AsyncClient(timeout=timeout)
    _DEFAULT_CLIENTS[timeout] = client
    return client


async def aclose_default_clients() -> None:
    clients = list(_DEFAULT_CLIENTS.values())
    _DEFAULT_CLIENTS.clear()
    for client in clients:
        await client.aclose()


async def default_fetch(
    url: str,
    *,
    method: str = "POST",
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
    body: Any = None,
    timeout_ms: int | None,
    stream: bool = False,
) -> ResponseLike:
    timeout = None if timeout_ms is None else timeout_ms / 1000
    client = _shared_client(timeout)
    request = client.build_request(method, url, headers=headers, **_build_request_kwargs(json_body, body))
    response = await client.send(request, stream=stream)
    if stream:
        return StreamingResponse(response=response)
    try:
        body_bytes = await response.aread()
        return BufferedResponse(
            status_code=response.status_code,
            body_bytes=body_bytes,
            headers=dict(getattr(response, "headers", {}) or {}),
        )
    finally:
        await response.aclose()
