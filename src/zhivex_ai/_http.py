from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, Self

import httpx

_LOOP_CLIENT_ATTRIBUTE = "_zhivex_ai_default_http_client"
DEFAULT_TIMEOUT_MS = 300_000
DEFAULT_MAX_BUFFERED_RESPONSE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_STREAM_CACHE_BYTES = 8 * 1024 * 1024


class ResponseTooLargeError(RuntimeError):
    pass


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
    def __init__(
        self,
        response: httpx.Response,
        *,
        max_cached_bytes: int = DEFAULT_MAX_STREAM_CACHE_BYTES,
        max_body_bytes: int = DEFAULT_MAX_BUFFERED_RESPONSE_BYTES,
    ) -> None:
        self.status_code = response.status_code
        self.headers: Mapping[str, str] = dict(getattr(response, "headers", {}) or {})
        self._response = response
        self._closed = False
        self._body_bytes: bytes | None = None
        self._max_cached_bytes = max_cached_bytes
        self._max_body_bytes = max_body_bytes

    async def _read_body(self) -> bytes:
        if self._body_bytes is not None:
            return self._body_bytes
        try:
            self._body_bytes = await _read_limited_body(
                self._response,
                max_bytes=self._max_body_bytes,
                label="streamed response body",
            )
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
        cached_bytes = 0
        cache_enabled = True
        try:
            async for line in self._response.aiter_lines():
                if cache_enabled:
                    cached_bytes += len(line.encode("utf-8")) + 1
                    if cached_bytes <= self._max_cached_bytes:
                        chunks.append(line)
                    else:
                        chunks = []
                        cache_enabled = False
                yield line
        finally:
            self._body_bytes = "\n".join(chunks).encode("utf-8") if cache_enabled else b""
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
    loop = asyncio.get_running_loop()
    client = getattr(loop, _LOOP_CLIENT_ATTRIBUTE, None)
    if client is not None and not getattr(client, "is_closed", False):
        return client
    client = httpx.AsyncClient(timeout=timeout)
    setattr(loop, _LOOP_CLIENT_ATTRIBUTE, client)
    return client


def _enforce_max_bytes(body: bytes, max_bytes: int, label: str) -> None:
    if len(body) > max_bytes:
        raise ResponseTooLargeError(f"{label} exceeded maximum size of {max_bytes} bytes.")


def _content_length(headers: Mapping[str, str]) -> int | None:
    raw_value = next((value for key, value in headers.items() if str(key).lower() == "content-length"), None)
    if raw_value is None:
        return None
    try:
        value = int(str(raw_value))
    except ValueError:
        return None
    return value if value >= 0 else None


async def _read_limited_body(response: Any, *, max_bytes: int, label: str) -> bytes:
    headers = dict(getattr(response, "headers", {}) or {})
    declared_length = _content_length(headers)
    if declared_length is not None and declared_length > max_bytes:
        raise ResponseTooLargeError(f"{label} exceeded maximum size of {max_bytes} bytes.")

    body = bytearray()
    if hasattr(response, "aiter_bytes"):
        async for chunk in response.aiter_bytes():
            body.extend(chunk)
            if len(body) > max_bytes:
                raise ResponseTooLargeError(f"{label} exceeded maximum size of {max_bytes} bytes.")
        return bytes(body)

    # Compatibility fallback for custom Fetcher response doubles. The default
    # httpx path always uses aiter_bytes and therefore enforces the limit while
    # the socket is being consumed.
    raw = await response.aread()
    _enforce_max_bytes(raw, max_bytes, label)
    return raw


async def aclose_default_clients() -> None:
    """Close the calling event loop's default HTTP pool before loop shutdown."""
    loop = asyncio.get_running_loop()
    client = getattr(loop, _LOOP_CLIENT_ATTRIBUTE, None)
    if client is not None:
        delattr(loop, _LOOP_CLIENT_ATTRIBUTE)
    if client is not None:
        await client.aclose()


class HTTPTransport:
    """An application-owned Fetcher with one connection pool and explicit shutdown.

    Pass this object as ``fetch=transport`` to provider factories. Create one
    per application lifespan/event loop and close it after in-flight work ends.
    A supplied httpx client is borrowed and remains owned by the caller.
    """

    def __init__(self, *, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def _get_client(self) -> httpx.AsyncClient:
        from .errors import ConfigurationError

        loop = asyncio.get_running_loop()
        if self._closed:
            raise ConfigurationError("HTTPTransport is closed.")
        if self._loop is not None and self._loop is not loop:
            raise ConfigurationError("HTTPTransport cannot be shared across event loops.")
        self._loop = loop
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_MS / 1000)
        return self._client

    async def __call__(
        self, url: str, *, method: str = "POST", headers: dict[str, str],
        json_body: dict[str, Any] | None = None, body: Any = None,
        timeout_ms: int | None, stream: bool = False,
    ) -> ResponseLike:
        return await _fetch_with_client(
            self._get_client(), url, method=method, headers=headers,
            json_body=json_body, body=body, timeout_ms=timeout_ms, stream=stream,
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        if self._client is not None:
            self._get_client()  # Reject closing an active pool from another loop.
        self._closed = True
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        self._get_client()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


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
    effective_timeout_ms = DEFAULT_TIMEOUT_MS if timeout_ms is None else timeout_ms
    timeout = effective_timeout_ms / 1000
    client = _shared_client(timeout)
    return await _fetch_with_client(
        client, url, method=method, headers=headers, json_body=json_body,
        body=body, timeout_ms=timeout_ms, stream=stream,
    )


async def _fetch_with_client(
    client: httpx.AsyncClient, url: str, *, method: str,
    headers: dict[str, str], json_body: dict[str, Any] | None, body: Any,
    timeout_ms: int | None, stream: bool,
) -> ResponseLike:
    timeout = (DEFAULT_TIMEOUT_MS if timeout_ms is None else timeout_ms) / 1000
    request = client.build_request(method, url, headers=headers, timeout=timeout, **_build_request_kwargs(json_body, body))
    # Always keep the transport in streaming mode. For non-streaming SDK calls
    # we consume it below with an incremental cap instead of allowing httpx to
    # buffer an unbounded response before the SDK can inspect its size.
    response = await client.send(request, stream=True)
    if stream:
        return StreamingResponse(response=response)
    try:
        body_bytes = await _read_limited_body(
            response,
            max_bytes=DEFAULT_MAX_BUFFERED_RESPONSE_BYTES,
            label="response body",
        )
        return BufferedResponse(
            status_code=response.status_code,
            body_bytes=body_bytes,
            headers=dict(getattr(response, "headers", {}) or {}),
        )
    finally:
        await response.aclose()
