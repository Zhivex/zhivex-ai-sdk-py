from __future__ import annotations

import json
from collections.abc import AsyncIterable, Callable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from ._http import ResponseLike
from .errors import ParseError, ProviderHTTPError
from .types import UIMessage, UIMessageChunk
from .ui import (
    deserialize_ui_message,
    serialize_ui_message,
    serialize_ui_message_chunk,
    to_ui_message_stream,
)


@dataclass(slots=True)
class HTTPResponse:
    body: str | bytes | AsyncIterable[bytes]
    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)


async def stream_sse(response: ResponseLike):
    if response.status_code >= 400:
        raise ProviderHTTPError(
            f"Streaming request failed with status {response.status_code}.",
            response.status_code,
            response_body=await response.text(),
        )

    async for event in _parse_sse_events(response.iter_lines()):
        yield event


async def _parse_sse_events(lines: AsyncIterable[str]):
    event_name: str | None = None
    data_lines: list[str] = []

    async for line in lines:
        if line == "":
            if data_lines:
                yield {"event": event_name, "data": "\n".join(data_lines)}
            event_name = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())

    if data_lines:
        yield {"event": event_name, "data": "\n".join(data_lines)}


def _normalize_sse_data(value: Any) -> str:
    if is_dataclass(value) and not isinstance(value, type):
        payload = json.dumps(asdict(value))
    else:
        payload = value if isinstance(value, str) else json.dumps(value)
    return "\n".join(f"data: {line}" for line in payload.split("\n"))


def to_sse_stream(source: AsyncIterable[Any], *, event: str | Callable[[Any], str | None] | None = None):
    async def generator():
        async for value in source:
            event_name = event(value) if callable(event) else event
            event_line = f"event: {event_name}\n" if event_name else ""
            yield f"{event_line}{_normalize_sse_data(value)}\n\n".encode("utf-8")

    return generator()


def to_sse_response(
    source: AsyncIterable[Any],
    *,
    event: str | Callable[[Any], str | None] | None = None,
    status_code: int = 200,
    headers: Mapping[str, str] | None = None,
) -> HTTPResponse:
    merged_headers = {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache, no-transform",
        "connection": "keep-alive",
        **dict(headers or {}),
    }
    return HTTPResponse(body=to_sse_stream(source, event=event), status_code=status_code, headers=merged_headers)


def to_text_stream(result: Any):
    async def generator():
        async for chunk in result.text_stream():
            yield chunk.encode("utf-8")

    return generator()


def to_text_stream_response(result: Any, *, status_code: int = 200, headers: Mapping[str, str] | None = None) -> HTTPResponse:
    return HTTPResponse(
        body=to_text_stream(result),
        status_code=status_code,
        headers={"content-type": "text/plain; charset=utf-8", **dict(headers or {})},
    )


def to_ui_message_stream_response(
    source: Any,
    *,
    message_id: str | None = None,
    status_code: int = 200,
    headers: Mapping[str, str] | None = None,
) -> HTTPResponse:
    ui_stream = to_ui_message_stream(source, message_id)
    return to_sse_response(ui_stream, event=lambda chunk: chunk.type, status_code=status_code, headers=headers)


async def _read_request_text(request: Any) -> tuple[str, str]:
    if isinstance(request, bytes):
        return request.decode("utf-8"), "application/json"
    if isinstance(request, str):
        return request, "application/json"

    headers = getattr(request, "headers", {}) or {}
    content_type = ""
    if isinstance(headers, Mapping):
        content_type = str(headers.get("content-type", ""))

    if hasattr(request, "json") and "application/json" in content_type:
        body = await request.json()
        return json.dumps(body), content_type
    if hasattr(request, "text"):
        return await request.text(), content_type
    if hasattr(request, "read"):
        raw = await request.read()
        return raw.decode("utf-8"), content_type

    raise ParseError("Unsupported UI message request type.")


async def parse_ui_message_request(request: Any) -> list[UIMessage]:
    text, content_type = await _read_request_text(request)
    if "application/json" in content_type or isinstance(request, (str, bytes)):
        payload = json.loads(text) if text.strip() else []
        return [deserialize_ui_message(json.dumps(item)) for item in payload]
    if not text.strip():
        return []
    return [deserialize_ui_message(line) for line in text.strip().splitlines()]


def create_ui_message_json_response(messages: list[UIMessage], *, status_code: int = 200, headers: Mapping[str, str] | None = None) -> HTTPResponse:
    return HTTPResponse(
        body=json.dumps([json.loads(serialize_ui_message(message)) for message in messages]),
        status_code=status_code,
        headers={"content-type": "application/json; charset=utf-8", **dict(headers or {})},
    )


def create_ui_message_lines_response(messages: list[UIMessage], *, status_code: int = 200, headers: Mapping[str, str] | None = None) -> HTTPResponse:
    return HTTPResponse(
        body="\n".join(serialize_ui_message(message) for message in messages),
        status_code=status_code,
        headers={"content-type": "application/x-ndjson; charset=utf-8", **dict(headers or {})},
    )


def to_ui_message_chunk_lines(source: AsyncIterable[UIMessageChunk]):
    async def generator():
        async for chunk in source:
            yield f"{serialize_ui_message_chunk(chunk)}\n".encode("utf-8")

    return generator()
