from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass


@dataclass(slots=True)
class SSEEvent:
    event: str | None
    data: str


DEFAULT_MAX_SSE_EVENT_BYTES = 1 * 1024 * 1024


async def parse_sse(lines: AsyncIterable[str], *, max_event_bytes: int = DEFAULT_MAX_SSE_EVENT_BYTES) -> AsyncIterable[SSEEvent]:
    event_name: str | None = None
    data_lines: list[str] = []
    data_bytes = 0

    async for line in lines:
        if line == "":
            if data_lines:
                yield SSEEvent(event=event_name, data="\n".join(data_lines))
            event_name = None
            data_lines = []
            data_bytes = 0
            continue

        if line.startswith(":"):
            continue

        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
            continue

        if line.startswith("data:"):
            data = line.removeprefix("data:").strip()
            data_bytes += len(data.encode("utf-8")) + 1
            if data_bytes > max_event_bytes:
                raise ValueError(f"SSE event exceeded maximum size of {max_event_bytes} bytes.")
            data_lines.append(data)

    if data_lines:
        yield SSEEvent(event=event_name, data="\n".join(data_lines))
