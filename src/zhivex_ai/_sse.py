from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass


@dataclass(slots=True)
class SSEEvent:
    event: str | None
    data: str


async def parse_sse(lines: AsyncIterable[str]) -> AsyncIterable[SSEEvent]:
    event_name: str | None = None
    data_lines: list[str] = []

    async for line in lines:
        if line == "":
            if data_lines:
                yield SSEEvent(event=event_name, data="\n".join(data_lines))
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
        yield SSEEvent(event=event_name, data="\n".join(data_lines))
