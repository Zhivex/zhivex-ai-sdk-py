from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable
from typing import Any, Literal, cast

from .errors import ParseError, UnsupportedFeatureError
from .generate_text import generate_text, stream_text
from .messages import create_text_message
from .schema import create_schema_adapter
from .types import (
    GenerateObjectOutput,
    ModelMessage,
    ObjectStreamEvent,
    StreamObjectCompleteEvent,
    StreamObjectDeltaEvent,
    StreamObjectPartialEvent,
    StreamTextDeltaEvent,
    StructuredOutputConfig,
)

ObjectMode = Literal["native", "prompted"]


def _resolve_object_mode(requested_mode: str, native_allowed: bool) -> ObjectMode:
    object_mode = "native" if requested_mode == "auto" and native_allowed else requested_mode
    if object_mode == "auto":
        object_mode = "prompted"
    if object_mode == "native" and not native_allowed:
        raise UnsupportedFeatureError("Model does not support native structured output.")
    return cast(ObjectMode, object_mode)


def _with_structured_prompt(
    *,
    prompt: str | None,
    messages: list[ModelMessage] | None,
    object_mode: ObjectMode,
) -> tuple[str | None, list[ModelMessage] | None]:
    if object_mode != "prompted":
        return prompt, messages
    suffix = "Return only valid JSON matching the requested schema."
    if messages is not None:
        return None, [*messages, create_text_message("system", suffix)]
    if prompt is not None:
        return f"{prompt}\n\n{suffix}", None
    return None, None


def _parse_object(text: str, schema: Any) -> Any:
    try:
        payload = json.loads(text)
    except Exception as error:
        raise ParseError("Model response is not valid JSON.") from error
    return create_schema_adapter(schema).validate_python(payload)


def _repair_partial_json(text: str) -> str | None:
    start = -1
    for idx, char in enumerate(text):
        if char in "{[":
            start = idx
            break
    if start == -1:
        return None
    slice_text = text[start:].strip()
    if not slice_text:
        return None
    closers: list[str] = []
    in_string = False
    escaped = False
    last_token = "other"
    for char in slice_text:
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
                last_token = "value"
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            closers.append("}")
            last_token = "other"
            continue
        if char == "[":
            closers.append("]")
            last_token = "other"
            continue
        if char in "}]":
            if closers and closers[-1] == char:
                closers.pop()
            last_token = "value"
            continue
        if char == ":":
            last_token = "colon"
            continue
        if char == ",":
            last_token = "comma"
            continue
        if not char.isspace():
            last_token = "value"
    repaired = slice_text.rstrip()
    if in_string:
        repaired += '"'
    if last_token == "colon":
        repaired += " null"
    if last_token == "comma":
        repaired = repaired.rstrip(", ")
    return repaired + "".join(reversed(closers))


def _parse_partial_object(text: str) -> Any | None:
    repaired = _repair_partial_json(text)
    if not repaired:
        return None
    try:
        return json.loads(repaired)
    except Exception:
        return None


class _StreamObjectResult:
    def __init__(self, task: asyncio.Task[GenerateObjectOutput], events: list[ObjectStreamEvent]) -> None:
        self._task = task
        self._events = events

    def event_stream(self) -> AsyncIterable[ObjectStreamEvent]:
        async def generator() -> AsyncIterable[ObjectStreamEvent]:
            index = 0
            while not self._task.done() or index < len(self._events):
                while index < len(self._events):
                    item = self._events[index]
                    index += 1
                    yield item
                if not self._task.done():
                    await asyncio.sleep(0)

        return generator()

    def text_stream(self) -> AsyncIterable[str]:
        async def generator() -> AsyncIterable[str]:
            async for event in self.event_stream():
                if isinstance(event, StreamObjectDeltaEvent):
                    yield event.text_delta

        return generator()

    def partial_object_stream(self) -> AsyncIterable[Any]:
        async def generator() -> AsyncIterable[Any]:
            async for event in self.event_stream():
                if isinstance(event, StreamObjectPartialEvent):
                    yield event.partial_object

        return generator()

    async def collect(self) -> GenerateObjectOutput:
        return await self._task


async def generate_object(
    *,
    model: Any,
    schema: Any,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    system: str | None = None,
    mode: str = "auto",
    schema_name: str | None = None,
    schema_description: str | None = None,
    **kwargs: Any,
) -> GenerateObjectOutput:
    object_mode = _resolve_object_mode(mode, model.capabilities.structured_output)
    prompt, messages = _with_structured_prompt(prompt=prompt, messages=messages, object_mode=object_mode)
    structured_output = (
        StructuredOutputConfig(
            schema=schema,
            mode="native",
            name=schema_name,
            description=schema_description,
        )
        if object_mode == "native"
        else None
    )
    text_result = await generate_text(
        model=model,
        prompt=prompt,
        messages=messages,
        system=system,
        structured_output=structured_output,
        **kwargs,
    )
    return GenerateObjectOutput(
        text=text_result.text,
        finish_reason=text_result.finish_reason,
        provider_finish_reason=text_result.provider_finish_reason,
        usage=text_result.usage,
        steps=text_result.steps,
        messages=text_result.messages,
        tool_results=text_result.tool_results,
        object=_parse_object(text_result.text, schema),
        object_mode=object_mode,
    )


def stream_object(
    *,
    model: Any,
    schema: Any,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    system: str | None = None,
    mode: str = "auto",
    schema_name: str | None = None,
    schema_description: str | None = None,
    **kwargs: Any,
) -> _StreamObjectResult:
    object_mode = _resolve_object_mode(mode, model.capabilities.structured_output)
    prompt, messages = _with_structured_prompt(prompt=prompt, messages=messages, object_mode=object_mode)
    structured_output = (
        StructuredOutputConfig(
            schema=schema,
            mode="native",
            name=schema_name,
            description=schema_description,
        )
        if object_mode == "native"
        else None
    )
    text_result = stream_text(
        model=model,
        prompt=prompt,
        messages=messages,
        system=system,
        structured_output=structured_output,
        **kwargs,
    )
    events: list[ObjectStreamEvent] = []

    async def runner() -> GenerateObjectOutput:
        text_buffer = ""
        last_partial: Any = None
        async for event in text_result.event_stream():
            events.append(event)
            if isinstance(event, StreamTextDeltaEvent):
                text_buffer += event.text_delta
                events.append(StreamObjectDeltaEvent(text_delta=event.text_delta, partial_text=text_buffer))
                partial = _parse_partial_object(text_buffer)
                if partial is not None and partial != last_partial:
                    last_partial = partial
                    events.append(StreamObjectPartialEvent(partial_object=partial))
        final = await text_result.collect()
        obj = _parse_object(final.text, schema)
        events.append(StreamObjectCompleteEvent(object=obj))
        return GenerateObjectOutput(
            text=final.text,
            finish_reason=final.finish_reason,
            provider_finish_reason=final.provider_finish_reason,
            usage=final.usage,
            steps=final.steps,
            messages=final.messages,
            tool_results=final.tool_results,
            object=obj,
            object_mode=object_mode,
        )

    return _StreamObjectResult(asyncio.create_task(runner()), events)
