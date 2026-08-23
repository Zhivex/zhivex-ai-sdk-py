from __future__ import annotations

import asyncio
import copy
import inspect
import json
import time
import uuid
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .agent import (
    Agent,
    AgentRunStartEvent,
    AgentTextDeltaEvent,
    run_agent,
    stream_agent,
)
from .errors import ParseError, ValidationError
from .messages import create_text_message
from .protocols import (
    HostedAgentRunOptions,
    ProtocolErrorMapper,
    ProtocolEventCallback,
    ProtocolInvocation,
    ProtocolLimits,
    ProtocolRunOptionsResolver,
    _map_protocol_error,
    _notify_protocol_event,
    _resolve_run_options,
)
from .types import JsonValue, ModelMessage, TokenUsage

DEFAULT_RESPONSES_REQUEST_BYTES = 1 * 1024 * 1024
AgentResolver = (
    Mapping[str, Agent] | Callable[[str], Agent | Awaitable[Agent | None] | None]
)
_RESPONSES_REQUEST_FIELDS = frozenset({"model", "input", "instructions", "stream"})
_RESPONSES_MESSAGE_FIELDS = frozenset({"type", "role", "content"})
_RESPONSES_CONTENT_FIELDS = frozenset({"type", "text"})


class _ResponsesExecutionError(Exception):
    """Carry an already-reviewed public failure through the ASGI boundary."""


@dataclass(slots=True)
class StoredResponsesRun:
    response_id: str
    model: str
    status: str = "in_progress"
    response: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    internal_run_id: str | None = None


class ResponsesEventStore(Protocol):
    async def create(
        self,
        record: StoredResponsesRun,
        *,
        invocation: ProtocolInvocation,
    ) -> None: ...

    async def append(
        self,
        response_id: str,
        event: Mapping[str, Any],
        *,
        invocation: ProtocolInvocation,
    ) -> None: ...

    async def complete(
        self,
        response_id: str,
        response: Mapping[str, Any],
        *,
        invocation: ProtocolInvocation,
        internal_run_id: str | None = None,
    ) -> None: ...

    async def get(
        self,
        response_id: str,
        *,
        invocation: ProtocolInvocation,
    ) -> StoredResponsesRun | None: ...

    async def replay(
        self,
        response_id: str,
        *,
        after_sequence: int,
        invocation: ProtocolInvocation,
    ) -> list[dict[str, Any]]: ...


class InMemoryResponsesEventStore:
    """Process-local reference store for tests and development.

    Multi-replica deployments must supply an application-owned implementation
    that scopes every operation using the trusted ``ProtocolInvocation``.
    """

    def __init__(self) -> None:
        self._records: dict[str, StoredResponsesRun] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        record: StoredResponsesRun,
        *,
        invocation: ProtocolInvocation,
    ) -> None:
        del invocation
        async with self._lock:
            if record.response_id in self._records:
                raise ValidationError(
                    f'Responses run "{record.response_id}" already exists.'
                )
            self._records[record.response_id] = copy.deepcopy(record)

    async def append(
        self,
        response_id: str,
        event: Mapping[str, Any],
        *,
        invocation: ProtocolInvocation,
    ) -> None:
        del invocation
        async with self._lock:
            record = self._records.get(response_id)
            if record is None:
                raise ValidationError(f'Responses run "{response_id}" was not found.')
            sequence = event.get("sequence_number")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
            ):
                raise ValidationError(
                    "Stored Responses events require a non-negative sequence_number."
                )
            expected = len(record.events)
            if sequence != expected:
                raise ValidationError(
                    f'Responses run "{response_id}" expected sequence {expected}, received {sequence}.'
                )
            record.events.append(copy.deepcopy(dict(event)))

    async def complete(
        self,
        response_id: str,
        response: Mapping[str, Any],
        *,
        invocation: ProtocolInvocation,
        internal_run_id: str | None = None,
    ) -> None:
        del invocation
        async with self._lock:
            record = self._records.get(response_id)
            if record is None:
                raise ValidationError(f'Responses run "{response_id}" was not found.')
            record.response = copy.deepcopy(dict(response))
            record.status = str(response.get("status") or record.status)
            record.internal_run_id = internal_run_id

    async def get(
        self,
        response_id: str,
        *,
        invocation: ProtocolInvocation,
    ) -> StoredResponsesRun | None:
        del invocation
        async with self._lock:
            record = self._records.get(response_id)
            return copy.deepcopy(record) if record is not None else None

    async def replay(
        self,
        response_id: str,
        *,
        after_sequence: int,
        invocation: ProtocolInvocation,
    ) -> list[dict[str, Any]]:
        del invocation
        async with self._lock:
            record = self._records.get(response_id)
            if record is None:
                raise KeyError(response_id)
            return [
                copy.deepcopy(event)
                for event in record.events
                if int(event["sequence_number"]) > after_sequence
            ]


@dataclass(slots=True)
class _TextBudget:
    limit: int
    used: int = 0

    def add(self, value: str) -> str:
        self.used += len(value)
        if self.used > self.limit:
            raise ValidationError(
                f"Responses input text exceeded the configured {self.limit}-character limit."
            )
        return value


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _usage(usage: TokenUsage | None) -> dict[str, JsonValue]:
    input_tokens = usage.input_tokens if usage and usage.input_tokens is not None else 0
    output_tokens = (
        usage.output_tokens if usage and usage.output_tokens is not None else 0
    )
    total_tokens = (
        usage.total_tokens
        if usage and usage.total_tokens is not None
        else input_tokens + output_tokens
    )
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": total_tokens,
    }


def _reject_unknown_fields(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ValidationError(f"Unsupported {context} field(s): {', '.join(unknown)}.")


def _content_text(
    value: Any,
    *,
    limits: ProtocolLimits,
    budget: _TextBudget,
) -> str:
    if isinstance(value, str):
        return budget.add(value)
    if isinstance(value, list):
        if len(value) > limits.max_parts_per_message:
            raise ValidationError(
                "Responses message content exceeded the configured "
                f"{limits.max_parts_per_message}-part limit."
            )
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(budget.add(part))
            elif isinstance(part, Mapping):
                _reject_unknown_fields(
                    part,
                    allowed=_RESPONSES_CONTENT_FIELDS,
                    context="Responses content-part",
                )
                part_type = part.get("type")
                if part_type not in {"input_text", "text"}:
                    raise ValidationError(
                        f'Unsupported Responses input content type "{part_type}".'
                    )
                text = part.get("text")
                if not isinstance(text, str):
                    raise ValidationError(
                        "Responses text content requires a string text field."
                    )
                parts.append(budget.add(text))
            else:
                raise ValidationError(
                    "Responses input content items must be strings or objects."
                )
        return "\n".join(parts)
    raise ValidationError(
        "Responses input content must be a string or content-part list."
    )


def _input_messages(
    payload: Mapping[str, Any], *, limits: ProtocolLimits
) -> list[ModelMessage]:
    _reject_unknown_fields(
        payload,
        allowed=_RESPONSES_REQUEST_FIELDS,
        context="Responses request",
    )
    if "stream" in payload and not isinstance(payload.get("stream"), bool):
        raise ValidationError("Responses stream must be a boolean.")
    raw_input = payload.get("input")
    if raw_input is None:
        raise ValidationError('Responses requests require an "input" field.')
    messages: list[ModelMessage] = []
    budget = _TextBudget(limits.max_text_chars)
    instructions = payload.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise ValidationError("Responses instructions must be a string.")
        if instructions:
            messages.append(create_text_message("system", budget.add(instructions)))
    if isinstance(raw_input, str):
        messages.append(create_text_message("user", budget.add(raw_input)))
        return messages
    if not isinstance(raw_input, list) or not raw_input:
        raise ValidationError(
            "Responses input must be a non-empty string or item list."
        )
    if len(raw_input) > limits.max_messages:
        raise ValidationError(
            f"Responses input exceeded the configured {limits.max_messages}-message limit."
        )
    for item in raw_input:
        if not isinstance(item, Mapping):
            raise ValidationError("Responses input items must be objects.")
        _reject_unknown_fields(
            item,
            allowed=_RESPONSES_MESSAGE_FIELDS,
            context="Responses message",
        )
        item_type = item.get("type", "message")
        if item_type != "message":
            raise ValidationError(
                f'Unsupported Responses input item type "{item_type}".'
            )
        role = item.get("role", "user")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValidationError(f'Unsupported Responses input role "{role}".')
        messages.append(
            create_text_message(
                "system" if role == "developer" else role,
                _content_text(
                    item.get("content", ""),
                    limits=limits,
                    budget=budget,
                ),
            )
        )
    return messages


def _output_message(
    *, item_id: str, text: str, status: str = "completed"
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": text,
            }
        ],
    }


def _response_object(
    *,
    response_id: str,
    model: str,
    created_at: int,
    output: list[dict[str, Any]],
    usage: TokenUsage | None,
    status: str = "completed",
    error: dict[str, JsonValue] | None = None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "background": False,
        "billing": {"payer": "developer"},
        "completed_at": int(time.time()) if status == "completed" else None,
        "error": error,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "prompt_cache_retention": None,
        "reasoning": {"effort": None, "summary": None},
        "safety_identifier": None,
        "service_tier": "default",
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_logprobs": 0,
        "top_p": None,
        "truncation": "disabled",
        "usage": None if usage is None and status != "completed" else _usage(usage),
        "user": None,
        "metadata": {},
    }


class ResponsesAgentHost:
    """Host configured agents behind the OpenAI Responses create/stream shape.

    The request ``model`` is an application-owned alias. It never accepts or
    constructs arbitrary provider credentials or provider model identifiers.
    """

    def __init__(
        self,
        agents: AgentResolver,
        *,
        run_options_resolver: ProtocolRunOptionsResolver | None = None,
        error_mapper: ProtocolErrorMapper | None = None,
        on_protocol_event: ProtocolEventCallback | None = None,
        limits: ProtocolLimits | None = None,
        event_store: ResponsesEventStore | None = None,
    ) -> None:
        self.agents = agents
        self.run_options_resolver = run_options_resolver
        self.error_mapper = error_mapper
        self.on_protocol_event = on_protocol_event
        self.limits = limits or ProtocolLimits(
            max_request_bytes=DEFAULT_RESPONSES_REQUEST_BYTES
        )
        self.event_store = event_store

    async def resolve(self, model: str) -> Agent:
        agent: Any
        if isinstance(self.agents, Mapping):
            agent = self.agents.get(model)
        else:
            agent = self.agents(model)
            if inspect.isawaitable(agent):
                agent = await agent
        if agent is None:
            raise KeyError(model)
        if not isinstance(agent, Agent):
            raise TypeError("Responses agent resolvers must return an Agent or None.")
        return agent

    async def prepare(
        self, payload: Mapping[str, Any]
    ) -> tuple[str, Agent, list[ModelMessage]]:
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise ValidationError(
                'Responses requests require a non-empty "model" alias.'
            )
        if len(model) > self.limits.max_alias_chars:
            raise ValidationError(
                "Responses model alias exceeded the configured "
                f"{self.limits.max_alias_chars}-character limit."
            )
        messages = _input_messages(payload, limits=self.limits)
        return model, await self.resolve(model), messages

    def invocation(
        self,
        payload: Mapping[str, Any],
        *,
        action: str,
        request: Any = None,
        response_id: str | None = None,
    ) -> ProtocolInvocation:
        model = payload.get("model")
        external_ids = {"response_id": response_id} if response_id is not None else {}
        return ProtocolInvocation(
            protocol="responses",
            action=action,
            agent_alias=model if isinstance(model, str) else None,
            external_ids=external_ids,
            request=request,
            payload=payload,
        )

    async def _run_options(
        self, invocation: ProtocolInvocation
    ) -> HostedAgentRunOptions:
        return await _resolve_run_options(
            invocation, resolver=self.run_options_resolver
        )

    async def _create_store_record(
        self,
        *,
        response_id: str,
        model: str,
        response: dict[str, Any],
        invocation: ProtocolInvocation,
    ) -> None:
        if self.event_store is not None:
            await self.event_store.create(
                StoredResponsesRun(
                    response_id=response_id,
                    model=model,
                    status=str(response["status"]),
                    response=copy.deepcopy(response),
                ),
                invocation=invocation,
            )

    async def _append_event(
        self,
        *,
        response_id: str,
        event: dict[str, Any],
        invocation: ProtocolInvocation,
    ) -> dict[str, Any]:
        if self.event_store is not None:
            await self.event_store.append(response_id, event, invocation=invocation)
        return event

    async def _complete_store_record(
        self,
        *,
        response_id: str,
        response: dict[str, Any],
        invocation: ProtocolInvocation,
        internal_run_id: str | None,
    ) -> None:
        if self.event_store is not None:
            await self.event_store.complete(
                response_id,
                response,
                invocation=invocation,
                internal_run_id=internal_run_id,
            )

    async def create(
        self,
        payload: Mapping[str, Any],
        *,
        invocation: ProtocolInvocation | None = None,
    ) -> dict[str, Any]:
        if payload.get("stream") is True:
            raise ValidationError(
                "Use ResponsesAgentHost.stream() for streaming requests."
            )
        model, agent, messages = await self.prepare(payload)
        response_id = _new_id("resp")
        created_at = int(time.time())
        resolved_invocation = invocation or self.invocation(
            payload,
            action="create",
            response_id=response_id,
        )
        resolved_invocation.external_ids["response_id"] = response_id
        options = await self._run_options(resolved_invocation)
        in_progress = _response_object(
            response_id=response_id,
            model=model,
            created_at=created_at,
            output=[],
            usage=None,
            status="in_progress",
        )
        await self._create_store_record(
            response_id=response_id,
            model=model,
            response=in_progress,
            invocation=resolved_invocation,
        )
        await _notify_protocol_event(
            self.on_protocol_event,
            resolved_invocation,
            status="started",
        )
        try:
            result = await run_agent(
                agent=agent,
                session=options.session,
                messages=messages,
                deps=options.deps,
                runtime=options.runtime,
                idempotency_key=options.idempotency_key,
            )
        except Exception as error:
            public_message = await _map_protocol_error(
                error, resolved_invocation, self.error_mapper
            )
            failed = _response_object(
                response_id=response_id,
                model=model,
                created_at=created_at,
                output=[],
                usage=None,
                status="failed",
                error={"code": "agent_execution_error", "message": public_message},
            )
            await self._complete_store_record(
                response_id=response_id,
                response=failed,
                invocation=resolved_invocation,
                internal_run_id=None,
            )
            await _notify_protocol_event(
                self.on_protocol_event,
                resolved_invocation,
                status="failed",
                error_code=type(error).__name__,
            )
            raise _ResponsesExecutionError(public_message) from None
        item = _output_message(item_id=_new_id("msg"), text=result.text)
        completed = _response_object(
            response_id=response_id,
            model=model,
            created_at=created_at,
            output=[item],
            usage=result.usage,
        )
        await self._complete_store_record(
            response_id=response_id,
            response=completed,
            invocation=resolved_invocation,
            internal_run_id=result.run_id,
        )
        await _notify_protocol_event(
            self.on_protocol_event,
            resolved_invocation,
            status="completed",
            internal_run_id=result.run_id,
        )
        return completed

    async def stream(
        self,
        payload: Mapping[str, Any],
        *,
        invocation: ProtocolInvocation | None = None,
    ) -> AsyncIterable[dict[str, Any]]:
        prepared = await self.prepare(payload)
        async for event in self.stream_prepared(
            payload,
            prepared=prepared,
            invocation=invocation,
        ):
            yield event

    async def stream_prepared(
        self,
        payload: Mapping[str, Any],
        *,
        prepared: tuple[str, Agent, list[ModelMessage]],
        invocation: ProtocolInvocation | None = None,
    ) -> AsyncIterable[dict[str, Any]]:
        model, agent, messages = prepared
        response_id = _new_id("resp")
        item_id = _new_id("msg")
        created_at = int(time.time())
        sequence = 0
        empty = _response_object(
            response_id=response_id,
            model=model,
            created_at=created_at,
            output=[],
            usage=None,
            status="in_progress",
        )
        resolved_invocation = invocation or self.invocation(
            payload,
            action="stream",
            response_id=response_id,
        )
        resolved_invocation.external_ids["response_id"] = response_id
        options = await self._run_options(resolved_invocation)
        await self._create_store_record(
            response_id=response_id,
            model=model,
            response=empty,
            invocation=resolved_invocation,
        )
        await _notify_protocol_event(
            self.on_protocol_event,
            resolved_invocation,
            status="started",
        )
        yield await self._append_event(
            response_id=response_id,
            event={
                "type": "response.created",
                "sequence_number": sequence,
                "response": empty,
            },
            invocation=resolved_invocation,
        )
        sequence += 1
        yield await self._append_event(
            response_id=response_id,
            event={
                "type": "response.in_progress",
                "sequence_number": sequence,
                "response": empty,
            },
            invocation=resolved_invocation,
        )
        sequence += 1
        output = _output_message(item_id=item_id, text="", status="in_progress")
        yield await self._append_event(
            response_id=response_id,
            event={
                "type": "response.output_item.added",
                "sequence_number": sequence,
                "output_index": 0,
                "item": output,
            },
            invocation=resolved_invocation,
        )
        sequence += 1
        content = {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}
        yield await self._append_event(
            response_id=response_id,
            event={
                "type": "response.content_part.added",
                "sequence_number": sequence,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": content,
            },
            invocation=resolved_invocation,
        )
        sequence += 1
        stream = stream_agent(
            agent=agent,
            session=options.session,
            messages=messages,
            deps=options.deps,
            runtime=options.runtime,
            idempotency_key=options.idempotency_key,
        )
        internal_run_id: str | None = None
        try:
            async for event in stream.event_stream():
                if isinstance(event, AgentRunStartEvent):
                    internal_run_id = event.run_id
                    await _notify_protocol_event(
                        self.on_protocol_event,
                        resolved_invocation,
                        status="running",
                        internal_run_id=internal_run_id,
                    )
                elif isinstance(event, AgentTextDeltaEvent) and event.text_delta:
                    yield await self._append_event(
                        response_id=response_id,
                        event={
                            "type": "response.output_text.delta",
                            "sequence_number": sequence,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": event.text_delta,
                            "logprobs": [],
                        },
                        invocation=resolved_invocation,
                    )
                    sequence += 1
            result = await stream.collect()
        except Exception as error:
            public_message = await _map_protocol_error(
                error, resolved_invocation, self.error_mapper
            )
            failed = _response_object(
                response_id=response_id,
                model=model,
                created_at=created_at,
                output=[],
                usage=None,
                status="failed",
                error={"code": "agent_execution_error", "message": public_message},
            )
            failed_event = await self._append_event(
                response_id=response_id,
                event={
                    "type": "response.failed",
                    "sequence_number": sequence,
                    "response": failed,
                },
                invocation=resolved_invocation,
            )
            await self._complete_store_record(
                response_id=response_id,
                response=failed,
                invocation=resolved_invocation,
                internal_run_id=internal_run_id,
            )
            await _notify_protocol_event(
                self.on_protocol_event,
                resolved_invocation,
                status="failed",
                internal_run_id=internal_run_id,
                error_code=type(error).__name__,
            )
            yield failed_event
            return
        final_part = {
            "type": "output_text",
            "annotations": [],
            "logprobs": [],
            "text": result.text,
        }
        yield await self._append_event(
            response_id=response_id,
            event={
                "type": "response.output_text.done",
                "sequence_number": sequence,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": result.text,
                "logprobs": [],
            },
            invocation=resolved_invocation,
        )
        sequence += 1
        yield await self._append_event(
            response_id=response_id,
            event={
                "type": "response.content_part.done",
                "sequence_number": sequence,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": final_part,
            },
            invocation=resolved_invocation,
        )
        sequence += 1
        final_item = _output_message(item_id=item_id, text=result.text)
        yield await self._append_event(
            response_id=response_id,
            event={
                "type": "response.output_item.done",
                "sequence_number": sequence,
                "output_index": 0,
                "item": final_item,
            },
            invocation=resolved_invocation,
        )
        sequence += 1
        completed = _response_object(
            response_id=response_id,
            model=model,
            created_at=created_at,
            output=[final_item],
            usage=result.usage,
        )
        completed_event = await self._append_event(
            response_id=response_id,
            event={
                "type": "response.completed",
                "sequence_number": sequence,
                "response": completed,
            },
            invocation=resolved_invocation,
        )
        await self._complete_store_record(
            response_id=response_id,
            response=completed,
            invocation=resolved_invocation,
            internal_run_id=result.run_id,
        )
        await _notify_protocol_event(
            self.on_protocol_event,
            resolved_invocation,
            status="completed",
            internal_run_id=result.run_id,
        )
        yield completed_event

    async def get(
        self,
        response_id: str,
        *,
        invocation: ProtocolInvocation | None = None,
    ) -> StoredResponsesRun | None:
        if self.event_store is None:
            return None
        resolved_invocation = invocation or ProtocolInvocation(
            protocol="responses",
            action="get",
            external_ids={"response_id": response_id},
        )
        return await self.event_store.get(response_id, invocation=resolved_invocation)

    async def replay(
        self,
        response_id: str,
        *,
        after_sequence: int = -1,
        invocation: ProtocolInvocation | None = None,
    ) -> list[dict[str, Any]]:
        if (
            isinstance(after_sequence, bool)
            or not isinstance(after_sequence, int)
            or after_sequence < -1
        ):
            raise ValidationError(
                "Responses replay cursor must be an integer greater than or equal to -1."
            )
        if self.event_store is None:
            raise KeyError(response_id)
        resolved_invocation = invocation or ProtocolInvocation(
            protocol="responses",
            action="replay",
            external_ids={"response_id": response_id},
        )
        return await self.event_store.replay(
            response_id,
            after_sequence=after_sequence,
            invocation=resolved_invocation,
        )


async def _read_payload(request: Any, *, max_request_bytes: int) -> dict[str, Any]:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > max_request_bytes:
                raise ParseError(
                    "Responses request body exceeded the configured maximum size."
                )
        except ValueError:
            raise ParseError(
                "Responses request contained an invalid Content-Length header."
            ) from None
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > max_request_bytes:
            raise ParseError(
                "Responses request body exceeded the configured maximum size."
            )
    payload = json.loads(bytes(raw) or b"{}")
    if not isinstance(payload, dict):
        raise ParseError("Responses request body must be a JSON object.")
    return payload


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _install_responses_routes(
    app: Any,
    *,
    host: ResponsesAgentHost,
    authorize: Callable[[Any], bool | Awaitable[bool]] | None,
) -> None:
    from fastapi import Request  # type: ignore[import-not-found]
    from fastapi.responses import JSONResponse, StreamingResponse  # type: ignore[import-not-found]

    async def endpoint(request: Any) -> Any:
        if authorize is not None and not await _maybe_await(authorize(request)):
            return JSONResponse(
                {"error": {"type": "authentication_error", "message": "Unauthorized."}},
                status_code=401,
            )
        invocation: ProtocolInvocation | None = None
        try:
            payload = await _read_payload(
                request,
                max_request_bytes=host.limits.max_request_bytes,
            )
            invocation = host.invocation(
                payload,
                action="stream" if payload.get("stream") is True else "create",
                request=request,
            )
            if payload.get("stream") is True:
                prepared = await host.prepare(payload)
                source = host.stream_prepared(
                    payload,
                    prepared=prepared,
                    invocation=invocation,
                )

                async def encoded():
                    async for item in source:
                        event_name = item["type"]
                        event_id = item["sequence_number"]
                        data = json.dumps(
                            item, ensure_ascii=False, separators=(",", ":")
                        )
                        yield f"id: {event_id}\nevent: {event_name}\ndata: {data}\n\n"

                return StreamingResponse(
                    encoded(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache, no-transform",
                        "X-Accel-Buffering": "no",
                    },
                )
            return JSONResponse(await host.create(payload, invocation=invocation))
        except KeyError as error:
            return JSONResponse(
                {
                    "error": {
                        "type": "invalid_request_error",
                        "message": f'Unknown model alias "{error.args[0]}".',
                    }
                },
                status_code=404,
            )
        except (ParseError, ValidationError, json.JSONDecodeError) as error:
            return JSONResponse(
                {"error": {"type": "invalid_request_error", "message": str(error)}},
                status_code=400,
            )
        except _ResponsesExecutionError as error:
            return JSONResponse(
                {"error": {"type": "server_error", "message": str(error)}},
                status_code=500,
            )
        except Exception as error:
            resolved_invocation = invocation or ProtocolInvocation(
                protocol="responses",
                action="create",
                request=request,
            )
            public_message = await _map_protocol_error(
                error, resolved_invocation, host.error_mapper
            )
            return JSONResponse(
                {"error": {"type": "server_error", "message": public_message}},
                status_code=500,
            )

    async def replay_endpoint(request: Any, response_id: str) -> Any:
        if authorize is not None and not await _maybe_await(authorize(request)):
            return JSONResponse(
                {"error": {"type": "authentication_error", "message": "Unauthorized."}},
                status_code=401,
            )
        invocation = ProtocolInvocation(
            protocol="responses",
            action="replay",
            external_ids={"response_id": response_id},
            request=request,
        )
        raw_cursor = request.query_params.get("after_sequence")
        if raw_cursor is None:
            raw_cursor = request.headers.get("last-event-id")
        try:
            after_sequence = int(raw_cursor) if raw_cursor is not None else -1
            events = await host.replay(
                response_id,
                after_sequence=after_sequence,
                invocation=invocation,
            )
        except (TypeError, ValueError, ValidationError) as error:
            return JSONResponse(
                {"error": {"type": "invalid_request_error", "message": str(error)}},
                status_code=400,
            )
        except KeyError:
            return JSONResponse(
                {
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Response not found.",
                    }
                },
                status_code=404,
            )

        async def encoded():
            for item in events:
                event_name = item["type"]
                event_id = item["sequence_number"]
                data = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                yield f"id: {event_id}\nevent: {event_name}\ndata: {data}\n\n"

        return StreamingResponse(
            encoded(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    async def get_endpoint(request: Any, response_id: str) -> Any:
        if authorize is not None and not await _maybe_await(authorize(request)):
            return JSONResponse(
                {"error": {"type": "authentication_error", "message": "Unauthorized."}},
                status_code=401,
            )
        invocation = ProtocolInvocation(
            protocol="responses",
            action="get",
            external_ids={"response_id": response_id},
            request=request,
        )
        record = await host.get(response_id, invocation=invocation)
        if record is None or record.response is None:
            return JSONResponse(
                {
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Response not found.",
                    }
                },
                status_code=404,
            )
        return JSONResponse(record.response)

    endpoint.__annotations__["request"] = Request
    replay_endpoint.__annotations__["request"] = Request
    replay_endpoint.__annotations__["response_id"] = str
    get_endpoint.__annotations__["request"] = Request
    get_endpoint.__annotations__["response_id"] = str
    app.add_api_route("/v1/responses", endpoint, methods=["POST"])
    app.add_api_route(
        "/v1/responses/{response_id}/events", replay_endpoint, methods=["GET"]
    )
    app.add_api_route("/v1/responses/{response_id}", get_endpoint, methods=["GET"])


def create_responses_app(
    *,
    agents: AgentResolver,
    authorize: Callable[[Any], bool | Awaitable[bool]] | None = None,
    max_request_bytes: int | None = None,
    limits: ProtocolLimits | None = None,
    run_options_resolver: ProtocolRunOptionsResolver | None = None,
    error_mapper: ProtocolErrorMapper | None = None,
    on_protocol_event: ProtocolEventCallback | None = None,
    event_store: ResponsesEventStore | None = None,
):
    """Create an optional FastAPI app with ``POST /v1/responses``."""

    try:
        from fastapi import FastAPI  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            'Responses hosting requires `pip install "zhivex-ai-sdk[api]"`.'
        ) from error
    effective_limits = limits or ProtocolLimits(
        max_request_bytes=DEFAULT_RESPONSES_REQUEST_BYTES
    )
    if max_request_bytes is not None:
        effective_limits = replace(
            effective_limits, max_request_bytes=max_request_bytes
        )
    app = FastAPI(title="Zhivex Responses Host", version="0.20.0")
    _install_responses_routes(
        app,
        host=ResponsesAgentHost(
            agents,
            run_options_resolver=run_options_resolver,
            error_mapper=error_mapper,
            on_protocol_event=on_protocol_event,
            limits=effective_limits,
            event_store=event_store,
        ),
        authorize=authorize,
    )
    return app


_PLAYGROUND_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Zhivex Agent Playground</title>
<style>body{font:16px system-ui;max-width:880px;margin:3rem auto;padding:0 1rem;background:#111;color:#eee}
textarea,input,button,pre{box-sizing:border-box;width:100%;padding:.8rem;margin:.4rem 0;background:#1d1d1d;color:#eee;border:1px solid #444;border-radius:8px}
button{cursor:pointer;background:#6d4aff;border:0;font-weight:700}pre{min-height:12rem;white-space:pre-wrap}</style></head>
<body><h1>Zhivex Agent Playground</h1><label>Model alias<input id="model" value="default"></label>
<label>Prompt<textarea id="prompt" rows="7">Hello</textarea></label><button id="run">Run</button><pre id="output"></pre>
<script>document.querySelector('#run').onclick=async()=>{const out=document.querySelector('#output');out.textContent='';
const response=await fetch('/v1/responses',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({model:document.querySelector('#model').value,input:document.querySelector('#prompt').value,stream:true})});
if(!response.ok){out.textContent=await response.text();return}const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
while(true){const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});const events=buffer.split('\n\n');buffer=events.pop();for(const event of events){const line=event.split('\n').find(x=>x.startsWith('data: '));if(!line)continue;const data=JSON.parse(line.slice(6));if(data.type==='response.output_text.delta')out.textContent+=data.delta;}}};</script>
</body></html>"""


def create_agent_playground_app(
    *,
    agents: AgentResolver,
    authorize: Callable[[Any], bool | Awaitable[bool]] | None = None,
    max_request_bytes: int | None = None,
    limits: ProtocolLimits | None = None,
    run_options_resolver: ProtocolRunOptionsResolver | None = None,
    error_mapper: ProtocolErrorMapper | None = None,
    on_protocol_event: ProtocolEventCallback | None = None,
    event_store: ResponsesEventStore | None = None,
):
    """Create a local playground plus the Responses-compatible endpoint."""

    try:
        from fastapi import FastAPI  # type: ignore[import-not-found]
        from fastapi.responses import HTMLResponse, JSONResponse  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            'The playground requires `pip install "zhivex-ai-sdk[api]"`.'
        ) from error
    effective_limits = limits or ProtocolLimits(
        max_request_bytes=DEFAULT_RESPONSES_REQUEST_BYTES
    )
    if max_request_bytes is not None:
        effective_limits = replace(
            effective_limits, max_request_bytes=max_request_bytes
        )
    app = FastAPI(title="Zhivex Agent Playground", version="0.20.0")
    _install_responses_routes(
        app,
        host=ResponsesAgentHost(
            agents,
            run_options_resolver=run_options_resolver,
            error_mapper=error_mapper,
            on_protocol_event=on_protocol_event,
            limits=effective_limits,
            event_store=event_store,
        ),
        authorize=authorize,
    )

    async def index() -> Any:
        return HTMLResponse(_PLAYGROUND_HTML)

    async def health() -> Any:
        return JSONResponse({"status": "ok"})

    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/health", health, methods=["GET"])
    return app
