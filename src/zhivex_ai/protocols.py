from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .agent import (
    Agent,
    AgentErrorEvent,
    AgentFinishEvent,
    AgentRunStartEvent,
    AgentRuntime,
    AgentSession,
    AgentTextDeltaEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
    stream_agent,
)
from .agent_state import cancel_agent_run
from .errors import ParseError, ValidationError
from .transport import HTTPResponse
from .types import JsonValue

A2A_PROTOCOL_VERSION = "1.0"
A2A_HTTP_JSON_BINDING = "HTTP+JSON"
DEFAULT_PROTOCOL_REQUEST_BYTES = 1 * 1024 * 1024


@dataclass(slots=True)
class HostedAgentRunOptions:
    """Trusted, application-resolved options for one hosted agent run.

    Authentication and tenant ownership remain application responsibilities.
    In particular, ``deps`` is intentionally opaque and must never be persisted
    by a protocol adapter.
    """

    session: AgentSession | None = None
    deps: Any = field(default=None, repr=False, compare=False)
    idempotency_key: str | None = None
    runtime: AgentRuntime | None = field(default=None, repr=False, compare=False)


@dataclass(slots=True)
class ProtocolInvocation:
    """Trusted routing context passed only to application-owned resolvers/stores."""

    protocol: str
    action: str
    agent_alias: str | None = None
    external_ids: dict[str, str] = field(default_factory=dict)
    request: Any = field(default=None, repr=False, compare=False)
    payload: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ProtocolLimits:
    """Finite local parsing limits; not a rate-limit or tenancy policy."""

    max_request_bytes: int = DEFAULT_PROTOCOL_REQUEST_BYTES
    max_alias_chars: int = 128
    max_identifier_chars: int = 256
    max_messages: int = 128
    max_parts_per_message: int = 64
    max_text_chars: int = 256 * 1024

    def __post_init__(self) -> None:
        for name in (
            "max_request_bytes",
            "max_alias_chars",
            "max_identifier_chars",
            "max_messages",
            "max_parts_per_message",
            "max_text_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"ProtocolLimits.{name} must be a positive integer.")


ProtocolRunOptionsResolver = Callable[
    [ProtocolInvocation],
    HostedAgentRunOptions | Awaitable[HostedAgentRunOptions],
]
ProtocolErrorMapper = Callable[[Exception, ProtocolInvocation], str | Awaitable[str]]
ProtocolEventCallback = Callable[[Mapping[str, JsonValue]], None | Awaitable[None]]


async def _resolve_run_options(
    invocation: ProtocolInvocation,
    *,
    options: HostedAgentRunOptions | None = None,
    resolver: ProtocolRunOptionsResolver | None = None,
) -> HostedAgentRunOptions:
    if options is not None and resolver is not None:
        raise ValidationError(
            "Pass hosted run options or a run-options resolver, not both."
        )
    resolved: Any = options
    if resolver is not None:
        resolved = resolver(invocation)
        if inspect.isawaitable(resolved):
            resolved = await resolved
    if resolved is None:
        return HostedAgentRunOptions()
    if not isinstance(resolved, HostedAgentRunOptions):
        raise TypeError(
            "Protocol run-options resolvers must return HostedAgentRunOptions."
        )
    return resolved


async def _map_protocol_error(
    error: Exception,
    invocation: ProtocolInvocation,
    mapper: ProtocolErrorMapper | None,
) -> str:
    if mapper is None:
        return "Agent execution failed."
    mapped = mapper(error, invocation)
    if inspect.isawaitable(mapped):
        mapped = await mapped
    if not isinstance(mapped, str) or not mapped.strip():
        raise TypeError("Protocol error mappers must return a non-empty string.")
    return mapped.strip()


async def _notify_protocol_event(
    callback: ProtocolEventCallback | None,
    invocation: ProtocolInvocation,
    *,
    status: str,
    internal_run_id: str | None = None,
    error_code: str | None = None,
) -> None:
    if callback is None:
        return
    event: dict[str, JsonValue] = {
        "protocol": invocation.protocol,
        "action": invocation.action,
        "status": status,
        "external_ids": dict(invocation.external_ids),
    }
    if invocation.agent_alias is not None:
        event["agent_alias"] = invocation.agent_alias
    if internal_run_id is not None:
        event["internal_run_id"] = internal_run_id
    if error_code is not None:
        event["error_code"] = error_code
    result = callback(event)
    if inspect.isawaitable(result):
        await result


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _json_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


@dataclass(slots=True)
class A2AAgentSkill:
    id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    input_modes: list[str] | None = None
    output_modes: list[str] | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "examples": list(self.examples),
        }
        if self.input_modes is not None:
            value["inputModes"] = list(self.input_modes)
        if self.output_modes is not None:
            value["outputModes"] = list(self.output_modes)
        return value


@dataclass(slots=True)
class A2AAgentCard:
    name: str
    description: str
    url: str
    version: str
    skills: list[A2AAgentSkill]
    protocol_version: str = A2A_PROTOCOL_VERSION
    preferred_transport: str = A2A_HTTP_JSON_BINDING
    default_input_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    default_output_modes: list[str] = field(default_factory=lambda: ["text/plain"])
    streaming: bool = True
    provider: dict[str, str] | None = None
    documentation_url: str | None = None
    security_schemes: dict[str, JsonValue] = field(default_factory=dict)
    security: list[dict[str, list[str]]] = field(default_factory=list)

    def to_dict(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "supportedInterfaces": [
                {
                    "url": self.url.rstrip("/"),
                    "protocolBinding": self.preferred_transport,
                    "protocolVersion": self.protocol_version,
                }
            ],
            "capabilities": {"streaming": self.streaming},
            "defaultInputModes": list(self.default_input_modes),
            "defaultOutputModes": list(self.default_output_modes),
            "skills": [skill.to_dict() for skill in self.skills],
        }
        if self.provider is not None:
            value["provider"] = dict(self.provider)
        if self.documentation_url is not None:
            value["documentationUrl"] = self.documentation_url
        if self.security_schemes:
            value["securitySchemes"] = dict(self.security_schemes)
        if self.security:
            value["securityRequirements"] = _json_value(self.security)
        return value


def create_a2a_agent_card(
    agent: Agent,
    *,
    url: str,
    version: str,
    description: str | None = None,
    skills: list[A2AAgentSkill] | None = None,
    provider: dict[str, str] | None = None,
    documentation_url: str | None = None,
    security_schemes: dict[str, JsonValue] | None = None,
    security: list[dict[str, list[str]]] | None = None,
) -> A2AAgentCard:
    resolved_description = (
        description or f"{agent.name} agent exposed by Zhivex AI SDK."
    )
    resolved_skills = skills or [
        A2AAgentSkill(
            id=agent.name,
            name=agent.name,
            description=resolved_description,
            tags=["zhivex", "agent"],
        )
    ]
    return A2AAgentCard(
        name=agent.name,
        description=resolved_description,
        url=url,
        version=version,
        skills=resolved_skills,
        provider=provider,
        documentation_url=documentation_url,
        security_schemes=security_schemes or {},
        security=security or [],
    )


def _validate_identifier(value: str, *, name: str, limits: ProtocolLimits) -> str:
    if len(value) > limits.max_identifier_chars:
        raise ValidationError(
            f"{name} exceeded the configured {limits.max_identifier_chars}-character limit."
        )
    return value


def _a2a_message(
    payload: Mapping[str, Any],
    *,
    limits: ProtocolLimits | None = None,
) -> Mapping[str, Any]:
    effective_limits = limits or ProtocolLimits()
    message = payload.get("message")
    if not isinstance(message, Mapping):
        raise ValidationError('A2A SendMessageRequest requires a "message" object.')
    if message.get("role") not in {"user", "ROLE_USER"}:
        raise ValidationError("A2A input messages must use the A2A user role.")
    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValidationError("A2A input messages require at least one part.")
    if len(parts) > effective_limits.max_parts_per_message:
        raise ValidationError(
            "A2A input message exceeded the configured "
            f"{effective_limits.max_parts_per_message}-part limit."
        )
    for key in ("messageId", "taskId", "contextId"):
        value = message.get(key)
        if value is not None:
            _validate_identifier(str(value), name=f"A2A {key}", limits=effective_limits)
    return message


def _a2a_prompt(
    message: Mapping[str, Any],
    *,
    limits: ProtocolLimits | None = None,
) -> str:
    effective_limits = limits or ProtocolLimits()
    values: list[str] = []
    total_chars = 0
    for part in message.get("parts", []):
        if not isinstance(part, Mapping):
            raise ValidationError("A2A message parts must be objects.")
        members = [
            name
            for name in ("text", "raw", "url", "data")
            if part.get(name) is not None
        ]
        if len(members) != 1:
            raise ValidationError(
                "Each A2A v1 part must set exactly one of text, raw, url, or data."
            )
        member = members[0]
        value = part[member]
        if member == "text":
            rendered = str(value)
        elif member == "data":
            rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            rendered = str(value)
        total_chars += len(rendered)
        if total_chars > effective_limits.max_text_chars:
            raise ValidationError(
                "A2A input text exceeded the configured "
                f"{effective_limits.max_text_chars}-character limit."
            )
        values.append(rendered)
    return "\n".join(values)


def _a2a_status(state: str, *, message: str | None = None) -> dict[str, JsonValue]:
    status: dict[str, JsonValue] = {
        "state": state,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if message is not None:
        status["message"] = {
            "messageId": _new_id("msg"),
            "role": "ROLE_AGENT",
            "parts": [{"text": message}],
        }
    return status


class A2AAgentExecutor:
    """In-process A2A v1 executor backed by a configured Zhivex Agent."""

    def __init__(
        self,
        agent: Agent,
        *,
        run_options_resolver: ProtocolRunOptionsResolver | None = None,
        error_mapper: ProtocolErrorMapper | None = None,
        on_protocol_event: ProtocolEventCallback | None = None,
        limits: ProtocolLimits | None = None,
    ) -> None:
        self.agent = agent
        self.run_options_resolver = run_options_resolver
        self.error_mapper = error_mapper
        self.on_protocol_event = on_protocol_event
        self.limits = limits or ProtocolLimits()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}
        self._internal_run_ids: dict[str, str] = {}

    def _invocation(
        self,
        *,
        action: str,
        payload: Mapping[str, Any],
        task_id: str,
        context_id: str,
        request: Any = None,
    ) -> ProtocolInvocation:
        return ProtocolInvocation(
            protocol="a2a",
            action=action,
            agent_alias=self.agent.name,
            external_ids={"task_id": task_id, "context_id": context_id},
            request=request,
            payload=payload,
        )

    async def _run(
        self,
        *,
        prompt: str,
        invocation: ProtocolInvocation,
    ) -> Any:
        options = await _resolve_run_options(
            invocation, resolver=self.run_options_resolver
        )
        await _notify_protocol_event(
            self.on_protocol_event, invocation, status="started"
        )
        stream = stream_agent(
            agent=self.agent,
            session=options.session,
            prompt=prompt,
            deps=options.deps,
            runtime=options.runtime,
            idempotency_key=options.idempotency_key,
        )
        task_id = invocation.external_ids.get("task_id", "")
        runner = getattr(stream, "_runner", None)
        if isinstance(runner, asyncio.Task):
            self._active_tasks[task_id] = runner
        try:
            async for event in stream.event_stream():
                if isinstance(event, AgentRunStartEvent):
                    self._internal_run_ids[task_id] = event.run_id
                    await _notify_protocol_event(
                        self.on_protocol_event,
                        invocation,
                        status="running",
                        internal_run_id=event.run_id,
                    )
            result = await stream.collect()
        except Exception as error:
            await _notify_protocol_event(
                self.on_protocol_event,
                invocation,
                status="failed",
                internal_run_id=self._internal_run_ids.get(task_id),
                error_code=type(error).__name__,
            )
            raise
        await _notify_protocol_event(
            self.on_protocol_event,
            invocation,
            status="completed",
            internal_run_id=result.run_id,
        )
        return result

    async def cancel_active(
        self, task_id: str, *, reason: str = "A2A task cancelled."
    ) -> None:
        internal_run_id = self._internal_run_ids.get(task_id)
        if internal_run_id is not None and self.agent.run_store is not None:
            await cancel_agent_run(self.agent.run_store, internal_run_id, reason=reason)
        active = self._active_tasks.get(task_id)
        if active is not None and not active.done():
            active.cancel()

    async def send_message(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        message = _a2a_message(payload, limits=self.limits)
        task_id = str(message.get("taskId") or _new_id("task"))
        context_id = str(message.get("contextId") or _new_id("ctx"))
        invocation = self._invocation(
            action="send",
            payload=payload,
            task_id=task_id,
            context_id=context_id,
        )
        running: dict[str, Any] = {
            "id": task_id,
            "contextId": context_id,
            "status": _a2a_status("TASK_STATE_WORKING"),
            "history": [_json_value(message)],
            "artifacts": [],
        }
        self._tasks[task_id] = running
        current_task = asyncio.current_task()
        if current_task is not None:
            self._active_tasks[task_id] = current_task
        try:
            result = await self._run(
                prompt=_a2a_prompt(message, limits=self.limits),
                invocation=invocation,
            )
        except Exception as error:
            public_message = await _map_protocol_error(
                error, invocation, self.error_mapper
            )
            failed: dict[str, Any] = {
                **running,
                "status": _a2a_status("TASK_STATE_FAILED", message=public_message),
            }
            self._tasks[task_id] = failed
            return failed
        finally:
            self._active_tasks.pop(task_id, None)
        completed: dict[str, Any] = {
            **running,
            "status": _a2a_status("TASK_STATE_COMPLETED"),
            "artifacts": [
                {
                    "artifactId": _new_id("artifact"),
                    "name": "agent-output",
                    "parts": [{"text": result.text}],
                }
            ],
        }
        self._tasks[task_id] = completed
        return completed

    async def stream_message(
        self, payload: Mapping[str, Any]
    ) -> AsyncIterable[dict[str, Any]]:
        message = _a2a_message(payload, limits=self.limits)
        task_id = str(message.get("taskId") or _new_id("task"))
        context_id = str(message.get("contextId") or _new_id("ctx"))
        invocation = self._invocation(
            action="stream",
            payload=payload,
            task_id=task_id,
            context_id=context_id,
        )
        task: dict[str, Any] = {
            "id": task_id,
            "contextId": context_id,
            "status": _a2a_status("TASK_STATE_SUBMITTED"),
            "history": [_json_value(message)],
            "artifacts": [],
        }
        self._tasks[task_id] = task
        yield {"task": _json_value(task)}
        task["status"] = _a2a_status("TASK_STATE_WORKING")
        yield {
            "statusUpdate": {
                "taskId": task_id,
                "contextId": context_id,
                "status": task["status"],
                "final": False,
            }
        }
        artifact_id = _new_id("artifact")
        options = await _resolve_run_options(
            invocation, resolver=self.run_options_resolver
        )
        await _notify_protocol_event(
            self.on_protocol_event, invocation, status="started"
        )
        stream = stream_agent(
            agent=self.agent,
            session=options.session,
            prompt=_a2a_prompt(message, limits=self.limits),
            deps=options.deps,
            runtime=options.runtime,
            idempotency_key=options.idempotency_key,
        )
        runner = getattr(stream, "_runner", None)
        if isinstance(runner, asyncio.Task):
            self._active_tasks[task_id] = runner
        try:
            async for event in stream.event_stream():
                if isinstance(event, AgentRunStartEvent):
                    self._internal_run_ids[task_id] = event.run_id
                    await _notify_protocol_event(
                        self.on_protocol_event,
                        invocation,
                        status="running",
                        internal_run_id=event.run_id,
                    )
                elif isinstance(event, AgentTextDeltaEvent) and event.text_delta:
                    yield {
                        "artifactUpdate": {
                            "taskId": task_id,
                            "contextId": context_id,
                            "artifact": {
                                "artifactId": artifact_id,
                                "name": "agent-output",
                                "parts": [{"text": event.text_delta}],
                            },
                            "append": True,
                            "lastChunk": False,
                        }
                    }
            result = await stream.collect()
        except Exception as error:
            public_message = await _map_protocol_error(
                error, invocation, self.error_mapper
            )
            task["status"] = _a2a_status("TASK_STATE_FAILED", message=public_message)
            self._tasks[task_id] = task
            await _notify_protocol_event(
                self.on_protocol_event,
                invocation,
                status="failed",
                internal_run_id=self._internal_run_ids.get(task_id),
                error_code=type(error).__name__,
            )
            yield {
                "statusUpdate": {
                    "taskId": task_id,
                    "contextId": context_id,
                    "status": task["status"],
                    "final": True,
                }
            }
            return
        finally:
            self._active_tasks.pop(task_id, None)
        artifact: dict[str, Any] = {
            "artifactId": artifact_id,
            "name": "agent-output",
            "parts": [{"text": result.text}],
        }
        task["artifacts"] = [artifact]
        task["status"] = _a2a_status("TASK_STATE_COMPLETED")
        self._tasks[task_id] = task
        await _notify_protocol_event(
            self.on_protocol_event,
            invocation,
            status="completed",
            internal_run_id=result.run_id,
        )
        yield {
            "artifactUpdate": {
                "taskId": task_id,
                "contextId": context_id,
                "artifact": artifact,
                "append": False,
                "lastChunk": True,
            }
        }
        yield {
            "statusUpdate": {
                "taskId": task_id,
                "contextId": context_id,
                "status": task["status"],
                "final": True,
            }
        }

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(task_id)
        status = task.get("status")
        if isinstance(status, Mapping) and status.get("state") in {
            "TASK_STATE_COMPLETED",
            "TASK_STATE_FAILED",
            "TASK_STATE_CANCELED",
            "TASK_STATE_REJECTED",
        }:
            raise ValidationError(f'A2A task "{task_id}" is already terminal.')
        task["status"] = _a2a_status("TASK_STATE_CANCELED")
        active = self._active_tasks.get(task_id)
        if active is not None and not active.done():
            internal_run_id = self._internal_run_ids.get(task_id)
            if internal_run_id is not None and self.agent.run_store is not None:
                asyncio.create_task(
                    cancel_agent_run(
                        self.agent.run_store,
                        internal_run_id,
                        reason="A2A task cancelled.",
                    )
                )
            active.cancel()
        return task


@dataclass(slots=True)
class AGUIEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.data}


async def stream_agent_ag_ui(
    *,
    agent: Agent,
    prompt: str,
    thread_id: str,
    run_id: str | None = None,
    run_options: HostedAgentRunOptions | None = None,
    run_options_resolver: ProtocolRunOptionsResolver | None = None,
    error_mapper: ProtocolErrorMapper | None = None,
    on_protocol_event: ProtocolEventCallback | None = None,
    limits: ProtocolLimits | None = None,
) -> AsyncIterable[dict[str, Any]]:
    """Translate a Zhivex agent stream into canonical AG-UI lifecycle events."""

    effective_limits = limits or ProtocolLimits()
    _validate_identifier(thread_id, name="AG-UI thread_id", limits=effective_limits)
    if len(prompt) > effective_limits.max_text_chars:
        raise ValidationError(
            "AG-UI prompt exceeded the configured "
            f"{effective_limits.max_text_chars}-character limit."
        )
    external_run_id = run_id or _new_id("run")
    _validate_identifier(external_run_id, name="AG-UI run_id", limits=effective_limits)
    invocation = ProtocolInvocation(
        protocol="ag-ui",
        action="stream",
        agent_alias=agent.name,
        external_ids={"thread_id": thread_id, "run_id": external_run_id},
        payload={"prompt": prompt},
    )
    options = await _resolve_run_options(
        invocation,
        options=run_options,
        resolver=run_options_resolver,
    )
    message_id = _new_id("msg")
    text_started = False
    internal_run_id: str | None = None
    await _notify_protocol_event(on_protocol_event, invocation, status="started")
    yield AGUIEvent(
        "RUN_STARTED",
        {
            "threadId": thread_id,
            "runId": external_run_id,
            "input": {
                "threadId": thread_id,
                "runId": external_run_id,
                "state": {},
                "messages": [
                    {
                        "id": _new_id("msg"),
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "tools": [],
                "context": [],
                "forwardedProps": {},
            },
        },
    ).to_dict()
    stream = stream_agent(
        agent=agent,
        session=options.session,
        prompt=prompt,
        deps=options.deps,
        runtime=options.runtime,
        idempotency_key=options.idempotency_key,
    )
    try:
        async for event in stream.event_stream():
            if isinstance(event, AgentRunStartEvent):
                internal_run_id = event.run_id
                await _notify_protocol_event(
                    on_protocol_event,
                    invocation,
                    status="running",
                    internal_run_id=internal_run_id,
                )
            elif isinstance(event, AgentTextDeltaEvent) and event.text_delta:
                if not text_started:
                    text_started = True
                    yield AGUIEvent(
                        "TEXT_MESSAGE_START",
                        {"messageId": message_id, "role": "assistant"},
                    ).to_dict()
                yield AGUIEvent(
                    "TEXT_MESSAGE_CONTENT",
                    {"messageId": message_id, "delta": event.text_delta},
                ).to_dict()
            elif isinstance(event, AgentToolCallEvent):
                call = event.tool_call
                yield AGUIEvent(
                    "TOOL_CALL_START",
                    {
                        "toolCallId": call.id,
                        "toolCallName": call.name,
                        "parentMessageId": message_id,
                    },
                ).to_dict()
                yield AGUIEvent(
                    "TOOL_CALL_ARGS",
                    {
                        "toolCallId": call.id,
                        "delta": json.dumps(
                            call.input, ensure_ascii=False, separators=(",", ":")
                        ),
                    },
                ).to_dict()
                yield AGUIEvent("TOOL_CALL_END", {"toolCallId": call.id}).to_dict()
            elif isinstance(event, AgentToolResultEvent):
                tool_result = event.tool_result
                content = (
                    tool_result.error.message
                    if tool_result.error is not None
                    else tool_result.output
                )
                yield AGUIEvent(
                    "TOOL_CALL_RESULT",
                    {
                        "messageId": _new_id("msg"),
                        "toolCallId": tool_result.tool_call_id,
                        "content": content
                        if isinstance(content, str)
                        else json.dumps(_json_value(content)),
                        "role": "tool",
                    },
                ).to_dict()
            elif isinstance(event, AgentErrorEvent):
                raise event.error or RuntimeError("Agent stream failed.")
            elif isinstance(event, AgentFinishEvent):
                continue
        run_result = await stream.collect()
    except Exception as error:
        if text_started:
            yield AGUIEvent("TEXT_MESSAGE_END", {"messageId": message_id}).to_dict()
        public_message = await _map_protocol_error(error, invocation, error_mapper)
        await _notify_protocol_event(
            on_protocol_event,
            invocation,
            status="failed",
            internal_run_id=internal_run_id,
            error_code=type(error).__name__,
        )
        yield AGUIEvent(
            "RUN_ERROR",
            {"message": public_message, "code": "agent_execution_error"},
        ).to_dict()
        return
    if text_started:
        yield AGUIEvent("TEXT_MESSAGE_END", {"messageId": message_id}).to_dict()
    yield AGUIEvent(
        "RUN_FINISHED",
        {
            "threadId": thread_id,
            "runId": external_run_id,
            "result": {"text": run_result.text, "agentRunId": run_result.run_id},
        },
    ).to_dict()
    await _notify_protocol_event(
        on_protocol_event,
        invocation,
        status="completed",
        internal_run_id=run_result.run_id,
    )


def to_ag_ui_sse_response(source: AsyncIterable[dict[str, Any]]) -> HTTPResponse:
    """Encode AG-UI events with the official protocol encoder."""

    try:
        from ag_ui.core import (  # type: ignore[import-not-found]
            RunErrorEvent,
            RunFinishedEvent,
            RunStartedEvent,
            TextMessageContentEvent,
            TextMessageEndEvent,
            TextMessageStartEvent,
            ToolCallArgsEvent,
            ToolCallEndEvent,
            ToolCallResultEvent,
            ToolCallStartEvent,
        )
        from ag_ui.encoder import EventEncoder  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            'AG-UI encoding requires `pip install "zhivex-ai-sdk[ag-ui]"`.'
        ) from error

    event_types = {
        "RUN_STARTED": RunStartedEvent,
        "RUN_FINISHED": RunFinishedEvent,
        "RUN_ERROR": RunErrorEvent,
        "TEXT_MESSAGE_START": TextMessageStartEvent,
        "TEXT_MESSAGE_CONTENT": TextMessageContentEvent,
        "TEXT_MESSAGE_END": TextMessageEndEvent,
        "TOOL_CALL_START": ToolCallStartEvent,
        "TOOL_CALL_ARGS": ToolCallArgsEvent,
        "TOOL_CALL_END": ToolCallEndEvent,
        "TOOL_CALL_RESULT": ToolCallResultEvent,
    }
    encoder = EventEncoder()

    async def encoded():
        async for item in source:
            event_type = str(item.get("type", ""))
            model = event_types.get(event_type)
            if model is None:
                raise ValidationError(f'Unsupported AG-UI event type "{event_type}".')
            yield encoder.encode(model.model_validate(item)).encode("utf-8")  # type: ignore[attr-defined]

    return HTTPResponse(
        body=encoded(),
        headers={
            "content-type": encoder.get_content_type(),
            "cache-control": "no-cache, no-transform",
            "connection": "keep-alive",
        },
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _read_json_request(request: Any, *, max_request_bytes: int) -> dict[str, Any]:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > max_request_bytes:
                raise ParseError(
                    "Protocol request body exceeded the configured maximum size."
                )
        except ValueError:
            raise ParseError(
                "Protocol request contained an invalid Content-Length header."
            ) from None
    body = await request.body()
    if len(body) > max_request_bytes:
        raise ParseError("Protocol request body exceeded the configured maximum size.")
    payload = json.loads(body or b"{}")
    if not isinstance(payload, dict):
        raise ParseError("Protocol request body must be a JSON object.")
    return payload


def create_a2a_app(
    *,
    executor: A2AAgentExecutor,
    card: A2AAgentCard,
    authorize: Callable[[Any], bool | Awaitable[bool]] | None = None,
    max_request_bytes: int | None = None,
    limits: ProtocolLimits | None = None,
    task_store: Any = None,
    request_context_builder: Any = None,
    queue_manager: Any = None,
):
    """Create an A2A v1 server using the official Python SDK.

    The app exposes both HTTP+JSON under ``/a2a`` and JSON-RPC at
    ``/a2a/rpc``. The official task store and request handler own the wire
    protocol, task lifecycle, version headers, and streaming envelopes.
    """

    try:
        from a2a.helpers import new_task_from_user_message  # type: ignore[import-not-found]
        from a2a.server.agent_execution import AgentExecutor as OfficialAgentExecutor  # type: ignore[import-not-found]
        from a2a.server.request_handlers import DefaultRequestHandler  # type: ignore[import-not-found]
        from a2a.server.routes import (  # type: ignore[import-not-found]
            add_a2a_routes_to_fastapi,
            create_agent_card_routes,
            create_jsonrpc_routes,
            create_rest_routes,
        )
        from a2a.server.tasks import InMemoryTaskStore, TaskUpdater  # type: ignore[import-not-found]
        from a2a.types import AgentCard as OfficialAgentCard  # type: ignore[import-not-found]
        from a2a.types import Part  # type: ignore[import-not-found]
        from fastapi import FastAPI  # type: ignore[import-not-found]
        from fastapi.responses import JSONResponse  # type: ignore[import-not-found]
        from google.protobuf.json_format import MessageToDict, ParseDict  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError(
            'A2A hosting requires `pip install "zhivex-ai-sdk[a2a]"`.'
        ) from error

    official_card_payload = card.to_dict()
    rest_url = card.url.rstrip("/")
    official_card_payload["supportedInterfaces"] = [
        {
            "url": f"{rest_url}/rpc",
            "protocolBinding": "JSONRPC",
            "protocolVersion": card.protocol_version,
        },
        {
            "url": rest_url,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": card.protocol_version,
        },
    ]
    official_card = ParseDict(official_card_payload, OfficialAgentCard())

    class ZhivexA2AExecutor(OfficialAgentExecutor):
        async def execute(self, context: Any, event_queue: Any) -> None:
            if context.message is None:
                raise ValidationError("A2A execution requires a user message.")
            task = context.current_task
            if task is None:
                task = new_task_from_user_message(context.message)
                await event_queue.enqueue_event(task)
            updater = TaskUpdater(event_queue, task.id, task.context_id)
            await updater.start_work()
            message = MessageToDict(context.message)
            invocation = executor._invocation(
                action="execute",
                payload={"message": message},
                task_id=task.id,
                context_id=task.context_id,
                request=context,
            )
            current_task = asyncio.current_task()
            if current_task is not None:
                executor._active_tasks[task.id] = current_task
            try:
                result = await executor._run(
                    prompt=_a2a_prompt(message, limits=executor.limits),
                    invocation=invocation,
                )
            except Exception as error:
                public_message = await _map_protocol_error(
                    error, invocation, executor.error_mapper
                )
                raise RuntimeError(public_message) from None
            finally:
                executor._active_tasks.pop(task.id, None)
            await updater.add_artifact(
                [Part(text=result.text)],
                artifact_id=_new_id("artifact"),
                last_chunk=True,
            )
            await updater.complete()

        async def cancel(self, context: Any, event_queue: Any) -> None:
            if context.task_id is None or context.context_id is None:
                raise ValidationError(
                    "A2A cancellation requires task and context identifiers."
                )
            invocation = executor._invocation(
                action="cancel",
                payload={},
                task_id=context.task_id,
                context_id=context.context_id,
                request=context,
            )
            await executor.cancel_active(context.task_id)
            await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()
            await _notify_protocol_event(
                executor.on_protocol_event,
                invocation,
                status="cancelled",
                internal_run_id=executor._internal_run_ids.get(context.task_id),
            )

    effective_limits = limits or executor.limits
    if max_request_bytes is not None:
        effective_limits = replace(
            effective_limits, max_request_bytes=max_request_bytes
        )
    executor.limits = effective_limits

    handler_options: dict[str, Any] = {
        "agent_executor": ZhivexA2AExecutor(),
        "task_store": task_store if task_store is not None else InMemoryTaskStore(),
        "agent_card": official_card,
    }
    if request_context_builder is not None:
        handler_options["request_context_builder"] = request_context_builder
    if queue_manager is not None:
        handler_options["queue_manager"] = queue_manager
    handler = DefaultRequestHandler(**handler_options)
    app = FastAPI(title=f"{card.name} A2A", version=card.version)
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(official_card),
        jsonrpc_routes=create_jsonrpc_routes(
            handler,
            rpc_url="/a2a/rpc",
            enable_v0_3_compat=False,
        ),
        rest_routes=create_rest_routes(
            handler,
            path_prefix="/a2a",
            enable_v0_3_compat=False,
        ),
    )

    @app.middleware("http")
    async def protocol_guard(
        request: Any, call_next: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        public_agent_card = (
            request.method == "GET"
            and request.url.path == "/.well-known/agent-card.json"
        )
        if authorize is not None and not public_agent_card:
            if not await _maybe_await(authorize(request)):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        if request.method not in {"GET", "HEAD"}:
            declared = request.headers.get("content-length")
            if declared is not None:
                try:
                    if int(declared) > effective_limits.max_request_bytes:
                        return JSONResponse(
                            {"error": "request_too_large"}, status_code=413
                        )
                except ValueError:
                    return JSONResponse(
                        {"error": "invalid_content_length"}, status_code=400
                    )
            body = await request.body()
            if len(body) > effective_limits.max_request_bytes:
                return JSONResponse({"error": "request_too_large"}, status_code=413)
        return await call_next(request)

    async def close_handler() -> None:
        close = getattr(handler, "aclose", None)
        if callable(close):
            await close()

    app.router.add_event_handler("shutdown", close_handler)

    return app
