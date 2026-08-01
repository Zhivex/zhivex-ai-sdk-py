from __future__ import annotations

import inspect
import json
import time
import uuid
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .agent import (
    Agent,
    AgentErrorEvent,
    AgentFinishEvent,
    AgentTextDeltaEvent,
    AgentToolCallEvent,
    AgentToolResultEvent,
    run_agent,
    stream_agent,
)
from .errors import ParseError, ValidationError
from .transport import HTTPResponse
from .types import JsonValue

A2A_PROTOCOL_VERSION = "1.0"
A2A_HTTP_JSON_BINDING = "HTTP+JSON"
DEFAULT_PROTOCOL_REQUEST_BYTES = 1 * 1024 * 1024


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
    resolved_description = description or f"{agent.name} agent exposed by Zhivex AI SDK."
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


def _a2a_message(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    message = payload.get("message")
    if not isinstance(message, Mapping):
        raise ValidationError('A2A SendMessageRequest requires a "message" object.')
    if message.get("role") not in {"user", "ROLE_USER"}:
        raise ValidationError('A2A input messages must use the A2A user role.')
    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValidationError("A2A input messages require at least one part.")
    return message


def _a2a_prompt(message: Mapping[str, Any]) -> str:
    values: list[str] = []
    for part in message.get("parts", []):
        if not isinstance(part, Mapping):
            raise ValidationError("A2A message parts must be objects.")
        members = [name for name in ("text", "raw", "url", "data") if part.get(name) is not None]
        if len(members) != 1:
            raise ValidationError("Each A2A v1 part must set exactly one of text, raw, url, or data.")
        member = members[0]
        value = part[member]
        if member == "text":
            values.append(str(value))
        elif member == "data":
            values.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        else:
            values.append(str(value))
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

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self._tasks: dict[str, dict[str, Any]] = {}

    async def send_message(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        message = _a2a_message(payload)
        task_id = str(message.get("taskId") or _new_id("task"))
        context_id = str(message.get("contextId") or _new_id("ctx"))
        running: dict[str, Any] = {
            "id": task_id,
            "contextId": context_id,
            "status": _a2a_status("TASK_STATE_WORKING"),
            "history": [_json_value(message)],
            "artifacts": [],
        }
        self._tasks[task_id] = running
        try:
            result = await run_agent(agent=self.agent, prompt=_a2a_prompt(message))
        except Exception as error:
            failed: dict[str, Any] = {
                **running,
                "status": _a2a_status("TASK_STATE_FAILED", message=str(error)),
            }
            self._tasks[task_id] = failed
            return failed
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

    async def stream_message(self, payload: Mapping[str, Any]) -> AsyncIterable[dict[str, Any]]:
        message = _a2a_message(payload)
        task_id = str(message.get("taskId") or _new_id("task"))
        context_id = str(message.get("contextId") or _new_id("ctx"))
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
        stream = stream_agent(agent=self.agent, prompt=_a2a_prompt(message))
        try:
            async for event in stream.event_stream():
                if isinstance(event, AgentTextDeltaEvent) and event.text_delta:
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
            task["status"] = _a2a_status("TASK_STATE_FAILED", message=str(error))
            self._tasks[task_id] = task
            yield {
                "statusUpdate": {
                    "taskId": task_id,
                    "contextId": context_id,
                    "status": task["status"],
                    "final": True,
                }
            }
            return
        artifact: dict[str, Any] = {
            "artifactId": artifact_id,
            "name": "agent-output",
            "parts": [{"text": result.text}],
        }
        task["artifacts"] = [artifact]
        task["status"] = _a2a_status("TASK_STATE_COMPLETED")
        self._tasks[task_id] = task
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
) -> AsyncIterable[dict[str, Any]]:
    """Translate a Zhivex agent stream into canonical AG-UI lifecycle events."""

    external_run_id = run_id or _new_id("run")
    message_id = _new_id("msg")
    text_started = False
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
    stream = stream_agent(agent=agent, prompt=prompt)
    try:
        async for event in stream.event_stream():
            if isinstance(event, AgentTextDeltaEvent) and event.text_delta:
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
                        "delta": json.dumps(call.input, ensure_ascii=False, separators=(",", ":")),
                    },
                ).to_dict()
                yield AGUIEvent("TOOL_CALL_END", {"toolCallId": call.id}).to_dict()
            elif isinstance(event, AgentToolResultEvent):
                tool_result = event.tool_result
                content = tool_result.error.message if tool_result.error is not None else tool_result.output
                yield AGUIEvent(
                    "TOOL_CALL_RESULT",
                    {
                        "messageId": _new_id("msg"),
                        "toolCallId": tool_result.tool_call_id,
                        "content": content if isinstance(content, str) else json.dumps(_json_value(content)),
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
        yield AGUIEvent(
            "RUN_ERROR",
            {"message": str(error), "code": type(error).__name__},
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
        raise RuntimeError('AG-UI encoding requires `pip install "zhivex-ai-sdk[ag-ui]"`.') from error

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
            yield encoder.encode(model.model_validate(item)).encode("utf-8")

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
                raise ParseError("Protocol request body exceeded the configured maximum size.")
        except ValueError:
            raise ParseError("Protocol request contained an invalid Content-Length header.") from None
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
    max_request_bytes: int = DEFAULT_PROTOCOL_REQUEST_BYTES,
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
        raise RuntimeError('A2A hosting requires `pip install "zhivex-ai-sdk[a2a]"`.') from error

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
            result = await run_agent(agent=executor.agent, prompt=_a2a_prompt(message))
            await updater.add_artifact(
                [Part(text=result.text)],
                artifact_id=_new_id("artifact"),
                last_chunk=True,
            )
            await updater.complete()

        async def cancel(self, context: Any, event_queue: Any) -> None:
            if context.task_id is None or context.context_id is None:
                raise ValidationError("A2A cancellation requires task and context identifiers.")
            await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()

    handler = DefaultRequestHandler(
        agent_executor=ZhivexA2AExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=official_card,
    )
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
    async def protocol_guard(request: Any, call_next: Callable[[Any], Awaitable[Any]]) -> Any:
        if request.method != "GET":
            declared = request.headers.get("content-length")
            if declared is not None:
                try:
                    if int(declared) > max_request_bytes:
                        return JSONResponse({"error": "request_too_large"}, status_code=413)
                except ValueError:
                    return JSONResponse({"error": "invalid_content_length"}, status_code=400)
            if authorize is not None and not await _maybe_await(authorize(request)):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    return app
