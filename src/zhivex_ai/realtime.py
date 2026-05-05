from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

from .errors import UnsupportedFeatureError
from .types import (
    AudioFrame,
    ModelCapabilities,
    RealtimeConnectOptions,
    RealtimeEvent,
    RealtimeSession,
    RealtimeSessionEndedEvent,
    RealtimeSessionStartedEvent,
    RealtimeSessionConfig,
    RealtimeToolResultEvent,
    ToolExecutionResult,
)


class RealtimeConnection(Protocol):
    async def send_json(self, payload: dict[str, Any]) -> None: ...

    async def recv_json(self) -> Any: ...

    async def close(self) -> None: ...


RealtimeEventParser = Callable[[dict[str, Any]], list[RealtimeEvent]]
RealtimePayloadBuilder = Callable[..., list[dict[str, Any]]]
RealtimeConnectionFactory = Callable[[str, dict[str, str], RealtimeConnectOptions | None], Awaitable[RealtimeConnection]]


@dataclass(slots=True)
class RealtimeSessionCallbacks:
    parse_event: RealtimeEventParser
    build_audio_payloads: RealtimePayloadBuilder
    build_text_payloads: RealtimePayloadBuilder
    build_tool_result_payloads: RealtimePayloadBuilder
    build_update_payloads: RealtimePayloadBuilder
    build_initial_payloads: RealtimePayloadBuilder | None = None
    build_close_payloads: RealtimePayloadBuilder | None = None


@dataclass
class _Broadcast:
    history: list[RealtimeEvent]
    done: bool = False
    subscribers: list[asyncio.Queue[RealtimeEvent | None]] | None = None

    def __post_init__(self) -> None:
        self.subscribers = []

    async def publish(self, event: RealtimeEvent) -> None:
        self.history.append(event)
        for queue in list(self.subscribers or []):
            await queue.put(event)

    async def close(self) -> None:
        self.done = True
        for queue in list(self.subscribers or []):
            await queue.put(None)

    def stream(self) -> AsyncIterable[RealtimeEvent]:
        async def generator() -> AsyncIterable[RealtimeEvent]:
            queue: asyncio.Queue[RealtimeEvent | None] = asyncio.Queue()
            cursor = 0
            self.subscribers = self.subscribers or []
            self.subscribers.append(queue)
            try:
                while True:
                    while cursor < len(self.history):
                        event = self.history[cursor]
                        cursor += 1
                        yield event
                    if self.done:
                        return
                    item = await queue.get()
                    if item is None:
                        return
            finally:
                if self.subscribers and queue in self.subscribers:
                    self.subscribers.remove(queue)

        return generator()


class CallbackRealtimeSession(RealtimeSession):
    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        capabilities: ModelCapabilities,
        config: RealtimeSessionConfig,
        connection: RealtimeConnection,
        callbacks: RealtimeSessionCallbacks,
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.capabilities = capabilities
        self.config = config
        self._connection = connection
        self._callbacks = callbacks
        self._broadcast = _Broadcast(history=[])
        self._receiver_task: asyncio.Task[None] | None = None
        self._closed = False
        self._ended = False

    async def initialize(self) -> None:
        if self._receiver_task is None:
            self._receiver_task = asyncio.create_task(self._receive_loop())
        if self._callbacks.build_initial_payloads is not None:
            await self._send_payloads(self._callbacks.build_initial_payloads(self.config))
        await self._broadcast.publish(RealtimeSessionStartedEvent())

    async def send_audio(self, frame: AudioFrame) -> None:
        await self._send_payloads(self._callbacks.build_audio_payloads(frame, self.config))

    async def send_text(self, text: str) -> None:
        await self._send_payloads(self._callbacks.build_text_payloads(text, self.config))

    async def send_tool_result(self, result: ToolExecutionResult) -> None:
        await self._send_payloads(self._callbacks.build_tool_result_payloads(result, self.config))
        await self._broadcast.publish(RealtimeToolResultEvent(tool_result=result))

    async def update(
        self,
        *,
        instructions: str | None = None,
        voice: str | None = None,
        tools: dict[str, Any] | None = None,
        tool_choice: Any = None,
        turn_detection: dict[str, Any] | None = None,
        provider_options: dict[str, Any] | None = None,
    ) -> None:
        self.config = replace(
            self.config,
            instructions=instructions if instructions is not None else self.config.instructions,
            voice=voice if voice is not None else self.config.voice,
            tools=tools if tools is not None else self.config.tools,
            tool_choice=tool_choice if tool_choice is not None else self.config.tool_choice,
            turn_detection=turn_detection if turn_detection is not None else self.config.turn_detection,
            provider_options=provider_options if provider_options is not None else self.config.provider_options,
        )
        await self._send_payloads(self._callbacks.build_update_payloads(self.config))

    def event_stream(self) -> AsyncIterable[RealtimeEvent]:
        return self._broadcast.stream()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._callbacks.build_close_payloads is not None:
                await self._send_payloads(self._callbacks.build_close_payloads(self.config))
        finally:
            await self._connection.close()
            if self._receiver_task is not None:
                try:
                    await self._receiver_task
                except Exception:
                    pass
            if not self._ended:
                self._ended = True
                await self._broadcast.publish(RealtimeSessionEndedEvent(reason="client-close"))
            await self._broadcast.close()

    async def _send_payloads(self, payloads: list[dict[str, Any]]) -> None:
        for payload in payloads:
            await self._connection.send_json(payload)

    async def _receive_loop(self) -> None:
        try:
            while True:
                payload = await self._connection.recv_json()
                if payload is None:
                    break
                for event in self._callbacks.parse_event(dict(payload or {})):
                    if isinstance(event, RealtimeSessionEndedEvent):
                        self._ended = True
                    await self._broadcast.publish(event)
                    if isinstance(event, RealtimeSessionEndedEvent):
                        await self._broadcast.close()
                        return
        except Exception as error:
            if not self._ended:
                self._ended = True
                await self._broadcast.publish(
                    RealtimeSessionEndedEvent(
                        reason="error",
                        provider_metadata={"message": str(error)},
                    )
                )
        finally:
            await self._broadcast.close()


class WebSocketRealtimeConnection:
    def __init__(self, websocket: Any) -> None:
        self._websocket = websocket

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self._websocket.send(json.dumps(payload))

    async def recv_json(self) -> Any:
        message = await self._websocket.recv()
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        return json.loads(message)

    async def close(self) -> None:
        await self._websocket.close()


async def open_websocket_connection(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    options: RealtimeConnectOptions | None = None,
) -> RealtimeConnection:
    try:
        import websockets
    except Exception as error:
        raise RuntimeError(
            'Realtime support requires the "websockets" package. '
            'If you are running from this repo, install dependencies first with `make dev` '
            "or `pip install -e .`."
        ) from error

    websocket = await websockets.connect(
        url,
        additional_headers=headers or None,
        subprotocols=cast(Any, list(options.subprotocols)) if options and options.subprotocols else None,
        open_timeout=(options.timeout_ms / 1000) if options and options.timeout_ms is not None else None,
    )
    return WebSocketRealtimeConnection(websocket)


def encode_audio_frame(frame: AudioFrame) -> str:
    data = frame.data
    if isinstance(data, str):
        return data
    if isinstance(data, memoryview):
        raw = data.tobytes()
    else:
        raw = bytes(data)
    return base64.b64encode(raw).decode("ascii")


def tool_result_payload(result: ToolExecutionResult) -> Any:
    if result.is_error and result.error is not None:
        return {"message": result.error.message}
    return result.output


async def unsupported_browser_token(*_: Any, **__: Any) -> Any:
    raise UnsupportedFeatureError("This realtime model does not support browser session tokens.")
