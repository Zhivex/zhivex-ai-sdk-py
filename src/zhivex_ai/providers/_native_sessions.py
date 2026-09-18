"""Provider-owned session APIs. Payloads remain native; no portable guarantees."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, AsyncIterable
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from .._http import Fetcher, ResponseLike
from .._sse import parse_sse
from ..errors import ParseError, ProviderHTTPError, ValidationError
from ..realtime import (
    RealtimeConnection,
    RealtimeConnectionFactory,
    open_websocket_connection,
)
from ..types import RealtimeConnectOptions, RetryOptions
import json


def _id(value: str) -> str:
    if not value or value in {".", ".."}:
        raise ValidationError("A non-empty resource ID is required.")
    return quote(value, safe="")


@dataclass(slots=True)
class _NativeClient:
    provider: str
    base_url: str
    headers: dict[str, str] = field(repr=False)
    fetch: Fetcher

    async def _request(
        self,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        method: str = "POST",
        options: RetryOptions | None = None,
        stream: bool = False,
    ) -> ResponseLike:
        # Do not replay session creation or input submission automatically.
        response = await self.fetch(
            self.base_url.rstrip("/") + path,
            method=method,
            headers={
                **self.headers,
                **({"Accept": "text/event-stream"} if stream else {}),
            },
            json_body=deepcopy(body),
            timeout_ms=options.timeout_ms if options else None,
            stream=stream,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"{self.provider} native request failed ({response.status_code}).",
                response.status_code,
                response_body=await response.text(),
            )
        return response

    async def _json(
        self,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        method: str = "POST",
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await self._request(path, body, method=method, options=options)
        if response.status_code == 204:
            return {}
        payload = await response.json()
        if not isinstance(payload, dict):
            raise ParseError("Expected a native JSON object response.")
        return payload


class AnthropicMessagesClient(_NativeClient):
    """Beta Messages API, preserving signed compaction blocks without rewriting."""

    async def create(
        self, body: dict[str, Any], options: RetryOptions | None = None
    ) -> dict[str, Any]:
        if body.get("stream"):
            raise ValidationError(
                "Use the native language model for streaming Messages."
            )
        if body.get("compaction") is not None:
            compaction = body["compaction"]
            if (
                not isinstance(compaction, dict)
                or compaction.get("type") != "summarize"
            ):
                raise ValidationError("Compaction must have type=summarize.")
            if (
                body.get("context_management") is not None
                or body.get("stop_sequences") is not None
            ):
                raise ValidationError(
                    "On-demand compaction cannot use context_management or stop_sequences."
                )
            output = body.get("output_config") or {}
            choice = body.get("tool_choice") or {}
            if not isinstance(output, dict) or not isinstance(choice, dict):
                raise ValidationError("output_config and tool_choice must be objects.")
            if output.get("format") is not None or choice.get("type") in {
                "any",
                "tool",
            }:
                raise ValidationError(
                    "On-demand compaction cannot force tools or structured output."
                )
        return await self._json("/messages", body, options=options)

    async def compact(
        self, body: dict[str, Any], options: RetryOptions | None = None
    ) -> dict[str, Any]:
        payload = deepcopy(body)
        if payload.get("compaction") is None:
            payload["compaction"] = {"type": "summarize"}
        # Return even null/absent summaries verbatim. The application must retain
        # its original history until it has a usable signed block.
        return await self.create(payload, options)


class OpenAIAgentSessionsClient(_NativeClient):
    """Beta hosted Agents sessions; independent of the application-owned Agent."""

    async def create(
        self, body: dict[str, Any], options: RetryOptions | None = None
    ) -> dict[str, Any]:
        if body.get("stream"):
            raise ValidationError(
                "Create without stream, then subscribe using events()."
            )
        return await self._json("/agents/sessions", body, options=options)

    async def retrieve(
        self, session_id: str, options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await self._json(
            f"/agents/sessions/{_id(session_id)}", method="GET", options=options
        )

    async def list(
        self,
        *,
        limit: int = 20,
        after: str | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        if isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValidationError("limit must be between 1 and 100.")
        query: dict[str, Any] = {"limit": limit}
        if after is not None:
            query["after"] = after
        return await self._json(
            "/agents/sessions?" + urlencode(query), method="GET", options=options
        )

    async def send_events(
        self,
        session_id: str,
        events: Sequence[dict[str, Any]],
        options: RetryOptions | None = None,
    ) -> None:
        response = await self._request(
            f"/agents/sessions/{_id(session_id)}/events",
            {"events": list(events)},
            options=options,
        )
        await response.read()

    async def cancel(
        self, session_id: str, options: RetryOptions | None = None
    ) -> None:
        await self.send_events(
            session_id, [{"type": "agent.session.input.cancel"}], options
        )

    async def delete(
        self, session_id: str, options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await self._json(
            f"/agents/sessions/{_id(session_id)}", method="DELETE", options=options
        )

    async def items(
        self,
        session_id: str,
        *,
        after: str | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        query = urlencode({"after": after}) if after is not None else ""
        return await self._json(
            f"/agents/sessions/{_id(session_id)}/items"
            + ("?" + query if query else ""),
            method="GET",
            options=options,
        )

    async def events(
        self, session_id: str, options: RetryOptions | None = None
    ) -> AsyncIterable[dict[str, Any]]:
        response = await self._request(
            f"/agents/sessions/{_id(session_id)}/events",
            method="GET",
            options=options,
            stream=True,
        )
        lines = response.iter_lines()
        try:
            async for event in parse_sse(lines):
                if event.data == "[DONE]":
                    break
                payload = json.loads(event.data)
                if not isinstance(payload, dict):
                    raise ParseError("Expected an Agents event object.")
                yield payload
        finally:
            close = getattr(lines, "aclose", None)
            if close is not None:
                await close()
            close_response = getattr(response, "aclose", None) or getattr(
                response, "_close", None
            )
            if close_response is not None:
                await close_response()


class OpenAILiveSession:
    """Experimental single-reader native event connection, with explicit finalization."""

    def __init__(self, connection: RealtimeConnection, started: dict[str, Any]) -> None:
        self.connection = connection
        self.started = started
        self.final_event: dict[str, Any] | None = None
        self.closed = False
        self.closing = False

    async def send(self, event: dict[str, Any]) -> None:
        if self.closed or self.closing:
            raise ValidationError("Live session is closing or closed.")
        if event.get("type") == "session.start":
            raise ValidationError("Live session already started.")
        await self.connection.send_json(deepcopy(event))
        if event.get("type") == "session.close":
            self.closing = True

    async def receive(self) -> dict[str, Any]:
        payload = await self.connection.recv_json()
        if not isinstance(payload, dict):
            raise ParseError("Live connection ended without a session.closed event.")
        if payload.get("type") == "session.closed":
            self.final_event = payload
            self.closing = True
        return payload

    async def finish(self, *, timeout_ms: int = 10_000) -> dict[str, Any]:
        """Drain final events with no competing reader; always release the socket."""
        try:
            async with asyncio.timeout(timeout_ms / 1000):
                if self.final_event is None and not self.closing:
                    await self.send({"type": "session.close"})
                while self.final_event is None:
                    await self.receive()
            return self.final_event
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        """Release transport; this alone does not establish provider finalization."""
        if not self.closed:
            self.closed = True
            await self.connection.close()

    async def __aenter__(self) -> OpenAILiveSession:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


@dataclass(slots=True)
class OpenAILiveClient(_NativeClient):
    connection_factory: RealtimeConnectionFactory | None = None

    async def create(
        self,
        *,
        session: dict[str, Any],
        transport: dict[str, Any],
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        """Create a WebRTC session on a trusted backend; return the SDP answer."""
        if (
            not session.get("model")
            or transport.get("type") != "webrtc"
            or not transport.get("sdp")
        ):
            raise ValidationError(
                "WebRTC creation requires a model and a transport SDP offer."
            )
        return await self._json(
            "/live/sessions",
            {"session": session, "transport": transport},
            options=options,
        )

    async def connect(
        self, session: dict[str, Any], options: RealtimeConnectOptions | None = None
    ) -> OpenAILiveSession:
        if not session.get("model"):
            raise ValidationError("GPT-Live requires a model.")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or parsed.query or parsed.fragment:
            raise ValidationError(
                "GPT-Live base_url must be an HTTP(S) URL without query or fragment."
            )
        url = urlunsplit(
            (
                "wss" if parsed.scheme == "https" else "ws",
                parsed.netloc,
                parsed.path.rstrip("/") + "/live/sessions",
                "",
                "",
            )
        )
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() != "content-type"
        }
        factory = self.connection_factory
        connection = (
            await factory(url, headers, options)
            if factory
            else await open_websocket_connection(url, headers=headers, options=options)
        )
        try:
            timeout = (
                options.timeout_ms
                if options and options.timeout_ms is not None
                else 30_000
            )
            async with asyncio.timeout(timeout / 1000):
                await connection.send_json(
                    {"type": "session.start", "session": deepcopy(session)}
                )
                started = await connection.recv_json()
                if (
                    not isinstance(started, dict)
                    or started.get("type") != "session.started"
                ):
                    raise ParseError("GPT-Live did not acknowledge session startup.")
            return OpenAILiveSession(connection, started)
        except BaseException:
            await connection.close()
            raise

    async def hangup(
        self, session_id: str, options: RetryOptions | None = None
    ) -> None:
        response = await self._request(
            f"/live/sessions/{_id(session_id)}/hangup", options=options
        )
        await response.read()

    async def fork(
        self, session_id: str, body: dict[str, Any], options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await self._json(
            f"/live/sessions/{_id(session_id)}/fork", body, options=options
        )

    async def download_recording(
        self, session_id: str, options: RetryOptions | None = None
    ) -> bytes:
        response = await self._request(
            f"/live/sessions/{_id(session_id)}/content", method="GET", options=options
        )
        return await response.read()
