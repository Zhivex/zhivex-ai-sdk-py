"""Provider-owned session APIs. Payloads remain native; no portable guarantees."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterable
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

if TYPE_CHECKING:
    from ..agent import Agent, AgentRunResult


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
    """Single-reader GPT-Live WebSocket connection with bounded finalization.

    Audio playback and backend task ownership remain with the application.
    """

    def __init__(self, connection: RealtimeConnection, started: dict[str, Any]) -> None:
        self.connection = connection
        self.started = started
        self.final_event: dict[str, Any] | None = None
        self.closed = False
        self.closing = False
        self._delegations: set[str] = set()
        self._sending = asyncio.Lock()
        self._receiving = asyncio.Lock()

    async def append_audio(self, audio: bytes) -> None:
        """Append nonempty mono PCM16 audio in the session-configured sample rate."""
        if not isinstance(audio, bytes) or not audio or len(audio) % 2:
            raise ValidationError("Live audio must contain complete PCM16 samples.")
        await self.send({"type": "session.input_audio.append", "audio": base64.b64encode(audio).decode("ascii")})

    async def append_context(
        self, content: str, *, kind: str = "commentary",
        delegation_id: str | None = None, event_id: str,
    ) -> None:
        """Send a bounded update; receive the matching acknowledgment separately.

        The 500-byte UTF-8 ceiling is deliberately stricter than the provider
        500-token limit. An acknowledgment does not establish audio playback.
        """
        if kind not in {"commentary", "thinking", "instructions"}:
            raise ValidationError("Unknown Live context update kind.")
        if not isinstance(content, str) or not content.strip() or len(content.encode("utf-8")) > 500:
            raise ValidationError("Live context must contain 1 to 500 UTF-8 bytes.")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValidationError("Live context requires an event ID.")
        if delegation_id is not None and (not isinstance(delegation_id, str) or not delegation_id.strip()):
            raise ValidationError("Live delegation ID must be nonempty.")
        await self.send({"type": f"session.{kind}.append", "content": content,
                         "delegation_id": delegation_id, "event_id": event_id})

    async def run_delegation(
        self, event: dict[str, Any], *, agent: Agent[Any, Any], prompt: str,
        **run_options: Any,
    ) -> AgentRunResult[Any]:
        """Run one client delegation with explicit application-owned context.

        Requires a durable Agent store. Duplicate/failed/cancelled attempts are
        not replayed in this voice session. Suspended approvals are returned to
        the application without announcing success. Resume them using the normal
        agent APIs and explicitly publish their verified result with append_context.
        The caller owns this coroutine; keep receiving voice events concurrently.
        """
        from ..agent import run_agent

        delegation = event.get("delegation")
        session_id = self.started.get("session", {}).get("id")
        if (event.get("type") != "session.delegation.created"
                or not isinstance(delegation, dict) or delegation.get("target") != "client"
                or not isinstance(delegation.get("id"), str) or not delegation["id"].strip()
                or not isinstance(session_id, str) or not session_id):
            raise ValidationError("Expected a client delegation from an identified Live session.")
        delegation_id = delegation["id"]
        if self.closed or self.closing:
            raise ValidationError("Live session is closing or closed.")
        if agent.run_store is None or not isinstance(prompt, str) or not prompt.strip():
            raise ValidationError("Live delegation requires an Agent run store and explicit context.")
        if "idempotency_key" in run_options:
            raise ValidationError("Live delegation owns its idempotency key.")
        if delegation_id in self._delegations:
            raise ValidationError("Live delegation was already claimed; inspect its durable state.")
        if len(self._delegations) >= 256:
            raise ValidationError("Live session delegation limit reached.")
        self._delegations.add(delegation_id)
        result = await run_agent(
            agent=agent, prompt=prompt,
            idempotency_key=f"gpt-live:{session_id}:{delegation_id}", **run_options,
        )
        if result.state is not None and result.state.status == "completed":
            await self.append_context(result.text, delegation_id=delegation_id,
                                      event_id=f"result-{delegation_id}")
        return result

    async def send(self, event: dict[str, Any]) -> None:
        async with self._sending:
            if self.closed or self.closing:
                raise ValidationError("Live session is closing or closed.")
            if event.get("type") == "session.start":
                raise ValidationError("Live session already started.")
            async with asyncio.timeout(5):
                await self.connection.send_json(deepcopy(event))
            if event.get("type") == "session.close":
                self.closing = True

    async def receive(self) -> dict[str, Any]:
        if self.closed or self.final_event is not None:
            raise ValidationError("Live session is closed or finalized.")
        if self._receiving.locked():
            raise ValidationError("Live sessions permit only one event reader.")
        async with self._receiving:
            payload = await self.connection.recv_json()
            if not isinstance(payload, dict):
                raise ParseError("Live connection ended without a session.closed event.")
            if payload.get("type") == "session.closed":
                self.final_event = payload
                self.closing = True
            return payload

    async def finish(self, *, timeout_ms: int = 10_000) -> dict[str, Any]:
        """Drain final events with no competing reader; always release the socket."""
        if timeout_ms <= 0:
            raise ValidationError("Live finalization timeout must be positive.")
        if self._receiving.locked():
            raise ValidationError("Stop the event reader before finalizing Live.")
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
            async with asyncio.timeout(5):
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
            try:
                async with asyncio.timeout(5):
                    await connection.close()
            except Exception:
                pass  # Cleanup must not replace the startup failure.
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
