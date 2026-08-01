from __future__ import annotations

import inspect
import json
import time
import uuid
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from typing import Any

from .agent import Agent, AgentTextDeltaEvent, run_agent, stream_agent
from .errors import ParseError, ValidationError
from .messages import create_text_message
from .types import JsonValue, ModelMessage, TokenUsage

DEFAULT_RESPONSES_REQUEST_BYTES = 1 * 1024 * 1024
AgentResolver = Mapping[str, Agent] | Callable[[str], Agent | Awaitable[Agent | None] | None]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _usage(usage: TokenUsage | None) -> dict[str, JsonValue]:
    input_tokens = usage.input_tokens if usage and usage.input_tokens is not None else 0
    output_tokens = usage.output_tokens if usage and usage.output_tokens is not None else 0
    total_tokens = usage.total_tokens if usage and usage.total_tokens is not None else input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": total_tokens,
    }


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                part_type = part.get("type")
                if part_type not in {"input_text", "text"}:
                    raise ValidationError(f'Unsupported Responses input content type "{part_type}".')
                text = part.get("text")
                if not isinstance(text, str):
                    raise ValidationError("Responses text content requires a string text field.")
                parts.append(text)
            else:
                raise ValidationError("Responses input content items must be strings or objects.")
        return "\n".join(parts)
    raise ValidationError("Responses input content must be a string or content-part list.")


def _input_messages(payload: Mapping[str, Any]) -> list[ModelMessage]:
    raw_input = payload.get("input")
    if raw_input is None:
        raise ValidationError('Responses requests require an "input" field.')
    messages: list[ModelMessage] = []
    instructions = payload.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise ValidationError("Responses instructions must be a string.")
        if instructions:
            messages.append(create_text_message("system", instructions))
    if isinstance(raw_input, str):
        messages.append(create_text_message("user", raw_input))
        return messages
    if not isinstance(raw_input, list) or not raw_input:
        raise ValidationError("Responses input must be a non-empty string or item list.")
    for item in raw_input:
        if not isinstance(item, Mapping):
            raise ValidationError("Responses input items must be objects.")
        item_type = item.get("type", "message")
        if item_type != "message":
            raise ValidationError(f'Unsupported Responses input item type "{item_type}".')
        role = item.get("role", "user")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValidationError(f'Unsupported Responses input role "{role}".')
        messages.append(create_text_message("system" if role == "developer" else role, _content_text(item.get("content", ""))))
    return messages


def _output_message(*, item_id: str, text: str, status: str = "completed") -> dict[str, Any]:
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

    def __init__(self, agents: AgentResolver) -> None:
        self.agents = agents

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

    async def prepare(self, payload: Mapping[str, Any]) -> tuple[str, Agent, list[ModelMessage]]:
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise ValidationError('Responses requests require a non-empty "model" alias.')
        return model, await self.resolve(model), _input_messages(payload)

    async def create(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if payload.get("stream") is True:
            raise ValidationError("Use ResponsesAgentHost.stream() for streaming requests.")
        model, agent, messages = await self.prepare(payload)
        response_id = _new_id("resp")
        created_at = int(time.time())
        result = await run_agent(agent=agent, messages=messages)
        item = _output_message(item_id=_new_id("msg"), text=result.text)
        return _response_object(
            response_id=response_id,
            model=model,
            created_at=created_at,
            output=[item],
            usage=result.usage,
        )

    async def stream(self, payload: Mapping[str, Any]) -> AsyncIterable[dict[str, Any]]:
        prepared = await self.prepare(payload)
        async for event in self.stream_prepared(payload, prepared=prepared):
            yield event

    async def stream_prepared(
        self,
        payload: Mapping[str, Any],
        *,
        prepared: tuple[str, Agent, list[ModelMessage]],
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
        yield {"type": "response.created", "sequence_number": sequence, "response": empty}
        sequence += 1
        yield {"type": "response.in_progress", "sequence_number": sequence, "response": empty}
        sequence += 1
        output = _output_message(item_id=item_id, text="", status="in_progress")
        yield {
            "type": "response.output_item.added",
            "sequence_number": sequence,
            "output_index": 0,
            "item": output,
        }
        sequence += 1
        content = {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}
        yield {
            "type": "response.content_part.added",
            "sequence_number": sequence,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": content,
        }
        sequence += 1
        stream = stream_agent(agent=agent, messages=messages)
        try:
            async for event in stream.event_stream():
                if isinstance(event, AgentTextDeltaEvent) and event.text_delta:
                    yield {
                        "type": "response.output_text.delta",
                        "sequence_number": sequence,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": event.text_delta,
                        "logprobs": [],
                    }
                    sequence += 1
            result = await stream.collect()
        except Exception as error:
            failed = _response_object(
                response_id=response_id,
                model=model,
                created_at=created_at,
                output=[],
                usage=None,
                status="failed",
                error={"code": type(error).__name__, "message": str(error)},
            )
            yield {"type": "response.failed", "sequence_number": sequence, "response": failed}
            return
        final_part = {"type": "output_text", "annotations": [], "logprobs": [], "text": result.text}
        yield {
            "type": "response.output_text.done",
            "sequence_number": sequence,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": result.text,
            "logprobs": [],
        }
        sequence += 1
        yield {
            "type": "response.content_part.done",
            "sequence_number": sequence,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": final_part,
        }
        sequence += 1
        final_item = _output_message(item_id=item_id, text=result.text)
        yield {
            "type": "response.output_item.done",
            "sequence_number": sequence,
            "output_index": 0,
            "item": final_item,
        }
        sequence += 1
        completed = _response_object(
            response_id=response_id,
            model=model,
            created_at=created_at,
            output=[final_item],
            usage=result.usage,
        )
        yield {"type": "response.completed", "sequence_number": sequence, "response": completed}


async def _read_payload(request: Any, *, max_request_bytes: int) -> dict[str, Any]:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > max_request_bytes:
                raise ParseError("Responses request body exceeded the configured maximum size.")
        except ValueError:
            raise ParseError("Responses request contained an invalid Content-Length header.") from None
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw) > max_request_bytes:
            raise ParseError("Responses request body exceeded the configured maximum size.")
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
    max_request_bytes: int,
) -> None:
    from fastapi import Request  # type: ignore[import-not-found]
    from fastapi.responses import JSONResponse, StreamingResponse  # type: ignore[import-not-found]

    async def endpoint(request: Any) -> Any:
        if authorize is not None and not await _maybe_await(authorize(request)):
            return JSONResponse(
                {"error": {"type": "authentication_error", "message": "Unauthorized."}},
                status_code=401,
            )
        try:
            payload = await _read_payload(request, max_request_bytes=max_request_bytes)
            if payload.get("stream") is True:
                prepared = await host.prepare(payload)
                source = host.stream_prepared(payload, prepared=prepared)

                async def encoded():
                    async for item in source:
                        event_name = item["type"]
                        data = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                        yield f"event: {event_name}\ndata: {data}\n\n"

                return StreamingResponse(encoded(), media_type="text/event-stream")
            return JSONResponse(await host.create(payload))
        except KeyError as error:
            return JSONResponse(
                {"error": {"type": "invalid_request_error", "message": f'Unknown model alias "{error.args[0]}".'}},
                status_code=404,
            )
        except (ParseError, ValidationError, json.JSONDecodeError) as error:
            return JSONResponse(
                {"error": {"type": "invalid_request_error", "message": str(error)}},
                status_code=400,
            )

    endpoint.__annotations__["request"] = Request
    app.add_api_route("/v1/responses", endpoint, methods=["POST"])


def create_responses_app(
    *,
    agents: AgentResolver,
    authorize: Callable[[Any], bool | Awaitable[bool]] | None = None,
    max_request_bytes: int = DEFAULT_RESPONSES_REQUEST_BYTES,
):
    """Create an optional FastAPI app with ``POST /v1/responses``."""

    try:
        from fastapi import FastAPI  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError('Responses hosting requires `pip install "zhivex-ai-sdk[api]"`.') from error
    app = FastAPI(title="Zhivex Responses Host", version="0.16.0")
    _install_responses_routes(
        app,
        host=ResponsesAgentHost(agents),
        authorize=authorize,
        max_request_bytes=max_request_bytes,
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
    max_request_bytes: int = DEFAULT_RESPONSES_REQUEST_BYTES,
):
    """Create a local playground plus the Responses-compatible endpoint."""

    try:
        from fastapi import FastAPI  # type: ignore[import-not-found]
        from fastapi.responses import HTMLResponse, JSONResponse  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError('The playground requires `pip install "zhivex-ai-sdk[api]"`.') from error
    app = FastAPI(title="Zhivex Agent Playground", version="0.16.0")
    _install_responses_routes(
        app,
        host=ResponsesAgentHost(agents),
        authorize=authorize,
        max_request_bytes=max_request_bytes,
    )

    async def index() -> Any:
        return HTMLResponse(_PLAYGROUND_HTML)

    async def health() -> Any:
        return JSONResponse({"status": "ok"})

    app.add_api_route("/", index, methods=["GET"])
    app.add_api_route("/health", health, methods=["GET"])
    return app
