from __future__ import annotations

import json
import os
from collections.abc import AsyncIterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

from .._http import Fetcher, ResponseLike, default_fetch
from .._sse import parse_sse
from ..errors import ConfigurationError, ProviderHTTPError, UnsupportedFeatureError, ValidationError
from ..messages import hosted_tool, is_hosted_tool_definition, normalize_finish_reason, validate_file_part, validate_message_parts
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..types import (
    AgentCapabilities,
    FilePart,
    FinishReason,
    GenerateResult,
    HostedToolClass,
    HostedToolDefinition,
    ImagePart,
    LanguageModel,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    PortableSupport,
    ProviderDataPart,
    StreamEvent,
    StreamFinishEvent,
    StreamProviderDataEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    TextPart,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolChoiceName,
    ToolResultPart,
)
from ._payload import drop_none
from ._url_security import validate_provider_url
from .base import ProviderAdapter, ProviderBundle, create_provider_bundle
from .openai_compat import OpenAICompatibleFilesClient, OpenAICompatibleResponsesClient


META_DEFAULT_BASE_URL = "https://api.meta.ai/v1"
MetaApiMode = Literal["auto", "chat", "responses"]

META_AGENT_CAPABILITIES = AgentCapabilities(
    support_tier="tier-c",
    tool_choice_none=False,
    hosted_web_search=True,
    toolsets=True,
)

META_CAPABILITIES = ModelCapabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    parallel_tool_calls=True,
    vision=True,
    files=True,
    audio_input=True,
    audio_output=False,
    embeddings=False,
    reasoning=True,
    web_search=True,
    agent_capabilities=META_AGENT_CAPABILITIES,
)

_RESERVED_PROVIDER_OPTIONS = frozenset(
    {
        "model",
        "messages",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "response_format",
        "text",
        "reasoning",
        "reasoning_effort",
        "stream",
    }
)
_AUDIO_FORMATS = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}


def meta_hosted_tool(
    tool_type: str,
    /,
    *,
    name: str | None = None,
    tool_class: HostedToolClass | None = None,
    **config: Any,
) -> HostedToolDefinition:
    return hosted_tool(
        name=name or tool_type,
        provider="meta",
        type=tool_type,
        config=drop_none(deepcopy(config)) or None,
        tool_class=tool_class,
    )


def meta_web_search_tool(**config: Any) -> HostedToolDefinition:
    return meta_hosted_tool("web_search", tool_class="web-search", **config)


def meta_tool_search_tool(**config: Any) -> HostedToolDefinition:
    return meta_hosted_tool("tool_search", tool_class="toolset", **config)


def _validated_base_url(base_url: str) -> str:
    raw = str(base_url).rstrip("/")
    parsed = urlparse(raw)
    if parsed.query or parsed.fragment or any(part == ".." for part in parsed.path.split("/")):
        raise ValidationError('Provider "meta" requires a safe API base URL without query, fragment, or parent paths.')
    host = (parsed.hostname or "").strip(".").lower()
    suffix = "meta.ai" if host == "meta.ai" or host.endswith(".meta.ai") else host
    if not suffix:
        raise ValidationError('Provider "meta" requires a valid API base URL.')
    return validate_provider_url(
        raw,
        provider="meta",
        purpose="API base",
        allowed_suffixes=(suffix,),
    )


def _headers(api_key: str, *, json_content: bool = True) -> dict[str, str]:
    headers = {"authorization": f"Bearer {api_key}"}
    if json_content:
        headers["content-type"] = "application/json"
    return headers


def _provider_error(response: ResponseLike, body: str) -> ProviderHTTPError:
    return ProviderHTTPError(
        f"Meta request failed with status {response.status_code}.",
        response.status_code,
        response_body=body,
        response_headers=dict(getattr(response, "headers", {}) or {}),
    )


async def _checked_fetch(
    *,
    fetch: Fetcher,
    url: str,
    api_key: str,
    body: dict[str, Any],
    timeout_ms: int | None,
    stream: bool,
) -> ResponseLike:
    response = await fetch(
        url,
        headers=_headers(api_key),
        json_body=body,
        timeout_ms=timeout_ms,
        stream=stream,
    )
    if response.status_code >= 400:
        raise _provider_error(response, await response.text())
    return response


def _provider_options(input: ModelGenerateInput) -> tuple[MetaApiMode, dict[str, Any]]:
    options = deepcopy(input.provider_options or {})
    raw_mode = options.pop("api_mode", "auto")
    if raw_mode not in {"auto", "chat", "responses"}:
        raise ValidationError('Provider "meta" api_mode must be "auto", "chat", or "responses".')
    collisions = sorted(_RESERVED_PROVIDER_OPTIONS & options.keys())
    if collisions:
        raise ValidationError(
            f'Provider "meta" provider_options cannot override reserved field(s): {", ".join(collisions)}.'
        )
    return raw_mode, options


def _has_hosted_tools(input: ModelGenerateInput) -> bool:
    return any(is_hosted_tool_definition(tool) for tool in (input.tools or {}).values())


def _has_file_parts(input: ModelGenerateInput) -> bool:
    return any(isinstance(part, FilePart) for message in input.messages for part in message.parts)


def _api_mode(input: ModelGenerateInput) -> Literal["chat", "responses"]:
    mode, options = _provider_options(input)
    needs_responses = _has_hosted_tools(input) or _has_file_parts(input) or "previous_response_id" in options
    if mode == "chat" and needs_responses:
        raise UnsupportedFeatureError(
            'Provider "meta" requires api_mode="responses" for hosted tools, file/audio inputs, and previous_response_id.'
        )
    if mode == "responses" or needs_responses:
        return "responses"
    return "chat"


def _validate_generation_input(model: LanguageModel, input: ModelGenerateInput) -> None:
    validate_message_parts(model, input.messages)
    _provider_options(input)
    if input.tool_choice is not None and input.tool_choice != "auto":
        raise UnsupportedFeatureError('Provider "meta" supports only tool_choice="auto".')
    if isinstance(input.tool_choice, ToolChoiceName):
        raise UnsupportedFeatureError('Provider "meta" supports only tool_choice="auto".')
    if input.reasoning is not None:
        if input.reasoning.budget_tokens is not None:
            raise UnsupportedFeatureError('Provider "meta" does not support reasoning token budgets.')
        if input.reasoning.effort == "none":
            raise UnsupportedFeatureError('Provider "meta" does not support reasoning effort "none".')
    for message in input.messages:
        for part in message.parts:
            if not isinstance(part, FilePart):
                continue
            validate_file_part(part)
            media_type = (part.media_type or "").split(";", 1)[0].strip().lower()
            if media_type.startswith("audio/") and media_type not in _AUDIO_FORMATS:
                raise UnsupportedFeatureError('Provider "meta" accepts only MP3 and WAV audio input.')
            if media_type in _AUDIO_FORMATS and (part.url is not None or part.file_uri is not None):
                raise UnsupportedFeatureError(
                    'Provider "meta" accepts MP3/WAV audio as inline data or an uploaded file_id, not a remote URL.'
                )


def _system_instructions(messages: list[ModelMessage]) -> str | None:
    text = "\n".join(
        part.text
        for message in messages
        if message.role == "system"
        for part in message.parts
        if isinstance(part, TextPart)
    )
    return text or None


def _audio_payload(part: FilePart) -> dict[str, Any]:
    media_type = (part.media_type or "").split(";", 1)[0].strip().lower()
    audio_format = _AUDIO_FORMATS.get(media_type)
    if audio_format is None:
        raise UnsupportedFeatureError('Provider "meta" accepts only MP3 and WAV audio input.')
    if part.file_id is not None:
        return {"type": "input_file", "file_id": part.file_id}
    if part.data is None:
        raise UnsupportedFeatureError('Provider "meta" inline audio requires FilePart.data.')
    data = part.data
    if data.startswith("data:"):
        prefix, separator, encoded = data.partition(",")
        if not separator or ";base64" not in prefix.lower():
            raise ValidationError('Provider "meta" inline audio data URLs must be base64 encoded.')
        data = encoded
    return {"type": "input_audio", "input_audio": {"data": data, "format": audio_format}}


def _responses_file_payload(part: FilePart) -> dict[str, Any]:
    media_type = (part.media_type or "").split(";", 1)[0].strip().lower()
    if media_type.startswith("audio/"):
        return _audio_payload(part)
    if part.text is not None or part.document_content is not None or part.file_uri is not None:
        raise UnsupportedFeatureError(
            'Provider "meta" file inputs require data, file_id, or url; text, document_content, and file_uri are unsupported.'
        )
    payload: dict[str, Any] = {"type": "input_file"}
    if part.data is not None:
        payload["file_data"] = part.data
    if part.file_id is not None:
        payload["file_id"] = part.file_id
    if part.url is not None:
        payload["file_url"] = part.url
    if part.filename is not None:
        payload["filename"] = part.filename
    return payload


def _chat_content(message: ModelMessage) -> str | list[dict[str, Any]] | None:
    text_parts = [part.text for part in message.parts if isinstance(part, TextPart)]
    multimodal = any(isinstance(part, (ImagePart, FilePart)) for part in message.parts)
    if not multimodal:
        return "".join(text_parts) or None
    content: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            content.append({"type": "image_url", "image_url": {"url": part.image}})
        elif isinstance(part, FilePart):
            content.append(_audio_payload(part))
    return content or None


def _tool_output(part: ToolResultPart) -> str:
    value: Any
    if part.tool_result.is_error and part.tool_result.error is not None:
        value = {"message": part.tool_result.error.message}
    else:
        value = part.tool_result.output
    return json.dumps(value)


def _chat_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            for part in message.parts:
                if isinstance(part, ToolResultPart):
                    mapped.append(
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_result.tool_call_id,
                            "content": _tool_output(part),
                        }
                    )
            continue
        payload: dict[str, Any] = {"role": message.role, "content": _chat_content(message)}
        if message.role == "assistant":
            tool_calls = [
                {
                    "id": part.tool_call.id,
                    "type": "function",
                    "function": {
                        "name": part.tool_call.name,
                        "arguments": json.dumps(part.tool_call.input),
                    },
                }
                for part in message.parts
                if isinstance(part, ToolCallPart)
            ]
            if tool_calls:
                payload["tool_calls"] = tool_calls
        mapped.append(drop_none(payload))
    return mapped


def _responses_content(message: ModelMessage) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            content.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ImagePart):
            content.append({"type": "input_image", "image_url": part.image})
        elif isinstance(part, FilePart):
            content.append(_responses_file_payload(part))
    return content


def _responses_input(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            for part in message.parts:
                if isinstance(part, ToolResultPart):
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": part.tool_result.tool_call_id,
                            "output": _tool_output(part),
                        }
                    )
            continue
        content = _responses_content(message)
        if content:
            items.append({"type": "message", "role": message.role, "content": content})
        if message.role == "assistant":
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": part.tool_call.id,
                            "name": part.tool_call.name,
                            "arguments": json.dumps(part.tool_call.input),
                        }
                    )
    return items


def _map_tools(tools: dict[str, Any] | None, *, responses: bool) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    mapped: list[dict[str, Any]] = []
    for definition in tools.values():
        if is_hosted_tool_definition(definition):
            if not responses:
                raise UnsupportedFeatureError('Provider "meta" hosted tools require the Responses API.')
            if definition.provider not in {None, "meta"}:
                raise ValidationError(
                    f'Hosted tool "{definition.name}" targets provider "{definition.provider}", but this model uses "meta".'
                )
            payload: dict[str, Any] = {}
            if isinstance(definition.config, dict):
                payload.update(deepcopy(definition.config))
            elif definition.config is not None:
                payload["config"] = deepcopy(definition.config)
            payload["type"] = definition.type
            mapped.append(drop_none(payload))
            continue
        function = drop_none(
            {
                "name": definition.name,
                "description": definition.description,
                "parameters": create_schema_adapter(definition.schema).json_schema(),
                "strict": True if definition.strict is None else definition.strict,
            }
        )
        mapped.append({"type": "function", **function} if responses else {"type": "function", "function": function})
    return mapped


def _structured_output(input: ModelGenerateInput, *, responses: bool) -> dict[str, Any] | None:
    config = input.structured_output
    if config is None or config.mode != "native":
        return None
    schema = create_schema_adapter(config.schema).json_schema()
    if responses:
        return {
            "format": {
                "type": "json_schema",
                "name": config.name or "response",
                "strict": True,
                "schema": schema,
            }
        }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": config.name or "response",
            "strict": True,
            "schema": schema,
        },
    }


def _reasoning(input: ModelGenerateInput, *, responses: bool) -> Any:
    effort = input.reasoning.effort if input.reasoning is not None else None
    if effort is None:
        return None
    return {"effort": effort} if responses else effort


def _chat_body(model_id: str, input: ModelGenerateInput, *, stream: bool) -> dict[str, Any]:
    _, options = _provider_options(input)
    return drop_none(
        {
            "model": model_id,
            "messages": _chat_messages(input.messages),
            "tools": _map_tools(input.tools, responses=False),
            "tool_choice": input.tool_choice,
            "response_format": _structured_output(input, responses=False),
            "reasoning_effort": _reasoning(input, responses=False),
            "temperature": input.temperature,
            "max_tokens": input.max_tokens,
            "stream": True if stream else None,
            **options,
        }
    )


def _responses_body(model_id: str, input: ModelGenerateInput, *, stream: bool) -> dict[str, Any]:
    _, options = _provider_options(input)
    return drop_none(
        {
            "model": model_id,
            "instructions": _system_instructions(input.messages),
            "input": _responses_input(input.messages),
            "tools": _map_tools(input.tools, responses=True),
            "tool_choice": input.tool_choice,
            "text": _structured_output(input, responses=True),
            "reasoning": _reasoning(input, responses=True),
            "temperature": input.temperature,
            "max_output_tokens": input.max_tokens,
            "stream": True if stream else None,
            **options,
        }
    )


def _parse_tool_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return deepcopy(value) if value is not None else {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _chat_tool_call(value: dict[str, Any]) -> ToolCall:
    function = value.get("function") or {}
    return ToolCall(
        id=str(value.get("id") or ""),
        name=str(function.get("name") or ""),
        input=_parse_tool_arguments(function.get("arguments")),
        provider_metadata={"provider": "meta", "raw_tool_call": deepcopy(value)},
    )


def _usage(payload: dict[str, Any]) -> TokenUsage | None:
    usage = payload.get("usage") or {}
    if not usage:
        return None
    return TokenUsage(
        input_tokens=usage.get("input_tokens", usage.get("prompt_tokens")),
        output_tokens=usage.get("output_tokens", usage.get("completion_tokens")),
        total_tokens=usage.get("total_tokens"),
    )


def _response_finish_reason(status: str | None, *, has_tool_calls: bool = False) -> FinishReason | None:
    if has_tool_calls and status == "completed":
        return "tool-calls"
    if status == "completed":
        return "stop"
    if status == "failed":
        return "error"
    return normalize_finish_reason(status)


def _chat_result(payload: dict[str, Any]) -> GenerateResult:
    choice = ((payload.get("choices") or [{}])[0] or {})
    raw_message = choice.get("message") or {}
    parts: list[Any] = []
    content = raw_message.get("content")
    if isinstance(content, str) and content:
        parts.append(TextPart(text=content))
    for item in raw_message.get("tool_calls") or []:
        if isinstance(item, dict):
            parts.append(ToolCallPart(tool_call=_chat_tool_call(item)))
    message = ModelMessage(role="assistant", parts=parts)
    finish = choice.get("finish_reason")
    return GenerateResult(
        messages=[message],
        text="".join(part.text for part in parts if isinstance(part, TextPart)),
        finish_reason=normalize_finish_reason(finish),
        provider_finish_reason=finish,
        usage=_usage(payload),
        raw_response=payload,
    )


def _responses_result(payload: dict[str, Any]) -> GenerateResult:
    parts: list[Any] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message" and item.get("role") == "assistant":
            for content in item.get("content") or []:
                if isinstance(content, dict) and content.get("type") in {"output_text", "text"} and content.get("text"):
                    parts.append(TextPart(text=str(content["text"]), provider_metadata=dict(content)))
        elif item.get("type") == "function_call":
            parts.append(
                ToolCallPart(
                    tool_call=ToolCall(
                        id=str(item.get("call_id") or item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        input=_parse_tool_arguments(item.get("arguments")),
                        provider_metadata={"provider": "meta", "response_item_id": item.get("id")},
                    )
                )
            )
        elif item.get("type") in {"reasoning", "web_search_call", "tool_search_call"}:
            parts.append(ProviderDataPart(provider="meta", data=deepcopy(item)))
    if not parts and isinstance(payload.get("output_text"), str) and payload.get("output_text"):
        parts.append(TextPart(text=str(payload["output_text"])))
    status = str(payload.get("status") or "") or None
    has_tool_calls = any(isinstance(part, ToolCallPart) for part in parts)
    finish = _response_finish_reason(status, has_tool_calls=has_tool_calls)
    message = ModelMessage(role="assistant", parts=parts)
    return GenerateResult(
        messages=[message],
        text="".join(part.text for part in parts if isinstance(part, TextPart)),
        finish_reason=finish,
        provider_finish_reason=status,
        usage=_usage(payload),
        raw_response=payload,
    )


@dataclass(slots=True)
class MetaLanguageModel(LanguageModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: META_CAPABILITIES)

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        _validate_generation_input(self, input)
        mode = _api_mode(input)
        body = _responses_body(self.model_id, input, stream=False) if mode == "responses" else _chat_body(self.model_id, input, stream=False)
        response = await with_retry(
            lambda: _checked_fetch(
                fetch=self.fetch,
                url=f"{self.base_url}/{'responses' if mode == 'responses' else 'chat/completions'}",
                api_key=self.api_key,
                body=body,
                timeout_ms=input.timeout_ms,
                stream=False,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms if input.retry_backoff_ms is not None else 250,
        )
        payload = await response.json()
        return _responses_result(payload) if mode == "responses" else _chat_result(payload)

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        _validate_generation_input(self, input)
        mode = _api_mode(input)
        body = _responses_body(self.model_id, input, stream=True) if mode == "responses" else _chat_body(self.model_id, input, stream=True)
        response = await with_retry(
            lambda: _checked_fetch(
                fetch=self.fetch,
                url=f"{self.base_url}/{'responses' if mode == 'responses' else 'chat/completions'}",
                api_key=self.api_key,
                body=body,
                timeout_ms=input.timeout_ms,
                stream=True,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms if input.retry_backoff_ms is not None else 250,
        )
        return _responses_stream(response) if mode == "responses" else _chat_stream(response)


async def _chat_stream(response: ResponseLike) -> AsyncIterable[StreamEvent]:
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage_payload: dict[str, Any] | None = None
    flushed = False

    async def terminal() -> AsyncIterable[StreamEvent]:
        nonlocal flushed
        if flushed:
            return
        flushed = True
        for value in tool_calls.values():
            yield StreamToolCallEvent(tool_call=_chat_tool_call(value))
        if finish_reason is not None or usage_payload is not None or tool_calls:
            provider_finish = finish_reason or ("tool_calls" if tool_calls else None)
            yield StreamFinishEvent(
                finish_reason="tool-calls" if tool_calls else normalize_finish_reason(provider_finish),
                provider_finish_reason=provider_finish,
                usage=_usage({"usage": usage_payload or {}}),
            )

    async for event in parse_sse(response.iter_lines()):
        if event.data == "[DONE]":
            async for item in terminal():
                yield item
            return
        payload = json.loads(event.data)
        if isinstance(payload.get("usage"), dict):
            usage_payload = dict(payload["usage"])
        choice = ((payload.get("choices") or [{}])[0] or {})
        delta = choice.get("delta") or {}
        if delta.get("content"):
            yield StreamTextDeltaEvent(text_delta=str(delta["content"]))
        for item in delta.get("tool_calls") or []:
            index = int(item.get("index") or 0)
            current = tool_calls.setdefault(index, {"id": "", "function": {"name": "", "arguments": ""}})
            if item.get("id"):
                current["id"] = item["id"]
            function = item.get("function") or {}
            target = current.setdefault("function", {"name": "", "arguments": ""})
            if function.get("name"):
                target["name"] = function["name"]
            if function.get("arguments"):
                target["arguments"] = str(target.get("arguments") or "") + str(function["arguments"])
        if choice.get("finish_reason"):
            finish_reason = str(choice["finish_reason"])

    async for item in terminal():
        yield item


async def _responses_stream(response: ResponseLike) -> AsyncIterable[StreamEvent]:
    states: dict[str, dict[str, Any]] = {}
    emitted: set[str] = set()
    finished = False

    def state_for(payload: dict[str, Any], item: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        value = item or {}
        key = str(
            payload.get("item_id")
            or value.get("id")
            or payload.get("call_id")
            or value.get("call_id")
            or payload.get("output_index")
            or len(states)
        )
        state = states.setdefault(key, {"id": "", "name": "", "arguments": ""})
        state["id"] = str(value.get("call_id") or payload.get("call_id") or state["id"] or value.get("id") or key)
        state["name"] = str(value.get("name") or payload.get("name") or state["name"])
        return key, state

    def tool_event(key: str, state: dict[str, Any]) -> StreamToolCallEvent | None:
        if key in emitted or not state.get("name"):
            return None
        emitted.add(key)
        return StreamToolCallEvent(
            tool_call=ToolCall(
                id=str(state.get("id") or key),
                name=str(state.get("name") or ""),
                input=_parse_tool_arguments(state.get("arguments")),
                provider_metadata={"provider": "meta", "response_item_id": key},
            )
        )

    async def pending_tools() -> AsyncIterable[StreamEvent]:
        for key, state in states.items():
            tool = tool_event(key, state)
            if tool is not None:
                yield tool

    async for event in parse_sse(response.iter_lines()):
        if event.data == "[DONE]":
            async for tool in pending_tools():
                yield tool
            return
        payload = json.loads(event.data)
        event_type = str(payload.get("type") or "")
        if event_type == "response.output_text.delta":
            yield StreamTextDeltaEvent(text_delta=str(payload.get("delta") or ""))
            continue
        if event_type == "response.output_item.added":
            item = payload.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "function_call":
                _, state = state_for(payload, item)
                if item.get("arguments"):
                    state["arguments"] = str(item["arguments"])
            elif isinstance(item, dict) and item.get("type") in {"reasoning", "web_search_call", "tool_search_call"}:
                yield StreamProviderDataEvent(provider="meta", data=deepcopy(item))
            continue
        if event_type == "response.function_call_arguments.delta":
            _, state = state_for(payload)
            state["arguments"] = str(state.get("arguments") or "") + str(payload.get("delta") or "")
            continue
        if event_type == "response.function_call_arguments.done":
            key, state = state_for(payload)
            if payload.get("arguments") is not None:
                state["arguments"] = str(payload.get("arguments") or "")
            tool_call_event = tool_event(key, state)
            if tool_call_event is not None:
                yield tool_call_event
            continue
        if event_type == "response.output_item.done":
            item = payload.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "function_call":
                key, state = state_for(payload, item)
                if item.get("arguments") is not None:
                    state["arguments"] = str(item.get("arguments") or "")
                tool_call_event = tool_event(key, state)
                if tool_call_event is not None:
                    yield tool_call_event
            continue
        if event_type in {"response.completed", "response.failed", "response.incomplete"}:
            async for tool in pending_tools():
                yield tool
            response_payload = payload.get("response") or {}
            status = str(response_payload.get("status") or event_type.removeprefix("response."))
            has_tools = bool(emitted)
            yield StreamFinishEvent(
                finish_reason=_response_finish_reason(status, has_tool_calls=has_tools),
                provider_finish_reason=status,
                usage=_usage(response_payload),
            )
            finished = True

    async for tool in pending_tools():
        yield tool
    if not finished and emitted:
        yield StreamFinishEvent(finish_reason="tool-calls", provider_finish_reason="tool_calls")


@dataclass(slots=True)
class MetaFilesClient(OpenAICompatibleFilesClient):
    default_purpose: str = "user_data"


def create_meta(
    *,
    api_key: str | None = None,
    base_url: str = META_DEFAULT_BASE_URL,
    fetch: Fetcher | None = None,
) -> ProviderBundle:
    resolved_key = api_key or os.getenv("MODEL_API_KEY")
    if not resolved_key:
        raise ConfigurationError("Missing Meta Model API key. Set MODEL_API_KEY.")
    resolved_base = _validated_base_url(base_url)
    requester = fetch or default_fetch
    native = ProviderAdapter(
        name="meta",
        language_model_factory=lambda model_id: MetaLanguageModel(
            provider="meta",
            model_id=model_id,
            api_key=resolved_key,
            base_url=resolved_base,
            fetch=requester,
        ),
        files_client_factory=lambda: MetaFilesClient(
            provider="meta",
            api_key=resolved_key,
            base_url=resolved_base,
            fetch=requester,
        ),
        responses_client_factory=lambda: OpenAICompatibleResponsesClient(
            provider="meta",
            model_id="",
            api_key=resolved_key,
            base_url=resolved_base,
            fetch=requester,
        ),
    )
    return create_provider_bundle(
        name="meta",
        native=native,
        agent_capabilities=META_AGENT_CAPABILITIES,
        portable_support=PortableSupport(
            text_generation=True,
            streaming=True,
            structured_output=True,
            tools=True,
            embeddings=False,
            grounding=False,
            retrieval=True,
            transcription=False,
            speech=False,
            portable_badge=True,
            tier="portable",
        ),
    )
