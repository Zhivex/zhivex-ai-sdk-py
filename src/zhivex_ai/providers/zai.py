"""Z.ai GLM adapter.

The general API is OpenAI-compatible Chat Completions, not Responses.
``glm-5.3`` and ``glm-5.3-flash`` always think. ``glm-5.3`` is text-only.
``glm-5.3-flash`` accepts images. ``glm-5.2`` can disable thinking.

Official references:
https://docs.z.ai/api-reference/llm/chat-completion
https://docs.z.ai/guides/llm/glm-5.3
https://docs.z.ai/guides/overview/pricing
https://docs.z.ai/guides/vlm/glm-5.3-flash
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterable
from typing import Any
from urllib.parse import urlparse

from .._http import Fetcher, default_fetch
from .._sse import parse_sse
from ..errors import ConfigurationError, ParseError, ProviderHTTPError, UnsupportedFeatureError, ValidationError
from ..messages import is_hosted_tool_definition, normalize_finish_reason
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..types import (
    AgentCapabilities,
    GenerateResult,
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
from .base import ProviderAdapter, create_provider_bundle
from .openai_compat import _parse_json_error


ZAI_GENERAL_BASE_URL = "https://api.z.ai/api/paas/v4"
ZAI_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_GLM53_EFFORTS = frozenset({"low", "high", "max"})
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_RESERVED_BODY_KEYS = frozenset({
    "thinking", "reasoning_effort", "temperature", "max_tokens", "response_format",
    "tool_stream", "tool_choice", "model", "messages", "tools", "stream",
})
_AGENT = AgentCapabilities(support_tier="tier-b", tool_choice_none=False)


def _is_glm_53(model_id: str) -> bool:
    return model_id.strip().lower() == "glm-5.3"


def _is_glm_53_flash(model_id: str) -> bool:
    return model_id.strip().lower() == "glm-5.3-flash"


def _is_glm_53_family(model_id: str) -> bool:
    return _is_glm_53(model_id) or _is_glm_53_flash(model_id)


def _is_glm_52(model_id: str) -> bool:
    return model_id.strip().lower() == "glm-5.2"


def _capabilities(model_id: str) -> ModelCapabilities:
    return ModelCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=True,
        tool_choice=False,
        parallel_tool_calls=False,
        vision=_is_glm_53_flash(model_id),
        files=False,
        audio_input=False,
        audio_output=False,
        embeddings=False,
        reasoning=_is_glm_53_family(model_id) or _is_glm_52(model_id),
        web_search=False,
        agent_capabilities=_AGENT,
    )


def _resolve_base_url(*, base_url: str | None, endpoint: str, allow_unsafe_endpoints: bool) -> str:
    if endpoint not in {"general", "coding"}:
        raise ConfigurationError('Provider "zai" endpoint must be "general" or "coding".')
    selected = ZAI_CODING_BASE_URL if endpoint == "coding" else ZAI_GENERAL_BASE_URL
    raw = base_url or os.getenv("ZAI_BASE_URL") or selected
    parsed = urlparse(raw)
    if parsed.scheme != "https" and not allow_unsafe_endpoints:
        raise ConfigurationError('Provider "zai" base_url must use https.')
    if not parsed.netloc:
        raise ConfigurationError('Provider "zai" base_url must include a host.')
    return raw.rstrip("/")


def _text(message: ModelMessage) -> str:
    return "".join(part.text for part in message.parts if isinstance(part, TextPart))


def _image_url(image: str, media_type: str | None) -> str:
    normalized = image.strip()
    if not normalized:
        raise ValidationError("Z.ai image input must not be empty.")
    if media_type is not None and not media_type.lower().startswith("image/"):
        raise ValidationError('Z.ai ImagePart media_type must use an image/* MIME type.')
    if normalized.lower().startswith(("http://", "https://")):
        return normalized
    if normalized.lower().startswith("data:"):
        if not normalized.lower().startswith("data:image/"):
            raise ValidationError("Z.ai image Data URLs must use an image/* MIME type.")
        return normalized
    return f"data:{media_type or 'image/png'};base64,{normalized}"


def _content(model_id: str, message: ModelMessage) -> str | list[dict[str, Any]]:
    for part in message.parts:
        if part.type in {"audio", "file"}:
            raise UnsupportedFeatureError(f'Z.ai model "{model_id}" does not support {part.type} input.')
    images = [part for part in message.parts if isinstance(part, ImagePart)]
    if not images:
        return _text(message)
    if not _is_glm_53_flash(model_id):
        raise UnsupportedFeatureError(f'Z.ai model "{model_id}" does not support image input.')
    blocks: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append({"type": "image_url", "image_url": {"url": _image_url(part.image, part.media_type)}})
    return blocks


def _reasoning_content(message: ModelMessage) -> str:
    chunks: list[str] = []
    for part in message.parts:
        if isinstance(part, ProviderDataPart) and part.provider == "zai" and isinstance(part.data, dict):
            value = part.data.get("reasoningContent", part.data.get("reasoning_content"))
            if isinstance(value, str) and value:
                chunks.append(value)
    return "".join(chunks)


def _messages(model_id: str, messages: list[ModelMessage]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            for part in message.parts:
                if isinstance(part, ToolResultPart):
                    output = part.tool_result.error.message if part.tool_result.is_error and part.tool_result.error else part.tool_result.output
                    mapped.append({
                        "role": "tool",
                        "tool_call_id": part.tool_result.tool_call_id,
                        "content": output if isinstance(output, str) else json.dumps(output),
                    })
            continue
        payload: dict[str, Any] = {"role": message.role, "content": _content(model_id, message)}
        reasoning = _reasoning_content(message)
        if reasoning:
            payload["reasoning_content"] = reasoning
        tool_calls = [part.tool_call for part in message.parts if isinstance(part, ToolCallPart)]
        if tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.input)},
                }
                for call in tool_calls
            ]
        mapped.append(payload)
    return mapped


def _map_tools(tools: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    definitions = list(tools.values())
    if len(definitions) > 128:
        raise ValidationError("Z.ai accepts at most 128 function tools per request.")
    if any(is_hosted_tool_definition(tool) for tool in definitions):
        raise UnsupportedFeatureError('Provider "zai" does not support hosted tools.')
    mapped = []
    for tool in definitions:
        if not _TOOL_NAME.match(tool.name):
            raise ValidationError(
                f'Z.ai tool name "{tool.name}" must contain 1-64 letters, numbers, underscores, or hyphens.'
            )
        mapped.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or tool.name,
                "parameters": create_schema_adapter(tool.schema).json_schema(),
            },
        })
    return mapped


def _validate_options(model_id: str, input: ModelGenerateInput) -> None:
    options = input.provider_options or {}
    if input.reasoning is not None and (options.get("thinking") is not None or options.get("reasoning_effort") is not None):
        raise ValidationError(
            'Do not combine shared "reasoning" with Z.ai "provider_options.thinking" or "provider_options.reasoning_effort".'
        )
    thinking = options.get("thinking")
    if thinking is not None:
        if (
            not isinstance(thinking, dict)
            or thinking.get("type") not in {"enabled", "disabled"}
            or ("clear_thinking" in thinking and not isinstance(thinking.get("clear_thinking"), bool))
        ):
            raise ValidationError(
                'Z.ai "thinking" must include type "enabled" or "disabled" and optional boolean "clear_thinking".'
            )
    effort = options.get("reasoning_effort")
    if effort is not None and effort not in _EFFORTS:
        raise ValidationError(
            'Z.ai "reasoning_effort" must be "none", "minimal", "low", "medium", "high", "xhigh", or "max".'
        )
    if _is_glm_53_family(model_id):
        if isinstance(thinking, dict) and thinking.get("type") == "disabled":
            raise UnsupportedFeatureError('Z.ai GLM-5.3 requires thinking.type "enabled".')
        if effort is not None and effort not in _GLM53_EFFORTS:
            raise UnsupportedFeatureError('Z.ai GLM-5.3 supports reasoning_effort "low", "high", or "max" only.')
    if isinstance(thinking, dict) and thinking.get("type") == "disabled" and effort is not None:
        raise UnsupportedFeatureError('Z.ai does not accept "reasoning_effort" while thinking is disabled.')
    temperature = input.temperature if input.temperature is not None else options.get("temperature")
    if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= float(temperature) <= 1):
        raise ValidationError('Z.ai "temperature" must be between 0 and 1.')
    top_p = options.get("top_p")
    if top_p is not None and (isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0.01 <= float(top_p) <= 1):
        raise ValidationError('Z.ai "top_p" must be between 0.01 and 1.')
    max_tokens = input.max_tokens if input.max_tokens is not None else options.get("max_tokens")
    if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 131_072):
        raise ValidationError('Z.ai "max_tokens" must be a positive integer no greater than 131072.')
    stop = options.get("stop")
    if stop is not None and (not isinstance(stop, list) or len(stop) > 4 or any(not isinstance(item, str) for item in stop)):
        raise ValidationError('Z.ai "stop" must be an array of at most 4 strings.')
    provider_tool_choice = options.get("tool_choice")
    if (
        isinstance(input.tool_choice, ToolChoiceName)
        or input.tool_choice not in {None, "auto"}
        or not isinstance(provider_tool_choice, (str, type(None)))
        or provider_tool_choice not in {None, "auto"}
    ):
        raise UnsupportedFeatureError('Z.ai supports tool_choice "auto" only.')
    response_format = options.get("response_format")
    if response_format is not None and (
        not isinstance(response_format, dict) or response_format.get("type") not in {"text", "json_object"}
    ):
        raise ValidationError('Z.ai "response_format.type" must be "text" or "json_object".')


def _shared_reasoning(model_id: str, input: ModelGenerateInput) -> tuple[dict[str, Any] | None, str | None]:
    reasoning = input.reasoning
    if reasoning is None:
        return None, None
    if reasoning.budget_tokens is not None:
        raise UnsupportedFeatureError('Provider "zai" does not support "reasoning.budgetTokens".')
    effort = reasoning.effort
    if _is_glm_53_family(model_id):
        if effort is not None and effort not in _GLM53_EFFORTS:
            raise UnsupportedFeatureError('Z.ai GLM-5.3 supports shared reasoning effort "low", "high", or "max" only.')
        return {"type": "enabled"}, effort
    if _is_glm_52(model_id):
        if effort in {"none", "minimal"}:
            return {"type": "disabled"}, None
        mapped = "max" if effort in {"xhigh", "max"} else "high" if effort else None
        return {"type": "enabled"}, mapped
    raise UnsupportedFeatureError(f'Provider "zai" model "{model_id}" has no documented shared reasoning contract.')


def _request_options(model_id: str, input: ModelGenerateInput) -> dict[str, Any]:
    _validate_options(model_id, input)
    options = dict(input.provider_options or {})
    thinking, shared_effort = _shared_reasoning(model_id, input)
    if thinking is None and isinstance(options.get("thinking"), dict):
        thinking = dict(options["thinking"])
    if _is_glm_53_family(model_id) and thinking is None:
        thinking = {"type": "enabled"}
    if _is_glm_53_flash(model_id) and isinstance(thinking, dict) and thinking.get("type") == "enabled" and "clear_thinking" not in thinking:
        thinking = {**thinking, "clear_thinking": False}
    history = any(_reasoning_content(message) for message in input.messages)
    preserve = history or bool(input.tools)
    raw_disabled = (
        _is_glm_52(model_id)
        and input.reasoning is None
        and options.get("reasoning_effort") in {"none", "minimal"}
    )
    if (
        preserve
        and not raw_disabled
        and (
            (isinstance(thinking, dict) and thinking.get("type") == "enabled")
            or (thinking is None and (_is_glm_53_family(model_id) or _is_glm_52(model_id)))
        )
    ):
        thinking = {**(thinking or {"type": "enabled"}), "clear_thinking": False}
    effort = shared_effort if input.reasoning is not None else options.get("reasoning_effort")
    response_format = {"type": "json_object"} if input.structured_output is not None and input.structured_output.mode == "native" else options.get("response_format")
    body_options = {key: value for key, value in options.items() if key not in _RESERVED_BODY_KEYS}
    return {
        "body_options": body_options,
        "thinking": thinking,
        "reasoning_effort": effort,
        "temperature": input.temperature if input.temperature is not None else options.get("temperature"),
        "max_tokens": input.max_tokens if input.max_tokens is not None else options.get("max_tokens"),
        "response_format": response_format,
        "tool_stream": options.get("tool_stream"),
        "tool_choice": options.get("tool_choice"),
    }


def _with_json_instruction(messages: list[ModelMessage], response_format: Any, input: ModelGenerateInput) -> list[ModelMessage]:
    if not isinstance(response_format, dict) or response_format.get("type") != "json_object":
        return messages
    if any(re.search(r"\bjson\b", _text(message), re.IGNORECASE) for message in messages):
        return messages
    schema = None
    if input.structured_output is not None and input.structured_output.mode == "native":
        schema = json.dumps(create_schema_adapter(input.structured_output.schema).json_schema())
    instruction = f"Return only valid JSON matching this JSON Schema: {schema}" if schema else "Return only valid JSON."
    return [ModelMessage(role="system", parts=[TextPart(text=instruction)]), *messages]


def _usage(payload: Any) -> TokenUsage | None:
    if not isinstance(payload, dict):
        return None
    raw_details = payload.get("prompt_tokens_details")
    details = raw_details if isinstance(raw_details, dict) else {}
    cached = details.get("cached_tokens")
    return TokenUsage(
        input_tokens=payload.get("prompt_tokens"),
        output_tokens=payload.get("completion_tokens"),
        total_tokens=payload.get("total_tokens"),
        cached_input_tokens=cached if isinstance(cached, int) and not isinstance(cached, bool) else None,
    )


def _assistant_message(payload: Any) -> ModelMessage:
    if not isinstance(payload, dict):
        raise ParseError("Z.ai response did not contain an assistant message.")
    parts: list[Any] = []
    reasoning = payload.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        parts.append(ProviderDataPart(provider="zai", data={"type": "reasoning_content", "reasoningContent": reasoning}))
    content = payload.get("content")
    if isinstance(content, str) and content:
        parts.append(TextPart(text=content))
    for call in payload.get("tool_calls") or []:
        if not isinstance(call, dict):
            raise ParseError("Z.ai returned a malformed tool call.")
        raw_function = call.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}
        name = function.get("name")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
            raise ParseError("Z.ai tool calls require non-empty string id and function name fields.")
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise ParseError("Z.ai tool call arguments contained invalid JSON.") from error
        else:
            parsed = arguments
        if not isinstance(parsed, dict):
            raise ValidationError("Z.ai tool call arguments must be a JSON object.")
        parts.append(ToolCallPart(tool_call=ToolCall(id=call_id, name=name, input=parsed)))
    return ModelMessage(role="assistant", parts=parts)


class ZAILanguageModel(LanguageModel):
    provider = "zai"

    def __init__(self, model_id: str, api_key: str, base_url: str, fetch: Fetcher) -> None:
        self.model_id = model_id
        self.capabilities = _capabilities(model_id)
        self._api_key = api_key
        self._base_url = base_url
        self._fetch = fetch

    def _body(self, input: ModelGenerateInput, *, stream: bool) -> dict[str, Any]:
        resolved = _request_options(self.model_id, input)
        tools = _map_tools(input.tools)
        messages = _with_json_instruction(input.messages, resolved["response_format"], input)
        body = {
            **resolved["body_options"],
            "model": self.model_id,
            "messages": _messages(self.model_id, messages),
            "tools": tools,
            "tool_choice": (resolved["tool_choice"] or "auto") if tools else resolved["tool_choice"],
            "response_format": resolved["response_format"],
            "temperature": resolved["temperature"],
            "max_tokens": resolved["max_tokens"],
            "stream": stream,
        }
        if resolved["thinking"] is not None:
            body["thinking"] = resolved["thinking"]
        if resolved["reasoning_effort"]:
            body["reasoning_effort"] = resolved["reasoning_effort"]
        if stream and tools:
            body["tool_stream"] = True if resolved["tool_stream"] is None else resolved["tool_stream"]
        return {key: value for key, value in body.items() if value is not None}

    async def _post(self, body: dict[str, Any], input: ModelGenerateInput, *, stream: bool):
        response = await with_retry(
            lambda: self._fetch(
                f"{self._base_url}/chat/completions",
                headers={"authorization": f"Bearer {self._api_key}", "content-type": "application/json"},
                json_body=body,
                timeout_ms=input.timeout_ms,
                stream=stream,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 300:
            if 300 <= response.status_code < 400:
                raise ProviderHTTPError("Credentialed provider request was redirected.", response.status_code)
            raise _parse_json_error("zai", response.status_code, await response.text())
        return response

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        response = await self._post(self._body(input, stream=False), input, stream=False)
        payload = await response.json()
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ParseError("Z.ai response did not contain a completion choice.")
        choice = choices[0]
        message = _assistant_message(choice.get("message"))
        finish = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
        text = "".join(part.text for part in message.parts if isinstance(part, TextPart))
        return GenerateResult(
            messages=[message],
            text=text,
            finish_reason=normalize_finish_reason(finish),
            provider_finish_reason=finish,
            usage=_usage(payload.get("usage")),
            raw_response=payload,
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        response = await self._post(self._body(input, stream=True), input, stream=True)

        async def generator() -> AsyncIterable[StreamEvent]:
            finish_reason = None
            usage = None
            tool_buffers: dict[int, dict[str, str]] = {}
            async for event in parse_sse(response.iter_lines()):
                if event.data == "[DONE]":
                    break
                try:
                    payload = json.loads(event.data)
                except json.JSONDecodeError as error:
                    raise ParseError("Z.ai streaming response contained invalid JSON.") from error
                choice = (payload.get("choices") or [{}])[0] if isinstance(payload, dict) else {}
                delta = choice.get("delta") if isinstance(choice, dict) else None
                if isinstance(payload, dict) and payload.get("usage"):
                    usage = payload.get("usage")
                if isinstance(delta, dict) and delta.get("reasoning_content"):
                    yield StreamProviderDataEvent(
                        provider="zai",
                        data={"type": "reasoning_content", "reasoningContent": delta["reasoning_content"]},
                    )
                if isinstance(delta, dict) and delta.get("content"):
                    yield StreamTextDeltaEvent(text_delta=delta["content"])
                if isinstance(choice, dict) and choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                    if finish_reason == "tool_calls":
                        for index in sorted(tool_buffers):
                            buffered = tool_buffers[index]
                            if not buffered["id"] or not buffered["name"]:
                                raise ParseError("Z.ai fragmented tool calls require non-empty id and function name fields.")
                            arguments = json.loads(buffered["args"] or "{}")
                            yield StreamToolCallEvent(tool_call=ToolCall(id=buffered["id"], name=buffered["name"], input=arguments))
                        tool_buffers.clear()
                calls = delta.get("tool_calls") if isinstance(delta, dict) else None
                if calls is not None and not isinstance(calls, list):
                    raise ParseError("Z.ai streaming tool_calls must be an array.")
                for tool_call in calls or []:
                    if not isinstance(tool_call, dict) or isinstance(tool_call.get("index"), bool) or not isinstance(tool_call.get("index"), int) or tool_call["index"] < 0:
                        raise ParseError("Z.ai streaming tool calls require a non-negative integer index.")
                    slot = tool_buffers.setdefault(tool_call["index"], {"id": "", "name": "", "args": ""})
                    if isinstance(tool_call.get("id"), str):
                        slot["id"] += tool_call["id"]
                    raw_function = tool_call.get("function")
                    function = raw_function if isinstance(raw_function, dict) else {}
                    if isinstance(function.get("name"), str):
                        slot["name"] += function["name"]
                    if isinstance(function.get("arguments"), str):
                        slot["args"] += function["arguments"]
            if tool_buffers:
                raise ParseError("Z.ai streaming response ended before fragmented tool calls were completed.")
            if not finish_reason:
                raise ParseError("Z.ai streaming response ended without a finish reason.")
            yield StreamFinishEvent(
                finish_reason=normalize_finish_reason(finish_reason),
                provider_finish_reason=finish_reason,
                usage=_usage(usage),
            )

        return generator()


def create_zai(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    endpoint: str = "general",
    fetch: Fetcher | None = None,
    allow_unsafe_endpoints: bool = False,
):
    resolved_key = api_key or os.getenv("ZAI_API_KEY")
    if not resolved_key:
        raise ConfigurationError("Missing Z.ai API key.")
    resolved_base = _resolve_base_url(base_url=base_url, endpoint=endpoint, allow_unsafe_endpoints=allow_unsafe_endpoints)
    requester = fetch or default_fetch
    native = ProviderAdapter(
        name="zai",
        language_model_factory=lambda model_id: ZAILanguageModel(model_id, resolved_key, resolved_base, requester),
    )
    return create_provider_bundle(
        name="zai",
        native=native,
        agent_capabilities=_AGENT,
        portable_support=PortableSupport(
            text_generation=True,
            streaming=True,
            structured_output=True,
            tools=True,
            embeddings=False,
            grounding=False,
            retrieval=False,
            transcription=False,
            speech=False,
            portable_badge=False,
            tier="native-only",
        ),
    )
