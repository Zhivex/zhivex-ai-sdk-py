"""OpenAI-compatible Chat Completions provider.

This matches the TypeScript ``openaiCompatible`` factory: chat, streaming, tools,
and native JSON Schema structured output go to ``/chat/completions``. The
Responses helper used by first-party OpenAI adapters stays separate.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterable
from typing import Any
from urllib.parse import urlparse

from .._http import Fetcher, default_fetch
from .._sse import parse_sse
from ..errors import ParseError, ProviderHTTPError, UnsupportedFeatureError
from ..messages import is_hosted_tool_definition, normalize_finish_reason
from ..model_resolver import ModelResolutionError, register_model_provider
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..types import (
    GenerateResult,
    ImagePart,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    StreamEvent,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    TextPart,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolChoiceName,
    ToolResultPart,
)
from .base import ProviderAdapter

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_PRESETS = {
    "mistral": "https://api.mistral.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
}


def _capabilities(overrides: ModelCapabilities | None) -> ModelCapabilities:
    if overrides is not None:
        return overrides
    return ModelCapabilities(
        streaming=True,
        tools=True,
        structured_output=True,
        json_mode=True,
        tool_choice=True,
        parallel_tool_calls=True,
        vision=False,
        files=False,
        audio_input=False,
        audio_output=False,
        embeddings=False,
        reasoning=False,
        web_search=False,
    )


def _endpoint(base_url: str, *, name: str, allow_unsafe: bool) -> str:
    parsed = urlparse(base_url)
    if parsed.username or parsed.password:
        raise ModelResolutionError("invalid_configuration", f'Provider "{name}" base_url must not include credentials.')
    if parsed.scheme != "https" and not allow_unsafe:
        raise ModelResolutionError("invalid_configuration", f'Provider "{name}" base_url must use https.')
    if not parsed.netloc:
        raise ModelResolutionError("invalid_configuration", f'Provider "{name}" base_url must include a host.')
    return base_url.rstrip("/")


def _content(message: ModelMessage) -> str | list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, TextPart) and part.text:
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart) and part.image:
            url = part.image if part.image.startswith(("http://", "https://", "data:")) else f"data:{part.media_type or 'image/png'};base64,{part.image}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def _messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
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
        payload: dict[str, Any] = {"role": message.role, "content": _content(message)}
        tool_calls = [part.tool_call for part in message.parts if isinstance(part, ToolCallPart)]
        if tool_calls:
            payload["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.input)}}
                for call in tool_calls
            ]
        mapped.append(payload)
    return mapped


def _tools(provider: str, tools: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    if any(is_hosted_tool_definition(tool) for tool in tools.values()):
        raise UnsupportedFeatureError(f'Provider "{provider}" does not accept hosted tools on the Chat Completions route.')
    mapped = []
    for tool in tools.values():
        mapped.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or tool.name,
                "parameters": create_schema_adapter(tool.schema).json_schema(),
            },
        })
    return mapped


def _tool_choice(value: Any) -> Any:
    if isinstance(value, ToolChoiceName):
        return {"type": "function", "function": {"name": value.tool_name}}
    return value


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


def _assistant(provider: str, payload: Any) -> ModelMessage:
    if not isinstance(payload, dict):
        raise ParseError(f'Provider "{provider}" response did not contain an assistant message.')
    parts: list[Any] = []
    content = payload.get("content")
    if isinstance(content, str) and content:
        parts.append(TextPart(text=content))
    for call in payload.get("tool_calls") or []:
        if not isinstance(call, dict):
            raise ParseError(f'Provider "{provider}" returned a malformed tool call.')
        raw_function = call.get("function")
        function = raw_function if isinstance(raw_function, dict) else {}
        name = function.get("name")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
            raise ParseError(f'Provider "{provider}" tool calls require non-empty string id and function name fields.')
        arguments = function.get("arguments") or "{}"
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        parts.append(ToolCallPart(tool_call=ToolCall(id=call_id, name=name, input=parsed)))
    return ModelMessage(role="assistant", parts=parts)


class _ChatLanguageModel:
    def __init__(self, *, provider: str, model_id: str, api_key: str, base_url: str, fetch: Fetcher, headers: dict[str, str], capabilities: ModelCapabilities) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ModelResolutionError("invalid_identifier", f'Provider "{provider}" requires a non-empty model id.', {"provider": provider})
        self.provider = provider
        self.model_id = model_id
        self.capabilities = capabilities
        self._api_key = api_key
        self._base_url = base_url
        self._fetch = fetch
        self._headers = {"authorization": f"Bearer {api_key}", "content-type": "application/json", **headers}

    @property
    def base_url(self) -> str:
        """Configured destination used by both authorization and requests."""
        return self._base_url

    def _egress_request_url(self, *, stream: bool) -> str:
        return f"{self._base_url}/chat/completions"

    def _body(self, model_input: ModelGenerateInput, *, stream: bool) -> dict[str, Any]:
        if model_input.tools and not self.capabilities.tools:
            raise UnsupportedFeatureError(f'Model "{self.provider}/{self.model_id}" does not support tools.')
        options = dict(model_input.provider_options or {})
        for reserved in ("model", "messages", "stream"):
            options.pop(reserved, None)
        tools = _tools(self.provider, model_input.tools)
        body: dict[str, Any] = {
            **options,
            "model": self.model_id,
            "messages": _messages(model_input.messages),
            "stream": stream,
        }
        if model_input.temperature is not None:
            body["temperature"] = model_input.temperature
        if model_input.max_tokens is not None:
            body["max_tokens"] = model_input.max_tokens
        if tools is not None:
            body["tools"] = tools
        if model_input.tool_choice is not None:
            body["tool_choice"] = _tool_choice(model_input.tool_choice)
        if model_input.structured_output is not None and model_input.structured_output.mode == "native":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": model_input.structured_output.name or "response",
                    "schema": create_schema_adapter(model_input.structured_output.schema).json_schema(),
                },
            }
        return body

    async def _post(self, body: dict[str, Any], model_input: ModelGenerateInput, *, stream: bool) -> Any:
        response = await with_retry(
            lambda: self._fetch(
                self._egress_request_url(stream=stream),
                headers=self._headers,
                json_body=body,
                timeout_ms=model_input.timeout_ms,
                stream=stream,
            ),
            max_retries=model_input.max_retries or 0,
            retry_backoff_ms=model_input.retry_backoff_ms or 250,
        )
        if response.status_code >= 300:
            if response.status_code < 400:
                raise ProviderHTTPError("Credentialed provider request was redirected.", response.status_code)
            raise ProviderHTTPError(f'Provider "{self.provider}" request failed.', response.status_code, response_body=await response.text())
        return response

    async def generate(self, model_input: ModelGenerateInput) -> GenerateResult:
        response = await self._post(self._body(model_input, stream=False), model_input, stream=False)
        payload = await response.json()
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ParseError(f'Provider "{self.provider}" response did not contain a completion choice.')
        choice = choices[0]
        message = _assistant(self.provider, choice.get("message"))
        finish = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else None
        return GenerateResult(
            messages=[message],
            text="".join(part.text for part in message.parts if isinstance(part, TextPart)),
            finish_reason=normalize_finish_reason(finish),
            provider_finish_reason=finish,
            usage=_usage(payload.get("usage") if isinstance(payload, dict) else None),
            raw_response=payload,
        )

    async def stream(self, model_input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        response = await self._post(self._body(model_input, stream=True), model_input, stream=True)
        provider = self.provider

        async def generator() -> AsyncIterable[StreamEvent]:
            finish_reason = None
            usage = None
            buffers: dict[int, dict[str, str]] = {}
            async for event in parse_sse(response.iter_lines()):
                if event.data == "[DONE]":
                    break
                try:
                    payload = json.loads(event.data)
                except json.JSONDecodeError as error:
                    raise ParseError(f'Provider "{provider}" streaming response contained invalid JSON.') from error
                choice = (payload.get("choices") or [{}])[0] if isinstance(payload, dict) else {}
                delta = choice.get("delta") if isinstance(choice, dict) else None
                if isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
                    usage = payload["usage"]
                if isinstance(delta, dict) and isinstance(delta.get("content"), str) and delta["content"]:
                    yield StreamTextDeltaEvent(text_delta=delta["content"])
                calls = delta.get("tool_calls") if isinstance(delta, dict) else None
                for tool_call in calls or []:
                    if not isinstance(tool_call, dict) or isinstance(tool_call.get("index"), bool) or not isinstance(tool_call.get("index"), int):
                        raise ParseError(f'Provider "{provider}" streaming tool calls require an integer index.')
                    slot = buffers.setdefault(tool_call["index"], {"id": "", "name": "", "args": ""})
                    if isinstance(tool_call.get("id"), str):
                        slot["id"] += tool_call["id"]
                    raw_function = tool_call.get("function")
                    function = raw_function if isinstance(raw_function, dict) else {}
                    if isinstance(function.get("name"), str):
                        slot["name"] += function["name"]
                    if isinstance(function.get("arguments"), str):
                        slot["args"] += function["arguments"]
                if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str):
                    finish_reason = choice["finish_reason"]
            for index in sorted(buffers):
                slot = buffers[index]
                if not slot["id"] or not slot["name"]:
                    raise ParseError(f'Provider "{provider}" fragmented tool calls require non-empty id and function name fields.')
                yield StreamToolCallEvent(tool_call=ToolCall(id=slot["id"], name=slot["name"], input=json.loads(slot["args"] or "{}")))
            if not finish_reason:
                raise ParseError(f'Provider "{provider}" streaming response ended without a finish reason.')
            yield StreamFinishEvent(
                finish_reason=normalize_finish_reason(finish_reason),
                provider_finish_reason=finish_reason,
                usage=_usage(usage),
            )

        return generator()


def openai_compatible(
    *,
    name: str,
    base_url: str,
    api_key: str,
    headers: dict[str, str] | None = None,
    fetch: Fetcher | None = None,
    allow_unsafe_endpoints: bool = False,
    register: bool = False,
    capabilities: ModelCapabilities | None = None,
) -> ProviderAdapter:
    if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
        raise ModelResolutionError(
            "invalid_configuration",
            "openai_compatible requires a provider name of letters, numbers, dots, underscores, or hyphens.",
        )
    if not isinstance(api_key, str) or not api_key.strip():
        raise ModelResolutionError("missing_credentials", f'Provider "{name}" requires an api_key. No request was sent.', {"provider": name})
    resolved_base = _endpoint(base_url, name=name, allow_unsafe=allow_unsafe_endpoints)
    resolved_capabilities = _capabilities(capabilities)
    requester = fetch or default_fetch
    extra_headers = dict(headers or {})
    adapter = ProviderAdapter(
        name=name,
        language_model_factory=lambda model_id: _ChatLanguageModel(
            provider=name,
            model_id=model_id,
            api_key=api_key,
            base_url=resolved_base,
            fetch=requester,
            headers=extra_headers,
            capabilities=resolved_capabilities,
        ),
    )
    if register:
        register_model_provider(adapter)
    return adapter


def create_mistral_compatible(**options: Any) -> ProviderAdapter:
    base_url = options.pop("base_url", None) or _PRESETS["mistral"]
    return openai_compatible(name="mistral", base_url=base_url, **options)


def create_groq_compatible(**options: Any) -> ProviderAdapter:
    base_url = options.pop("base_url", None) or _PRESETS["groq"]
    return openai_compatible(name="groq", base_url=base_url, **options)
