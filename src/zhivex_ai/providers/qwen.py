from __future__ import annotations

import base64
from collections.abc import AsyncIterable
from copy import deepcopy
import json
import os
from dataclasses import dataclass, field
from dataclasses import replace
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

from .._http import Fetcher, default_fetch
from .._sse import parse_sse
from ..errors import ConfigurationError, ProviderHTTPError, UnsupportedFeatureError, ValidationError
from ..messages import (
    hosted_tool,
    is_hosted_tool_definition,
    normalize_finish_reason,
    validate_file_part,
    validate_message_parts,
)
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..types import (
    AgentCapabilities,
    AudioInput,
    FilePart,
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
    RetryOptions,
    SpeechModel,
    SpeechOutput,
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
    TranscriptionModel,
    TranscriptionOutput,
)
from .base import create_provider_bundle
from .openai_compat import (
    OPENAI_COMPAT_CAPABILITIES,
    OPENAI_COMPAT_SPEECH_CAPABILITIES,
    OPENAI_COMPAT_TRANSCRIPTION_CAPABILITIES,
    OpenAICompatibleBatchesClient,
    OpenAICompatibleFilesClient,
    OpenAICompatibleResponsesClient,
    _parse_json_error,
    create_openai_compatible_provider,
)
from ._url_security import validate_provider_url

QwenRegion = Literal["intl", "us", "cn"]

QWEN_REGION_BASE_URLS: dict[QwenRegion, str] = {
    "intl": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

QWEN_38_MAX_MODEL = "qwen3.8-max"
QWEN_38_MAX_REASONING_BUDGET = 262_144


def _qwen_base_url(region: QwenRegion) -> str:
    try:
        return QWEN_REGION_BASE_URLS[region]
    except KeyError as exc:
        supported = ", ".join(sorted(QWEN_REGION_BASE_URLS))
        raise ConfigurationError(f'Unsupported qwen region "{region}". Supported regions: {supported}.') from exc


def _qwen_responses_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _qwen_speech_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    for suffix in ("/api/v2/apps/protocols/compatible-mode/v1", "/compatible-mode/v1", "/compatible-mode", "/api/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            f"{path}/api/v1/services/aigc/multimodal-generation/generation",
            "",
            "",
            "",
        )
    )


def _qwen_audio_allowed_suffixes(base_url: str) -> tuple[str, ...]:
    host = (urlparse(base_url).hostname or "").strip(".").lower()
    suffixes = ["aliyuncs.com"]
    if host and host != "aliyuncs.com" and not host.endswith(".aliyuncs.com"):
        suffixes.append(host)
    return tuple(suffixes)


def _qwen_secure_audio_url(url: str, *, base_url: str, provider: str) -> str:
    candidate = str(url)
    parsed = urlparse(candidate)
    if parsed.scheme == "http":
        try:
            host = (parsed.hostname or "").strip(".").encode("idna").decode("ascii").lower()
            allowed_suffixes = tuple(
                suffix.strip(".").encode("idna").decode("ascii").lower()
                for suffix in _qwen_audio_allowed_suffixes(base_url)
            )
        except UnicodeError:
            host = ""
            allowed_suffixes = ()
        if host and any(host == suffix or host.endswith(f".{suffix}") for suffix in allowed_suffixes):
            candidate = parsed._replace(scheme="https").geturl()
    return validate_provider_url(
        candidate,
        provider=provider,
        purpose="audio download",
        allowed_suffixes=_qwen_audio_allowed_suffixes(base_url),
    )


def _infer_qwen_media_type(url: str | None) -> str:
    normalized = (url or "").lower()
    if normalized.endswith(".mp3"):
        return "audio/mpeg"
    if normalized.endswith(".ogg") or normalized.endswith(".opus"):
        return "audio/ogg"
    if normalized.endswith(".pcm"):
        return "audio/pcm"
    return "audio/wav"


def _is_qwen_38_max(model_id: str) -> bool:
    return model_id.strip().lower() == QWEN_38_MAX_MODEL


def _qwen_chat_headers(api_key: str) -> dict[str, str]:
    return {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }


def _qwen_chat_tool_result(part: ToolResultPart) -> str:
    result = part.tool_result
    if result.is_error and result.error is not None:
        return json.dumps({"message": result.error.message})
    return json.dumps(result.output)


def _qwen_chat_provider_data(message: ModelMessage, key: str) -> Any:
    values: list[Any] = []
    for part in message.parts:
        if (
            isinstance(part, ProviderDataPart)
            and part.provider == "qwen"
            and isinstance(part.data, dict)
            and key in part.data
        ):
            values.append(part.data[key])
    if not values:
        return None
    if all(isinstance(value, str) for value in values):
        return "".join(values)
    return deepcopy(values[0])


def _qwen_chat_file_content(part: FilePart) -> dict[str, Any]:
    validate_file_part(part)
    media_type = (part.media_type or "").strip().lower()
    if not media_type.startswith(("image/", "video/")):
        raise ValidationError(
            'Qwen3.8-Max FilePart inputs require an image/* or video/* "media_type".'
        )
    if part.url is not None:
        url = part.url
    elif part.data is not None:
        url = f"data:{media_type};base64,{part.data}"
    else:
        raise ValidationError(
            "Qwen3.8-Max image/video FilePart inputs currently require url=... or inline data=...."
        )
    if media_type.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": url}}
    return {"type": "video_url", "video_url": {"url": url}}


def _qwen_chat_message_content(message: ModelMessage) -> str | list[dict[str, Any]] | None:
    text_chunks: list[str] = []
    content: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            text_chunks.append(part.text)
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            content.append({"type": "image_url", "image_url": {"url": part.image}})
        elif isinstance(part, FilePart):
            content.append(_qwen_chat_file_content(part))
    if not content:
        return None
    if all(item.get("type") == "text" for item in content):
        return "".join(text_chunks)
    return content


def _qwen_chat_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            for part in message.parts:
                if isinstance(part, ToolResultPart):
                    mapped.append(
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_result.tool_call_id,
                            "content": _qwen_chat_tool_result(part),
                        }
                    )
            continue

        payload: dict[str, Any] = {"role": message.role}
        content = _qwen_chat_message_content(message)
        if content is not None:
            payload["content"] = content
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
            payload.setdefault("content", "")
        reasoning_content = _qwen_chat_provider_data(message, "reasoning_content")
        if reasoning_content is not None:
            payload["reasoning_content"] = reasoning_content
        if "content" in payload or "tool_calls" in payload:
            mapped.append(payload)
    return mapped


def _qwen_chat_tools(tools: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    mapped: list[dict[str, Any]] = []
    for definition in tools.values():
        if is_hosted_tool_definition(definition):
            raise UnsupportedFeatureError(
                "Qwen hosted tools require the Responses transport and cannot be combined with "
                "Chat-only structured output, reasoning budgets, or video input."
            )
        function: dict[str, Any] = {
            "name": definition.name,
            "description": definition.description,
            "parameters": create_schema_adapter(definition.schema).json_schema(),
        }
        if definition.strict is not None:
            function["strict"] = definition.strict
        mapped.append({"type": "function", "function": function})
    return mapped


def _qwen_chat_tool_choice(tool_choice: str | ToolChoiceName | None) -> str | dict[str, Any] | None:
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    return {"type": "function", "function": {"name": tool_choice.tool_name}}


def _qwen_38_reasoning_options(
    input: ModelGenerateInput,
    provider_options: dict[str, Any],
    *,
    structured_output: bool,
) -> dict[str, Any]:
    configured = {
        name: provider_options.get(name)
        for name in ("enable_thinking", "reasoning_effort", "thinking_budget")
        if name in provider_options
    }
    if input.reasoning is not None and configured:
        names = ", ".join(sorted(configured))
        raise ValidationError(
            f"Pass either reasoning=... or Qwen provider options ({names}), not both."
        )
    if "reasoning_effort" in configured and "thinking_budget" in configured:
        raise ValidationError(
            'Qwen3.8-Max does not accept "reasoning_effort" and "thinking_budget" together.'
        )

    if structured_output:
        if input.max_tokens is not None:
            raise UnsupportedFeatureError(
                "Qwen recommends leaving max_tokens unset for structured output to avoid truncated JSON."
            )
        if input.reasoning is not None and (
            input.reasoning.budget_tokens is not None or input.reasoning.effort != "none"
        ):
            raise UnsupportedFeatureError(
                'Qwen3.8-Max structured output requires reasoning=ReasoningConfig(effort="none") '
                "or no reasoning configuration."
            )
        if any(
            value not in {None, False, "none"}
            for value in configured.values()
        ):
            raise UnsupportedFeatureError(
                "Qwen3.8-Max structured output is available only with thinking disabled."
            )
        provider_options.pop("reasoning_effort", None)
        provider_options.pop("thinking_budget", None)
        provider_options["enable_thinking"] = False
        return {}

    if input.reasoning is None:
        return {}
    if input.reasoning.effort is not None and input.reasoning.budget_tokens is not None:
        raise ValidationError(
            'Qwen3.8-Max does not accept "reasoning.effort" and "reasoning.budget_tokens" together.'
        )
    if input.reasoning.budget_tokens is not None:
        if input.reasoning.budget_tokens > QWEN_38_MAX_REASONING_BUDGET:
            raise UnsupportedFeatureError(
                f"Qwen3.8-Max reasoning budget cannot exceed {QWEN_38_MAX_REASONING_BUDGET} tokens."
            )
        return {
            "enable_thinking": True,
            "thinking_budget": input.reasoning.budget_tokens,
        }
    effort = input.reasoning.effort
    if effort == "none":
        return {"enable_thinking": False}
    if effort is None:
        return {}
    mapped_effort = {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "xhigh",
        "xhigh": "xhigh",
        "max": "xhigh",
    }.get(effort)
    return {"enable_thinking": True, "reasoning_effort": mapped_effort} if mapped_effort else {}


def _qwen_38_explicit_thinking_enabled(
    provider_options: dict[str, Any],
    reasoning_options: dict[str, Any],
    *,
    structured_output: bool,
) -> bool:
    if structured_output:
        return False
    if reasoning_options.get("enable_thinking") is True:
        return True
    enable_thinking = provider_options.get("enable_thinking")
    if enable_thinking is not None and enable_thinking is not False:
        return True
    reasoning_effort = provider_options.get("reasoning_effort")
    if reasoning_effort is not None and reasoning_effort != "none":
        return True
    return provider_options.get("thinking_budget") is not None


def _qwen_chat_response_format(input: ModelGenerateInput) -> dict[str, Any] | None:
    structured = input.structured_output
    if structured is None or structured.mode != "native":
        return None
    schema_config: dict[str, Any] = {
        "name": structured.name or "response",
        "strict": True,
        "schema": create_schema_adapter(structured.schema).json_schema(),
    }
    if structured.description is not None:
        schema_config["description"] = structured.description
    return {
        "type": "json_schema",
        "json_schema": schema_config,
    }


def _qwen_chat_body(model_id: str, input: ModelGenerateInput, *, stream: bool) -> dict[str, Any]:
    provider_options = deepcopy(input.provider_options or {})
    reserved = {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "response_format",
        "max_completion_tokens",
        "temperature",
        "stream",
    }.intersection(provider_options)
    if reserved:
        raise ValidationError(
            "Qwen provider_options cannot override SDK-owned fields: " + ", ".join(sorted(reserved)) + "."
        )
    structured_output = input.structured_output is not None and input.structured_output.mode == "native"
    reasoning_options = _qwen_38_reasoning_options(
        input,
        provider_options,
        structured_output=structured_output,
    )
    forced_tool_choice = input.tool_choice == "required" or isinstance(input.tool_choice, ToolChoiceName)
    if forced_tool_choice:
        if _qwen_38_explicit_thinking_enabled(
            provider_options,
            reasoning_options,
            structured_output=structured_output,
        ):
            raise UnsupportedFeatureError(
                "Qwen3.8-Max cannot force a required or named tool while thinking is enabled. "
                'Use reasoning=ReasoningConfig(effort="none") or tool_choice="auto".'
            )
        reasoning_options.setdefault("enable_thinking", False)
    stream_options = deepcopy(provider_options.pop("stream_options", None))
    if stream:
        if stream_options is None:
            stream_options = {}
        if not isinstance(stream_options, dict):
            raise ValidationError('Qwen provider option "stream_options" must be an object.')
        stream_options.setdefault("include_usage", True)
    body = {
        "model": model_id,
        "messages": _qwen_chat_messages(input.messages),
        "tools": _qwen_chat_tools(input.tools),
        "tool_choice": _qwen_chat_tool_choice(input.tool_choice),
        "temperature": input.temperature,
        "max_completion_tokens": input.max_tokens,
        "response_format": _qwen_chat_response_format(input),
        **provider_options,
        **reasoning_options,
        "stream": True if stream else None,
        "stream_options": stream_options,
    }
    return {key: value for key, value in body.items() if value is not None}


def _qwen_chat_usage(payload: dict[str, Any]) -> TokenUsage | None:
    usage = payload.get("usage") or {}
    if not usage:
        return None
    return TokenUsage(
        input_tokens=usage.get("prompt_tokens") or usage.get("input_tokens"),
        output_tokens=usage.get("completion_tokens") or usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


def _qwen_chat_tool_call(value: dict[str, Any]) -> ToolCall:
    function = value.get("function") or {}
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            parsed_arguments: Any = json.loads(arguments)
        except json.JSONDecodeError:
            parsed_arguments = arguments
    else:
        parsed_arguments = deepcopy(arguments) if arguments is not None else {}
    return ToolCall(
        id=str(value.get("id") or ""),
        name=str(function.get("name") or ""),
        input=parsed_arguments,
        provider_metadata={"provider": "qwen", "raw_tool_call": deepcopy(value)},
    )


def _qwen_chat_message(payload: dict[str, Any]) -> ModelMessage:
    parts: list[Any] = []
    content = payload.get("content")
    if isinstance(content, str) and content:
        parts.append(TextPart(text=content))
    reasoning_content = payload.get("reasoning_content")
    if reasoning_content is not None:
        parts.append(
            ProviderDataPart(
                provider="qwen",
                data={"reasoning_content": reasoning_content},
            )
        )
    for item in payload.get("tool_calls") or []:
        if isinstance(item, dict):
            parts.append(ToolCallPart(tool_call=_qwen_chat_tool_call(item)))
    return ModelMessage(role="assistant", parts=parts)


@dataclass(slots=True)
class QwenChatLanguageModel(LanguageModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        validate_message_parts(self, input.messages)
        body = _qwen_chat_body(self.model_id, input, stream=False)
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/chat/completions",
                headers=_qwen_chat_headers(self.api_key),
                json_body=body,
                timeout_ms=input.timeout_ms,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        choice = ((payload.get("choices") or [{}])[0] or {})
        message = _qwen_chat_message(choice.get("message") or {})
        finish_reason = choice.get("finish_reason")
        return GenerateResult(
            messages=[message],
            text="".join(part.text for part in message.parts if isinstance(part, TextPart)),
            finish_reason=normalize_finish_reason(finish_reason),
            provider_finish_reason=finish_reason,
            usage=_qwen_chat_usage(payload),
            raw_response=payload,
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        validate_message_parts(self, input.messages)
        body = _qwen_chat_body(self.model_id, input, stream=True)
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/chat/completions",
                headers=_qwen_chat_headers(self.api_key),
                json_body=body,
                timeout_ms=input.timeout_ms,
                stream=True,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())

        async def generator() -> AsyncIterable[StreamEvent]:
            tool_calls: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None
            usage: dict[str, Any] | None = None

            async def terminal_events() -> AsyncIterable[StreamEvent]:
                for item in tool_calls.values():
                    yield StreamToolCallEvent(tool_call=_qwen_chat_tool_call(item))
                if finish_reason is not None or usage is not None:
                    yield StreamFinishEvent(
                        finish_reason=normalize_finish_reason(finish_reason),
                        provider_finish_reason=finish_reason,
                        usage=_qwen_chat_usage({"usage": usage or {}}),
                    )

            async for event in parse_sse(response.iter_lines()):
                if event.data == "[DONE]":
                    async for terminal in terminal_events():
                        yield terminal
                    return
                payload = json.loads(event.data)
                if isinstance(payload.get("usage"), dict):
                    usage = dict(payload["usage"])
                choices = payload.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] or {}
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    yield StreamTextDeltaEvent(text_delta=str(delta["content"]))
                if delta.get("reasoning_content") is not None:
                    yield StreamProviderDataEvent(
                        provider="qwen",
                        data={"reasoning_content": delta.get("reasoning_content")},
                    )
                for item in delta.get("tool_calls") or []:
                    index = int(item.get("index") or 0)
                    current = tool_calls.setdefault(
                        index,
                        {"id": "", "function": {"name": "", "arguments": ""}},
                    )
                    if item.get("id"):
                        current["id"] = item["id"]
                    function = item.get("function") or {}
                    current_function = current.setdefault(
                        "function",
                        {"name": "", "arguments": ""},
                    )
                    if function.get("name"):
                        current_function["name"] = function["name"]
                    if function.get("arguments"):
                        current_function["arguments"] = (
                            str(current_function.get("arguments") or "")
                            + str(function["arguments"])
                        )
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])

            async for terminal in terminal_events():
                yield terminal

        return generator()


def _qwen_38_requires_chat(input: ModelGenerateInput) -> bool:
    if input.structured_output is not None and input.structured_output.mode == "native":
        return True
    if input.reasoning is not None and input.reasoning.budget_tokens is not None:
        return True
    return any(isinstance(part, FilePart) for message in input.messages for part in message.parts)


@dataclass(slots=True)
class Qwen38LanguageModel(LanguageModel):
    provider: str
    model_id: str
    responses_model: LanguageModel
    chat_model: LanguageModel
    capabilities: ModelCapabilities

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        model = self.chat_model if _qwen_38_requires_chat(input) else self.responses_model
        return await model.generate(input)

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        model = self.chat_model if _qwen_38_requires_chat(input) else self.responses_model
        return await model.stream(input)


def qwen_hosted_tool(
    type: str,
    *,
    name: str | None = None,
    tool_class: HostedToolClass | None = None,
    **config: Any,
) -> HostedToolDefinition:
    return hosted_tool(
        name=name or type,
        provider="qwen",
        type=type,
        config=config or None,
        tool_class=tool_class,
    )


def qwen_web_search_tool(**config: Any) -> HostedToolDefinition:
    return qwen_hosted_tool("web_search", tool_class="web-search", **config)


def qwen_web_extractor_tool(**config: Any) -> HostedToolDefinition:
    return qwen_hosted_tool("web_extractor", tool_class="web-search", **config)


def qwen_code_interpreter_tool(**config: Any) -> HostedToolDefinition:
    return qwen_hosted_tool("code_interpreter", tool_class="code-execution", **config)


def qwen_file_search_tool(*, vector_store_ids: list[str], **config: Any) -> HostedToolDefinition:
    return qwen_hosted_tool(
        "file_search",
        tool_class="file-search",
        vector_store_ids=list(vector_store_ids),
        **config,
    )


def qwen_web_search_image_tool(**config: Any) -> HostedToolDefinition:
    return qwen_hosted_tool("web_search_image", tool_class="web-search", **config)


def qwen_image_search_tool(**config: Any) -> HostedToolDefinition:
    return qwen_hosted_tool("image_search", **config)


def qwen_mcp_tool(
    *,
    server_label: str,
    server_url: str,
    server_protocol: str = "sse",
    server_description: str | None = None,
    headers: dict[str, str] | None = None,
    **config: Any,
) -> HostedToolDefinition:
    return qwen_hosted_tool(
        "mcp",
        tool_class="remote-mcp",
        server_label=server_label,
        server_url=server_url,
        server_protocol=server_protocol,
        server_description=server_description,
        headers=headers,
        **config,
    )


def _qwen_asr_audio_data(audio: AudioInput) -> str:
    if isinstance(audio.data, str):
        if audio.data.startswith(("http://", "https://", "data:")):
            return audio.data
        return f"data:{audio.media_type};base64,{audio.data}"
    if isinstance(audio.data, memoryview):
        raw = audio.data.tobytes()
    else:
        raw = bytes(audio.data)
    return f"data:{audio.media_type};base64,{base64.b64encode(raw).decode('ascii')}"


@dataclass(slots=True)
class QwenTranscriptionModel(TranscriptionModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: OPENAI_COMPAT_TRANSCRIPTION_CAPABILITIES)

    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }

    async def transcribe(
        self,
        *,
        audio: AudioInput,
        prompt: str | None = None,
        language: str | None = None,
        provider_options: dict[str, Any] | None = None,
        options: RetryOptions | None = None,
    ) -> TranscriptionOutput:
        input_options = dict(provider_options or {})
        asr_options = dict(input_options.pop("asr_options", {}))
        if language:
            asr_options["language"] = language
        messages: list[dict[str, Any]] = []
        if prompt:
            messages.append({"role": "system", "content": [{"text": prompt}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": _qwen_asr_audio_data(audio)},
                    }
                ],
            }
        )
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json_body={
                    "model": self.model_id,
                    "messages": messages,
                    "asr_options": asr_options or None,
                    **input_options,
                },
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f'Provider "{self.provider}" transcription request failed with status {response.status_code}.',
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        message = (((payload.get("choices") or [{}])[0]).get("message") or {})
        return TranscriptionOutput(text=str(message.get("content") or ""), audio=audio, raw_response=payload)


@dataclass(slots=True)
class QwenSpeechModel(SpeechModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: OPENAI_COMPAT_SPEECH_CAPABILITIES)

    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }

    async def generate_speech(
        self,
        *,
        input: str,
        voice: str | None = None,
        provider_options: dict[str, Any] | None = None,
        options: RetryOptions | None = None,
    ) -> SpeechOutput:
        input_options = dict(provider_options or {})
        response = await with_retry(
            lambda: self.fetch(
                _qwen_speech_url(self.base_url),
                headers=self._headers(),
                json_body={
                    "model": self.model_id,
                    "input": {
                        **input_options,
                        "text": input,
                        "voice": voice or input_options.get("voice") or "Cherry",
                    },
                },
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f'Provider "{self.provider}" request failed with status {response.status_code}.',
                response.status_code,
                response_body=await response.text(),
            )

        payload = await response.json()
        audio_info = ((payload.get("output") or {}).get("audio") or {})
        if isinstance(audio_info.get("url"), str) and audio_info.get("url"):
            audio_url = _qwen_secure_audio_url(
                str(audio_info["url"]),
                base_url=self.base_url,
                provider=self.provider,
            )
            audio_response = await with_retry(
                lambda: self.fetch(
                    audio_url,
                    method="GET",
                    headers={},
                    json_body=None,
                    body=None,
                    timeout_ms=options.timeout_ms if options else None,
                ),
                max_retries=options.max_retries if options and options.max_retries is not None else 0,
                retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
            )
            if audio_response.status_code >= 400:
                raise ProviderHTTPError(
                    f'Provider "{self.provider}" audio download failed with status {audio_response.status_code}.',
                    audio_response.status_code,
                    response_body=await audio_response.text(),
                )
            return SpeechOutput(
                audio=await audio_response.read(),
                media_type=audio_response.headers.get("content-type", _infer_qwen_media_type(str(audio_info.get("url")))),
                raw_response=payload,
            )

        if isinstance(audio_info.get("data"), str) and audio_info.get("data"):
            return SpeechOutput(
                audio=base64.b64decode(str(audio_info.get("data"))),
                media_type=_infer_qwen_media_type(str(audio_info.get("url"))),
                raw_response=payload,
            )

        raise ValidationError('Provider "qwen" did not return audio data for speech generation.')


def create_qwen(
    *,
    api_key: str | None = None,
    region: QwenRegion = "intl",
    base_url: str | None = None,
    responses_base_url: str | None = None,
    fetch: Fetcher | None = None,
):
    resolved_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not resolved_key:
        raise ConfigurationError("Missing qwen API key.")
    requester = fetch or default_fetch
    resolved_base_url = (base_url or _qwen_base_url(region)).rstrip("/")
    resolved_responses_base_url = (responses_base_url or _qwen_responses_base_url(resolved_base_url)).rstrip("/")
    capabilities = replace(
        OPENAI_COMPAT_CAPABILITIES,
        tools=True,
        tool_choice=True,
        parallel_tool_calls=False,
        web_search=True,
    )
    native = create_openai_compatible_provider(
        provider_name="qwen",
        env_var="QWEN_API_KEY",
        api_key=resolved_key,
        base_url=resolved_base_url,
        responses_base_url=resolved_responses_base_url,
        fetch=requester,
        capabilities=capabilities,
        supports_grounding=True,
        default_grounding_tool={"type": "web_search"},
        files_client_factory=lambda: OpenAICompatibleFilesClient(
            provider="qwen",
            api_key=resolved_key,
            base_url=resolved_base_url,
            fetch=requester,
            default_purpose="batch",
        ),
        batches_client_factory=lambda: OpenAICompatibleBatchesClient(
            provider="qwen",
            api_key=resolved_key,
            base_url=resolved_base_url,
            fetch=requester,
        ),
        responses_client_factory=lambda: OpenAICompatibleResponsesClient(
            provider="qwen",
            model_id="",
            api_key=resolved_key,
            base_url=resolved_responses_base_url,
            fetch=requester,
        ),
    )
    responses_language_model_factory = native.language_model_factory
    shared_capabilities = native.language_model("").capabilities

    def qwen_language_model(model_id: str) -> LanguageModel:
        responses_model = responses_language_model_factory(model_id)
        if not _is_qwen_38_max(model_id):
            return responses_model
        qwen_38_capabilities = replace(shared_capabilities, files=True)
        return Qwen38LanguageModel(
            provider="qwen",
            model_id=model_id,
            responses_model=responses_model,
            chat_model=QwenChatLanguageModel(
                provider="qwen",
                model_id=model_id,
                api_key=resolved_key,
                base_url=resolved_base_url,
                fetch=requester,
                capabilities=qwen_38_capabilities,
            ),
            capabilities=qwen_38_capabilities,
        )

    native = replace(
        native,
        language_model_factory=qwen_language_model,
        transcription_model_factory=lambda model_id: QwenTranscriptionModel(
            provider="qwen",
            model_id=model_id,
            api_key=resolved_key,
            base_url=resolved_base_url,
            fetch=requester,
        ),
        speech_model_factory=lambda model_id: QwenSpeechModel(
            provider="qwen",
            model_id=model_id,
            api_key=resolved_key,
            base_url=resolved_base_url,
            fetch=requester,
        ),
    )
    return create_provider_bundle(
        name="qwen",
        native=native,
        agent_capabilities=native.language_model("").capabilities.agent_capabilities or AgentCapabilities(),
        portable_support=PortableSupport(
            text_generation=True,
            streaming=True,
            structured_output=True,
            tools=True,
            embeddings=True,
            grounding=False,
            retrieval=False,
            transcription=False,
            speech=False,
            portable_badge=True,
            tier="portable",
        ),
    )
