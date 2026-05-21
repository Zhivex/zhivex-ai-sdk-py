from __future__ import annotations

from copy import deepcopy
import json
import os
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any

from .._http import Fetcher, default_fetch
from .._sse import parse_sse
from ..errors import ConfigurationError, UnsupportedFeatureError, ValidationError
from ..messages import normalize_finish_reason, tool, validate_file_part, validate_message_parts
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..types import (
    AgentCapabilities,
    CountTokensResult,
    FilePart,
    GenerateResult,
    ImagePart,
    LanguageModel,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    PortableSupport,
    ProviderDataPart,
    ProviderFile,
    RetryOptions,
    StreamEvent,
    StreamFinishEvent,
    StreamProviderDataEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    TextPart,
    TokenCountDetail,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolChoiceName,
    ToolDefinition,
    ToolResultPart,
)
from .base import ProviderAdapter, create_provider_bundle
from .openai_compat import OpenAICompatibleBatchesClient, OpenAICompatibleFilesClient, _normalize_binary, _parse_json_error


KIMI_DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
KIMI_OFFICIAL_TOOL_URIS: tuple[str, ...] = (
    "moonshot/convert:latest",
    "moonshot/web-search:latest",
    "moonshot/rethink:latest",
    "moonshot/random-choice:latest",
    "moonshot/memory:latest",
    "moonshot/excel:latest",
    "moonshot/date:latest",
    "moonshot/base64:latest",
    "moonshot/fetch:latest",
    "moonshot/quickjs:latest",
    "moonshot/code_runner:latest",
)


KIMI_AGENT_CAPABILITIES = AgentCapabilities(support_tier="tier-b", tool_choice_none=True, toolsets=True)

KIMI_CHAT_CAPABILITIES = ModelCapabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    parallel_tool_calls=True,
    vision=True,
    files=True,
    audio_input=False,
    audio_output=False,
    embeddings=False,
    reasoning=True,
    web_search=False,
    agent_capabilities=KIMI_AGENT_CAPABILITIES,
)


def _headers(api_key: str, *, json_content: bool = True) -> dict[str, str]:
    headers = {"authorization": f"Bearer {api_key}"}
    if json_content:
        headers["content-type"] = "application/json"
    return headers


def _is_kimi_multimodal_model(model_id: str) -> bool:
    normalized = model_id.strip().lower()
    return normalized in {"kimi-k2.6", "kimi-k2.5"}


def _is_image_media(media_type: str | None) -> bool:
    return (media_type or "").lower().startswith("image/")


def _is_video_media(media_type: str | None) -> bool:
    return (media_type or "").lower().startswith("video/")


def _image_content(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def _video_content(url: str) -> dict[str, Any]:
    return {"type": "video_url", "video_url": {"url": url}}


def _file_part_content(part: FilePart) -> dict[str, Any]:
    validate_file_part(part)
    if part.file_id is not None:
        if _is_image_media(part.media_type):
            return _image_content(f"ms://{part.file_id}")
        if _is_video_media(part.media_type):
            return _video_content(f"ms://{part.file_id}")
        raise ValidationError('Kimi FilePart(file_id=...) inputs require image/* or video/* "media_type".')
    if part.file_uri is not None and part.file_uri.startswith("ms://"):
        if _is_image_media(part.media_type):
            return _image_content(part.file_uri)
        if _is_video_media(part.media_type):
            return _video_content(part.file_uri)
    if part.data is not None and (_is_image_media(part.media_type) or _is_video_media(part.media_type)):
        data_url = f"data:{part.media_type};base64,{part.data}"
        return _image_content(data_url) if _is_image_media(part.media_type) else _video_content(data_url)
    raise ValidationError(
        "Kimi chat inputs only map ImagePart values and FilePart image/video references. "
        "For file-extract Q&A, upload with provider.files() and pass the downloaded text as a system message."
    )


def _message_content(message: ModelMessage) -> str | list[dict[str, Any]] | None:
    text_chunks: list[str] = []
    content: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            text_chunks.append(part.text)
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            content.append(_image_content(part.image))
        elif isinstance(part, FilePart):
            content.append(_file_part_content(part))

    if not content:
        return None
    if all(item.get("type") == "text" for item in content):
        return "".join(text_chunks)
    return content


def _tool_result_content(part: ToolResultPart) -> str:
    result = part.tool_result
    if result.is_error and result.error is not None:
        return json.dumps({"message": result.error.message})
    return json.dumps(result.output)


def _provider_data(message: ModelMessage, key: str) -> Any:
    for part in message.parts:
        if isinstance(part, ProviderDataPart) and part.provider == "kimi" and isinstance(part.data, dict) and key in part.data:
            return deepcopy(part.data[key])
    return None


def _to_chat_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            for part in message.parts:
                if isinstance(part, ToolResultPart):
                    mapped.append(
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_result.tool_call_id,
                            "content": _tool_result_content(part),
                        }
                    )
            continue

        payload: dict[str, Any] = {"role": message.role}
        content = _message_content(message)
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
        reasoning_content = _provider_data(message, "reasoning_content")
        if reasoning_content:
            payload["reasoning_content"] = reasoning_content
        if "content" in payload or "tool_calls" in payload:
            mapped.append(payload)
    return mapped


def _map_tools(tools: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    mapped = []
    for definition in tools.values():
        if getattr(definition, "kind", None) == "hosted":
            provider = getattr(definition, "provider", None)
            if provider not in {None, "kimi"}:
                raise ValidationError(
                    f'Hosted tool "{getattr(definition, "name", "")}" targets provider "{provider}", but this model uses "kimi".'
                )
            payload = {"type": getattr(definition, "type", "")}
            config = getattr(definition, "config", None)
            if isinstance(config, dict):
                payload.update(deepcopy(config))
            elif config is not None:
                payload["config"] = deepcopy(config)
            mapped.append(payload)
            continue

        parameters = create_schema_adapter(definition.schema).json_schema()
        function = {
            "name": definition.name,
            "description": definition.description,
            "parameters": parameters,
        }
        if definition.strict is not None:
            function["strict"] = definition.strict
        mapped.append({"type": "function", "function": function})
    return mapped


def _map_tool_choice(tool_choice: str | ToolChoiceName | None) -> str | dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    return {"type": "function", "function": {"name": tool_choice.tool_name}}


def _thinking_from_options(provider_options: dict[str, Any]) -> dict[str, Any] | None:
    value = provider_options.get("thinking")
    return dict(value) if isinstance(value, dict) else None


def _thinking_enabled(input: ModelGenerateInput, model_id: str, provider_options: dict[str, Any]) -> bool:
    if not _is_kimi_multimodal_model(model_id):
        return False
    explicit = _thinking_from_options(provider_options)
    if explicit is not None:
        return explicit.get("type") != "disabled"
    if input.reasoning is None:
        return True
    return input.reasoning.effort != "none"


def _kimi_reasoning_options(input: ModelGenerateInput, provider_options: dict[str, Any]) -> dict[str, Any]:
    if input.reasoning is None:
        return {}
    if input.reasoning.budget_tokens is not None:
        raise UnsupportedFeatureError('Provider "kimi" does not support "reasoning.budgetTokens".')
    if "thinking" in provider_options:
        raise ValidationError('Pass either reasoning=... or provider_options={"thinking": ...}, not both.')
    return {"thinking": {"type": "disabled" if input.reasoning.effort == "none" else "enabled"}}


def _validate_kimi_request(model_id: str, input: ModelGenerateInput, provider_options: dict[str, Any]) -> None:
    if _is_kimi_multimodal_model(model_id):
        blocked = {"temperature", "top_p", "n", "presence_penalty", "frequency_penalty"}
        if input.temperature is not None:
            raise UnsupportedFeatureError(f'Provider "kimi" does not allow overriding "temperature" for model "{model_id}".')
        configured = sorted(blocked.intersection(provider_options))
        if configured:
            joined = ", ".join(configured)
            raise UnsupportedFeatureError(f'Provider "kimi" does not allow overriding {joined} for model "{model_id}".')
        if _thinking_enabled(input, model_id, provider_options) and isinstance(input.tool_choice, ToolChoiceName):
            raise UnsupportedFeatureError('Kimi thinking mode only supports tool_choice="auto" or tool_choice="none".')
        if _thinking_enabled(input, model_id, provider_options) and input.tool_choice == "required":
            raise UnsupportedFeatureError('Kimi thinking mode only supports tool_choice="auto" or tool_choice="none".')


def _response_format(input: ModelGenerateInput) -> dict[str, Any] | None:
    if input.structured_output is None or input.structured_output.mode != "native":
        return None
    return {"type": "json_object"}


def _chat_body(model_id: str, input: ModelGenerateInput, *, stream: bool) -> dict[str, Any]:
    provider_options = deepcopy(input.provider_options or {})
    _validate_kimi_request(model_id, input, provider_options)
    tools = _map_tools(input.tools)
    body = {
        "model": model_id,
        "messages": _to_chat_messages(input.messages),
        "tools": tools,
        "tool_choice": _map_tool_choice(input.tool_choice),
        "max_completion_tokens": input.max_tokens,
        "response_format": _response_format(input),
        **provider_options,
        **_kimi_reasoning_options(input, provider_options),
        "stream": True if stream else None,
    }
    return {key: value for key, value in body.items() if value is not None}


def _parse_usage(payload: dict[str, Any]) -> TokenUsage | None:
    usage = payload.get("usage") or {}
    if not usage:
        return None
    return TokenUsage(
        input_tokens=usage.get("prompt_tokens") or usage.get("input_tokens"),
        output_tokens=usage.get("completion_tokens") or usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


def _parse_tool_call(value: dict[str, Any]) -> ToolCall:
    function = value.get("function") or {}
    return ToolCall(
        id=str(value.get("id") or ""),
        name=str(function.get("name") or ""),
        input=_parse_tool_arguments(function.get("arguments")),
        provider_metadata={"provider": "kimi", "raw_tool_call": deepcopy(value)},
    )


def _parse_tool_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return deepcopy(value) if value is not None else {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_message(payload: dict[str, Any]) -> ModelMessage:
    parts: list[Any] = []
    content = payload.get("content")
    if isinstance(content, str) and content:
        parts.append(TextPart(text=content))
    reasoning_content = payload.get("reasoning_content")
    if reasoning_content:
        parts.append(ProviderDataPart(provider="kimi", data={"reasoning_content": reasoning_content}))
    for item in payload.get("tool_calls") or []:
        if isinstance(item, dict):
            parts.append(ToolCallPart(tool_call=_parse_tool_call(item)))
    return ModelMessage(role="assistant", parts=parts)


def _parse_generate_payload(payload: dict[str, Any]) -> GenerateResult:
    choice = ((payload.get("choices") or [{}])[0] or {})
    message_payload = choice.get("message") or {}
    message = _parse_message(message_payload)
    finish = choice.get("finish_reason")
    return GenerateResult(
        messages=[message],
        text="".join(part.text for part in message.parts if isinstance(part, TextPart)),
        finish_reason=normalize_finish_reason(finish),
        provider_finish_reason=finish,
        usage=_parse_usage(payload),
        raw_response=payload,
    )


@dataclass(slots=True)
class KimiChatLanguageModel(LanguageModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: KIMI_CHAT_CAPABILITIES)

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        validate_message_parts(self, input.messages)
        body = _chat_body(self.model_id, input, stream=False)
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/chat/completions",
                headers=_headers(self.api_key),
                json_body=body,
                timeout_ms=input.timeout_ms,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _parse_generate_payload(await response.json())

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        validate_message_parts(self, input.messages)
        body = _chat_body(self.model_id, input, stream=True)
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/chat/completions",
                headers=_headers(self.api_key),
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
            tool_call_accumulator: dict[int, dict[str, Any]] = {}
            async for event in parse_sse(response.iter_lines()):
                if event.data == "[DONE]":
                    return
                payload = json.loads(event.data)
                choice = ((payload.get("choices") or [{}])[0] or {})
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    yield StreamTextDeltaEvent(text_delta=str(delta.get("content") or ""))
                if delta.get("reasoning_content"):
                    yield StreamProviderDataEvent(provider="kimi", data={"reasoning_content": delta.get("reasoning_content")})
                for item in delta.get("tool_calls") or []:
                    index = int(item.get("index") or 0)
                    current = tool_call_accumulator.setdefault(index, {"id": "", "function": {"name": "", "arguments": ""}})
                    if item.get("id"):
                        current["id"] = item["id"]
                    function = item.get("function") or {}
                    current_function = current.setdefault("function", {"name": "", "arguments": ""})
                    if function.get("name"):
                        current_function["name"] = function["name"]
                    if function.get("arguments"):
                        current_function["arguments"] = str(current_function.get("arguments") or "") + str(function.get("arguments") or "")
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    for current in tool_call_accumulator.values():
                        yield StreamToolCallEvent(tool_call=_parse_tool_call(current))
                    yield StreamFinishEvent(
                        finish_reason=normalize_finish_reason(finish_reason),
                        provider_finish_reason=finish_reason,
                        usage=_parse_usage(payload),
                    )

        return generator()


@dataclass(slots=True)
class KimiFilesClient(OpenAICompatibleFilesClient):
    default_purpose: str = "file-extract"

    async def upload(
        self,
        *,
        data: bytes | bytearray | memoryview,
        filename: str,
        media_type: str = "application/pdf",
        purpose: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderFile:
        resolved_purpose = purpose or self.default_purpose
        if resolved_purpose not in {"file-extract", "image", "video", "batch"}:
            raise ValidationError('Kimi files purpose must be "file-extract", "image", "video", or "batch".')
        if metadata:
            raise UnsupportedFeatureError('Provider "kimi" does not support file metadata in the Files API.')
        response = await self.fetch(
            f"{self.base_url}/files",
            headers=self._headers(json_content=False),
            body={
                "data": {"purpose": resolved_purpose},
                "files": {"file": (filename, _normalize_binary(data), media_type)},
            },
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        from .openai_compat import _normalize_openai_file

        return _normalize_openai_file(await response.json(), provider=self.provider)


@dataclass(slots=True)
class KimiCountTokensClient:
    provider: str
    api_key: str
    base_url: str
    fetch: Fetcher

    async def count(
        self,
        *,
        model_id: str,
        prompt: str | None = None,
        messages: list[ModelMessage] | None = None,
        system: str | None = None,
        tools: dict[str, Any] | None = None,
        provider_options: dict[str, Any] | None = None,
        options: RetryOptions | None = None,
    ) -> CountTokensResult:
        request_messages = list(messages or [])
        if system is not None:
            request_messages.insert(0, ModelMessage(role="system", parts=[TextPart(text=system)]))
        if prompt is not None:
            request_messages.append(ModelMessage(role="user", parts=[TextPart(text=prompt)]))
        body = {
            "model": model_id,
            "messages": _to_chat_messages(request_messages),
            "tools": _map_tools(tools),
            **deepcopy(provider_options or {}),
        }
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/tokenizers/estimate-token-count",
                headers=_headers(self.api_key),
                json_body={key: value for key, value in body.items() if value is not None},
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        total = payload.get("total_tokens") or payload.get("token_count") or (payload.get("data") or {}).get("total_tokens")
        details_payload = payload.get("details") or payload.get("data", {}).get("details") or []
        details = [
            TokenCountDetail(
                modality=item.get("modality"),
                token_count=item.get("token_count"),
                billable_characters=item.get("billable_characters"),
                provider_metadata=dict(item),
            )
            for item in details_payload
            if isinstance(item, dict)
        ]
        return CountTokensResult(total_tokens=total, details=details, raw_response=payload)


@dataclass(slots=True)
class KimiFormulaClient:
    api_key: str
    base_url: str
    fetch: Fetcher
    provider: str = "kimi"

    async def list_tools(self, formula_uri: str, options: RetryOptions | None = None) -> list[dict[str, Any]]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/formulas/{formula_uri}/tools",
                method="GET",
                headers=_headers(self.api_key),
                json_body=None,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        tools = payload.get("tools") if isinstance(payload, dict) else None
        return [dict(item) for item in tools or [] if isinstance(item, dict)]

    async def call_tool(
        self,
        formula_uri: str,
        function_name: str,
        arguments: dict[str, Any],
        options: RetryOptions | None = None,
    ) -> Any:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/formulas/{formula_uri}/fibers",
                headers=_headers(self.api_key),
                json_body={"name": function_name, "arguments": json.dumps(arguments)},
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        context = payload.get("context") if isinstance(payload, dict) else None
        if isinstance(context, dict) and ("output" in context or "encrypted_output" in context):
            return context.get("output") if "output" in context else context.get("encrypted_output")
        if isinstance(payload, dict) and "error" in payload:
            return {"error": payload.get("error")}
        if isinstance(context, dict) and "error" in context:
            return {"error": context.get("error")}
        return payload

    async def toolset(self, formula_uris: list[str], options: RetryOptions | None = None) -> dict[str, ToolDefinition]:
        result: dict[str, ToolDefinition] = {}
        for formula_uri in formula_uris:
            for definition in await self.list_tools(formula_uri, options=options):
                function = definition.get("function") if isinstance(definition.get("function"), dict) else {}
                name = str(function.get("name") or "")
                if not name:
                    continue
                schema = function.get("parameters") or {"type": "object", "properties": {}}
                description = function.get("description")

                async def execute(input: dict[str, Any], *, _formula_uri: str = formula_uri, _name: str = name) -> Any:
                    return await self.call_tool(_formula_uri, _name, dict(input))

                result[name] = tool(
                    name=name,
                    description=description,
                    schema=schema,
                    execute=execute,
                    strict=False,
                    metadata={"provider": "kimi", "formula_uri": formula_uri},
                )
        return result


async def kimi_formula_toolset(
    client: KimiFormulaClient,
    formula_uris: list[str] | tuple[str, ...],
    options: RetryOptions | None = None,
) -> dict[str, ToolDefinition]:
    return await client.toolset(list(formula_uris), options=options)


def create_kimi(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    fetch: Fetcher | None = None,
):
    resolved_key = api_key or os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
    if not resolved_key:
        raise ConfigurationError("Missing kimi API key. Set MOONSHOT_API_KEY or KIMI_API_KEY.")
    resolved_base_url = (base_url or os.getenv("MOONSHOT_BASE_URL") or KIMI_DEFAULT_BASE_URL).rstrip("/")
    requester = fetch or default_fetch

    native = ProviderAdapter(
        name="kimi",
        language_model_factory=lambda model_id: KimiChatLanguageModel(
            provider="kimi",
            model_id=model_id,
            api_key=resolved_key,
            base_url=resolved_base_url,
            fetch=requester,
        ),
        files_client_factory=lambda: KimiFilesClient(
            provider="kimi",
            api_key=resolved_key,
            base_url=resolved_base_url,
            fetch=requester,
        ),
        batches_client_factory=lambda: OpenAICompatibleBatchesClient(
            provider="kimi",
            api_key=resolved_key,
            base_url=resolved_base_url,
            fetch=requester,
        ),
        count_tokens_client_factory=lambda: KimiCountTokensClient(
            provider="kimi",
            api_key=resolved_key,
            base_url=resolved_base_url,
            fetch=requester,
        ),
        formulas_client_factory=lambda: KimiFormulaClient(
            api_key=resolved_key,
            base_url=resolved_base_url,
            fetch=requester,
        ),
    )
    return create_provider_bundle(
        name="kimi",
        native=native,
        agent_capabilities=KIMI_AGENT_CAPABILITIES,
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
            portable_badge=True,
            tier="portable",
        ),
    )
