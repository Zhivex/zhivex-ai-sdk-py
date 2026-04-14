from __future__ import annotations

import asyncio
import base64
import json
from copy import deepcopy
from datetime import datetime
import os
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from dataclasses import replace
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

from .._http import Fetcher, default_fetch
from .._sse import parse_sse
from ..errors import ConfigurationError, ProviderHTTPError, UnsupportedFeatureError, ValidationError
from ..messages import normalize_finish_reason, validate_file_part, validate_message_parts
from ..realtime import (
    CallbackRealtimeSession,
    RealtimeConnectionFactory,
    RealtimeSessionCallbacks,
    encode_audio_frame,
    open_websocket_connection,
    tool_result_payload,
)
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..types import (
    AudioFrame,
    AudioInput,
    CodeExecutionResultPart,
    CountTokensClient,
    CountTokensResult,
    EmbedResult,
    EmbeddingModel,
    FilePart,
    FileSearchBatch,
    FileSearchDocument,
    FileSearchDocumentListResult,
    FileSearchOperation,
    FileSearchSearchResult,
    FileSearchStore,
    FileSearchStoreListResult,
    FileSearchStoresClient,
    FilesClient,
    GenerateResult,
    GeneratedCodePart,
    GroundedGenerateResult,
    GroundedLanguageModel,
    GroundingSupport,
    GroundedModelGenerateInput,
    GroundingSource,
    LanguageModel,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    ProviderFile,
    RealtimeAudioOutputEvent,
    RealtimeConnectOptions,
    RealtimeGoAwayEvent,
    RealtimeModel,
    RealtimeResponseCompletedEvent,
    RealtimeSession,
    RealtimeSessionConfig,
    RealtimeSessionEndedEvent,
    RealtimeSessionResumptionEvent,
    RealtimeTextDeltaEvent,
    RealtimeTokenResult,
    RealtimeToolCallEvent,
    RealtimeTranscriptEvent,
    RetryOptions,
    SpeechModel,
    SpeechOutput,
    StreamEvent,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    TextPart,
    TokenUsage,
    TokenCountDetail,
    ToolCall,
    ToolChoiceName,
    ToolCallPart,
    ToolExecutionResult,
    TranscriptionModel,
    TranscriptionOutput,
    PortableSupport,
)
from .base import ProviderAdapter, create_provider_bundle
from ._payload import drop_none

GEMINI_CAPABILITIES = ModelCapabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    parallel_tool_calls=False,
    vision=True,
    files=True,
    audio_input=True,
    audio_output=False,
    embeddings=True,
    reasoning=True,
    web_search=False,
)

GEMINI_GROUNDED_CAPABILITIES = ModelCapabilities(
    streaming=False,
    tools=False,
    structured_output=False,
    json_mode=False,
    tool_choice=False,
    parallel_tool_calls=False,
    vision=True,
    files=False,
    audio_input=False,
    audio_output=False,
    embeddings=False,
    reasoning=True,
    web_search=True,
)

GEMINI_SPEECH_CAPABILITIES = ModelCapabilities(
    streaming=False,
    tools=False,
    structured_output=False,
    json_mode=False,
    tool_choice=False,
    parallel_tool_calls=False,
    vision=False,
    files=False,
    audio_input=False,
    audio_output=True,
    embeddings=False,
    reasoning=False,
    web_search=False,
)

GEMINI_TRANSCRIPTION_CAPABILITIES = replace(
    GEMINI_SPEECH_CAPABILITIES,
    audio_input=True,
    audio_output=False,
)

GEMINI_REALTIME_CAPABILITIES = ModelCapabilities(
    streaming=False,
    tools=False,
    structured_output=False,
    json_mode=False,
    tool_choice=False,
    parallel_tool_calls=False,
    vision=True,
    files=False,
    audio_input=True,
    audio_output=True,
    embeddings=False,
    reasoning=True,
    web_search=False,
    realtime=True,
    realtime_audio_input=True,
    realtime_audio_output=True,
    realtime_tools=True,
    realtime_browser_tokens=True,
)

_RAW_TOOLS_PROVIDER_OPTION = "tools"
_BUILT_IN_TOOLS_PROVIDER_OPTIONS = {"built_in_tools", "builtInTools"}
_BUILT_IN_TOOL_NAME_MAP = {
    "google_search": "googleSearch",
    "googlesearch": "googleSearch",
    "google_maps": "googleMaps",
    "googlemaps": "googleMaps",
    "url_context": "urlContext",
    "urlcontext": "urlContext",
    "code_execution": "codeExecution",
    "codeexecution": "codeExecution",
    "file_search": "fileSearch",
    "filesearch": "fileSearch",
    "computer_use": "computerUse",
    "computeruse": "computerUse",
}


def gemini_hosted_tool(tool_type: str, /, **config: Any) -> dict[str, Any]:
    return {tool_type: drop_none(deepcopy(config)) if config else {}}


def gemini_google_search_tool(*, exclude_domains: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    return gemini_hosted_tool("google_search", excludeDomains=list(exclude_domains or []) if exclude_domains else None, **extra)


def gemini_google_maps_tool(**config: Any) -> dict[str, Any]:
    return gemini_hosted_tool("google_maps", **config)


def gemini_url_context_tool(**config: Any) -> dict[str, Any]:
    return gemini_hosted_tool("url_context", **config)


def gemini_code_execution_tool(**config: Any) -> dict[str, Any]:
    return gemini_hosted_tool("code_execution", **config)


def gemini_computer_use_tool(**config: Any) -> dict[str, Any]:
    return gemini_hosted_tool("computer_use", **config)


def gemini_file_search_tool(
    *,
    file_search_store_names: list[str],
    filters: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return gemini_hosted_tool(
        "file_search",
        fileSearchStoreNames=list(file_search_store_names),
        filters=deepcopy(filters) if filters is not None else None,
        **extra,
    )


def _system_instruction(messages: list[ModelMessage]) -> dict[str, Any] | None:
    text = "\n".join(
        part.text for message in messages if message.role == "system" for part in message.parts if part.type == "text"
    )
    return {"parts": [{"text": text}]} if text else None


def _map_file_part(part: FilePart) -> dict[str, Any]:
    validate_file_part(part)
    if part.text is not None or part.document_content is not None:
        raise ValidationError('Provider "gemini" does not support FilePart "text" or "document_content".')
    if part.file_id is not None:
        raise ValidationError('Provider "gemini" does not support "file_id". Use "file_uri" instead.')
    if part.data is not None:
        return {"inlineData": {"mimeType": part.media_type or "application/octet-stream", "data": part.data}}
    file_uri = part.file_uri or part.url
    if file_uri is None:
        raise ValidationError('Provider "gemini" requires "file_uri" or "url" for remote file references.')
    payload = {"fileUri": file_uri}
    if part.media_type:
        payload["mimeType"] = part.media_type
    return {"fileData": payload}


def _map_part(part: Any) -> dict[str, Any]:
    if part.type == "text":
        return {"text": part.text}
    if part.type == "image":
        data = part.image
        media_type = part.media_type or "image/jpeg"
        if data.startswith("data:") and ";base64," in data:
            header, body = data[len("data:"):].split(";base64,", 1)
            media_type = part.media_type or header.lower()
            data = body
        return {"inlineData": {"mimeType": media_type, "data": data}}
    if part.type == "file":
        return _map_file_part(part)
    if part.type == "tool-call":
        function_call = {"name": part.tool_call.name, "args": part.tool_call.input}
        thought_signature = part.tool_call.provider_metadata.get("thought_signature")
        payload = {"functionCall": function_call}
        if thought_signature is not None:
            payload["thought_signature"] = thought_signature
        return payload
    if part.type == "tool-result":
        return {
            "functionResponse": {
                "name": part.tool_result.tool_name,
                "response": {
                    "name": part.tool_result.tool_name,
                    "content": part.tool_result.error.__dict__ if part.tool_result.is_error else part.tool_result.output,
                },
            }
        }
    if part.type == "generated-code":
        return {"executableCode": {"language": part.language or "python", "code": part.code}}
    if part.type == "code-result":
        payload = {"output": part.output}
        if part.outcome is not None:
            payload["outcome"] = part.outcome
        return {"codeExecutionResult": payload}
    return {"text": json.dumps(str(part))}


def _map_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    return [
        {
            "role": "model" if message.role == "assistant" else "user",
            "parts": [_map_part(part) for part in message.parts],
        }
        for message in messages
        if message.role != "system"
    ]


def _google_search_tool() -> dict[str, Any]:
    return {"googleSearch": {}}


def _provider_option_value(options: Any, *names: str) -> Any:
    if options is None:
        return None
    if isinstance(options, dict):
        for name in names:
            if name in options:
                return options[name]
        return None
    for name in names:
        if hasattr(options, name):
            return getattr(options, name)
    return None


def _normalize_builtin_tool_name(name: str) -> str | None:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in name).strip("_")
    normalized = "_".join(part for part in normalized.split("_") if part)
    return _BUILT_IN_TOOL_NAME_MAP.get(normalized)


def _normalize_builtin_tool_config(value: Any) -> dict[str, Any] | None:
    if value in (None, False):
        return None
    if value is True:
        return {}
    if not isinstance(value, dict):
        raise ValidationError("Gemini built-in tool config values must be booleans or dictionaries.")
    return deepcopy(value)


def _extract_builtin_tools(provider_options: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not provider_options:
        return []

    normalized: dict[str, dict[str, Any]] = {}
    for key, value in provider_options.items():
        canonical = _normalize_builtin_tool_name(key)
        if canonical is None:
            continue
        config = _normalize_builtin_tool_config(value)
        if config is not None:
            normalized[canonical] = config

    built_in_tools = provider_options.get("built_in_tools") or provider_options.get("builtInTools")
    if built_in_tools is not None:
        if not isinstance(built_in_tools, list):
            raise ValidationError('Gemini provider_options["built_in_tools"] must be a list.')
        for item in built_in_tools:
            if isinstance(item, str):
                canonical = _normalize_builtin_tool_name(item)
                if canonical is None:
                    raise ValidationError(f'Unsupported Gemini built-in tool "{item}".')
                normalized[canonical] = {}
                continue
            if not isinstance(item, dict) or len(item) != 1:
                raise ValidationError('Each Gemini built-in tool entry must be a string or a single-key dictionary.')
            raw_name, raw_config = next(iter(item.items()))
            canonical = _normalize_builtin_tool_name(str(raw_name))
            if canonical is None:
                raise ValidationError(f'Unsupported Gemini built-in tool "{raw_name}".')
            config = _normalize_builtin_tool_config(raw_config)
            if config is not None:
                normalized[canonical] = config

    return [{name: config} for name, config in normalized.items()]


def _extract_raw_tools(provider_options: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not provider_options:
        return []
    raw = provider_options.get(_RAW_TOOLS_PROVIDER_OPTION)
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValidationError('Gemini provider_options["tools"] must be a list when provided.')
    return [deepcopy(item) for item in raw]


def _validate_builtin_tool_combination(function_tools: dict[str, Any] | None, builtin_tools: list[dict[str, Any]]) -> None:
    names = [next(iter(tool.keys())) for tool in builtin_tools if isinstance(tool, dict) and tool]
    if "fileSearch" in names and (len(names) > 1 or bool(function_tools)):
        raise UnsupportedFeatureError('Provider "gemini" does not support combining "file_search" with other tools.')
    if "urlContext" in names and function_tools:
        raise UnsupportedFeatureError('Provider "gemini" does not support combining "url_context" with function calling.')


_GEMINI_SUPPORTED_SCHEMA_KEYS = {
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "properties",
    "required",
    "items",
    "anyOf",
    "oneOf",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
}


def _normalize_gemini_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    def visit(node: Any) -> Any:
        if isinstance(node, list):
            return [visit(item) for item in node]
        if not isinstance(node, dict):
            return node

        normalized: dict[str, Any] = {}
        for key, value in node.items():
            if key not in _GEMINI_SUPPORTED_SCHEMA_KEYS:
                continue
            if key == "properties" and isinstance(value, dict):
                normalized[key] = {str(name): visit(property_schema) for name, property_schema in value.items()}
                continue
            normalized[key] = visit(value)
        return normalized

    return visit(deepcopy(schema))


def _map_tools(
    tools: dict[str, Any] | None,
    provider_options: dict[str, Any] | None = None,
    *,
    force_google_search: bool = False,
) -> list[dict[str, Any]] | None:
    mapped: list[dict[str, Any]] = []
    if tools:
        mapped.append(
            {
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": (
                            _normalize_gemini_tool_schema(create_schema_adapter(tool.schema).json_schema())
                            if getattr(tool, "source", None) == "mcp"
                            else create_schema_adapter(tool.schema).json_schema()
                        ),
                    }
                    for tool in tools.values()
                ]
            }
        )
    builtin_tools = _extract_builtin_tools(provider_options)
    if force_google_search and not any("googleSearch" in tool for tool in builtin_tools):
        builtin_tools.insert(0, _google_search_tool())
    _validate_builtin_tool_combination(tools, builtin_tools)
    mapped.extend(builtin_tools)
    mapped.extend(_extract_raw_tools(provider_options))
    return mapped or None


def _map_tool_config(tools: dict[str, Any] | None, tool_choice: str | ToolChoiceName | None) -> dict[str, Any] | None:
    if not tools or tool_choice is None or tool_choice == "auto":
        return None
    if tool_choice == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if tool_choice == "required":
        return {"functionCallingConfig": {"mode": "ANY"}}
    return {
        "functionCallingConfig": {
            "mode": "ANY",
            "allowedFunctionNames": [tool_choice.tool_name],
        }
    }


def _provider_options_without_mapped_tools(provider_options: dict[str, Any] | None) -> dict[str, Any] | None:
    if not provider_options:
        return None
    stripped_keys = set(_BUILT_IN_TOOLS_PROVIDER_OPTIONS)
    stripped_keys.add(_RAW_TOOLS_PROVIDER_OPTION)
    stripped_keys.update(key for key in provider_options if _normalize_builtin_tool_name(key) is not None)
    remaining = {key: value for key, value in provider_options.items() if key not in stripped_keys}
    return remaining or None


def _is_gemini_3_model(model_id: str) -> bool:
    return model_id.startswith("gemini-3")


def _is_gemini_3_pro_model(model_id: str) -> bool:
    return _is_gemini_3_model(model_id) and "pro" in model_id


def _map_reasoning(model_id: str, input: ModelGenerateInput) -> dict[str, Any] | None:
    if input.reasoning is None:
        return None
    if _is_gemini_3_model(model_id):
        if input.reasoning.budget_tokens is not None:
            raise UnsupportedFeatureError(
                'Provider "gemini" uses "reasoning.effort" for Gemini 3 models and does not support "reasoning.budgetTokens".'
            )
        if input.reasoning.effort == "none":
            raise UnsupportedFeatureError('Provider "gemini" does not support "reasoning.effort=none" for Gemini 3 models.')
        if input.reasoning.effort == "xhigh":
            raise UnsupportedFeatureError('Provider "gemini" does not support "reasoning.effort=xhigh".')
        if input.reasoning.effort == "minimal" and _is_gemini_3_pro_model(model_id):
            raise UnsupportedFeatureError(
                'Provider "gemini" does not support "reasoning.effort=minimal" for Gemini 3 Pro models.'
            )
        return {"thinkingLevel": input.reasoning.effort} if input.reasoning.effort is not None else None
    if input.reasoning.effort is not None:
        raise UnsupportedFeatureError(
            'Provider "gemini" does not support "reasoning.effort" for models earlier than Gemini 3.'
        )
    return {"thinkingBudget": input.reasoning.budget_tokens} if input.reasoning.budget_tokens is not None else None


def _generation_config(model_id: str, input: ModelGenerateInput) -> dict[str, Any]:
    config: dict[str, Any] = {"temperature": input.temperature, "maxOutputTokens": input.max_tokens}
    if input.reasoning is not None:
        config["thinkingConfig"] = _map_reasoning(model_id, input)
    if input.structured_output is not None and input.structured_output.mode == "native":
        config["responseMimeType"] = "application/json"
        config["responseJsonSchema"] = create_schema_adapter(input.structured_output.schema).json_schema()
    return drop_none(config)


def _gemini_speech_generation_config(
    *,
    provider: str,
    voice: str | None,
    provider_options: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    remaining = deepcopy(provider_options or {})
    generation_config = deepcopy(dict(remaining.pop("generationConfig", {}) or {}))

    speech_config = deepcopy(dict(generation_config.get("speechConfig") or remaining.pop("speechConfig", {}) or {}))
    if voice:
        if "multiSpeakerVoiceConfig" in speech_config:
            raise ValidationError(
                f'Provider "{provider}" does not support passing both "voice" and provider_options["speechConfig"]["multiSpeakerVoiceConfig"].'
            )
        speech_config["voiceConfig"] = {
            **dict(speech_config.get("voiceConfig") or {}),
            "prebuiltVoiceConfig": {
                **dict((speech_config.get("voiceConfig") or {}).get("prebuiltVoiceConfig") or {}),
                "voiceName": voice,
            },
        }
    if not speech_config:
        raise ValidationError(
            f'Provider "{provider}" requires a "voice" argument or provider_options["speechConfig"] for speech generation.'
        )

    generation_config["responseModalities"] = list(generation_config.get("responseModalities") or ["AUDIO"])
    if "AUDIO" not in generation_config["responseModalities"]:
        generation_config["responseModalities"].append("AUDIO")
    generation_config["speechConfig"] = speech_config
    return remaining, generation_config


def _extract_gemini_audio_part(
    payload: dict[str, Any],
    *,
    provider: str,
) -> tuple[bytes, str]:
    candidate = (payload.get("candidates") or [None])[0] or {}
    for part in ((candidate.get("content") or {}).get("parts") or []):
        inline = part.get("inlineData") or part.get("inline_data")
        if not isinstance(inline, dict) or not inline.get("data"):
            continue
        return (
            base64.b64decode(str(inline.get("data"))),
            str(inline.get("mimeType") or inline.get("mime_type") or "audio/pcm"),
        )
    raise ValidationError(f'Provider "{provider}" did not return audio data for speech generation.')


def _normalize_binary(data: bytes | bytearray | memoryview) -> bytes:
    if isinstance(data, memoryview):
        return data.tobytes()
    return bytes(data)


def _parse_timestamp_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return int(datetime.fromisoformat(normalized).timestamp() * 1000)
    except ValueError:
        return None


def _normalize_file_search_store(payload: dict[str, Any]) -> FileSearchStore:
    return FileSearchStore(
        name=str(payload.get("name") or ""),
        display_name=payload.get("displayName"),
        create_time=payload.get("createTime"),
        update_time=payload.get("updateTime"),
        metadata=dict(payload),
    )


def _normalize_file_search_document(payload: dict[str, Any]) -> FileSearchDocument:
    size_bytes = payload.get("sizeBytes")
    try:
        parsed_size = int(size_bytes) if size_bytes is not None else None
    except (TypeError, ValueError):
        parsed_size = None
    return FileSearchDocument(
        name=str(payload.get("name") or ""),
        display_name=payload.get("displayName"),
        custom_metadata=list(payload.get("customMetadata") or []),
        state=payload.get("state"),
        size_bytes=parsed_size,
        media_type=payload.get("mimeType"),
        create_time=payload.get("createTime"),
        update_time=payload.get("updateTime"),
        metadata=dict(payload),
    )


def _normalize_file_search_operation(payload: dict[str, Any]) -> FileSearchOperation:
    return FileSearchOperation(
        name=str(payload.get("name") or ""),
        done=bool(payload.get("done")),
        metadata=dict(payload.get("metadata") or {}),
        response=dict(payload.get("response") or {}) if isinstance(payload.get("response"), dict) else payload.get("response"),
        error=dict(payload.get("error") or {}) if isinstance(payload.get("error"), dict) else payload.get("error"),
        raw_response=payload,
    )


def _embedding_request_options(options: Any) -> dict[str, Any]:
    task_type = _provider_option_value(options, "task_type", "taskType")
    title = _provider_option_value(options, "title")
    output_dimensionality = _provider_option_value(options, "output_dimensionality", "outputDimensionality")
    auto_truncate = _provider_option_value(options, "auto_truncate", "autoTruncate")
    task_types = _provider_option_value(options, "task_types", "taskTypes")
    titles = _provider_option_value(options, "titles")
    return drop_none(
        {
            "task_type": task_type,
            "title": title,
            "output_dimensionality": output_dimensionality,
            "auto_truncate": auto_truncate,
            "task_types": task_types,
            "titles": titles,
        }
    )


def _build_messages_for_request(
    *,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    system: str | None = None,
) -> list[ModelMessage]:
    if prompt is not None and messages is not None:
        raise ValidationError('Pass either "prompt" or "messages", but not both.')
    built = list(messages or [])
    if system:
        built.insert(0, ModelMessage(role="system", parts=[TextPart(text=system)]))
    if prompt:
        built.append(ModelMessage(role="user", parts=[TextPart(text=prompt)]))
    return built


def _normalize_gemini_file(payload: dict[str, Any], *, provider: str) -> ProviderFile:
    return ProviderFile(
        provider=provider,
        id=str(payload.get("name") or ""),
        filename=payload.get("displayName"),
        media_type=payload.get("mimeType"),
        size_bytes=payload.get("sizeBytes"),
        status=payload.get("state"),
        file_uri=payload.get("uri"),
        created_at=payload.get("createTime"),
        metadata=dict(payload),
    )


@dataclass(slots=True)
class GeminiFilesClient(FilesClient):
    provider: str
    api_key: str
    base_url: str
    fetch: Fetcher

    def _json_url(self, path: str) -> str:
        return f"{self.base_url}{path}?key={self.api_key}"

    def _upload_url(self) -> str:
        return f"{self.base_url.replace('/v1beta', '')}/upload/v1beta/files?key={self.api_key}"

    async def upload(
        self,
        *,
        data: bytes | bytearray | memoryview,
        filename: str,
        media_type: str = "application/pdf",
        purpose: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderFile:
        del purpose
        raw = _normalize_binary(data)
        start_body = {"file": {"display_name": filename}}
        if metadata:
            start_body["file"]["custom_metadata"] = metadata
        start = await self.fetch(
            self._upload_url(),
            headers={
                "content-type": "application/json",
                "x-goog-upload-protocol": "resumable",
                "x-goog-upload-command": "start",
                "x-goog-upload-header-content-length": str(len(raw)),
                "x-goog-upload-header-content-type": media_type,
            },
            json_body=start_body,
            timeout_ms=None,
        )
        if start.status_code >= 400:
            raise ProviderHTTPError(
                f'Provider "{self.provider}" request failed with status {start.status_code}.',
                start.status_code,
                response_body=await start.text(),
            )
        upload_url = start.headers.get("x-goog-upload-url") or start.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise ValidationError('Provider "gemini" did not return an upload URL for the Files API.')
        finalize = await self.fetch(
            str(upload_url),
            headers={
                "content-length": str(len(raw)),
                "x-goog-upload-offset": "0",
                "x-goog-upload-command": "upload, finalize",
            },
            json_body=None,
            body=raw,
            timeout_ms=None,
        )
        if finalize.status_code >= 400:
            raise ProviderHTTPError(
                f'Provider "{self.provider}" request failed with status {finalize.status_code}.',
                finalize.status_code,
                response_body=await finalize.text(),
            )
        payload = await finalize.json()
        return _normalize_gemini_file(dict(payload.get("file") or payload), provider=self.provider)

    async def list(self) -> list[ProviderFile]:
        response = await self.fetch(
            self._json_url("/files"),
            method="GET",
            headers={"content-type": "application/json"},
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f'Provider "{self.provider}" request failed with status {response.status_code}.',
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        return [_normalize_gemini_file(dict(item), provider=self.provider) for item in payload.get("files") or []]

    async def get(self, file_id: str) -> ProviderFile:
        response = await self.fetch(
            self._json_url(f"/files/{file_id}"),
            method="GET",
            headers={"content-type": "application/json"},
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f'Provider "{self.provider}" request failed with status {response.status_code}.',
                response.status_code,
                response_body=await response.text(),
            )
        return _normalize_gemini_file(await response.json(), provider=self.provider)

    async def download(self, file_id: str) -> bytes:
        raise UnsupportedFeatureError(
            f'Provider "{self.provider}" does not support downloading uploaded files through the Files API.'
        )

    async def delete(self, file_id: str) -> bool:
        response = await self.fetch(
            self._json_url(f"/files/{file_id}"),
            method="DELETE",
            headers={"content-type": "application/json"},
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f'Provider "{self.provider}" request failed with status {response.status_code}.',
                response.status_code,
                response_body=await response.text(),
            )
        return True


@dataclass(slots=True)
class GeminiCountTokensClient(CountTokensClient):
    provider: str
    api_key: str
    base_url: str
    fetch: Fetcher

    def _url(self, model_id: str) -> str:
        return f"{self.base_url}/models/{model_id}:countTokens?key={self.api_key}"

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
        built_messages = _build_messages_for_request(prompt=prompt, messages=messages, system=system)
        request = drop_none(
            {
                "contents": _map_messages(built_messages),
                "systemInstruction": _system_instruction(built_messages),
                "tools": _map_tools(tools, provider_options),
                **(_provider_options_without_mapped_tools(provider_options) or {}),
            }
        )
        response = await with_retry(
            lambda: self.fetch(
                self._url(model_id),
                method="POST",
                headers={"content-type": "application/json"},
                json_body={"generateContentRequest": request},
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        return CountTokensResult(
            total_tokens=payload.get("totalTokens"),
            cached_content_token_count=payload.get("cachedContentTokenCount"),
            total_billable_characters=payload.get("totalBillableCharacters"),
            details=[
                TokenCountDetail(
                    modality=item.get("modality"),
                    token_count=item.get("tokenCount"),
                    billable_characters=item.get("billableCharacters"),
                    provider_metadata=dict(item),
                )
                for item in payload.get("promptTokensDetails") or []
                if isinstance(item, dict)
            ],
            raw_response=payload,
        )


@dataclass(slots=True)
class GeminiFileSearchStoresClient(FileSearchStoresClient):
    api_key: str
    base_url: str
    fetch: Fetcher

    def _json_url(self, path: str) -> str:
        return f"{self.base_url}{path}?key={self.api_key}"

    def _upload_url(self, store_name: str) -> str:
        return f"{self.base_url.replace('/v1beta', '')}/upload/v1beta/{store_name}:uploadToFileSearchStore?key={self.api_key}"

    async def create(
        self,
        *,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FileSearchStore:
        body = dict(metadata or {})
        if display_name is not None:
            body["displayName"] = display_name
        response = await self.fetch(
            self._json_url("/fileSearchStores"),
            method="POST",
            headers={"content-type": "application/json"},
            json_body=body or {},
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        return _normalize_file_search_store(await response.json())

    async def list(
        self,
        *,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> FileSearchStoreListResult:
        url = self._json_url("/fileSearchStores")
        params = []
        if page_size is not None:
            params.append(f"pageSize={page_size}")
        if page_token is not None:
            params.append(f"pageToken={page_token}")
        if params:
            url = f"{url}&{'&'.join(params)}"
        response = await self.fetch(
            url,
            method="GET",
            headers={"content-type": "application/json"},
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        return FileSearchStoreListResult(
            stores=[_normalize_file_search_store(item) for item in payload.get("fileSearchStores") or []],
            next_page_token=payload.get("nextPageToken"),
            raw_response=payload,
        )

    async def get(self, name: str) -> FileSearchStore:
        response = await self.fetch(
            self._json_url(f"/{name}"),
            method="GET",
            headers={"content-type": "application/json"},
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        return _normalize_file_search_store(await response.json())

    async def update(
        self,
        name: str,
        *,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        expires_after: dict[str, Any] | None = None,
    ) -> FileSearchStore:
        del display_name, metadata, expires_after
        raise UnsupportedFeatureError('Provider "gemini" does not expose file search store update operations through this SDK.')

    async def delete(self, name: str, *, force: bool = False) -> bool:
        url = self._json_url(f"/{name}")
        if force:
            url = f"{url}&force=true"
        response = await self.fetch(
            url,
            method="DELETE",
            headers={"content-type": "application/json"},
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        return True

    async def upload(
        self,
        *,
        file_search_store_name: str,
        data: bytes | bytearray | memoryview,
        filename: str,
        media_type: str | None = None,
        display_name: str | None = None,
        custom_metadata: list[dict[str, Any]] | None = None,
        chunking_config: dict[str, Any] | None = None,
    ) -> FileSearchOperation:
        raw = _normalize_binary(data)
        metadata_body = drop_none(
            {
                "displayName": display_name or filename,
                "customMetadata": deepcopy(custom_metadata),
                "chunkingConfig": deepcopy(chunking_config),
                "mimeType": media_type,
            }
        )
        start = await self.fetch(
            self._upload_url(file_search_store_name),
            method="POST",
            headers={
                "content-type": "application/json",
                "x-goog-upload-protocol": "resumable",
                "x-goog-upload-command": "start",
                "x-goog-upload-header-content-length": str(len(raw)),
                **({"x-goog-upload-header-content-type": media_type} if media_type else {}),
            },
            json_body=metadata_body,
            timeout_ms=None,
        )
        if start.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {start.status_code}.",
                start.status_code,
                response_body=await start.text(),
            )
        upload_url = start.headers.get("x-goog-upload-url") or start.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise ValidationError('Provider "gemini" did not return an upload URL for File Search.')
        finalize = await self.fetch(
            str(upload_url),
            method="POST",
            headers={
                "content-length": str(len(raw)),
                "x-goog-upload-offset": "0",
                "x-goog-upload-command": "upload, finalize",
            },
            json_body=None,
            body=raw,
            timeout_ms=None,
        )
        if finalize.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {finalize.status_code}.",
                finalize.status_code,
                response_body=await finalize.text(),
            )
        return _normalize_file_search_operation(await finalize.json())

    async def import_file(
        self,
        *,
        file_search_store_name: str,
        file_name: str,
        custom_metadata: list[dict[str, Any]] | None = None,
        chunking_config: dict[str, Any] | None = None,
    ) -> FileSearchOperation:
        response = await self.fetch(
            self._json_url(f"/{file_search_store_name}:importFile"),
            method="POST",
            headers={"content-type": "application/json"},
            json_body=drop_none(
                {
                    "fileName": file_name,
                    "customMetadata": deepcopy(custom_metadata),
                    "chunkingConfig": deepcopy(chunking_config),
                }
            ),
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        return _normalize_file_search_operation(await response.json())

    async def list_documents(
        self,
        *,
        file_search_store_name: str,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> FileSearchDocumentListResult:
        url = self._json_url(f"/{file_search_store_name}/documents")
        params = []
        if page_size is not None:
            params.append(f"pageSize={page_size}")
        if page_token is not None:
            params.append(f"pageToken={page_token}")
        if params:
            url = f"{url}&{'&'.join(params)}"
        response = await self.fetch(
            url,
            method="GET",
            headers={"content-type": "application/json"},
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        return FileSearchDocumentListResult(
            documents=[_normalize_file_search_document(item) for item in payload.get("documents") or []],
            next_page_token=payload.get("nextPageToken"),
            raw_response=payload,
        )

    async def get_document(self, name: str) -> FileSearchDocument:
        response = await self.fetch(
            self._json_url(f"/{name}"),
            method="GET",
            headers={"content-type": "application/json"},
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        return _normalize_file_search_document(await response.json())

    async def download_document(self, name: str) -> bytes:
        del name
        raise UnsupportedFeatureError('Provider "gemini" does not expose document download through the File Search API.')

    async def delete_document(self, name: str) -> bool:
        response = await self.fetch(
            self._json_url(f"/{name}"),
            method="DELETE",
            headers={"content-type": "application/json"},
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        return True

    async def update_document(
        self,
        name: str,
        *,
        custom_metadata: list[dict[str, Any]] | None = None,
        chunking_config: dict[str, Any] | None = None,
    ) -> FileSearchDocument:
        del name, custom_metadata, chunking_config
        raise UnsupportedFeatureError('Provider "gemini" does not expose file search document update operations through this SDK.')

    async def search(
        self,
        *,
        file_search_store_name: str,
        query: str | list[str],
        filters: dict[str, Any] | None = None,
        max_num_results: int | None = None,
        ranking_options: dict[str, Any] | None = None,
        rewrite_query: bool | None = None,
    ) -> FileSearchSearchResult:
        del file_search_store_name, query, filters, max_num_results, ranking_options, rewrite_query
        raise UnsupportedFeatureError('Provider "gemini" does not expose a standalone file search query endpoint through this SDK.')

    async def create_batch(
        self,
        *,
        file_search_store_name: str,
        file_names: list[str],
        custom_metadata: list[dict[str, Any]] | None = None,
        chunking_config: dict[str, Any] | None = None,
    ) -> FileSearchBatch:
        del file_search_store_name, file_names, custom_metadata, chunking_config
        raise UnsupportedFeatureError('Provider "gemini" does not expose OpenAI-style file search batch operations through this SDK.')

    async def get_batch(self, name: str) -> FileSearchBatch:
        del name
        raise UnsupportedFeatureError('Provider "gemini" does not expose OpenAI-style file search batch operations through this SDK.')

    async def cancel_batch(self, name: str) -> FileSearchBatch:
        del name
        raise UnsupportedFeatureError('Provider "gemini" does not expose OpenAI-style file search batch operations through this SDK.')

    async def list_batch_documents(
        self,
        *,
        name: str,
        page_size: int | None = None,
        page_token: str | None = None,
        state_filter: str | None = None,
    ) -> FileSearchDocumentListResult:
        del name, page_size, page_token, state_filter
        raise UnsupportedFeatureError('Provider "gemini" does not expose OpenAI-style file search batch operations through this SDK.')

    async def wait_batch(
        self,
        name: str,
        *,
        poll_interval_ms: int = 500,
        timeout_ms: int | None = None,
    ) -> FileSearchBatch:
        del name, poll_interval_ms, timeout_ms
        raise UnsupportedFeatureError('Provider "gemini" does not expose OpenAI-style file search batch operations through this SDK.')

    async def get_operation(self, name: str) -> FileSearchOperation:
        response = await self.fetch(
            self._json_url(f"/{name}"),
            method="GET",
            headers={"content-type": "application/json"},
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        return _normalize_file_search_operation(await response.json())

    async def wait_operation(
        self,
        name: str,
        *,
        poll_interval_ms: int = 500,
        timeout_ms: int | None = None,
    ) -> FileSearchOperation:
        deadline = None if timeout_ms is None else (asyncio.get_running_loop().time() + timeout_ms / 1000)
        while True:
            operation = await self.get_operation(name)
            if operation.done:
                return operation
            if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f'Waiting for file search operation "{name}" timed out.')
            await asyncio.sleep(max(poll_interval_ms, 1) / 1000)


def _gemini_realtime_url(base_url: str, api_key: str, provider_options: dict[str, Any] | None = None) -> str:
    override = (provider_options or {}).get("realtime_url")
    if isinstance(override, str) and override:
        return override
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = dict((provider_options or {}).get("realtime_query") or {})
    access_token = (provider_options or {}).get("access_token") or (provider_options or {}).get("accessToken")
    if access_token:
        query.setdefault("access_token", str(access_token))
    else:
        query.setdefault("key", api_key)
    return urlunparse((scheme, parsed.netloc, "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent", "", urlencode(query), ""))


def _gemini_realtime_headers(provider_options: dict[str, Any] | None = None) -> dict[str, str]:
    return dict((provider_options or {}).get("headers") or {})


def _gemini_realtime_tools(config: RealtimeSessionConfig) -> list[dict[str, Any]] | None:
    return _map_tools(config.tools, config.provider_options)


def _gemini_realtime_provider_options(provider_options: dict[str, Any] | None) -> dict[str, Any] | None:
    remaining = _provider_options_without_mapped_tools(provider_options) or {}
    cleaned = {
        key: value
        for key, value in remaining.items()
        if key not in {"headers", "realtime_url", "realtime_query", "access_token", "accessToken"}
    }
    return cleaned or None


def _gemini_realtime_setup(config: RealtimeSessionConfig, model_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": f"models/{model_id}",
        "generationConfig": drop_none({
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": config.voice}}} if config.voice else None,
            "responseModalities": ["AUDIO"] if config.output_audio_media_type else ["TEXT"],
        }),
        "tools": _gemini_realtime_tools(config),
        "systemInstruction": {"parts": [{"text": config.instructions}]} if config.instructions else None,
        **(_gemini_realtime_provider_options(config.provider_options) or {}),
    }
    return {"setup": drop_none(payload)}


def _gemini_realtime_build_audio(frame: AudioFrame, _config: RealtimeSessionConfig) -> list[dict[str, Any]]:
    return [
        {
            "realtimeInput": {
                "audio": {
                    "mimeType": frame.media_type,
                    "data": encode_audio_frame(frame),
                }
            }
        }
    ]


def _gemini_realtime_build_text(text: str, _config: RealtimeSessionConfig) -> list[dict[str, Any]]:
    return [
        {
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": text}]}],
                "turnComplete": True,
            }
        }
    ]


def _gemini_realtime_build_tool_result(result: ToolExecutionResult, _config: RealtimeSessionConfig) -> list[dict[str, Any]]:
    return [
        {
            "toolResponse": {
                "functionResponses": [
                    {
                        "id": result.tool_call_id,
                        "name": result.tool_name,
                        "response": tool_result_payload(result),
                    }
                ]
            }
        }
    ]


def _gemini_realtime_build_update(config: RealtimeSessionConfig, model_id: str) -> list[dict[str, Any]]:
    return [_gemini_realtime_setup(config, model_id)]


def _gemini_realtime_parse_event(payload: dict[str, Any]) -> list[Any]:
    if "setupComplete" in payload:
        return []
    if isinstance(payload.get("serverContent") or payload.get("server_content"), dict):
        content = payload.get("serverContent") or payload.get("server_content") or {}
        model_turn = dict(content.get("modelTurn") or content.get("model_turn") or {})
        parts = list(model_turn.get("parts") or [])
        events: list[Any] = []
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                events.append(
                    RealtimeTextDeltaEvent(
                        text_delta=str(part.get("text") or ""),
                        provider_metadata=payload,
                    )
                )
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                audio = base64.b64decode(str(inline.get("data")))
                events.append(
                    RealtimeAudioOutputEvent(
                        audio=audio,
                        media_type=str(inline.get("mimeType") or inline.get("mime_type") or "audio/pcm"),
                        provider_metadata=payload,
                    )
                )
            if isinstance(part, dict) and part.get("functionCall"):
                call = dict(part.get("functionCall") or {})
                events.append(
                    RealtimeToolCallEvent(
                        tool_call=ToolCall(
                            id=str(call.get("id") or f'{call.get("name", "")}-0'),
                            name=str(call.get("name") or ""),
                            input=call.get("args") or {},
                        )
                    )
                )
        input_transcription = content.get("inputTranscription") or content.get("input_transcription")
        if isinstance(input_transcription, dict) and input_transcription.get("text"):
            events.append(
                RealtimeTranscriptEvent(
                    text=str(input_transcription.get("text") or ""),
                    role="user",
                    is_final=bool(content.get("turnComplete") or content.get("turn_complete")),
                    provider_metadata=payload,
                )
            )
        output_transcription = content.get("outputTranscription") or content.get("output_transcription")
        if isinstance(output_transcription, dict) and output_transcription.get("text"):
            events.append(
                RealtimeTranscriptEvent(
                    text=str(output_transcription.get("text") or ""),
                    role="assistant",
                    is_final=bool(content.get("turnComplete") or content.get("turn_complete")),
                    provider_metadata=payload,
                )
            )
        if content.get("generationComplete") or content.get("generation_complete"):
            events.append(RealtimeResponseCompletedEvent(reason="generation-complete", provider_metadata=payload))
        if content.get("turnComplete") or content.get("turn_complete"):
            events.append(RealtimeResponseCompletedEvent(reason="turn-complete", provider_metadata=payload))
        return events
    tool_call = payload.get("toolCall") or payload.get("tool_call")
    if isinstance(tool_call, dict):
        calls = tool_call.get("functionCalls") or tool_call.get("function_calls") or [tool_call]
        return [
            RealtimeToolCallEvent(
                tool_call=ToolCall(
                    id=str(call.get("id") or f'{call.get("name", "")}-0'),
                    name=str(call.get("name") or ""),
                    input=call.get("args") or {},
                )
            )
            for call in calls
            if isinstance(call, dict)
        ]
    session_resumption = payload.get("sessionResumptionUpdate") or payload.get("session_resumption_update")
    if isinstance(session_resumption, dict):
        return [
            RealtimeSessionResumptionEvent(
                new_handle=session_resumption.get("newHandle") or session_resumption.get("new_handle"),
                resumable=session_resumption.get("resumable"),
                provider_metadata=payload,
            )
        ]
    go_away = payload.get("goAway") or payload.get("go_away")
    if isinstance(go_away, dict):
        return [
            RealtimeGoAwayEvent(
                time_left_ms=go_away.get("timeLeftMs") or go_away.get("time_left_ms"),
                provider_metadata=payload,
            )
        ]
    if isinstance(payload.get("error"), dict):
        return [RealtimeSessionEndedEvent(reason="error", provider_metadata=payload)]
    return []


def _parse_assistant_message(candidate: dict[str, Any] | None) -> ModelMessage:
    parts = []
    for part in ((candidate or {}).get("content") or {}).get("parts", []):
        if part.get("text"):
            parts.append(TextPart(text=part["text"]))
        elif part.get("executableCode") or part.get("executable_code"):
            code = part.get("executableCode") or part.get("executable_code") or {}
            parts.append(
                GeneratedCodePart(
                    code=str(code.get("code") or ""),
                    language=code.get("language"),
                )
            )
        elif part.get("codeExecutionResult") or part.get("code_execution_result"):
            result = part.get("codeExecutionResult") or part.get("code_execution_result") or {}
            parts.append(
                CodeExecutionResultPart(
                    output=str(result.get("output") or ""),
                    outcome=result.get("outcome"),
                )
            )
        elif part.get("functionCall"):
            call = part["functionCall"]
            provider_metadata: dict[str, Any] = {}
            if part.get("thoughtSignature") is not None:
                provider_metadata["thought_signature"] = part["thoughtSignature"]
            parts.append(
                ToolCallPart(
                    tool_call=ToolCall(
                        id=f'{call["name"]}-0',
                        name=call["name"],
                        input=call.get("args") or {},
                        provider_metadata=provider_metadata,
                    )
                )
            )
    return ModelMessage(role="assistant", parts=parts)


def _extract_grounding_sources(payload: dict[str, Any]) -> list[GroundingSource]:
    candidate = (payload.get("candidates") or [None])[0] or {}
    grounding_metadata = candidate.get("groundingMetadata") or {}
    sources: list[GroundingSource] = []
    seen: set[str] = set()
    for chunk in grounding_metadata.get("groundingChunks") or []:
        source_kind = next((key for key in ("web", "retrievedContext", "maps") if isinstance(chunk.get(key), dict)), None)
        source_payload = chunk.get(source_kind) if source_kind else None
        if not isinstance(source_payload, dict):
            continue
        url = source_payload.get("uri") or source_payload.get("url") or source_payload.get("fileUri")
        if not isinstance(url, str) or not url or url in seen:
            continue
        seen.add(url)
        sources.append(
            GroundingSource(
                url=url,
                title=source_payload.get("title") or source_payload.get("displayName"),
                snippet=source_payload.get("text") or source_payload.get("snippet"),
                kind=source_kind,
                provider_metadata=source_payload,
            )
        )
    return sources


def _extract_grounding_queries(payload: dict[str, Any]) -> list[str]:
    candidate = (payload.get("candidates") or [None])[0] or {}
    grounding_metadata = candidate.get("groundingMetadata") or {}
    queries = grounding_metadata.get("webSearchQueries") or grounding_metadata.get("searchQueries") or []
    return [str(query) for query in queries if isinstance(query, str) and query]


def _extract_grounding_supports(payload: dict[str, Any]) -> list[GroundingSupport]:
    candidate = (payload.get("candidates") or [None])[0] or {}
    grounding_metadata = candidate.get("groundingMetadata") or {}
    supports: list[GroundingSupport] = []
    for item in grounding_metadata.get("groundingSupports") or []:
        if not isinstance(item, dict):
            continue
        segment = item.get("segment") or {}
        supports.append(
            GroundingSupport(
                start_index=segment.get("startIndex"),
                end_index=segment.get("endIndex"),
                segment_text=segment.get("text"),
                source_indices=[int(index) for index in item.get("groundingChunkIndices") or [] if isinstance(index, int)],
                provider_metadata=item,
            )
        )
    return supports


def _extract_search_entry_point(payload: dict[str, Any]) -> dict[str, Any] | None:
    candidate = (payload.get("candidates") or [None])[0] or {}
    grounding_metadata = candidate.get("groundingMetadata") or {}
    entry_point = grounding_metadata.get("searchEntryPoint")
    return dict(entry_point) if isinstance(entry_point, dict) else None


@dataclass(slots=True)
class GeminiLanguageModel(LanguageModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: GEMINI_CAPABILITIES)

    def _url(self, action: str) -> str:
        separator = "&" if "?" in action else "?"
        return f"{self.base_url}/models/{self.model_id}:{action}{separator}key={self.api_key}"

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        validate_message_parts(self, input.messages)
        generation_config = _generation_config(self.model_id, input) or None
        response = await with_retry(
            lambda: self.fetch(
                self._url("generateContent"),
                headers={"content-type": "application/json"},
                json_body=drop_none({
                    "contents": _map_messages(input.messages),
                    "systemInstruction": _system_instruction(input.messages),
                    "tools": _map_tools(input.tools, input.provider_options),
                    "toolConfig": _map_tool_config(input.tools, input.tool_choice),
                    **(_provider_options_without_mapped_tools(input.provider_options) or {}),
                    "generationConfig": generation_config,
                }),
                timeout_ms=input.timeout_ms,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(f"Gemini request failed with status {response.status_code}.", response.status_code, response_body=await response.text())
        payload = await response.json()
        candidate = (payload.get("candidates") or [None])[0]
        assistant_message = _parse_assistant_message(candidate)
        usage = payload.get("usageMetadata") or {}
        return GenerateResult(
            messages=[assistant_message],
            text="".join(part.text for part in assistant_message.parts if part.type == "text"),
            finish_reason=normalize_finish_reason(candidate.get("finishReason") if candidate else None),
            provider_finish_reason=candidate.get("finishReason") if candidate else None,
            usage=TokenUsage(
                input_tokens=usage.get("promptTokenCount"),
                output_tokens=usage.get("candidatesTokenCount"),
                total_tokens=usage.get("totalTokenCount")
                or ((usage.get("promptTokenCount") or 0) + (usage.get("candidatesTokenCount") or 0)),
            )
            if usage
            else None,
            raw_response=payload,
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        validate_message_parts(self, input.messages)
        generation_config = _generation_config(self.model_id, input) or None
        response = await with_retry(
            lambda: self.fetch(
                self._url("streamGenerateContent?alt=sse"),
                headers={"content-type": "application/json"},
                json_body=drop_none({
                    "contents": _map_messages(input.messages),
                    "systemInstruction": _system_instruction(input.messages),
                    "tools": _map_tools(input.tools, input.provider_options),
                    "toolConfig": _map_tool_config(input.tools, input.tool_choice),
                    **(_provider_options_without_mapped_tools(input.provider_options) or {}),
                    "generationConfig": generation_config,
                }),
                timeout_ms=input.timeout_ms,
                stream=True,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(f"Gemini request failed with status {response.status_code}.", response.status_code, response_body=await response.text())

        async def generator() -> AsyncIterable[StreamEvent]:
            async for event in parse_sse(response.iter_lines()):
                payload = json.loads(event.data)
                candidate = (payload.get("candidates") or [None])[0]
                parts = ((candidate or {}).get("content") or {}).get("parts", [])
                for part in parts:
                    if part.get("text"):
                        yield StreamTextDeltaEvent(text_delta=part["text"])
                    if part.get("functionCall"):
                        call = part["functionCall"]
                        provider_metadata: dict[str, Any] = {}
                        if part.get("thoughtSignature") is not None:
                            provider_metadata["thought_signature"] = part["thoughtSignature"]
                        yield StreamToolCallEvent(
                            tool_call=ToolCall(
                                id=f'{call["name"]}-0',
                                name=call["name"],
                                input=call.get("args") or {},
                                provider_metadata=provider_metadata,
                            )
                        )
                if candidate and candidate.get("finishReason"):
                    yield StreamFinishEvent(
                        finish_reason=normalize_finish_reason(candidate["finishReason"]),
                        provider_finish_reason=candidate["finishReason"],
                    )

        return generator()


@dataclass(slots=True)
class GeminiSpeechModel(SpeechModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: GEMINI_SPEECH_CAPABILITIES)

    def _url(self, action: str) -> str:
        separator = "&" if "?" in action else "?"
        return f"{self.base_url}/models/{self.model_id}:{action}{separator}key={self.api_key}"

    async def generate_speech(
        self,
        *,
        input: str,
        voice: str | None = None,
        provider_options: dict[str, Any] | None = None,
        options: RetryOptions | None = None,
    ) -> SpeechOutput:
        remaining_options, generation_config = _gemini_speech_generation_config(
            provider=self.provider,
            voice=voice or "Kore",
            provider_options=provider_options,
        )
        response = await with_retry(
            lambda: self.fetch(
                self._url("generateContent"),
                headers={"content-type": "application/json"},
                json_body=drop_none({
                    "contents": [{"role": "user", "parts": [{"text": input}]}],
                    "generationConfig": generation_config,
                    **remaining_options,
                }),
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        audio, media_type = _extract_gemini_audio_part(payload, provider=self.provider)
        return SpeechOutput(audio=audio, media_type=media_type, raw_response=payload)


@dataclass(slots=True)
class GeminiTranscriptionModel(TranscriptionModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: GEMINI_TRANSCRIPTION_CAPABILITIES)

    def _url(self, action: str) -> str:
        separator = "&" if "?" in action else "?"
        return f"{self.base_url}/models/{self.model_id}:{action}{separator}key={self.api_key}"

    async def transcribe(
        self,
        *,
        audio: AudioInput,
        prompt: str | None = None,
        language: str | None = None,
        provider_options: dict[str, Any] | None = None,
        options: RetryOptions | None = None,
    ) -> TranscriptionOutput:
        prompt_text = prompt or "Transcribe the provided audio."
        if language:
            prompt_text = f"{prompt_text} Use language code {language}."
        audio_data = audio.data
        if isinstance(audio_data, str):
            encoded_audio = audio_data
        elif isinstance(audio_data, memoryview):
            encoded_audio = base64.b64encode(audio_data.tobytes()).decode("ascii")
        else:
            encoded_audio = base64.b64encode(bytes(audio_data)).decode("ascii")
        response = await with_retry(
            lambda: self.fetch(
                self._url("generateContent"),
                headers={"content-type": "application/json"},
                json_body=drop_none(
                    {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": prompt_text},
                                    {"inlineData": {"mimeType": audio.media_type, "data": encoded_audio}},
                                ],
                            }
                        ],
                        **dict(provider_options or {}),
                    }
                ),
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        candidate = (payload.get("candidates") or [None])[0] or {}
        parts = ((candidate.get("content") or {}).get("parts") or [])
        text = "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
        return TranscriptionOutput(text=text, raw_response=payload)


@dataclass(slots=True)
class GeminiGroundedLanguageModel(GroundedLanguageModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: GEMINI_GROUNDED_CAPABILITIES)

    def _url(self, action: str) -> str:
        separator = "&" if "?" in action else "?"
        return f"{self.base_url}/models/{self.model_id}:{action}{separator}key={self.api_key}"

    async def generate(self, input: GroundedModelGenerateInput) -> GroundedGenerateResult:
        response = await with_retry(
            lambda: self.fetch(
                self._url("generateContent"),
                headers={"content-type": "application/json"},
                json_body=drop_none({
                    "contents": _map_messages(input.messages),
                    "systemInstruction": _system_instruction(input.messages),
                    "tools": _map_tools(None, input.provider_options, force_google_search=True),
                    **(_provider_options_without_mapped_tools(input.provider_options) or {}),
                    "generationConfig": drop_none({
                        "temperature": input.temperature,
                        "maxOutputTokens": input.max_tokens,
                        "thinkingConfig": _map_reasoning(
                            self.model_id,
                            ModelGenerateInput(messages=input.messages, reasoning=input.reasoning),
                        )
                        if input.reasoning is not None
                        else None,
                    }),
                }),
                timeout_ms=input.timeout_ms,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(f"Gemini request failed with status {response.status_code}.", response.status_code, response_body=await response.text())
        payload = await response.json()
        candidate = (payload.get("candidates") or [None])[0]
        assistant_message = _parse_assistant_message(candidate)
        usage = payload.get("usageMetadata") or {}
        return GroundedGenerateResult(
            messages=[assistant_message],
            text="".join(part.text for part in assistant_message.parts if part.type == "text"),
            finish_reason=normalize_finish_reason(candidate.get("finishReason") if candidate else None),
            provider_finish_reason=candidate.get("finishReason") if candidate else None,
            usage=TokenUsage(
                input_tokens=usage.get("promptTokenCount"),
                output_tokens=usage.get("candidatesTokenCount"),
                total_tokens=usage.get("totalTokenCount")
                or ((usage.get("promptTokenCount") or 0) + (usage.get("candidatesTokenCount") or 0)),
            )
            if usage
            else None,
            raw_response=payload,
            sources=_extract_grounding_sources(payload),
            queries=_extract_grounding_queries(payload),
            supports=_extract_grounding_supports(payload),
            search_entry_point=_extract_search_entry_point(payload),
        )


@dataclass(slots=True)
class GeminiEmbeddingModel(EmbeddingModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: GEMINI_CAPABILITIES)

    async def embed(self, values: list[str], options: Any = None) -> EmbedResult:
        config = _embedding_request_options(options)
        task_type = config.get("task_type")
        title = config.get("title")
        output_dimensionality = config.get("output_dimensionality")
        task_types = config.get("task_types")
        titles = config.get("titles")

        if len(values) == 1:
            response = await with_retry(
                lambda: self.fetch(
                    f"{self.base_url}/models/{self.model_id}:embedContent?key={self.api_key}",
                    headers={"content-type": "application/json"},
                    json_body=drop_none(
                        {
                            "content": {"parts": [{"text": values[0]}]},
                            "taskType": task_type,
                            "title": title,
                            "outputDimensionality": output_dimensionality,
                        }
                    ),
                    timeout_ms=_provider_option_value(options, "timeout_ms", "timeoutMs"),
                ),
                max_retries=_provider_option_value(options, "max_retries", "maxRetries") or 0,
                retry_backoff_ms=_provider_option_value(options, "retry_backoff_ms", "retryBackoffMs") or 250,
            )
            if response.status_code >= 400:
                raise ProviderHTTPError(f"Gemini request failed with status {response.status_code}.", response.status_code, response_body=await response.text())
            payload = await response.json()
            return EmbedResult(embeddings=[payload["embedding"]["values"]], raw_response=payload)

        requests = []
        for index, value in enumerate(values):
            requests.append(
                drop_none(
                    {
                        "model": f"models/{self.model_id}",
                        "content": {"parts": [{"text": value}]},
                        "taskType": task_types[index] if isinstance(task_types, list) and index < len(task_types) else task_type,
                        "title": titles[index] if isinstance(titles, list) and index < len(titles) else title,
                        "outputDimensionality": output_dimensionality,
                    }
                )
            )
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/models/{self.model_id}:batchEmbedContents?key={self.api_key}",
                headers={"content-type": "application/json"},
                json_body={"requests": requests},
                timeout_ms=_provider_option_value(options, "timeout_ms", "timeoutMs"),
            ),
            max_retries=_provider_option_value(options, "max_retries", "maxRetries") or 0,
            retry_backoff_ms=_provider_option_value(options, "retry_backoff_ms", "retryBackoffMs") or 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(f"Gemini request failed with status {response.status_code}.", response.status_code, response_body=await response.text())
        payload = await response.json()
        return EmbedResult(embeddings=[item["values"] for item in payload.get("embeddings") or []], raw_response=payload)


@dataclass(slots=True)
class GeminiRealtimeModel(RealtimeModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    realtime_url: str | None = None
    auth_token_url: str | None = None
    connection_factory: RealtimeConnectionFactory | None = None
    capabilities: ModelCapabilities = field(default_factory=lambda: GEMINI_REALTIME_CAPABILITIES)

    async def connect(
        self,
        config: RealtimeSessionConfig | None = None,
        options: RealtimeConnectOptions | None = None,
    ) -> RealtimeSession:
        resolved_config = config or RealtimeSessionConfig()
        url = self.realtime_url or _gemini_realtime_url(self.base_url, self.api_key, resolved_config.provider_options)
        headers = _gemini_realtime_headers(resolved_config.provider_options)
        factory = self.connection_factory or (lambda u, h, o: open_websocket_connection(u, headers=h, options=o))
        connection = await factory(url, headers, options)
        session = CallbackRealtimeSession(
            provider=self.provider,
            model_id=self.model_id,
            capabilities=self.capabilities,
            config=resolved_config,
            connection=connection,
            callbacks=RealtimeSessionCallbacks(
                parse_event=_gemini_realtime_parse_event,
                build_audio_payloads=_gemini_realtime_build_audio,
                build_text_payloads=_gemini_realtime_build_text,
                build_tool_result_payloads=_gemini_realtime_build_tool_result,
                build_update_payloads=lambda session_config: _gemini_realtime_build_update(session_config, self.model_id),
                build_initial_payloads=lambda session_config: _gemini_realtime_build_update(session_config, self.model_id),
            ),
        )
        await session.initialize()
        return session

    async def create_browser_token(
        self,
        config: RealtimeSessionConfig | None = None,
        options: RealtimeConnectOptions | None = None,
    ) -> RealtimeTokenResult:
        resolved_config = config or RealtimeSessionConfig()
        url = self.auth_token_url or f"{self.base_url.replace('/v1beta', '')}/v1alpha/authTokens?key={self.api_key}"
        provider_options = resolved_config.provider_options or {}
        payload = drop_none(
            {
                "authToken": drop_none(
                    {
                        "expireTime": provider_options.get("expireTime") or provider_options.get("expire_time"),
                        "newSessionExpireTime": provider_options.get("newSessionExpireTime") or provider_options.get("new_session_expire_time"),
                        "uses": provider_options.get("uses"),
                    }
                )
            }
        )
        response = await with_retry(
            lambda: self.fetch(
                url,
                method="POST",
                headers={"content-type": "application/json"},
                json_body=payload,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Gemini request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        body = await response.json()
        auth_token = dict(body.get("authToken") or body)
        token_value = auth_token.get("name") or auth_token.get("token") or auth_token.get("accessToken")
        if not isinstance(token_value, str) or not token_value:
            raise ValidationError('Provider "gemini" did not return a valid ephemeral token.')
        return RealtimeTokenResult(
            value=token_value,
            expires_at_ms=_parse_timestamp_ms(auth_token.get("expireTime") or auth_token.get("expire_time")),
            raw_response=body,
        )


def create_gemini(
    *,
    api_key: str | None = None,
    base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    fetch: Fetcher | None = None,
    realtime_url: str | None = None,
    auth_token_url: str | None = None,
    realtime_connection_factory: RealtimeConnectionFactory | None = None,
):
    resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if not resolved_key:
        raise ConfigurationError("Missing Gemini API key.")
    requester = fetch or default_fetch
    base = base_url.rstrip("/")
    native = ProviderAdapter(
        name="gemini",
        language_model_factory=lambda model_id: GeminiLanguageModel(
            provider="gemini", model_id=model_id, api_key=resolved_key, base_url=base, fetch=requester
        ),
        embedding_model_factory=lambda model_id: GeminiEmbeddingModel(
            provider="gemini", model_id=model_id, api_key=resolved_key, base_url=base, fetch=requester
        ),
        transcription_model_factory=lambda model_id: GeminiTranscriptionModel(
            provider="gemini", model_id=model_id, api_key=resolved_key, base_url=base, fetch=requester
        ),
        grounded_language_model_factory=lambda model_id: GeminiGroundedLanguageModel(
            provider="gemini", model_id=model_id, api_key=resolved_key, base_url=base, fetch=requester
        ),
        speech_model_factory=lambda model_id: GeminiSpeechModel(
            provider="gemini", model_id=model_id, api_key=resolved_key, base_url=base, fetch=requester
        ),
        realtime_model_factory=lambda model_id: GeminiRealtimeModel(
            provider="gemini",
            model_id=model_id,
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
            realtime_url=realtime_url,
            auth_token_url=auth_token_url,
            connection_factory=realtime_connection_factory,
        ),
        files_client_factory=lambda: GeminiFilesClient(
            provider="gemini",
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
        ),
        count_tokens_client_factory=lambda: GeminiCountTokensClient(
            provider="gemini",
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
        ),
        file_search_stores_client_factory=lambda: GeminiFileSearchStoresClient(
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
        ),
    )
    return create_provider_bundle(
        name="gemini",
        native=native,
        portable_support=PortableSupport(
            text_generation=True,
            streaming=True,
            structured_output=True,
            tools=True,
            embeddings=True,
            grounding=True,
            retrieval=True,
            transcription=True,
            speech=True,
            portable_badge=True,
            tier="portable",
        ),
    )
