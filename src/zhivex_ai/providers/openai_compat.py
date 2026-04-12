from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import json
import os
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass, field, replace
import time
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

from .._http import Fetcher, default_fetch
from .._sse import parse_sse
from ..errors import ConfigurationError, ProviderHTTPError, UnsupportedFeatureError, ValidationError
from ..messages import normalize_finish_reason, validate_file_part, validate_message_parts
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..realtime import (
    CallbackRealtimeSession,
    RealtimeConnectionFactory,
    RealtimeSessionCallbacks,
    encode_audio_frame,
    open_websocket_connection,
    tool_result_payload,
    unsupported_browser_token,
)
from ..types import (
    AudioFrame,
    AudioInput,
    CodeExecutionResultPart,
    ConversationsClient,
    EmbedResult,
    EmbeddingModel,
    FilePart,
    FilesClient,
    GenerateResult,
    GeneratedCodePart,
    GroundedGenerateResult,
    GroundedLanguageModel,
    GroundedModelGenerateInput,
    GroundingSource,
    ImagePart,
    LanguageModel,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    ProviderFile,
    RealtimeAudioOutputEvent,
    RealtimeConnectOptions,
    RealtimeModel,
    RealtimeSession,
    RealtimeSessionConfig,
    RealtimeSessionEndedEvent,
    RealtimeTextDeltaEvent,
    RealtimeTokenResult,
    RealtimeToolCallEvent,
    RealtimeTranscriptEvent,
    RetryOptions,
    ResponsesClient,
    SpeechModel,
    SpeechOutput,
    StreamEvent,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    TokenUsage,
    ToolCall,
    ToolChoiceName,
    ToolCallPart,
    ToolExecutionResult,
    ToolResultPart,
    TextPart,
    TranscriptionModel,
    TranscriptionOutput,
)
from .base import ProviderAdapter
from ._payload import drop_none

OPENAI_COMPAT_CAPABILITIES = ModelCapabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    parallel_tool_calls=True,
    vision=True,
    files=False,
    audio_input=False,
    audio_output=False,
    embeddings=True,
    reasoning=True,
    web_search=False,
)

OPENAI_COMPAT_TRANSCRIPTION_CAPABILITIES = replace(
    OPENAI_COMPAT_CAPABILITIES,
    streaming=False,
    tools=False,
    structured_output=False,
    json_mode=False,
    tool_choice=False,
    parallel_tool_calls=False,
    vision=False,
    audio_input=True,
    audio_output=False,
    embeddings=False,
    reasoning=False,
    web_search=False,
)

OPENAI_COMPAT_SPEECH_CAPABILITIES = replace(
    OPENAI_COMPAT_TRANSCRIPTION_CAPABILITIES,
    audio_input=False,
    audio_output=True,
)

OPENAI_COMPAT_GROUNDED_CAPABILITIES = replace(
    OPENAI_COMPAT_CAPABILITIES,
    web_search=True,
)

OPENAI_COMPAT_REALTIME_CAPABILITIES = replace(
    OPENAI_COMPAT_CAPABILITIES,
    audio_input=True,
    audio_output=True,
    realtime=True,
    realtime_audio_input=True,
    realtime_audio_output=True,
    realtime_tools=True,
    realtime_browser_tokens=True,
)

_PROVIDER_MANAGED_TOOL_NAMES = {
    "apply_patch_call": "apply_patch",
    "code_interpreter_call": "code_interpreter",
    "computer_call": "computer_use",
    "computer_call_output": "computer_use",
    "file_search_call": "file_search",
    "image_generation_call": "image_generation",
    "local_shell_call": "local_shell",
    "mcp_call": "mcp",
    "shell_call": "shell",
    "web_search_call": "web_search",
}

_TERMINAL_RESPONSE_STATUSES = {"completed", "failed", "incomplete", "cancelled"}


def _parse_json_error(provider_name: str, status_code: int, body: str) -> ProviderHTTPError:
    return ProviderHTTPError(f"{provider_name} request failed with status {status_code}.", status_code, response_body=body)


def _request_url(base_url: str, path: str, params: dict[str, Any] | None = None) -> str:
    if not params:
        return f"{base_url}{path}"
    pairs: list[tuple[str, str]] = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            pairs.extend((key, str(item)) for item in value if item is not None)
            continue
        pairs.append((key, str(value)))
    query = urlencode(pairs)
    return f"{base_url}{path}" if not query else f"{base_url}{path}?{query}"


def _normalize_tool_call_input(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"value": value}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"value": value}


def _image_media_type_from_format(value: str | None, *, default: str = "image/png") -> str:
    normalized = (value or "").strip().lower()
    if not normalized:
        return default
    if "/" in normalized:
        return normalized
    if normalized in {"jpg", "jpeg"}:
        return "image/jpeg"
    return f"image/{normalized}"


def _output_image_part(
    *,
    data: str | None,
    url: str | None = None,
    media_type: str | None = None,
    default_media_type: str = "image/png",
) -> ImagePart | None:
    if isinstance(url, str) and url:
        return ImagePart(image=url, media_type=media_type or default_media_type)
    if not isinstance(data, str) or not data:
        return None
    resolved_media_type = media_type or default_media_type
    image = data if data.startswith("data:") else f"data:{resolved_media_type};base64,{data}"
    return ImagePart(image=image, media_type=resolved_media_type)


def _provider_managed_tool_name(item_type: str) -> str:
    if item_type in _PROVIDER_MANAGED_TOOL_NAMES:
        return _PROVIDER_MANAGED_TOOL_NAMES[item_type]
    if item_type.endswith("_call"):
        return item_type[: -len("_call")]
    return item_type


def _provider_managed_tool_call(item: dict[str, Any]) -> ToolCall:
    item_type = str(item.get("type") or "")
    payload = item.get("arguments")
    if payload is None:
        payload = item.get("input")
    if payload is None and isinstance(item.get("action"), dict):
        payload = item.get("action")
    metadata = dict(item)
    metadata["provider_managed"] = True
    metadata["item_type"] = item_type
    return ToolCall(
        id=str(item.get("call_id") or item.get("id") or item_type),
        name=_provider_managed_tool_name(item_type),
        input=_normalize_tool_call_input(payload),
        provider_metadata=metadata,
    )


def _is_provider_managed_output_item(item: dict[str, Any]) -> bool:
    item_type = str(item.get("type") or "")
    return item_type in _PROVIDER_MANAGED_TOOL_NAMES or item_type.endswith("_call")


def _system_instructions(messages: list[ModelMessage]) -> str | None:
    text = "\n".join(
        part.text for message in messages if message.role == "system" for part in message.parts if part.type == "text"
    )
    return text or None


def _map_file_part(part: FilePart) -> dict[str, Any]:
    validate_file_part(part)
    if part.text is not None or part.document_content is not None:
        raise ValidationError('This SDK does not map FilePart "text" or "document_content" to OpenAI-compatible file inputs.')
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


def _map_message_content(message: ModelMessage) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in message.parts:
        if part.type == "text":
            content.append({"type": "input_text", "text": part.text})
        elif part.type == "image":
            content.append({"type": "input_image", "image_url": part.image})
        elif part.type == "file":
            content.append(_map_file_part(part))
    return content


def _serialize_tool_output(tool_result: ToolExecutionResult) -> str:
    value = (
        {"message": tool_result.error.message}
        if tool_result.is_error and tool_result.error is not None
        else tool_result.output
    )
    return json.dumps(value)


def _to_responses_input(messages: list[ModelMessage]) -> list[dict[str, Any]]:
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
                            "output": _serialize_tool_output(part.tool_result),
                        }
                    )
            continue

        content = _map_message_content(message)
        if content:
            items.append(
                {
                    "type": "message",
                    "role": "assistant" if message.role == "assistant" else "user",
                    "content": content,
                }
            )

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


def _map_tools(tools: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    mapped = []
    for tool in tools.values():
        parameters = create_schema_adapter(tool.schema).json_schema()
        if getattr(tool, "source", None) == "mcp":
            parameters = _normalize_openai_strict_tool_schema(parameters)
        _validate_openai_strict_tool_schema(tool.name, parameters)
        mapped.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "strict": True,
                "parameters": parameters,
            }
        )
    return mapped


def _resolve_json_schema_ref(ref: str, root: dict[str, Any]) -> dict[str, Any] | None:
    if not ref.startswith("#/"):
        return None
    current: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current if isinstance(current, dict) else None


def _visit_json_schema_nodes(schema: Any, root: dict[str, Any], visit: Any, seen: set[int]) -> None:
    if not isinstance(schema, dict):
        if isinstance(schema, list):
            for item in schema:
                _visit_json_schema_nodes(item, root, visit, seen)
        return
    marker = id(schema)
    if marker in seen:
        return
    seen.add(marker)
    visit(schema)
    ref = schema.get("$ref")
    if isinstance(ref, str):
        resolved = _resolve_json_schema_ref(ref, root)
        if resolved is not None:
            _visit_json_schema_nodes(resolved, root, visit, seen)
    for key in ("properties", "$defs", "definitions", "patternProperties"):
        nested = schema.get(key)
        if isinstance(nested, dict):
            for value in nested.values():
                _visit_json_schema_nodes(value, root, visit, seen)
    if isinstance(schema.get("additionalProperties"), dict):
        _visit_json_schema_nodes(schema["additionalProperties"], root, visit, seen)
    if isinstance(schema.get("items"), dict):
        _visit_json_schema_nodes(schema["items"], root, visit, seen)
    elif isinstance(schema.get("items"), list):
        for item in schema["items"]:
            _visit_json_schema_nodes(item, root, visit, seen)
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        nested = schema.get(key)
        if isinstance(nested, list):
            for item in nested:
                _visit_json_schema_nodes(item, root, visit, seen)
    if isinstance(schema.get("not"), dict):
        _visit_json_schema_nodes(schema["not"], root, visit, seen)


def _is_object_schema(schema: dict[str, Any]) -> bool:
    kind = schema.get("type")
    if kind == "object":
        return True
    return isinstance(kind, list) and "object" in kind


def _schema_allows_null(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    kind = schema.get("type")
    if kind == "null":
        return True
    if isinstance(kind, list) and "null" in kind:
        return True
    for key in ("anyOf", "oneOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for item in variants:
                if isinstance(item, dict) and item.get("type") == "null":
                    return True
    return False


def _make_schema_nullable(schema: dict[str, Any]) -> dict[str, Any]:
    if _schema_allows_null(schema):
        return schema
    return {"anyOf": [schema, {"type": "null"}]}


def _normalize_openai_strict_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(schema)

    def visit(node: Any) -> Any:
        if isinstance(node, list):
            return [visit(item) for item in node]
        if not isinstance(node, dict):
            return node

        updated = {key: visit(value) for key, value in node.items()}

        properties = updated.get("properties")
        if isinstance(properties, dict):
            required = updated.get("required")
            required_names = set(required) if isinstance(required, list) else set()
            normalized_properties: dict[str, Any] = {}
            for name, property_schema in properties.items():
                if name not in required_names and isinstance(property_schema, dict):
                    normalized_properties[name] = _make_schema_nullable(property_schema)
                else:
                    normalized_properties[name] = property_schema
            updated["properties"] = normalized_properties
            updated["required"] = list(normalized_properties.keys())

        if _is_object_schema(updated):
            updated["additionalProperties"] = False

        return updated

    return visit(normalized)


def _validate_openai_strict_tool_schema(tool_name: str, schema: dict[str, Any]) -> None:
    errors: list[str] = []

    def visit(node: dict[str, Any]) -> None:
        if not _is_object_schema(node):
            return
        if node.get("additionalProperties") is not False:
            errors.append('set "additionalProperties": false on every object')
        properties = node.get("properties")
        if isinstance(properties, dict) and properties:
            required = node.get("required")
            required_names = set(required) if isinstance(required, list) else set()
            missing = sorted(name for name in properties if name not in required_names)
            if missing:
                errors.append(f'mark every property as required (missing: {", ".join(missing)})')

    _visit_json_schema_nodes(schema, schema, visit, set())
    if not errors:
        return
    unique_errors: list[str] = []
    for error in errors:
        if error not in unique_errors:
            unique_errors.append(error)
    details = "; ".join(unique_errors)
    raise ValidationError(
        f'OpenAI tool "{tool_name}" uses a schema that is incompatible with strict mode: {details}. '
        'Use a closed object schema, for example a Pydantic BaseModel with extra="forbid", '
        "instead of an open-ended mapping like dict[str, str]."
    )


def _map_tool_choice(
    tool_choice: str | ToolChoiceName | None,
    *,
    provider_name: str,
) -> str | dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if provider_name == "openrouter" and tool_choice == "required":
            raise UnsupportedFeatureError('Provider "openrouter" does not support "tool_choice=\\"required\\"" in Responses API.')
        return tool_choice
    return {
        "type": "function",
        "name": tool_choice.tool_name,
    }


def _map_structured_output(input: ModelGenerateInput) -> dict[str, Any] | None:
    if input.structured_output is None or input.structured_output.mode != "native":
        return None
    return {
        "format": {
            "type": "json_schema",
            "name": input.structured_output.name or "response",
            "strict": True,
            "schema": create_schema_adapter(input.structured_output.schema).json_schema(),
        }
    }


def _map_reasoning(input: ModelGenerateInput, provider_name: str) -> dict[str, Any] | None:
    if input.reasoning is None:
        return None
    if input.reasoning.budget_tokens is not None:
        raise UnsupportedFeatureError(f'Provider "{provider_name}" does not support "reasoning.budgetTokens".')
    return {"effort": input.reasoning.effort}


def _parse_responses_usage(payload: dict[str, Any]) -> TokenUsage | None:
    usage = payload.get("usage") or {}
    if not usage:
        return None
    return TokenUsage(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


def _parse_response_finish_reason(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    status = payload.get("status")
    if status == "completed":
        return "stop", status
    if status == "failed":
        return "error", status
    if status == "incomplete":
        reason = (payload.get("incomplete_details") or {}).get("reason")
        return normalize_finish_reason(reason or status), reason or status
    return normalize_finish_reason(status), status


def _parse_output_content_part(content: dict[str, Any]) -> list[Any]:
    content_type = str(content.get("type") or "")
    if content_type in {"output_text", "text"} and content.get("text"):
        provider_metadata = dict(content)
        return [TextPart(text=str(content["text"]), provider_metadata=provider_metadata)]
    if content_type in {"output_image", "image"}:
        image_part = _output_image_part(
            data=content.get("result") or content.get("b64_json") or content.get("image_base64"),
            url=content.get("image_url") or content.get("url"),
            media_type=content.get("media_type"),
            default_media_type=_image_media_type_from_format(content.get("output_format")),
        )
        if image_part is not None:
            image_part.provider_metadata = dict(content)
        return [image_part] if image_part is not None else []
    return []


def _parse_output_item(item: dict[str, Any]) -> list[Any]:
    parts: list[Any] = []
    item_type = str(item.get("type") or "")
    if item_type == "message" and item.get("role") == "assistant":
        for content in item.get("content") or []:
            if isinstance(content, dict):
                parts.extend(_parse_output_content_part(content))
    elif item_type == "function_call":
        parts.append(
            ToolCallPart(
                tool_call=ToolCall(
                    id=item.get("call_id") or item.get("id", ""),
                    name=item.get("name", ""),
                    input=_normalize_tool_call_input(item.get("arguments")),
                )
            )
        )
    elif item_type == "image_generation_call":
        image_part = _output_image_part(
            data=item.get("result"),
            url=item.get("url"),
            media_type=item.get("media_type"),
            default_media_type=_image_media_type_from_format(item.get("output_format")),
        )
        if image_part is not None:
            image_part.provider_metadata = dict(item)
            parts.append(image_part)
        parts.append(ToolCallPart(tool_call=_provider_managed_tool_call(item)))
    elif item_type == "code_interpreter_call":
        if item.get("code"):
            parts.append(
                GeneratedCodePart(
                    code=str(item.get("code") or ""),
                    language=item.get("language") or "python",
                )
            )
        for output in item.get("outputs") or []:
            if not isinstance(output, dict):
                continue
            if output.get("type") == "logs" and output.get("logs") is not None:
                parts.append(
                    CodeExecutionResultPart(
                        output=str(output.get("logs") or ""),
                        outcome=item.get("status"),
                    )
                )
            elif output.get("type") == "image" and output.get("url"):
                parts.append(ImagePart(image=str(output["url"]), provider_metadata=dict(output)))
        parts.append(ToolCallPart(tool_call=_provider_managed_tool_call(item)))
    elif item_type == "computer_call_output":
        output = item.get("output") or {}
        if isinstance(output, dict):
            if output.get("image_url"):
                parts.append(ImagePart(image=str(output["image_url"]), provider_metadata=dict(item)))
            elif output.get("file_id"):
                parts.append(
                    FilePart(
                        file_id=str(output["file_id"]),
                        provider_metadata=dict(item),
                    )
                )
        parts.append(ToolCallPart(tool_call=_provider_managed_tool_call(item)))
    elif item_type == "file_search_call":
        for result in item.get("results") or []:
            if not isinstance(result, dict):
                continue
            source = result.get("text")
            if source:
                parts.append(
                    FilePart(
                        text=str(source),
                        title=result.get("filename"),
                        context=result.get("filename"),
                        provider_metadata=dict(result),
                    )
                )
        parts.append(ToolCallPart(tool_call=_provider_managed_tool_call(item)))
    elif _is_provider_managed_output_item(item):
        parts.append(ToolCallPart(tool_call=_provider_managed_tool_call(item)))
    return parts


def _parse_responses_message(payload: dict[str, Any]) -> ModelMessage:
    parts: list[Any] = []
    for item in payload.get("output") or []:
        parts.extend(_parse_output_item(item))
    return ModelMessage(role="assistant", parts=parts)


def _responses_body(model_id: str, provider_name: str, input: ModelGenerateInput, *, stream: bool) -> dict[str, Any]:
    if provider_name == "qwen" and input.tools:
        raise UnsupportedFeatureError(
            'Provider "qwen" tool calling is not currently supported through this Responses-compatible adapter. '
            "Use a Qwen chat-completions-compatible path for tool calling."
        )
    provider_options = deepcopy(input.provider_options or {})
    provider_tools = provider_options.pop("tools", None)
    if provider_tools is not None and not isinstance(provider_tools, list):
        raise ValidationError('The OpenAI-compatible "provider_options.tools" field must be a list.')
    merged_tools = []
    mapped_tools = _map_tools(input.tools) or []
    if mapped_tools:
        merged_tools.extend(mapped_tools)
    if provider_tools:
        merged_tools.extend(deepcopy(provider_tools))
    body = {
        "model": model_id,
        "instructions": _system_instructions(input.messages),
        "input": _to_responses_input(input.messages),
        "tools": merged_tools or None,
        "tool_choice": _map_tool_choice(input.tool_choice, provider_name=provider_name),
        "text": _map_structured_output(input),
        "temperature": input.temperature,
        "max_output_tokens": input.max_tokens,
        "reasoning": _map_reasoning(input, provider_name),
        "parallel_tool_calls": True if merged_tools else None,
        **provider_options,
        "stream": True if stream else None,
    }
    return drop_none(body)


def _extract_sources(value: Any) -> list[GroundingSource]:
    found: list[GroundingSource] = []

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            if isinstance(node, list):
                for item in node:
                    visit(item)
            return
        if isinstance(node.get("url"), str):
            found.append(
                GroundingSource(
                    url=node["url"],
                    title=node.get("title"),
                    snippet=node.get("snippet"),
                    provider_metadata=node,
                )
            )
        for child in node.values():
            visit(child)

    visit(value)
    deduped: list[GroundingSource] = []
    seen: set[str] = set()
    for item in found:
        if item.url in seen:
            continue
        seen.add(item.url)
        deduped.append(item)
    return deduped


def _normalize_binary(data: bytes | bytearray | memoryview) -> bytes:
    if isinstance(data, memoryview):
        return data.tobytes()
    return bytes(data)


def _normalize_openai_file(payload: dict[str, Any], *, provider: str) -> ProviderFile:
    return ProviderFile(
        provider=provider,
        id=str(payload.get("id") or ""),
        filename=payload.get("filename"),
        media_type=payload.get("mime_type"),
        size_bytes=payload.get("bytes"),
        status=payload.get("status"),
        url=payload.get("url"),
        created_at=payload.get("created_at"),
        metadata=dict(payload),
    )


@dataclass(slots=True)
class OpenAICompatibleFilesClient(FilesClient):
    provider: str
    api_key: str
    base_url: str
    fetch: Fetcher
    auth_header: str = "authorization"
    auth_prefix: str = "Bearer "
    default_purpose: str = "assistants"

    def _headers(self, *, json_content: bool = True) -> dict[str, str]:
        value = self.api_key if not self.auth_prefix else f"{self.auth_prefix}{self.api_key}"
        headers = {self.auth_header: value}
        if json_content:
            headers["content-type"] = "application/json"
        return headers

    async def upload(
        self,
        *,
        data: bytes | bytearray | memoryview,
        filename: str,
        media_type: str = "application/pdf",
        purpose: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderFile:
        response = await self.fetch(
            f"{self.base_url}/files",
            headers=self._headers(json_content=False),
            body={
                "data": {"purpose": purpose or self.default_purpose, **({"metadata": json.dumps(metadata)} if metadata else {})},
                "files": {"file": (filename, _normalize_binary(data), media_type)},
            },
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_file(await response.json(), provider=self.provider)

    async def list(self) -> list[ProviderFile]:
        response = await self.fetch(
            f"{self.base_url}/files",
            method="GET",
            headers=self._headers(),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        return [_normalize_openai_file(dict(item), provider=self.provider) for item in payload.get("data") or []]

    async def get(self, file_id: str) -> ProviderFile:
        response = await self.fetch(
            f"{self.base_url}/files/{file_id}",
            method="GET",
            headers=self._headers(),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_file(await response.json(), provider=self.provider)

    async def download(self, file_id: str) -> bytes:
        response = await self.fetch(
            f"{self.base_url}/files/{file_id}/content",
            method="GET",
            headers=self._headers(json_content=False),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.read()

    async def delete(self, file_id: str) -> bool:
        response = await self.fetch(
            f"{self.base_url}/files/{file_id}",
            method="DELETE",
            headers=self._headers(),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        return bool(payload.get("deleted"))


def _audio_payload(audio: AudioInput) -> dict[str, Any]:
    data = audio.data
    if isinstance(data, str):
        encoded = data
    elif isinstance(data, memoryview):
        encoded = base64.b64encode(data.tobytes()).decode("ascii")
    else:
        encoded = base64.b64encode(bytes(data)).decode("ascii")
    return {
        "data": {
            "model": None,
            "prompt": None,
            "language": None,
        },
        "files": {
            "file": (audio.filename or "audio", base64.b64decode(encoded), audio.media_type),
        },
    }


def _openai_realtime_url(base_url: str, model_id: str, provider_options: dict[str, Any] | None = None) -> str:
    override = (provider_options or {}).get("realtime_url")
    if isinstance(override, str) and override:
        return override
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/")
    query = dict((provider_options or {}).get("realtime_query") or {})
    query.setdefault("model", model_id)
    return urlunparse((scheme, parsed.netloc, f"{path}/realtime", "", urlencode(query), ""))


def _openai_realtime_headers(
    api_key: str,
    *,
    auth_header: str,
    auth_prefix: str,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    value = api_key if not auth_prefix else f"{auth_prefix}{api_key}"
    headers = {
        auth_header: value,
        "OpenAI-Beta": "realtime=v1",
    }
    headers.update(dict(extra_headers or {}))
    return headers


def _openai_realtime_tools(config: RealtimeSessionConfig) -> list[dict[str, Any]] | None:
    return _map_tools(config.tools)


def _openai_realtime_session_payload(config: RealtimeSessionConfig) -> dict[str, Any]:
    session: dict[str, Any] = {
        "instructions": config.instructions,
        "voice": config.voice,
        "tools": _openai_realtime_tools(config),
        "tool_choice": _map_tool_choice(config.tool_choice, provider_name="openai") if config.tool_choice is not None else None,
        "input_audio_format": config.input_audio_media_type,
        "output_audio_format": config.output_audio_media_type,
        "turn_detection": config.turn_detection,
        **(config.provider_options or {}),
    }
    return {"type": "session.update", "session": drop_none(session)}


def _openai_realtime_build_audio(frame: AudioFrame, config: RealtimeSessionConfig) -> list[dict[str, Any]]:
    payloads = [{"type": "input_audio_buffer.append", "audio": encode_audio_frame(frame)}]
    if frame.is_final:
        payloads.append({"type": "input_audio_buffer.commit"})
        if config.auto_response:
            payloads.append({"type": "response.create"})
    return payloads


def _openai_realtime_build_text(text: str, config: RealtimeSessionConfig) -> list[dict[str, Any]]:
    payloads = [
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }
    ]
    if config.auto_response:
        payloads.append({"type": "response.create"})
    return payloads


def _openai_realtime_build_tool_result(result: ToolExecutionResult, config: RealtimeSessionConfig) -> list[dict[str, Any]]:
    payloads = [
        {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": result.tool_call_id,
                "output": json.dumps(tool_result_payload(result)),
            },
        }
    ]
    if config.auto_response:
        payloads.append({"type": "response.create"})
    return payloads


def _openai_realtime_build_update(config: RealtimeSessionConfig) -> list[dict[str, Any]]:
    return [_openai_realtime_session_payload(config)]


def _openai_realtime_parse_event(payload: dict[str, Any]) -> list[Any]:
    event_type = str(payload.get("type") or "")
    if event_type in {"response.text.delta", "response.output_text.delta"}:
        return [
            RealtimeTextDeltaEvent(
                text_delta=str(payload.get("delta") or ""),
                item_id=payload.get("item_id"),
                response_id=payload.get("response_id"),
                provider_metadata=payload,
            )
        ]
    if event_type == "response.audio.delta":
        delta = payload.get("delta")
        audio = base64.b64decode(delta) if isinstance(delta, str) and delta else b""
        return [
            RealtimeAudioOutputEvent(
                audio=audio,
                media_type=str(payload.get("media_type") or "audio/pcm"),
                sample_rate_hz=payload.get("sample_rate_hz"),
                channels=payload.get("channels"),
                item_id=payload.get("item_id"),
                response_id=payload.get("response_id"),
                provider_metadata=payload,
            )
        ]
    if event_type in {"conversation.item.input_audio_transcription.completed", "input_audio_buffer.transcription.completed"}:
        return [
            RealtimeTranscriptEvent(
                text=str(payload.get("transcript") or ""),
                role="user",
                is_final=True,
                item_id=payload.get("item_id"),
                response_id=payload.get("response_id"),
                provider_metadata=payload,
            )
        ]
    if event_type in {"response.audio_transcript.delta", "response.audio_transcription.delta"}:
        return [
            RealtimeTranscriptEvent(
                text=str(payload.get("delta") or ""),
                role="assistant",
                is_final=False,
                item_id=payload.get("item_id"),
                response_id=payload.get("response_id"),
                provider_metadata=payload,
            )
        ]
    if event_type in {"response.audio_transcript.done", "response.output_text.done"}:
        text = payload.get("transcript")
        if text is None:
            text = payload.get("text")
        return [
            RealtimeTranscriptEvent(
                text=str(text or ""),
                role="assistant",
                is_final=True,
                item_id=payload.get("item_id"),
                response_id=payload.get("response_id"),
                provider_metadata=payload,
            )
        ]
    if event_type in {"response.output_item.done", "response.function_call_arguments.done"}:
        item = dict(payload.get("item") or {})
        if not item and payload.get("name"):
            item = payload
        if item.get("type") == "function_call" or item.get("name"):
            arguments = item.get("arguments") or "{}"
            return [
                RealtimeToolCallEvent(
                    tool_call=ToolCall(
                        id=item.get("call_id") or item.get("id", ""),
                        name=item.get("name", ""),
                        input=json.loads(arguments),
                    )
                )
            ]
    if event_type in {"response.done", "response.completed", "response.incomplete", "response.failed", "session.closed"}:
        return [RealtimeSessionEndedEvent(reason=event_type, provider_metadata=payload)]
    return []


@dataclass(slots=True)
class _BaseOpenAICompatible:
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    auth_header: str = "authorization"
    auth_prefix: str = "Bearer "

    def _headers(self, *, json_content: bool = True) -> dict[str, str]:
        value = self.api_key if not self.auth_prefix else f"{self.auth_prefix}{self.api_key}"
        headers = {self.auth_header: value}
        if json_content:
            headers["content-type"] = "application/json"
        return headers


@dataclass(slots=True)
class OpenAICompatibleLanguageModel(_BaseOpenAICompatible, LanguageModel):
    capabilities: ModelCapabilities = field(default_factory=lambda: OPENAI_COMPAT_CAPABILITIES)

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        validate_message_parts(self, input.messages)
        body = _responses_body(self.model_id, self.provider, input, stream=False)
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/responses",
                headers=self._headers(),
                json_body=body,
                timeout_ms=input.timeout_ms,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        assistant_message = _parse_responses_message(payload)
        finish_reason, provider_finish_reason = _parse_response_finish_reason(payload)
        return GenerateResult(
            messages=[assistant_message],
            text="".join(part.text for part in assistant_message.parts if part.type == "text"),
            finish_reason=finish_reason,
            provider_finish_reason=provider_finish_reason,
            usage=_parse_responses_usage(payload),
            raw_response=payload,
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        validate_message_parts(self, input.messages)
        body = _responses_body(self.model_id, self.provider, input, stream=True)
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/responses",
                headers=self._headers(),
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
            async for event in parse_sse(response.iter_lines()):
                if event.data == "[DONE]":
                    return
                payload = json.loads(event.data)
                if payload.get("type") == "response.output_text.delta":
                    yield StreamTextDeltaEvent(text_delta=payload.get("delta", ""))
                    continue
                if payload.get("type") == "response.output_item.done":
                    item = payload.get("item") or {}
                    if item.get("type") == "function_call":
                        yield StreamToolCallEvent(
                            tool_call=ToolCall(
                                id=item.get("call_id") or item.get("id", ""),
                                name=item.get("name", ""),
                                input=_normalize_tool_call_input(item.get("arguments")),
                            )
                        )
                    elif isinstance(item, dict) and _is_provider_managed_output_item(item):
                        yield StreamToolCallEvent(tool_call=_provider_managed_tool_call(item))
                    continue
                if payload.get("type") in {"response.completed", "response.incomplete", "response.failed"}:
                    response_payload = payload.get("response") or {}
                    finish_reason, provider_finish_reason = _parse_response_finish_reason(response_payload)
                    yield StreamFinishEvent(
                        finish_reason=finish_reason,
                        provider_finish_reason=provider_finish_reason,
                        usage=_parse_responses_usage(response_payload),
                    )

        return generator()


@dataclass(slots=True)
class OpenAICompatibleEmbeddingModel(_BaseOpenAICompatible, EmbeddingModel):
    capabilities: ModelCapabilities = field(default_factory=lambda: OPENAI_COMPAT_CAPABILITIES)

    async def embed(self, values: list[str], options: RetryOptions | None = None) -> EmbedResult:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json_body={"model": self.model_id, "input": values},
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        usage = payload.get("usage") or {}
        return EmbedResult(
            embeddings=[entry["embedding"] for entry in payload.get("data", [])],
            usage=TokenUsage(
                input_tokens=usage.get("prompt_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
            raw_response=payload,
        )


@dataclass(slots=True)
class OpenAICompatibleTranscriptionModel(_BaseOpenAICompatible, TranscriptionModel):
    capabilities: ModelCapabilities = field(default_factory=lambda: OPENAI_COMPAT_TRANSCRIPTION_CAPABILITIES)

    async def transcribe(
        self,
        *,
        audio: AudioInput,
        prompt: str | None = None,
        language: str | None = None,
        provider_options: dict[str, Any] | None = None,
        options: RetryOptions | None = None,
    ) -> TranscriptionOutput:
        form = _audio_payload(audio)
        form["data"]["model"] = self.model_id
        if prompt:
            form["data"]["prompt"] = prompt
        if language:
            form["data"]["language"] = language
        for key, value in (provider_options or {}).items():
            form["data"][key] = value if isinstance(value, str) else json.dumps(value)

        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/audio/transcriptions",
                headers=self._headers(json_content=False),
                body=form,
                json_body=None,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        return TranscriptionOutput(text=payload.get("text", ""), raw_response=payload)


@dataclass(slots=True)
class OpenAICompatibleSpeechModel(_BaseOpenAICompatible, SpeechModel):
    capabilities: ModelCapabilities = field(default_factory=lambda: OPENAI_COMPAT_SPEECH_CAPABILITIES)

    async def generate_speech(
        self,
        *,
        input: str,
        voice: str | None = None,
        provider_options: dict[str, Any] | None = None,
        options: RetryOptions | None = None,
    ) -> SpeechOutput:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/audio/speech",
                headers=self._headers(),
                json_body={
                    "model": self.model_id,
                    "input": input,
                    "voice": voice or "alloy",
                    **(provider_options or {}),
                },
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return SpeechOutput(
            audio=await response.read(),
            media_type=response.headers.get("content-type", "audio/mpeg"),
            raw_response=None,
        )


def _audio_format_to_media_type(value: str | None, *, default: str = "audio/wav") -> str:
    normalized = (value or "").strip().lower()
    if normalized in {"wav", "wave"}:
        return "audio/wav"
    if normalized == "mp3":
        return "audio/mpeg"
    if normalized == "flac":
        return "audio/flac"
    if normalized in {"opus", "ogg_opus"}:
        return "audio/ogg"
    if normalized in {"pcm", "pcm16"}:
        return "audio/pcm"
    return default


@dataclass(slots=True)
class OpenAICompatibleChatSpeechModel(_BaseOpenAICompatible, SpeechModel):
    capabilities: ModelCapabilities = field(default_factory=lambda: OPENAI_COMPAT_SPEECH_CAPABILITIES)

    async def generate_speech(
        self,
        *,
        input: str,
        voice: str | None = None,
        provider_options: dict[str, Any] | None = None,
        options: RetryOptions | None = None,
    ) -> SpeechOutput:
        remaining_options = deepcopy(provider_options or {})
        audio_config = deepcopy(dict(remaining_options.pop("audio", {}) or {}))
        audio_config.setdefault("format", "wav")
        audio_config["voice"] = voice or audio_config.get("voice") or "alloy"
        modalities = list(remaining_options.pop("modalities", []) or ["text", "audio"])
        if "audio" not in modalities:
            modalities.append("audio")

        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json_body={
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": input}],
                    "modalities": modalities,
                    "audio": audio_config,
                    "stream": True,
                    **remaining_options,
                },
                timeout_ms=options.timeout_ms if options else None,
                stream=True,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())

        raw_chunks: list[dict[str, Any]] = []
        audio_chunks: list[str] = []
        async for event in parse_sse(response.iter_lines()):
            if event.data == "[DONE]":
                break
            payload = json.loads(event.data)
            raw_chunks.append(payload)
            delta = ((payload.get("choices") or [{}])[0] or {}).get("delta") or {}
            audio = delta.get("audio") or {}
            data = audio.get("data")
            if isinstance(data, str) and data:
                audio_chunks.append(data)

        if not audio_chunks:
            raise ValidationError(f'Provider "{self.provider}" did not return audio data for speech generation.')
        return SpeechOutput(
            audio=base64.b64decode("".join(audio_chunks)),
            media_type=_audio_format_to_media_type(str(audio_config.get("format") or "")),
            raw_response=raw_chunks,
        )


@dataclass(slots=True)
class OpenAICompatibleGroundedLanguageModel(_BaseOpenAICompatible, GroundedLanguageModel):
    capabilities: ModelCapabilities = field(default_factory=lambda: OPENAI_COMPAT_GROUNDED_CAPABILITIES)
    default_web_search_tool: dict[str, Any] = field(default_factory=lambda: {"type": "web_search"})

    async def generate(self, input: GroundedModelGenerateInput) -> GroundedGenerateResult:
        provider_options = deepcopy(input.provider_options or {})
        web_search_tool = provider_options.pop("web_search", None)
        if web_search_tool is None:
            web_search_tool = deepcopy(self.default_web_search_tool)
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/responses",
                headers=self._headers(),
                json_body={
                    "model": self.model_id,
                    "input": _to_responses_input(input.messages),
                    "tools": [web_search_tool],
                    "temperature": input.temperature,
                    "max_output_tokens": input.max_tokens,
                    **provider_options,
                },
                timeout_ms=input.timeout_ms,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        usage = payload.get("usage") or {}
        return GroundedGenerateResult(
            text=payload.get("output_text"),
            sources=_extract_sources(payload),
            finish_reason=normalize_finish_reason(payload.get("status")),
            provider_finish_reason=payload.get("status"),
            usage=TokenUsage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
            raw_response=payload,
        )


@dataclass(slots=True)
class OpenAICompatibleResponsesClient(_BaseOpenAICompatible, ResponsesClient):
    async def create(self, body: dict[str, Any], options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/responses",
                headers=self._headers(),
                json_body=body,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def create_background(self, body: dict[str, Any], options: RetryOptions | None = None) -> dict[str, Any]:
        payload = dict(body)
        payload["background"] = True
        return await self.create(payload, options=options)

    async def retrieve(
        self,
        response_id: str,
        *,
        include: list[str] | None = None,
        stream: bool | None = None,
        starting_after: int | None = None,
        include_obfuscation: bool | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                _request_url(
                    self.base_url,
                    f"/responses/{response_id}",
                    {
                        "include": include,
                        "stream": stream,
                        "starting_after": starting_after,
                        "include_obfuscation": include_obfuscation,
                    },
                ),
                method="GET",
                headers=self._headers(),
                json_body=None,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def wait(
        self,
        response_id: str,
        *,
        include: list[str] | None = None,
        stream: bool | None = None,
        starting_after: int | None = None,
        include_obfuscation: bool | None = None,
        poll_interval_ms: int = 500,
        timeout_ms: int | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        deadline = None if timeout_ms is None else time.monotonic() + max(0, timeout_ms) / 1000
        while True:
            payload = await self.retrieve(
                response_id,
                include=include,
                stream=stream,
                starting_after=starting_after,
                include_obfuscation=include_obfuscation,
                options=options,
            )
            status = str(payload.get("status") or "").lower()
            if not status or status in _TERMINAL_RESPONSE_STATUSES:
                return payload
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'OpenAI response "{response_id}" did not reach a terminal state before timeout.')
                await asyncio.sleep(min(max(poll_interval_ms, 0) / 1000, remaining))
            else:
                await asyncio.sleep(max(poll_interval_ms, 0) / 1000)

    async def delete(self, response_id: str, options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/responses/{response_id}",
                method="DELETE",
                headers=self._headers(),
                json_body=None,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def list_input_items(
        self,
        response_id: str,
        *,
        after: str | None = None,
        before: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        include: list[str] | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                _request_url(
                    self.base_url,
                    f"/responses/{response_id}/input_items",
                    {
                        "after": after,
                        "before": before,
                        "include": include,
                        "limit": limit,
                        "order": order,
                    },
                ),
                method="GET",
                headers=self._headers(),
                json_body=None,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def count_input_tokens(
        self,
        body: dict[str, Any],
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/responses/input_tokens",
                headers=self._headers(),
                json_body=body,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def cancel(self, response_id: str, options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/responses/{response_id}/cancel",
                headers=self._headers(),
                json_body={},
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def compact(
        self,
        body: dict[str, Any],
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/responses/compact",
                headers=self._headers(),
                json_body=body,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()


@dataclass(slots=True)
class OpenAICompatibleConversationsClient(_BaseOpenAICompatible, ConversationsClient):
    async def create(self, body: dict[str, Any], options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/conversations",
                headers=self._headers(),
                json_body=body,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def retrieve(self, conversation_id: str, options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/conversations/{conversation_id}",
                method="GET",
                headers=self._headers(),
                json_body=None,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def update(
        self,
        conversation_id: str,
        body: dict[str, Any],
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/conversations/{conversation_id}",
                method="POST",
                headers=self._headers(),
                json_body=body,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def delete(self, conversation_id: str, options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/conversations/{conversation_id}",
                method="DELETE",
                headers=self._headers(),
                json_body=None,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def create_item(
        self,
        conversation_id: str,
        body: dict[str, Any],
        *,
        include: list[str] | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                _request_url(self.base_url, f"/conversations/{conversation_id}/items", {"include": include}),
                headers=self._headers(),
                json_body=body,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def retrieve_item(
        self,
        conversation_id: str,
        item_id: str,
        *,
        include: list[str] | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                _request_url(
                    self.base_url,
                    f"/conversations/{conversation_id}/items/{item_id}",
                    {"include": include},
                ),
                method="GET",
                headers=self._headers(),
                json_body=None,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def delete_item(
        self,
        conversation_id: str,
        item_id: str,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/conversations/{conversation_id}/items/{item_id}",
                method="DELETE",
                headers=self._headers(),
                json_body=None,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def list_items(
        self,
        conversation_id: str,
        *,
        after: str | None = None,
        before: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        include: list[str] | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                _request_url(
                    self.base_url,
                    f"/conversations/{conversation_id}/items",
                    {
                        "after": after,
                        "before": before,
                        "include": include,
                        "limit": limit,
                        "order": order,
                    },
                ),
                method="GET",
                headers=self._headers(),
                json_body=None,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()


@dataclass(slots=True)
class OpenAICompatibleRealtimeModel(_BaseOpenAICompatible, RealtimeModel):
    capabilities: ModelCapabilities = field(default_factory=lambda: OPENAI_COMPAT_REALTIME_CAPABILITIES)
    realtime_url: str | None = None
    browser_token_url: str | None = None
    connection_factory: RealtimeConnectionFactory | None = None

    async def connect(
        self,
        config: RealtimeSessionConfig | None = None,
        options: RealtimeConnectOptions | None = None,
    ) -> RealtimeSession:
        resolved_config = config or RealtimeSessionConfig()
        headers = _openai_realtime_headers(
            self.api_key,
            auth_header=self.auth_header,
            auth_prefix=self.auth_prefix,
            extra_headers=dict((resolved_config.provider_options or {}).get("headers") or {}),
        )
        url = self.realtime_url or _openai_realtime_url(self.base_url, self.model_id, resolved_config.provider_options)
        factory = self.connection_factory or (lambda u, h, o: open_websocket_connection(u, headers=h, options=o))
        connection = await factory(url, headers, options)
        session = CallbackRealtimeSession(
            provider=self.provider,
            model_id=self.model_id,
            capabilities=self.capabilities,
            config=resolved_config,
            connection=connection,
            callbacks=RealtimeSessionCallbacks(
                parse_event=_openai_realtime_parse_event,
                build_audio_payloads=_openai_realtime_build_audio,
                build_text_payloads=_openai_realtime_build_text,
                build_tool_result_payloads=_openai_realtime_build_tool_result,
                build_update_payloads=_openai_realtime_build_update,
                build_initial_payloads=lambda session_config: _openai_realtime_build_update(session_config),
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
        url = self.browser_token_url or f"{self.base_url}/realtime/sessions"
        response = await with_retry(
            lambda: self.fetch(
                url,
                headers=self._headers(),
                json_body=drop_none({
                    "model": self.model_id,
                    "voice": resolved_config.voice,
                    "instructions": resolved_config.instructions,
                    "tools": _openai_realtime_tools(resolved_config),
                    **(resolved_config.provider_options or {}),
                }),
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        secret = payload.get("client_secret")
        if isinstance(secret, dict):
            value = str(secret.get("value") or "")
            expires_at_ms = secret.get("expires_at_ms")
        else:
            value = str(payload.get("token") or payload.get("value") or "")
            expires_at_ms = payload.get("expires_at_ms")
        return RealtimeTokenResult(value=value, expires_at_ms=expires_at_ms, raw_response=payload)


def create_openai_compatible_provider(
    *,
    provider_name: str,
    env_var: str,
    base_url: str,
    api_key: str | None = None,
    fetch: Fetcher | None = None,
    auth_header: str = "authorization",
    auth_prefix: str = "Bearer ",
    capabilities: ModelCapabilities | None = None,
    supports_audio: bool = False,
    supports_transcription: bool | None = None,
    supports_speech: bool | None = None,
    supports_grounding: bool = False,
    supports_realtime: bool = False,
    speech_transport: str = "audio_speech",
    realtime_url: str | None = None,
    browser_token_url: str | None = None,
    realtime_connection_factory: RealtimeConnectionFactory | None = None,
    default_grounding_tool: dict[str, Any] | None = None,
    files_client_factory: Callable[[], FilesClient] | None = None,
    responses_client_factory: Callable[[], ResponsesClient] | None = None,
    conversations_client_factory: Callable[[], ConversationsClient] | None = None,
) -> ProviderAdapter:
    resolved_key = api_key or os.getenv(env_var)
    if not resolved_key:
        raise ConfigurationError(f"Missing {provider_name} API key.")
    requester = fetch or default_fetch
    base = base_url.rstrip("/")
    shared_capabilities = capabilities or OPENAI_COMPAT_CAPABILITIES
    resolved_supports_transcription = supports_audio if supports_transcription is None else supports_transcription
    resolved_supports_speech = supports_audio if supports_speech is None else supports_speech
    return ProviderAdapter(
        name=provider_name,
        language_model_factory=lambda model_id: OpenAICompatibleLanguageModel(
            provider=provider_name,
            model_id=model_id,
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
            capabilities=shared_capabilities,
        ),
        embedding_model_factory=lambda model_id: OpenAICompatibleEmbeddingModel(
            provider=provider_name,
            model_id=model_id,
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
            capabilities=shared_capabilities,
        ),
        transcription_model_factory=(
            (lambda model_id: OpenAICompatibleTranscriptionModel(
                provider=provider_name,
                model_id=model_id,
                api_key=resolved_key,
                base_url=base,
                fetch=requester,
                auth_header=auth_header,
                auth_prefix=auth_prefix,
            ))
            if resolved_supports_transcription
            else None
        ),
        speech_model_factory=(
            (
                (lambda model_id: OpenAICompatibleSpeechModel(
                    provider=provider_name,
                    model_id=model_id,
                    api_key=resolved_key,
                    base_url=base,
                    fetch=requester,
                    auth_header=auth_header,
                    auth_prefix=auth_prefix,
                ))
                if speech_transport == "audio_speech"
                else (lambda model_id: OpenAICompatibleChatSpeechModel(
                    provider=provider_name,
                    model_id=model_id,
                    api_key=resolved_key,
                    base_url=base,
                    fetch=requester,
                    auth_header=auth_header,
                    auth_prefix=auth_prefix,
                ))
            )
            if resolved_supports_speech
            else None
        ),
        grounded_language_model_factory=(
            (lambda model_id: OpenAICompatibleGroundedLanguageModel(
                provider=provider_name,
                model_id=model_id,
                api_key=resolved_key,
                base_url=base,
                fetch=requester,
                auth_header=auth_header,
                auth_prefix=auth_prefix,
                default_web_search_tool=deepcopy(default_grounding_tool) if default_grounding_tool is not None else {"type": "web_search"},
            ))
            if supports_grounding
            else None
        ),
        realtime_model_factory=(
            (lambda model_id: OpenAICompatibleRealtimeModel(
                provider=provider_name,
                model_id=model_id,
                api_key=resolved_key,
                base_url=base,
                fetch=requester,
                auth_header=auth_header,
                auth_prefix=auth_prefix,
                realtime_url=realtime_url,
                browser_token_url=browser_token_url,
                connection_factory=realtime_connection_factory,
            ))
            if supports_realtime
            else None
        ),
        files_client_factory=files_client_factory,
        responses_client_factory=responses_client_factory,
        conversations_client_factory=conversations_client_factory,
    )
