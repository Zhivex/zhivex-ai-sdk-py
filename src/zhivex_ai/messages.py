from __future__ import annotations

import json
from dataclasses import is_dataclass
from typing import Any

from .errors import UnsupportedFeatureError, ValidationError
from .types import (
    ContentPart,
    FilePart,
    FinishReason,
    GenerateResult,
    ImagePart,
    LanguageModel,
    MCPToolConfig,
    MessageRole,
    ModelMessage,
    RemoteHTTPToolConfig,
    TextPart,
    ToolCall,
    ToolCallPart,
    ToolDefinition,
    ToolSource,
    ToolExecutionResult,
    ToolResultPart,
)


def text_part(text: str) -> TextPart:
    return TextPart(text=text)


def tool_call_part(tool_call: ToolCall) -> ToolCallPart:
    return ToolCallPart(tool_call=tool_call)


def tool_result_part(tool_result: ToolExecutionResult) -> ToolResultPart:
    return ToolResultPart(tool_result=tool_result)


def _normalize_message_parts(input: str | list[ContentPart]) -> list[ContentPart]:
    return [text_part(input)] if isinstance(input, str) else input


def create_text_message(role: MessageRole, text: str) -> ModelMessage:
    return ModelMessage(role=role, parts=[text_part(text)])


def system(text: str) -> ModelMessage:
    return ModelMessage(role="system", parts=[text_part(text)])


def user(input: str | list[ContentPart]) -> ModelMessage:
    return ModelMessage(role="user", parts=_normalize_message_parts(input))


def assistant(input: str | list[ContentPart]) -> ModelMessage:
    return ModelMessage(role="assistant", parts=_normalize_message_parts(input))


def get_text_from_parts(parts: list[ContentPart]) -> str:
    return "".join(part.text for part in parts if isinstance(part, TextPart))


def get_text_from_messages(messages: list[ModelMessage]) -> str:
    return "".join(get_text_from_parts(message.parts) for message in messages if message.role == "assistant")


def get_text_from_result(result: GenerateResult) -> str:
    return get_text_from_messages(result_messages(result))


def _to_json_compatible(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _to_json_compatible(v) for k, v in value.__dict__.items()}
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(v) for v in value]
    return value


def serialize_json_value(value: Any) -> Any:
    return json.loads(json.dumps(_to_json_compatible(value)))


def tool(
    definition: ToolDefinition | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: Any = None,
    execute: Any = None,
    input_examples: list[Any] | None = None,
    strict: bool | None = None,
    defer_loading: bool | None = None,
    eager_input_streaming: bool | None = None,
    allowed_callers: list[str] | None = None,
    cache_control: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    requires_approval: bool | None = None,
    permissions: list[str] | None = None,
    source: ToolSource = "local",
    metadata: dict[str, Any] | None = None,
    supports_streaming: bool = False,
    remote_config: RemoteHTTPToolConfig | None = None,
    mcp_config: MCPToolConfig | None = None,
) -> ToolDefinition:
    if definition is not None:
        return definition
    if not name:
        raise ValueError('Pass either an existing ToolDefinition or at least a "name".')
    if source == "local" and execute is None:
        raise ValueError('Local tools require an "execute" callable.')
    if source == "remote" and remote_config is None:
        raise ValueError('Remote tools require a "remote_config".')
    if source == "mcp" and mcp_config is None:
        raise ValueError('MCP tools require an "mcp_config".')
    return ToolDefinition(
        name=name,
        description=description,
        schema=schema,
        execute=execute,
        input_examples=[serialize_json_value(item) for item in (input_examples or [])],
        strict=strict,
        defer_loading=defer_loading,
        eager_input_streaming=eager_input_streaming,
        allowed_callers=list(allowed_callers or []),
        cache_control=serialize_json_value(cache_control) if cache_control is not None else None,
        tags=list(tags or []),
        requires_approval=requires_approval,
        permissions=list(permissions or []),
        source=source,
        metadata=dict(metadata or {}),
        supports_streaming=supports_streaming,
        remote_config=remote_config,
        mcp_config=mcp_config,
    )


def remote_tool(
    *,
    name: str,
    url: str,
    schema: Any,
    description: str | None = None,
    headers: dict[str, str] | None = None,
    timeout_ms: int | None = None,
    tags: list[str] | None = None,
    requires_approval: bool | None = None,
    permissions: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolDefinition:
    return tool(
        name=name,
        description=description,
        schema=schema,
        execute=None,
        tags=tags,
        requires_approval=requires_approval,
        permissions=permissions,
        source="remote",
        metadata=metadata,
        remote_config=RemoteHTTPToolConfig(
            url=url,
            headers=dict(headers or {}),
            timeout_ms=timeout_ms,
        ),
    )


def normalize_finish_reason(reason: str | None) -> FinishReason | None:
    if not reason:
        return None
    lowered = reason.lower()
    if lowered in {"stop", "end_turn"}:
        return "stop"
    if lowered in {"length", "max_tokens"}:
        return "length"
    if lowered in {"tool_calls", "tool_use"}:
        return "tool-calls"
    if lowered == "content_filter":
        return "content-filter"
    if lowered == "error":
        return "error"
    return "unknown"


def validate_file_part(part: FilePart) -> None:
    sources = [
        name
        for name, value in (
            ("data", part.data),
            ("text", part.text),
            ("document_content", part.document_content),
            ("file_id", part.file_id),
            ("file_uri", part.file_uri),
            ("url", part.url),
        )
        if value
    ]
    if len(sources) != 1:
        raise ValidationError(
            'FilePart requires exactly one source: "data", "text", "document_content", "file_id", "file_uri", or "url".'
        )

    media_type = (part.media_type or "").strip().lower()
    if sources == ["data"] and not media_type:
        raise ValidationError('Inline FilePart values require "media_type".')


def result_messages(result: GenerateResult) -> list[ModelMessage]:
    if result.messages:
        return result.messages
    if result.message:
        return [result.message]
    if result.text:
        return [create_text_message("assistant", result.text)]
    return []


def validate_message_parts(model: LanguageModel, messages: list[ModelMessage]) -> None:
    for message in messages:
        for part in message.parts:
            if isinstance(part, ImagePart) and not model.capabilities.vision:
                raise UnsupportedFeatureError(
                    f'Model "{model.provider}/{model.model_id}" does not support image inputs.'
                )
            if part.type == "file" and not model.capabilities.files:
                raise UnsupportedFeatureError(
                    f'Model "{model.provider}/{model.model_id}" does not support file inputs.'
                )
            if part.type == "file":
                validate_file_part(part)
            if part.type in {"tool-call", "tool-result"} and not model.capabilities.tools:
                raise UnsupportedFeatureError(
                    f'Model "{model.provider}/{model.model_id}" does not support tool calling.'
                )
