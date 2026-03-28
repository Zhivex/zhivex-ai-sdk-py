from __future__ import annotations

import json
from dataclasses import is_dataclass
from typing import Any

from .errors import UnsupportedFeatureError
from .types import (
    ContentPart,
    FinishReason,
    GenerateResult,
    ImagePart,
    LanguageModel,
    MessageRole,
    ModelMessage,
    TextPart,
    ToolCall,
    ToolCallPart,
    ToolDefinition,
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
) -> ToolDefinition:
    if definition is not None:
        return definition
    if not name or execute is None:
        raise ValueError('Pass either an existing ToolDefinition or "name" plus "execute".')
    return ToolDefinition(name=name, description=description, schema=schema, execute=execute)


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
            if part.type in {"tool-call", "tool-result"} and not model.capabilities.tools:
                raise UnsupportedFeatureError(
                    f'Model "{model.provider}/{model.model_id}" does not support tool calling.'
                )
