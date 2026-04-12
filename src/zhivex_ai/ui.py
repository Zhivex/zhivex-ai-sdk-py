from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from typing import Any

from .types import (
    CodeExecutionResultPart,
    FilePart,
    FinishReason,
    GeneratedCodePart,
    ImagePart,
    MessageRole,
    ModelMessage,
    TextPart,
    ToolCall,
    ToolCallPart,
    ToolExecutionError,
    ToolExecutionResult,
    ToolResultPart,
    TokenUsage,
    UIMessage,
    UIMessageChunk,
    UIMessageErrorChunk,
    UIMessageFinishChunk,
    UIMessageTextChunk,
    UIMessageToolCallChunk,
    UIMessageToolResultChunk,
)


def _random_id() -> str:
    return f"msg_{random.randrange(36**8):08x}"


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def _tool_call_from_dict(value: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=value["id"],
        name=value["name"],
        input=value.get("input") or {},
        provider_metadata=value.get("provider_metadata") or {},
    )


def _tool_result_from_dict(value: dict[str, Any]) -> ToolExecutionResult:
    error = value.get("error")
    return ToolExecutionResult(
        tool_call_id=value["tool_call_id"],
        tool_name=value["tool_name"],
        output=value.get("output"),
        error=ToolExecutionError(message=error["message"]) if error else None,
        is_error=value.get("is_error", False),
    )


def _part_from_dict(
    value: dict[str, Any],
) -> TextPart | ImagePart | FilePart | ToolCallPart | ToolResultPart | GeneratedCodePart | CodeExecutionResultPart:
    part_type = value["type"]
    if part_type == "text":
        return TextPart(
            text=value.get("text", ""),
            provider_metadata=value.get("provider_metadata") or {},
        )
    if part_type == "image":
        return ImagePart(
            image=value.get("image", ""),
            media_type=value.get("media_type"),
            provider_metadata=value.get("provider_metadata") or {},
        )
    if part_type == "file":
        return FilePart(
            data=value.get("data"),
            text=value.get("text"),
            document_content=value.get("document_content"),
            media_type=value.get("media_type"),
            filename=value.get("filename"),
            file_id=value.get("file_id"),
            file_uri=value.get("file_uri"),
            url=value.get("url"),
            title=value.get("title"),
            context=value.get("context"),
            citations_enabled=value.get("citations_enabled"),
            cache_control=value.get("cache_control"),
            provider_metadata=value.get("provider_metadata") or {},
        )
    if part_type == "tool-call":
        return ToolCallPart(tool_call=_tool_call_from_dict(value["tool_call"]))
    if part_type == "tool-result":
        return ToolResultPart(tool_result=_tool_result_from_dict(value["tool_result"]))
    if part_type == "generated-code":
        return GeneratedCodePart(
            code=value.get("code", ""),
            language=value.get("language"),
        )
    if part_type == "code-result":
        return CodeExecutionResultPart(
            output=value.get("output", ""),
            outcome=value.get("outcome"),
        )
    raise ValueError(f"Unsupported content part type: {part_type}")


def _usage_from_dict(value: dict[str, Any] | None) -> TokenUsage | None:
    if value is None:
        return None
    return TokenUsage(
        input_tokens=value.get("input_tokens"),
        output_tokens=value.get("output_tokens"),
        total_tokens=value.get("total_tokens"),
    )


def to_ui_message(message: ModelMessage, id: str | None = None) -> UIMessage:
    return UIMessage(id=id or _random_id(), role=message.role, parts=message.parts)


def to_ui_messages(messages: list[ModelMessage]) -> list[UIMessage]:
    return [to_ui_message(message) for message in messages]


def from_ui_message(message: UIMessage) -> ModelMessage:
    return ModelMessage(role=message.role, parts=message.parts)


def from_ui_messages(messages: list[UIMessage]) -> list[ModelMessage]:
    return [from_ui_message(message) for message in messages]


def serialize_ui_message(message: UIMessage) -> str:
    return json.dumps(_to_plain(message))


def deserialize_ui_message(value: str) -> UIMessage:
    payload = json.loads(value)
    return UIMessage(
        id=payload["id"],
        role=payload["role"],
        parts=[_part_from_dict(part) for part in payload.get("parts", [])],
    )


def serialize_ui_message_chunk(chunk: UIMessageChunk) -> str:
    return json.dumps(_to_plain(chunk))


def deserialize_ui_message_chunk(value: str) -> UIMessageChunk:
    payload = json.loads(value)
    chunk_type = payload["type"]
    if chunk_type == "text-delta":
        return UIMessageTextChunk(
            message_id=payload["message_id"],
            role=payload.get("role", "assistant"),
            text_delta=payload.get("text_delta", ""),
        )
    if chunk_type == "tool-call":
        return UIMessageToolCallChunk(
            message_id=payload["message_id"],
            role=payload.get("role", "assistant"),
            tool_call=_tool_call_from_dict(payload["tool_call"]),
        )
    if chunk_type == "tool-result":
        return UIMessageToolResultChunk(
            message_id=payload["message_id"],
            role=payload.get("role", "tool"),
            tool_result=_tool_result_from_dict(payload["tool_result"]),
        )
    if chunk_type == "finish":
        return UIMessageFinishChunk(
            message_id=payload["message_id"],
            finish_reason=payload.get("finish_reason"),
            provider_finish_reason=payload.get("provider_finish_reason"),
            usage=_usage_from_dict(payload.get("usage")),
        )
    if chunk_type == "error":
        return UIMessageErrorChunk(
            message_id=payload["message_id"],
            error=ToolExecutionError(message=((payload.get("error") or {}).get("message") or "")),
        )
    raise ValueError(f"Unsupported UI message chunk type: {chunk_type}")


def to_ui_message_stream(source: Any, message_id: str | None = None):
    event_stream = source.event_stream() if hasattr(source, "event_stream") else source
    resolved_id = message_id or _random_id()

    async def generator():
        async for event in event_stream:
            if event.type == "text-delta":
                yield UIMessageTextChunk(message_id=resolved_id, text_delta=event.text_delta)
            elif event.type == "tool-call":
                yield UIMessageToolCallChunk(message_id=resolved_id, tool_call=event.tool_call)
            elif event.type == "tool-result":
                yield UIMessageToolResultChunk(message_id=resolved_id, tool_result=event.tool_result)
            elif event.type == "finish":
                yield UIMessageFinishChunk(
                    message_id=resolved_id,
                    finish_reason=event.finish_reason,
                    provider_finish_reason=event.provider_finish_reason,
                    usage=event.usage,
                )
            elif event.type == "error":
                message = str(event.error) if event.error is not None else ""
                yield UIMessageErrorChunk(message_id=resolved_id, error=ToolExecutionError(message=message))

    return generator()


def collect_ui_message(result: Any, message_id: str | None = None) -> UIMessage:
    if not getattr(result, "messages", None):
        return UIMessage(id=message_id or _random_id(), role="assistant", parts=[])
    return to_ui_message(result.messages[-1], message_id)
