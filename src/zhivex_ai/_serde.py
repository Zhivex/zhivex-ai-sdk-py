from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

from .schema import create_schema_adapter
from .types import (
    AnyToolDefinition,
    CodeExecutionResultPart,
    GenerateResult,
    GeneratedCodePart,
    HostedToolDefinition,
    MCPServerConfig,
    MCPToolConfig,
    ModelGenerateInput,
    ModelMessage,
    ProviderDataPart,
    RemoteHTTPToolConfig,
    StructuredOutputConfig,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolDefinition,
    ToolExecutionContext,
    ToolExecutionError,
    ToolExecutionResult,
    ToolResultPart,
    MessageRole,
)


def _json_compatible(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _json_compatible(item) for key, item in asdict(cast("DataclassInstance", value)).items()}
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def serialize_token_usage(usage: TokenUsage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def deserialize_token_usage(payload: dict[str, Any] | None) -> TokenUsage | None:
    if payload is None:
        return None
    return TokenUsage(
        input_tokens=payload.get("input_tokens"),
        output_tokens=payload.get("output_tokens"),
        total_tokens=payload.get("total_tokens"),
    )


def serialize_tool_call(call: ToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "name": call.name,
        "input": _json_compatible(call.input),
        "provider_metadata": _json_compatible(call.provider_metadata),
    }


def deserialize_tool_call(payload: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=str(payload.get("id", "")),
        name=str(payload.get("name", "")),
        input=payload.get("input", {}),
        provider_metadata=dict(payload.get("provider_metadata") or {}),
    )


def serialize_tool_execution_result(result: ToolExecutionResult) -> dict[str, Any]:
    return {
        "tool_call_id": result.tool_call_id,
        "tool_name": result.tool_name,
        "output": _json_compatible(result.output),
        "error": {"message": result.error.message} if result.error is not None else None,
        "is_error": result.is_error,
    }


def deserialize_tool_execution_result(payload: dict[str, Any]) -> ToolExecutionResult:
    error_payload = payload.get("error")
    return ToolExecutionResult(
        tool_call_id=str(payload.get("tool_call_id", "")),
        tool_name=str(payload.get("tool_name", "")),
        output=payload.get("output"),
        error=ToolExecutionError(message=str(error_payload.get("message", ""))) if isinstance(error_payload, dict) else None,
        is_error=bool(payload.get("is_error", False)),
    )


def serialize_content_part(part: Any) -> dict[str, Any]:
    if getattr(part, "type", None) == "text":
        return {
            "type": "text",
            "text": getattr(part, "text", ""),
            "provider_metadata": _json_compatible(getattr(part, "provider_metadata", {})),
        }
    if getattr(part, "type", None) == "image":
        return {
            "type": "image",
            "image": getattr(part, "image", ""),
            "media_type": getattr(part, "media_type", None),
            "provider_metadata": _json_compatible(getattr(part, "provider_metadata", {})),
        }
    if getattr(part, "type", None) == "file":
        return {
            "type": "file",
            "data": getattr(part, "data", None),
            "text": getattr(part, "text", None),
            "document_content": _json_compatible(getattr(part, "document_content", None)),
            "media_type": getattr(part, "media_type", None),
            "filename": getattr(part, "filename", None),
            "file_id": getattr(part, "file_id", None),
            "file_uri": getattr(part, "file_uri", None),
            "url": getattr(part, "url", None),
            "title": getattr(part, "title", None),
            "context": getattr(part, "context", None),
            "citations_enabled": getattr(part, "citations_enabled", None),
            "cache_control": _json_compatible(getattr(part, "cache_control", None)),
            "provider_metadata": _json_compatible(getattr(part, "provider_metadata", {})),
        }
    if getattr(part, "type", None) == "provider-data":
        return {
            "type": "provider-data",
            "provider": getattr(part, "provider", ""),
            "data": _json_compatible(getattr(part, "data", None)),
        }
    if getattr(part, "type", None) == "tool-call":
        return {"type": "tool-call", "tool_call": serialize_tool_call(part.tool_call)}
    if getattr(part, "type", None) == "tool-result":
        return {"type": "tool-result", "tool_result": serialize_tool_execution_result(part.tool_result)}
    if getattr(part, "type", None) == "generated-code":
        return {
            "type": "generated-code",
            "code": getattr(part, "code", ""),
            "language": getattr(part, "language", None),
        }
    if getattr(part, "type", None) == "code-result":
        return {
            "type": "code-result",
            "output": getattr(part, "output", ""),
            "outcome": getattr(part, "outcome", None),
        }
    raise TypeError(f"Unsupported content part type: {getattr(part, 'type', type(part).__name__)}")


def deserialize_content_part(payload: dict[str, Any]) -> Any:
    part_type = payload.get("type")
    if part_type == "text":
        from .types import TextPart

        return TextPart(
            text=str(payload.get("text", "")),
            provider_metadata=dict(payload.get("provider_metadata") or {}),
        )
    if part_type == "image":
        from .types import ImagePart

        return ImagePart(
            image=str(payload.get("image", "")),
            media_type=payload.get("media_type"),
            provider_metadata=dict(payload.get("provider_metadata") or {}),
        )
    if part_type == "file":
        from .types import FilePart

        return FilePart(
            data=payload.get("data"),
            text=payload.get("text"),
            document_content=payload.get("document_content"),
            media_type=payload.get("media_type"),
            filename=payload.get("filename"),
            file_id=payload.get("file_id"),
            file_uri=payload.get("file_uri"),
            url=payload.get("url"),
            title=payload.get("title"),
            context=payload.get("context"),
            citations_enabled=payload.get("citations_enabled"),
            cache_control=payload.get("cache_control"),
            provider_metadata=dict(payload.get("provider_metadata") or {}),
        )
    if part_type == "tool-call":
        return ToolCallPart(tool_call=deserialize_tool_call(dict(payload.get("tool_call") or {})))
    if part_type == "provider-data":
        return ProviderDataPart(
            provider=str(payload.get("provider", "")),
            data=payload.get("data"),
        )
    if part_type == "tool-result":
        return ToolResultPart(tool_result=deserialize_tool_execution_result(dict(payload.get("tool_result") or {})))
    if part_type == "generated-code":
        return GeneratedCodePart(
            code=str(payload.get("code", "")),
            language=payload.get("language"),
        )
    if part_type == "code-result":
        return CodeExecutionResultPart(
            output=str(payload.get("output", "")),
            outcome=payload.get("outcome"),
        )
    raise TypeError(f"Unsupported content part type: {part_type}")


def serialize_message(message: ModelMessage) -> dict[str, Any]:
    return {
        "role": message.role,
        "parts": [serialize_content_part(part) for part in message.parts],
    }


def deserialize_message(payload: dict[str, Any]) -> ModelMessage:
    return ModelMessage(
        role=cast("MessageRole", str(payload.get("role", "user"))),
        parts=[deserialize_content_part(dict(part)) for part in payload.get("parts") or []],
    )


def serialize_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    return [serialize_message(message) for message in messages]


def deserialize_messages(messages: list[dict[str, Any]] | None) -> list[ModelMessage]:
    return [deserialize_message(dict(message)) for message in (messages or [])]


def _serialize_schema(schema: Any) -> dict[str, Any]:
    try:
        return {"json_schema": create_schema_adapter(schema).json_schema()}
    except Exception:
        return {"repr": repr(schema)}


def serialize_remote_http_tool_config(config: RemoteHTTPToolConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "url": config.url,
        "headers": dict(config.headers),
        "timeout_ms": config.timeout_ms,
    }


def deserialize_remote_http_tool_config(payload: dict[str, Any] | None) -> RemoteHTTPToolConfig | None:
    if payload is None:
        return None
    return RemoteHTTPToolConfig(
        url=str(payload.get("url", "")),
        headers={str(key): str(value) for key, value in dict(payload.get("headers") or {}).items()},
        timeout_ms=payload.get("timeout_ms"),
    )


def serialize_mcp_server_config(config: MCPServerConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "transport": config.transport,
        "name": config.name,
        "command": config.command,
        "args": list(config.args),
        "env": dict(config.env),
        "url": config.url,
        "headers": dict(config.headers),
        "timeout_ms": config.timeout_ms,
    }


def deserialize_mcp_server_config(payload: dict[str, Any] | None) -> MCPServerConfig | None:
    if payload is None:
        return None
    return MCPServerConfig(
        transport=str(payload.get("transport", "stdio")),  # type: ignore[arg-type]
        name=str(payload.get("name", "default")),
        command=payload.get("command"),
        args=[str(item) for item in payload.get("args") or []],
        env={str(key): str(value) for key, value in dict(payload.get("env") or {}).items()},
        url=payload.get("url"),
        headers={str(key): str(value) for key, value in dict(payload.get("headers") or {}).items()},
        timeout_ms=payload.get("timeout_ms"),
    )


def serialize_mcp_tool_config(config: MCPToolConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "server": serialize_mcp_server_config(config.server),
        "tool_name": config.tool_name,
    }


def deserialize_mcp_tool_config(payload: dict[str, Any] | None) -> MCPToolConfig | None:
    if payload is None:
        return None
    server = deserialize_mcp_server_config(dict(payload.get("server") or {}))
    if server is None:
        return None
    return MCPToolConfig(server=server, tool_name=str(payload.get("tool_name", "")))


def serialize_tool_definition(definition: AnyToolDefinition) -> dict[str, Any]:
    if isinstance(definition, HostedToolDefinition) or getattr(definition, "kind", None) == "hosted":
        hosted = cast(HostedToolDefinition, definition)
        return {
            "kind": "hosted",
            "name": hosted.name,
            "provider": hosted.provider,
            "type": hosted.type,
            "config": _json_compatible(hosted.config),
            "tool_class": hosted.tool_class,
            "requires_approval": hosted.requires_approval,
            "metadata": _json_compatible(hosted.metadata),
        }
    callable_definition = cast(ToolDefinition, definition)
    return {
        "name": callable_definition.name,
        "description": callable_definition.description,
        "schema": _serialize_schema(callable_definition.schema),
        "input_examples": _json_compatible(callable_definition.input_examples),
        "strict": callable_definition.strict,
        "defer_loading": callable_definition.defer_loading,
        "eager_input_streaming": callable_definition.eager_input_streaming,
        "allowed_callers": list(callable_definition.allowed_callers),
        "cache_control": _json_compatible(callable_definition.cache_control),
        "tags": list(callable_definition.tags),
        "requires_approval": callable_definition.requires_approval,
        "permissions": list(callable_definition.permissions),
        "source": callable_definition.source,
        "metadata": _json_compatible(callable_definition.metadata),
        "supports_streaming": callable_definition.supports_streaming,
        "remote_config": serialize_remote_http_tool_config(callable_definition.remote_config),
        "mcp_config": serialize_mcp_tool_config(callable_definition.mcp_config),
    }


def deserialize_tool_definition(payload: dict[str, Any]) -> AnyToolDefinition:
    if payload.get("kind") == "hosted":
        return HostedToolDefinition(
            name=str(payload.get("name", "")),
            provider=payload.get("provider"),
            type=str(payload.get("type", "")),
            config=payload.get("config"),
            tool_class=payload.get("tool_class"),
            requires_approval=payload.get("requires_approval"),
            metadata=dict(payload.get("metadata") or {}),
        )
    schema_payload = payload.get("schema")
    schema: Any = {}
    if isinstance(schema_payload, dict):
        schema = schema_payload.get("json_schema") or schema_payload.get("repr") or {}
    return ToolDefinition(
        name=str(payload.get("name", "")),
        description=payload.get("description"),
        schema=schema,
        execute=None,
        input_examples=list(payload.get("input_examples") or []),
        strict=payload.get("strict"),
        defer_loading=payload.get("defer_loading"),
        eager_input_streaming=payload.get("eager_input_streaming"),
        allowed_callers=[str(item) for item in payload.get("allowed_callers") or []],
        cache_control=payload.get("cache_control"),
        tags=[str(item) for item in payload.get("tags") or []],
        requires_approval=payload.get("requires_approval"),
        permissions=[str(item) for item in payload.get("permissions") or []],
        source=str(payload.get("source", "local")),  # type: ignore[arg-type]
        metadata=dict(payload.get("metadata") or {}),
        supports_streaming=bool(payload.get("supports_streaming", False)),
        remote_config=deserialize_remote_http_tool_config(payload.get("remote_config")),
        mcp_config=deserialize_mcp_tool_config(payload.get("mcp_config")),
    )


def serialize_structured_output(config: StructuredOutputConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "schema": _serialize_schema(config.schema),
        "mode": config.mode,
        "name": config.name,
        "description": config.description,
    }


def deserialize_structured_output(payload: dict[str, Any] | None) -> StructuredOutputConfig | None:
    if payload is None:
        return None
    schema_payload = payload.get("schema")
    schema: Any = {}
    if isinstance(schema_payload, dict):
        schema = schema_payload.get("json_schema") or schema_payload.get("repr") or {}
    return StructuredOutputConfig(
        schema=schema,
        mode=str(payload.get("mode", "prompted")),  # type: ignore[arg-type]
        name=payload.get("name"),
        description=payload.get("description"),
    )


def serialize_model_generate_input(value: ModelGenerateInput) -> dict[str, Any]:
    return {
        "messages": serialize_messages(value.messages),
        "tools": {name: serialize_tool_definition(tool) for name, tool in (value.tools or {}).items()},
        "tool_choice": _json_compatible(value.tool_choice),
        "temperature": value.temperature,
        "max_tokens": value.max_tokens,
        "reasoning": _json_compatible(value.reasoning),
        "provider_options": _json_compatible(value.provider_options),
        "structured_output": serialize_structured_output(value.structured_output),
        "timeout_ms": value.timeout_ms,
        "max_retries": value.max_retries,
        "retry_backoff_ms": value.retry_backoff_ms,
    }


def deserialize_model_generate_input(payload: dict[str, Any]) -> ModelGenerateInput:
    return ModelGenerateInput(
        messages=deserialize_messages(payload.get("messages")),
        tools={name: deserialize_tool_definition(tool) for name, tool in dict(payload.get("tools") or {}).items()},
        tool_choice=payload.get("tool_choice"),
        temperature=payload.get("temperature"),
        max_tokens=payload.get("max_tokens"),
        reasoning=payload.get("reasoning"),
        provider_options=payload.get("provider_options"),
        structured_output=deserialize_structured_output(payload.get("structured_output")),
        timeout_ms=payload.get("timeout_ms"),
        max_retries=payload.get("max_retries"),
        retry_backoff_ms=payload.get("retry_backoff_ms"),
    )


def serialize_generate_result(value: GenerateResult) -> dict[str, Any]:
    return {
        "message": serialize_message(value.message) if value.message is not None else None,
        "messages": serialize_messages(value.messages) if value.messages is not None else None,
        "text": value.text,
        "finish_reason": value.finish_reason,
        "provider_finish_reason": value.provider_finish_reason,
        "usage": serialize_token_usage(value.usage),
        "raw_response": _json_compatible(value.raw_response),
    }


def deserialize_generate_result(payload: dict[str, Any]) -> GenerateResult:
    message_payload = payload.get("message")
    messages_payload = payload.get("messages")
    return GenerateResult(
        message=deserialize_message(dict(message_payload)) if isinstance(message_payload, dict) else None,
        messages=deserialize_messages(messages_payload) if isinstance(messages_payload, list) else None,
        text=payload.get("text"),
        finish_reason=payload.get("finish_reason"),
        provider_finish_reason=payload.get("provider_finish_reason"),
        usage=deserialize_token_usage(payload.get("usage")),
        raw_response=payload.get("raw_response"),
    )


def serialize_tool_execution_context(context: ToolExecutionContext) -> dict[str, Any]:
    return {
        "tool_name": context.tool_name,
        "tool_call_id": context.tool_call_id,
        "run_id": context.run_id,
        "session_id": context.session_id,
        "agent_name": context.agent_name,
        "memory_summary": context.memory_summary,
        "permissions": list(context.permissions),
        "source": context.source,
        "metadata": _json_compatible(context.metadata),
        "handoff_path": list(context.handoff_path),
    }
