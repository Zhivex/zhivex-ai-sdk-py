from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass, replace
from typing import TYPE_CHECKING, Any, TypeGuard, cast

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

from .errors import UnsupportedFeatureError, ValidationError
from .types import (
    AgentCapabilities,
    AgentSupportTier,
    ContentPart,
    FilePart,
    FinishReason,
    GenerateResult,
    HostedToolClass,
    HostedToolDefinition,
    ImagePart,
    LanguageModel,
    MCPToolConfig,
    MessageRole,
    ModelMessage,
    ProviderDataPart,
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


def provider_data_part(provider: str, data: Any) -> ProviderDataPart:
    return ProviderDataPart(provider=provider, data=data)


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


def get_agent_capabilities(model: LanguageModel) -> AgentCapabilities:
    capabilities = getattr(model, "capabilities", None)
    configured = getattr(capabilities, "agent_capabilities", None)
    if configured is None:
        return AgentCapabilities()
    return replace(configured)


def get_agent_support_tier(model: LanguageModel) -> AgentSupportTier:
    return get_agent_capabilities(model).support_tier


def get_text_from_parts(parts: list[ContentPart]) -> str:
    return "".join(part.text for part in parts if isinstance(part, TextPart))


def get_text_from_messages(messages: list[ModelMessage]) -> str:
    return "".join(get_text_from_parts(message.parts) for message in messages if message.role == "assistant")


def get_text_from_result(result: GenerateResult) -> str:
    return get_text_from_messages(result_messages(result))


def get_provider_data_parts(value: Any, *, provider: str | None = None) -> list[ProviderDataPart]:
    collected: list[ProviderDataPart] = []

    def visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, ProviderDataPart):
            if provider is None or node.provider == provider:
                collected.append(node)
            return
        if isinstance(node, ModelMessage):
            for part in node.parts:
                visit(part)
            return
        if isinstance(node, GenerateResult):
            for message in result_messages(node):
                visit(message)
            return
        if hasattr(node, "messages") and isinstance(getattr(node, "messages"), list):
            for message in getattr(node, "messages"):
                visit(message)
            return
        if hasattr(node, "steps") and isinstance(getattr(node, "steps"), list):
            for step in getattr(node, "steps"):
                response = getattr(step, "response", None)
                if response is not None:
                    visit(response)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(value)
    return collected


def get_last_provider_data_part(value: Any, *, provider: str | None = None) -> ProviderDataPart | None:
    parts = get_provider_data_parts(value, provider=provider)
    return parts[-1] if parts else None


def get_provider_data_entries(
    value: Any,
    *,
    provider: str | None = None,
    parser: Any | None = None,
    data_type: str | None = None,
) -> list[Any]:
    entries: list[Any] = []
    for part in get_provider_data_parts(value, provider=provider):
        parsed = parser(part) if parser is not None else getattr(part, "data", None)
        if parsed is None:
            continue
        if data_type is not None and getattr(parsed, "type", None) != data_type:
            continue
        entries.append(parsed)
    return entries


def get_last_provider_data_entry(
    value: Any,
    *,
    provider: str | None = None,
    parser: Any | None = None,
    data_type: str | None = None,
) -> Any:
    entries = get_provider_data_entries(value, provider=provider, parser=parser, data_type=data_type)
    return entries[-1] if entries else None


def _to_json_compatible(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _to_json_compatible(v) for k, v in asdict(cast("DataclassInstance", value)).items()}
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(v) for v in value]
    return value


def serialize_json_value(value: Any) -> Any:
    return json.loads(json.dumps(_to_json_compatible(value)))


def _infer_hosted_tool_class(
    *,
    tool_type: str,
    config: Any = None,
) -> HostedToolClass:
    normalized_type = tool_type.lower()

    if (
        "web_search" in normalized_type
        or "web-search" in normalized_type
        or "googlesearch" in normalized_type
        or normalized_type == "google_search"
    ):
        return "web-search"
    if "file_search" in normalized_type or "file-search" in normalized_type or normalized_type == "filesearch":
        return "file-search"
    if normalized_type == "mcp":
        return "remote-mcp"
    if "mcp_toolset" in normalized_type or "mcp-toolset" in normalized_type:
        return "toolset"
    if "computer_use" in normalized_type or "computer-use" in normalized_type:
        return "computer-use"
    if "codeexecution" in normalized_type or "code_execution" in normalized_type or "code-execution" in normalized_type:
        return "code-execution"
    if isinstance(config, dict):
        if "toolset" in normalized_type or "tools" in config:
            return "toolset"
    return "custom"


def hosted_tool(
    *,
    name: str,
    type: str,
    provider: str | None = None,
    config: Any = None,
    tool_class: HostedToolClass | None = None,
    requires_approval: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> HostedToolDefinition:
    return HostedToolDefinition(
        name=name,
        provider=provider,
        type=type,
        config=serialize_json_value(config) if config is not None else None,
        tool_class=tool_class or _infer_hosted_tool_class(tool_type=type, config=config),
        requires_approval=requires_approval,
        metadata=dict(metadata or {}),
    )


def is_hosted_tool_definition(tool_definition: ToolDefinition | HostedToolDefinition) -> TypeGuard[HostedToolDefinition]:
    return isinstance(tool_definition, HostedToolDefinition) or getattr(tool_definition, "kind", None) == "hosted"


def is_callable_tool_definition(tool_definition: ToolDefinition | HostedToolDefinition) -> TypeGuard[ToolDefinition]:
    return not is_hosted_tool_definition(tool_definition)


def get_hosted_tool_class(tool_definition: HostedToolDefinition) -> HostedToolClass:
    return tool_definition.tool_class or _infer_hosted_tool_class(tool_type=tool_definition.type, config=tool_definition.config)


def is_hosted_tool_class(tool_definition: HostedToolDefinition, tool_class: HostedToolClass) -> bool:
    return get_hosted_tool_class(tool_definition) == tool_class


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
    output_schema: Any = None,
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
        output_schema=output_schema,
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
    if lowered == "refusal":
        return "refusal"
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
            if part.type == "provider-data" and not isinstance(part, ProviderDataPart):
                raise ValidationError("Provider data parts must be created with provider_data_part(...).")
