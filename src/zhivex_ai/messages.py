from __future__ import annotations

import inspect
import json
from dataclasses import asdict, is_dataclass, replace
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeGuard, cast, get_origin, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, create_model

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
    ToolExecutionContext,
    ToolInputGuardrail,
    ToolOutputGuardrail,
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


def _callable_type_hints(execute: Any) -> dict[str, Any]:
    try:
        return get_type_hints(execute, include_extras=True)
    except (NameError, TypeError):
        return {}


def _is_tool_context_parameter(name: str, annotation: Any) -> bool:
    return name == "context" or annotation is ToolExecutionContext or get_origin(annotation) is ToolExecutionContext


def _tool_from_callable(
    execute: Any,
    *,
    name: str | None,
    description: str | None,
    schema: Any,
    output_schema: Any,
    input_examples: list[Any] | None,
    strict: bool | None,
    defer_loading: bool | None,
    eager_input_streaming: bool | None,
    allowed_callers: list[str] | None,
    cache_control: dict[str, Any] | None,
    tags: list[str] | None,
    requires_approval: bool | None,
    permissions: list[str] | None,
    metadata: dict[str, Any] | None,
    supports_streaming: bool,
    input_guardrails: list[ToolInputGuardrail] | None,
    output_guardrails: list[ToolOutputGuardrail] | None,
) -> ToolDefinition:
    if not callable(execute):
        raise TypeError("@tool can only decorate callables.")
    try:
        signature = inspect.signature(execute)
    except (TypeError, ValueError) as error:
        raise TypeError("@tool requires a callable with an inspectable signature.") from error

    hints = _callable_type_hints(execute)
    user_parameters: list[inspect.Parameter] = []
    context_parameter_names: set[str] = set()
    fields: dict[str, tuple[Any, Any]] = {}
    for parameter in signature.parameters.values():
        annotation = hints.get(parameter.name, parameter.annotation)
        if _is_tool_context_parameter(parameter.name, annotation):
            context_parameter_names.add(parameter.name)
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError("@tool does not support variadic *args or **kwargs parameters.")
        user_parameters.append(parameter)
        if annotation is inspect.Parameter.empty or isinstance(annotation, str):
            annotation = Any
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (annotation, default)

    resolved_name = name or str(getattr(execute, "__name__", execute.__class__.__name__))
    if not resolved_name:
        raise ValueError("Tool callables require a non-empty name.")
    resolved_description = description
    if resolved_description is None:
        resolved_description = inspect.getdoc(execute) or None
    resolved_schema = schema
    if resolved_schema is None:
        model_name = "".join(part.title() for part in resolved_name.replace("-", "_").split("_")) or "Tool"
        resolved_schema = create_model(
            f"{model_name}ToolInput",
            __config__=ConfigDict(extra="forbid"),
            **cast(dict[str, Any], fields),
        )

    resolved_output_schema = output_schema
    if resolved_output_schema is None:
        return_annotation = hints.get("return", signature.return_annotation)
        if return_annotation not in (inspect.Signature.empty, None, type(None)) and not isinstance(
            return_annotation, str
        ):
            resolved_output_schema = return_annotation

    def call_arguments(input: Any, context: ToolExecutionContext[Any] | None) -> tuple[list[Any], dict[str, Any]]:
        if isinstance(input, BaseModel):
            values = {parameter.name: getattr(input, parameter.name) for parameter in user_parameters}
        elif isinstance(input, dict):
            values = input
        elif len(user_parameters) == 1:
            values = {user_parameters[0].name: input}
        else:
            raise TypeError(f'Tool "{resolved_name}" expected an object input.')
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        for parameter in signature.parameters.values():
            value = context if parameter.name in context_parameter_names else values[parameter.name]
            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                args.append(value)
            else:
                kwargs[parameter.name] = value
        return args, kwargs

    is_async = inspect.iscoroutinefunction(execute) or inspect.iscoroutinefunction(getattr(execute, "__call__", None))
    if is_async:

        @wraps(execute)
        async def decorated_execute(input: Any, context: ToolExecutionContext[Any] | None = None) -> Any:
            args, kwargs = call_arguments(input, context)
            return await execute(*args, **kwargs)

    else:

        @wraps(execute)
        def decorated_execute(input: Any, context: ToolExecutionContext[Any] | None = None) -> Any:
            args, kwargs = call_arguments(input, context)
            return execute(*args, **kwargs)

    # Runtime invocation is object-based even when the decorated callable exposes
    # multiple typed Python parameters. Keep the original metadata via wraps,
    # while exposing the adapter signature to the tool executor.
    setattr(
        decorated_execute,
        "__signature__",
        inspect.Signature(
            parameters=[
                inspect.Parameter("input", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Any),
                inspect.Parameter(
                    "context",
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=ToolExecutionContext[Any] | None,
                ),
            ],
            return_annotation=Any,
        ),
    )

    return ToolDefinition(
        name=resolved_name,
        description=resolved_description,
        schema=resolved_schema,
        execute=decorated_execute,
        input_examples=[serialize_json_value(item) for item in (input_examples or [])],
        strict=strict,
        defer_loading=defer_loading,
        eager_input_streaming=eager_input_streaming,
        allowed_callers=list(allowed_callers or []),
        output_schema=resolved_output_schema,
        cache_control=serialize_json_value(cache_control) if cache_control is not None else None,
        tags=list(tags or []),
        requires_approval=requires_approval,
        permissions=list(permissions or []),
        source="local",
        metadata=dict(metadata or {}),
        supports_streaming=supports_streaming,
        input_guardrails=list(input_guardrails or []),
        output_guardrails=list(output_guardrails or []),
    )


ToolDecorator = Callable[[Callable[..., Any]], ToolDefinition]


@overload
def tool(definition: ToolDefinition, **kwargs: Any) -> ToolDefinition: ...


@overload
def tool(definition: Callable[..., Any], **kwargs: Any) -> ToolDefinition: ...


@overload
def tool(
    definition: None = None,
    *,
    execute: Callable[..., Any],
    **kwargs: Any,
) -> ToolDefinition: ...


@overload
def tool(
    definition: None = None,
    *,
    source: Literal["remote", "mcp"],
    **kwargs: Any,
) -> ToolDefinition: ...


@overload
def tool(
    definition: None = None,
    *,
    source: Literal["local"] = "local",
    execute: None = None,
    **kwargs: Any,
) -> ToolDecorator: ...


def tool(
    definition: ToolDefinition | Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: Any = None,
    execute: Callable[..., Any] | None = None,
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
    input_guardrails: list[ToolInputGuardrail] | None = None,
    output_guardrails: list[ToolOutputGuardrail] | None = None,
    **kwargs: Any,
) -> ToolDefinition | ToolDecorator:
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"tool() got an unexpected keyword argument {unexpected!r}")
    if isinstance(definition, ToolDefinition):
        return definition
    if definition is not None:
        if source != "local" or remote_config is not None or mcp_config is not None:
            raise ValueError("@tool only supports local callable tools.")
        return _tool_from_callable(
            definition,
            name=name,
            description=description,
            schema=schema,
            output_schema=output_schema,
            input_examples=input_examples,
            strict=strict,
            defer_loading=defer_loading,
            eager_input_streaming=eager_input_streaming,
            allowed_callers=allowed_callers,
            cache_control=cache_control,
            tags=tags,
            requires_approval=requires_approval,
            permissions=permissions,
            metadata=metadata,
            supports_streaming=supports_streaming,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
        )
    if source == "local" and execute is None:
        def decorate(callable_execute: Callable[..., Any]) -> ToolDefinition:
            return _tool_from_callable(
                callable_execute,
                name=name,
                description=description,
                schema=schema,
                output_schema=output_schema,
                input_examples=input_examples,
                strict=strict,
                defer_loading=defer_loading,
                eager_input_streaming=eager_input_streaming,
                allowed_callers=allowed_callers,
                cache_control=cache_control,
                tags=tags,
                requires_approval=requires_approval,
                permissions=permissions,
                metadata=metadata,
                supports_streaming=supports_streaming,
                input_guardrails=input_guardrails,
                output_guardrails=output_guardrails,
            )

        return decorate
    if not name:
        raise ValueError('Pass either an existing ToolDefinition or at least a "name".')
    if source == "remote" and remote_config is None:
        raise ValueError('Remote tools require a "remote_config".')
    if source == "mcp" and mcp_config is None:
        raise ValueError('MCP tools require an "mcp_config".')
    resolved_requires_approval = requires_approval
    resolved_permissions = list(permissions or [])
    resolved_metadata = dict(metadata or {})
    if source in {"remote", "mcp"}:
        if resolved_requires_approval is None:
            resolved_requires_approval = True
        if permissions is None:
            resolved_permissions = ["network", "external-side-effect"]
    if source == "remote":
        resolved_metadata["remote_trust"] = (
            "application" if resolved_requires_approval is False else "approval-required"
        )
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
        requires_approval=resolved_requires_approval,
        permissions=resolved_permissions,
        source=source,
        metadata=resolved_metadata,
        supports_streaming=supports_streaming,
        remote_config=remote_config,
        mcp_config=mcp_config,
        input_guardrails=list(input_guardrails or []),
        output_guardrails=list(output_guardrails or []),
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
    input_guardrails: list[ToolInputGuardrail] | None = None,
    output_guardrails: list[ToolOutputGuardrail] | None = None,
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
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
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
    if lowered in {"length", "max_tokens", "model_context_window_exceeded"}:
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
