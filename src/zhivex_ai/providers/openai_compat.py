from __future__ import annotations

import asyncio
import base64
from copy import deepcopy
import json
import os
from collections.abc import AsyncIterable, Callable
from dataclasses import asdict, dataclass, field, replace
import time
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

from .._http import Fetcher, default_fetch
from .._sse import parse_sse
from ..errors import ConfigurationError, ProviderHTTPError, UnsupportedFeatureError, ValidationError
from ..messages import is_hosted_tool_definition, normalize_finish_reason, validate_file_part, validate_message_parts
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..realtime import (
    CallbackRealtimeSession,
    RealtimeConnectionFactory,
    RealtimeSessionCallbacks,
    encode_audio_frame,
    open_websocket_connection,
    tool_result_payload,
)
from ..types import (
    AgentCapabilities,
    AudioFrame,
    AudioInput,
    BatchesClient,
    CodeExecutionResultPart,
    ContainersClient,
    ConversationsClient,
    EmbedResult,
    EmbeddingModel,
    FileSearchBatch,
    FileSearchDocument,
    FileSearchDocumentListResult,
    FileSearchOperation,
    FileSearchSearchResult,
    FileSearchStore,
    FileSearchStoreListResult,
    FileSearchStoresClient,
    FilePart,
    FilesClient,
    GenerateResult,
    GeneratedCodePart,
    GroundedGenerateResult,
    GroundedLanguageModel,
    GroundedModelGenerateInput,
    GroundingSource,
    ImagePart,
    ImagesClient,
    ImagesResult,
    LanguageModel,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    OpenAIMcpApprovalRequest,
    OpenAIMcpApprovalResponse,
    OpenAIMcpCall,
    OpenAIMcpListTools,
    OpenAIResponseReference,
    ModerationsClient,
    ProviderDataPart,
    ProviderFile,
    ProviderImage,
    ProviderUpload,
    ProviderUploadPart,
    RealtimeAudioOutputEvent,
    RealtimeConnectOptions,
    RealtimeModel,
    RealtimeResponseCompletedEvent,
    RealtimeSession,
    RealtimeSessionConfig,
    RealtimeSessionEndedEvent,
    RealtimeTextDeltaEvent,
    RealtimeTokenResult,
    RealtimeToolCallEvent,
    RealtimeTranscriptEvent,
    RetryOptions,
    ResponsesClient,
    SkillsClient,
    SpeechModel,
    SpeechOutput,
    StreamEvent,
    StreamFinishEvent,
    StreamProviderDataEvent,
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
    UploadsClient,
    AzureOpenAIMcpApprovalRequest,
    AzureOpenAIMcpApprovalResponse,
    AzureOpenAIMcpCall,
    AzureOpenAIMcpListTools,
    AzureOpenAIResponseReference,
)
from .base import ProviderAdapter
from ._payload import drop_none

def _openai_compat_agent_capabilities(provider_name: str) -> AgentCapabilities:
    if provider_name in {"openai", "azure-openai"}:
        return AgentCapabilities(
            support_tier="tier-a",
            tool_choice_none=True,
            approval_requests=True,
            hosted_web_search=True,
            hosted_file_search=True,
            remote_mcp=True,
            computer_use=True,
        )
    if provider_name == "openrouter":
        return AgentCapabilities(
            support_tier="tier-c",
            tool_choice_none=True,
            hosted_web_search=True,
        )
    if provider_name == "qwen":
        return AgentCapabilities(
            support_tier="tier-c",
            tool_choice_none=True,
        )
    if provider_name == "kimi":
        return AgentCapabilities(
            support_tier="tier-c",
            tool_choice_none=True,
        )
    if provider_name == "ollama":
        return AgentCapabilities(
            support_tier="tier-c",
            tool_choice_none=False,
        )
    return AgentCapabilities()


def _with_agent_capabilities(capabilities: ModelCapabilities, agent_capabilities: AgentCapabilities) -> ModelCapabilities:
    return replace(capabilities, agent_capabilities=agent_capabilities)


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
    agent_capabilities=AgentCapabilities(),
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
    "tool_search_call": "tool_search",
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


def _serialize_provider_data_input(message: ModelMessage, provider_name: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    accepted_providers = {provider_name}
    if provider_name == "azure-openai":
        accepted_providers.add("openai")
    for part in message.parts:
        if getattr(part, "type", None) != "provider-data":
            continue
        provider = getattr(part, "provider", "")
        data = getattr(part, "data", None)
        if provider not in accepted_providers:
            continue
        parsed = _parse_provider_data_value(data, provider_name)
        if parsed is None:
            continue
        if getattr(parsed, "type", None) == "mcp_approval_response":
            items.append(drop_none(asdict(parsed)))
    return items


def _provider_data_classes(
    provider_name: str,
) -> tuple[type[Any], type[Any], type[Any], type[Any], type[Any]]:
    if provider_name == "azure-openai":
        return (
            AzureOpenAIResponseReference,
            AzureOpenAIMcpApprovalRequest,
            AzureOpenAIMcpApprovalResponse,
            AzureOpenAIMcpCall,
            AzureOpenAIMcpListTools,
        )
    return (
        OpenAIResponseReference,
        OpenAIMcpApprovalRequest,
        OpenAIMcpApprovalResponse,
        OpenAIMcpCall,
        OpenAIMcpListTools,
    )


def _parse_provider_data_value(value: Any, provider_name: str) -> Any | None:
    response_reference_cls, approval_request_cls, approval_response_cls, mcp_call_cls, mcp_list_tools_cls = (
        _provider_data_classes(provider_name)
    )
    if isinstance(
        value,
        (
            response_reference_cls,
            approval_request_cls,
            approval_response_cls,
            mcp_call_cls,
            mcp_list_tools_cls,
        ),
    ):
        return value
    if not isinstance(value, dict):
        return None
    if "response_id" in value or "responseId" in value:
        response_id = value.get("response_id", value.get("responseId"))
        if isinstance(response_id, str) and response_id:
            return response_reference_cls(response_id=response_id)
    item_type = value.get("type")
    if item_type == "mcp_approval_request":
        return approval_request_cls(
            id=str(value.get("id") or ""),
            arguments=str(value.get("arguments") or ""),
            name=str(value.get("name") or ""),
            server_label=str(value.get("server_label") or ""),
        )
    if item_type == "mcp_approval_response":
        return approval_response_cls(
            approval_request_id=str(value.get("approval_request_id") or ""),
            approve=bool(value.get("approve")),
            id=str(value.get("id")) if value.get("id") is not None else None,
            reason=str(value.get("reason")) if value.get("reason") is not None else None,
        )
    if item_type == "mcp_call":
        return mcp_call_cls(
            id=str(value.get("id") or ""),
            arguments=str(value.get("arguments") or ""),
            name=str(value.get("name") or ""),
            server_label=str(value.get("server_label") or ""),
            approval_request_id=str(value.get("approval_request_id")) if value.get("approval_request_id") is not None else None,
            error=str(value.get("error")) if value.get("error") is not None else None,
            output=str(value.get("output")) if value.get("output") is not None else None,
            status=value.get("status"),
        )
    if item_type == "mcp_list_tools":
        return mcp_list_tools_cls(
            id=str(value.get("id")) if value.get("id") is not None else None,
            server_label=str(value.get("server_label")) if value.get("server_label") is not None else None,
            tools=deepcopy(value.get("tools")),
        )
    return None


def _provider_data_part_for(provider_name: str, data: Any) -> ProviderDataPart:
    return ProviderDataPart(provider=provider_name, data=data)


def _response_reference_part(payload: dict[str, Any], provider_name: str) -> ProviderDataPart | None:
    response_id = payload.get("id")
    if not isinstance(response_id, str) or not response_id:
        return None
    response_reference_cls, _, _, _, _ = _provider_data_classes(provider_name)
    return _provider_data_part_for(provider_name, response_reference_cls(response_id=response_id))


def _parse_provider_data_output_item(item: dict[str, Any], provider_name: str) -> ProviderDataPart | None:
    parsed = _parse_provider_data_value(item, provider_name)
    if parsed is None:
        return None
    return _provider_data_part_for(provider_name, parsed)


def _to_responses_input(messages: list[ModelMessage], provider_name: str) -> list[dict[str, Any]]:
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
            items.extend(_serialize_provider_data_input(message, provider_name))
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
        items.extend(_serialize_provider_data_input(message, provider_name))
    return items


def _map_hosted_tool(tool: Any, provider_name: str) -> dict[str, Any]:
    provider = getattr(tool, "provider", None)
    accepted_providers = {None, provider_name}
    if provider_name == "azure-openai":
        accepted_providers.add("openai")
    if provider not in accepted_providers:
        raise ValidationError(
            f'Hosted tool "{getattr(tool, "name", "")}" targets provider "{provider}", but this model uses "{provider_name}".'
        )
    payload = {"type": tool.type}
    if isinstance(tool.config, dict):
        payload.update(deepcopy(tool.config))
    elif tool.config is not None:
        payload["config"] = deepcopy(tool.config)
    return drop_none(payload)


def _map_tools(tools: dict[str, Any] | None, *, provider_name: str) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    mapped = []
    for tool in tools.values():
        if is_hosted_tool_definition(tool):
            mapped.append(_map_hosted_tool(tool, provider_name))
            continue
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


def _parse_output_item(item: dict[str, Any], provider_name: str) -> list[Any]:
    parts: list[Any] = []
    item_type = str(item.get("type") or "")
    provider_data_part = _parse_provider_data_output_item(item, provider_name)
    if provider_data_part is not None:
        parts.append(provider_data_part)
        return parts
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


def _parse_responses_message(payload: dict[str, Any], provider_name: str) -> ModelMessage:
    parts: list[Any] = []
    response_reference = _response_reference_part(payload, provider_name)
    if response_reference is not None:
        parts.append(response_reference)
    for item in payload.get("output") or []:
        parts.extend(_parse_output_item(item, provider_name))
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
    mapped_tools = _map_tools(input.tools, provider_name=provider_name) or []
    if mapped_tools:
        merged_tools.extend(mapped_tools)
    if provider_tools:
        merged_tools.extend(deepcopy(provider_tools))
    body = {
        "model": model_id,
        "instructions": _system_instructions(input.messages),
        "input": _to_responses_input(input.messages, provider_name),
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


def _normalize_openai_image(payload: dict[str, Any], *, provider: str, default_media_type: str | None = None) -> ProviderImage:
    media_type = payload.get("media_type") or _image_media_type_from_format(payload.get("output_format")) or default_media_type
    return ProviderImage(
        provider=provider,
        b64_json=payload.get("b64_json"),
        url=payload.get("url"),
        revised_prompt=payload.get("revised_prompt"),
        media_type=media_type,
        metadata=dict(payload),
    )


def _normalize_openai_upload(payload: dict[str, Any], *, provider: str) -> ProviderUpload:
    file_payload = payload.get("file")
    return ProviderUpload(
        provider=provider,
        id=str(payload.get("id") or ""),
        filename=payload.get("filename"),
        purpose=payload.get("purpose"),
        bytes=payload.get("bytes"),
        status=payload.get("status"),
        mime_type=payload.get("mime_type"),
        created_at=payload.get("created_at"),
        expires_at=payload.get("expires_at"),
        completed_at=payload.get("completed_at"),
        cancelled_at=payload.get("cancelled_at"),
        file=_normalize_openai_file(dict(file_payload), provider=provider) if isinstance(file_payload, dict) else None,
        metadata=dict(payload),
    )


def _normalize_openai_upload_part(payload: dict[str, Any], *, provider: str) -> ProviderUploadPart:
    return ProviderUploadPart(
        provider=provider,
        id=str(payload.get("id") or ""),
        upload_id=payload.get("upload_id"),
        created_at=payload.get("created_at"),
        metadata=dict(payload),
    )


def _normalize_openai_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_openai_vector_store(payload: dict[str, Any]) -> FileSearchStore:
    return FileSearchStore(
        name=str(payload.get("id") or ""),
        display_name=payload.get("name"),
        create_time=_normalize_openai_timestamp(payload.get("created_at")),
        update_time=_normalize_openai_timestamp(payload.get("last_active_at")),
        metadata=dict(payload),
    )


def _normalize_openai_vector_store_attributes(custom_metadata: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not custom_metadata:
        return None
    attributes: dict[str, Any] = {}
    for item in custom_metadata:
        if not isinstance(item, dict):
            continue
        if "key" in item:
            key = str(item.get("key") or "")
            if not key:
                continue
            if "value" in item:
                attributes[key] = deepcopy(item.get("value"))
                continue
            for value_key in ("string_value", "number_value", "boolean_value", "bool_value"):
                if value_key in item:
                    attributes[key] = deepcopy(item.get(value_key))
                    break
            continue
        for key, value in item.items():
            attributes[str(key)] = deepcopy(value)
    return attributes or None


def _openai_vector_store_file_name(store_id: str, file_id: str) -> str:
    return f"vector_stores/{store_id}/files/{file_id}"


def _openai_vector_store_file_batch_name(store_id: str, batch_id: str) -> str:
    return f"vector_stores/{store_id}/file_batches/{batch_id}"


def _parse_openai_vector_store_file_name(name: str) -> tuple[str, str]:
    parts = [part for part in name.split("/") if part]
    if len(parts) >= 4 and parts[0] == "vector_stores" and parts[2] == "files":
        return parts[1], parts[3]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValidationError(
        'OpenAI vector store document names must look like "vector_stores/<store_id>/files/<file_id>" or "<store_id>/<file_id>".'
    )


def _parse_openai_vector_store_file_batch_name(name: str) -> tuple[str, str]:
    parts = [part for part in name.split("/") if part]
    if len(parts) >= 4 and parts[0] == "vector_stores" and parts[2] == "file_batches":
        return parts[1], parts[3]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValidationError(
        'OpenAI vector store batch names must look like "vector_stores/<store_id>/file_batches/<batch_id>" or "<store_id>/<batch_id>".'
    )


def _normalize_openai_vector_store_file(payload: dict[str, Any], *, store_id: str | None = None) -> FileSearchDocument:
    resolved_store_id = store_id or str(payload.get("vector_store_id") or "")
    file_id = str(payload.get("id") or payload.get("file_id") or "")
    name = _openai_vector_store_file_name(resolved_store_id, file_id) if resolved_store_id and file_id else file_id
    size_bytes = payload.get("usage_bytes")
    try:
        parsed_size = int(size_bytes) if size_bytes is not None else None
    except (TypeError, ValueError):
        parsed_size = None
    attributes = payload.get("attributes")
    custom_metadata = [dict(attributes)] if isinstance(attributes, dict) and attributes else []
    return FileSearchDocument(
        name=name,
        display_name=payload.get("filename") or payload.get("display_name") or file_id or None,
        custom_metadata=custom_metadata,
        state=payload.get("status"),
        size_bytes=parsed_size,
        media_type=payload.get("mime_type"),
        create_time=_normalize_openai_timestamp(payload.get("created_at")),
        update_time=_normalize_openai_timestamp(payload.get("last_active_at")),
        metadata=dict(payload),
    )


def _normalize_openai_vector_store_operation(payload: dict[str, Any], *, store_id: str | None = None) -> FileSearchOperation:
    status = str(payload.get("status") or "").lower()
    file_name = None
    resolved_store_id = store_id or str(payload.get("vector_store_id") or "")
    file_id = str(payload.get("id") or payload.get("file_id") or "")
    if resolved_store_id and file_id:
        file_name = _openai_vector_store_file_name(resolved_store_id, file_id)
    return FileSearchOperation(
        name=file_name or str(payload.get("id") or ""),
        done=bool(payload.get("deleted")) or status in {"completed", "failed", "cancelled"},
        metadata=dict(payload),
        response=dict(payload),
        error=dict(payload.get("last_error") or {}) if isinstance(payload.get("last_error"), dict) else None,
        raw_response=payload,
    )


def _normalize_openai_vector_store_batch(payload: dict[str, Any], *, store_id: str | None = None) -> FileSearchBatch:
    resolved_store_id = store_id or str(payload.get("vector_store_id") or "")
    batch_id = str(payload.get("id") or "")
    name = _openai_vector_store_file_batch_name(resolved_store_id, batch_id) if resolved_store_id and batch_id else batch_id
    return FileSearchBatch(
        name=name,
        file_search_store_name=resolved_store_id or None,
        state=payload.get("status"),
        create_time=_normalize_openai_timestamp(payload.get("created_at")),
        metadata=dict(payload),
        raw_response=payload,
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


@dataclass(slots=True)
class OpenAICompatibleImagesClient(ImagesClient):
    provider: str
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

    async def generate(
        self,
        *,
        prompt: str,
        model: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        moderation: str | None = None,
        user: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ImagesResult:
        response = await self.fetch(
            f"{self.base_url}/images/generations",
            headers=self._headers(),
            json_body=drop_none(
                {
                    "prompt": prompt,
                    "model": model,
                    "size": size,
                    "quality": quality,
                    "background": background,
                    "output_format": output_format,
                    "moderation": moderation,
                    "user": user,
                    **deepcopy(extra_body or {}),
                }
            ),
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        return ImagesResult(
            images=[
                _normalize_openai_image(dict(item), provider=self.provider, default_media_type=_image_media_type_from_format(output_format))
                for item in payload.get("data") or []
                if isinstance(item, dict)
            ],
            created_at=payload.get("created"),
            raw_response=payload,
        )

    async def edit(
        self,
        *,
        prompt: str,
        image: bytes | bytearray | memoryview | list[bytes | bytearray | memoryview],
        image_filenames: str | list[str] | None = None,
        image_media_type: str | list[str] | None = None,
        model: str | None = None,
        mask: bytes | bytearray | memoryview | None = None,
        mask_filename: str | None = None,
        mask_media_type: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        moderation: str | None = None,
        user: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ImagesResult:
        images = image if isinstance(image, list) else [image]
        filenames = image_filenames if isinstance(image_filenames, list) else [image_filenames] * len(images)
        media_types = image_media_type if isinstance(image_media_type, list) else [image_media_type] * len(images)
        files_payload: list[tuple[str, tuple[str, bytes, str]]] = []
        for index, image_item in enumerate(images):
            files_payload.append(
                (
                    "image[]",
                    (
                        filenames[index] or f"image-{index + 1}.png",
                        _normalize_binary(image_item),
                        media_types[index] or "image/png",
                    ),
                )
            )
        if mask is not None:
            files_payload.append(
                (
                    "mask",
                    (
                        mask_filename or "mask.png",
                        _normalize_binary(mask),
                        mask_media_type or "image/png",
                    ),
                )
            )
        response = await self.fetch(
            f"{self.base_url}/images/edits",
            headers=self._headers(json_content=False),
            body={
                "data": drop_none(
                    {
                        "prompt": prompt,
                        "model": model,
                        "size": size,
                        "quality": quality,
                        "background": background,
                        "output_format": output_format,
                        "moderation": moderation,
                        "user": user,
                        **deepcopy(extra_body or {}),
                    }
                ),
                "files": files_payload,
            },
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        return ImagesResult(
            images=[
                _normalize_openai_image(dict(item), provider=self.provider, default_media_type=_image_media_type_from_format(output_format))
                for item in payload.get("data") or []
                if isinstance(item, dict)
            ],
            created_at=payload.get("created"),
            raw_response=payload,
        )

    async def variation(
        self,
        *,
        image: bytes | bytearray | memoryview,
        image_filename: str | None = None,
        image_media_type: str | None = None,
        model: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        user: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ImagesResult:
        response = await self.fetch(
            f"{self.base_url}/images/variations",
            headers=self._headers(json_content=False),
            body={
                "data": drop_none(
                    {
                        "model": model,
                        "size": size,
                        "quality": quality,
                        "background": background,
                        "output_format": output_format,
                        "user": user,
                        **deepcopy(extra_body or {}),
                    }
                ),
                "files": {
                    "image": (
                        image_filename or "image.png",
                        _normalize_binary(image),
                        image_media_type or "image/png",
                    )
                },
            },
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        return ImagesResult(
            images=[
                _normalize_openai_image(dict(item), provider=self.provider, default_media_type=_image_media_type_from_format(output_format))
                for item in payload.get("data") or []
                if isinstance(item, dict)
            ],
            created_at=payload.get("created"),
            raw_response=payload,
        )


@dataclass(slots=True)
class OpenAICompatibleUploadsClient(UploadsClient):
    provider: str
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

    async def create(
        self,
        *,
        filename: str,
        bytes: int,
        mime_type: str,
        purpose: str,
        expires_after: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderUpload:
        response = await self.fetch(
            f"{self.base_url}/uploads",
            headers=self._headers(),
            json_body=drop_none(
                {
                    "filename": filename,
                    "bytes": bytes,
                    "mime_type": mime_type,
                    "purpose": purpose,
                    "expires_after": deepcopy(expires_after),
                    "metadata": deepcopy(metadata),
                }
            ),
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_upload(await response.json(), provider=self.provider)

    async def add_part(
        self,
        *,
        upload_id: str,
        data: bytes | bytearray | memoryview,
        filename: str | None = None,
        media_type: str | None = None,
    ) -> ProviderUploadPart:
        response = await self.fetch(
            f"{self.base_url}/uploads/{upload_id}/parts",
            headers=self._headers(json_content=False),
            body={
                "data": None,
                "files": {
                    "data": (
                        filename or "part.bin",
                        _normalize_binary(data),
                        media_type or "application/octet-stream",
                    )
                },
            },
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_upload_part(await response.json(), provider=self.provider)

    async def complete(
        self,
        upload_id: str,
        *,
        part_ids: list[str],
        md5: str | None = None,
    ) -> ProviderUpload:
        response = await self.fetch(
            f"{self.base_url}/uploads/{upload_id}/complete",
            headers=self._headers(),
            json_body=drop_none({"part_ids": list(part_ids), "md5": md5}),
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_upload(await response.json(), provider=self.provider)

    async def cancel(self, upload_id: str) -> ProviderUpload:
        response = await self.fetch(
            f"{self.base_url}/uploads/{upload_id}/cancel",
            headers=self._headers(),
            json_body={},
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_upload(await response.json(), provider=self.provider)

    async def upload_bytes(
        self,
        *,
        data: bytes | bytearray | memoryview,
        filename: str,
        mime_type: str,
        purpose: str,
        part_size_bytes: int = 64 * 1024 * 1024,
        expires_after: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        md5: str | None = None,
    ) -> ProviderFile:
        raw = _normalize_binary(data)
        if part_size_bytes <= 0:
            raise ValidationError('"part_size_bytes" must be greater than zero.')
        upload = await self.create(
            filename=filename,
            bytes=len(raw),
            mime_type=mime_type,
            purpose=purpose,
            expires_after=deepcopy(expires_after),
            metadata=deepcopy(metadata),
        )
        part_ids: list[str] = []
        for index in range(0, len(raw), part_size_bytes):
            part = await self.add_part(
                upload_id=upload.id,
                data=raw[index:index + part_size_bytes],
                filename=f"{filename}.part-{len(part_ids) + 1}",
                media_type="application/octet-stream",
            )
            part_ids.append(part.id)
        completed = await self.complete(upload.id, part_ids=part_ids, md5=md5)
        if completed.file is None:
            raise ValidationError('OpenAI upload completed without returning a created file object.')
        return completed.file


@dataclass(slots=True)
class OpenAICompatibleModerationsClient(ModerationsClient):
    provider: str
    api_key: str
    base_url: str
    fetch: Fetcher
    auth_header: str = "authorization"
    auth_prefix: str = "Bearer "

    def _headers(self) -> dict[str, str]:
        value = self.api_key if not self.auth_prefix else f"{self.auth_prefix}{self.api_key}"
        return {self.auth_header: value, "content-type": "application/json"}

    async def create(self, body: dict[str, Any], options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/moderations",
                headers=self._headers(),
                json_body=deepcopy(body),
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()


@dataclass(slots=True)
class OpenAICompatibleBatchesClient(BatchesClient):
    provider: str
    api_key: str
    base_url: str
    fetch: Fetcher
    auth_header: str = "authorization"
    auth_prefix: str = "Bearer "

    def _headers(self) -> dict[str, str]:
        value = self.api_key if not self.auth_prefix else f"{self.auth_prefix}{self.api_key}"
        return {self.auth_header: value, "content-type": "application/json"}

    async def create(self, body: dict[str, Any], options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/batches",
                headers=self._headers(),
                json_body=deepcopy(body),
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def retrieve(self, batch_id: str, options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/batches/{batch_id}",
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

    async def list(
        self,
        *,
        after: str | None = None,
        limit: int | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                _request_url(self.base_url, "/batches", {"after": after, "limit": limit}),
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

    async def cancel(self, batch_id: str, options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/batches/{batch_id}/cancel",
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


@dataclass(slots=True)
class OpenAICompatibleContainersClient(ContainersClient):
    provider: str
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

    async def create(self, body: dict[str, Any], options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/containers",
                headers=self._headers(),
                json_body=deepcopy(body),
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def retrieve(self, container_id: str, options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/containers/{container_id}",
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

    async def delete(self, container_id: str, options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/containers/{container_id}",
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

    async def list(
        self,
        *,
        after: str | None = None,
        limit: int | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                _request_url(self.base_url, "/containers", {"after": after, "limit": limit}),
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

    async def create_file(
        self,
        *,
        container_id: str,
        data: bytes | bytearray | memoryview,
        filename: str,
        media_type: str = "application/octet-stream",
        options: RetryOptions | None = None,
    ) -> ProviderFile:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/containers/{container_id}/files",
                headers=self._headers(json_content=False),
                body={"data": None, "files": {"file": (filename, _normalize_binary(data), media_type)}},
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_file(await response.json(), provider=self.provider)

    async def list_files(
        self,
        container_id: str,
        *,
        after: str | None = None,
        limit: int | None = None,
        options: RetryOptions | None = None,
    ) -> list[ProviderFile]:
        response = await with_retry(
            lambda: self.fetch(
                _request_url(self.base_url, f"/containers/{container_id}/files", {"after": after, "limit": limit}),
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
        payload = await response.json()
        return [_normalize_openai_file(dict(item), provider=self.provider) for item in payload.get("data") or []]

    async def retrieve_file(
        self,
        container_id: str,
        file_id: str,
        options: RetryOptions | None = None,
    ) -> ProviderFile:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/containers/{container_id}/files/{file_id}",
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
        return _normalize_openai_file(await response.json(), provider=self.provider)

    async def delete_file(
        self,
        container_id: str,
        file_id: str,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/containers/{container_id}/files/{file_id}",
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

    async def retrieve_file_content(
        self,
        container_id: str,
        file_id: str,
        options: RetryOptions | None = None,
    ) -> bytes:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/containers/{container_id}/files/{file_id}/content",
                method="GET",
                headers=self._headers(json_content=False),
                json_body=None,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.read()


@dataclass(slots=True)
class OpenAICompatibleSkillsClient(SkillsClient):
    provider: str
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

    async def create(self, body: dict[str, Any], options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/skills",
                headers=self._headers(),
                json_body=deepcopy(body),
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def retrieve(self, skill_id: str, options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/skills/{skill_id}",
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

    async def retrieve_content(self, skill_id: str, options: RetryOptions | None = None) -> bytes:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/skills/{skill_id}/content",
                method="GET",
                headers=self._headers(json_content=False),
                json_body=None,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.read()

    async def update(self, skill_id: str, body: dict[str, Any], options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/skills/{skill_id}",
                headers=self._headers(),
                json_body=deepcopy(body),
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def delete(self, skill_id: str, options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/skills/{skill_id}",
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

    async def list(
        self,
        *,
        after: str | None = None,
        limit: int | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                _request_url(self.base_url, "/skills", {"after": after, "limit": limit}),
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

    async def create_version(self, skill_id: str, body: dict[str, Any], options: RetryOptions | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/skills/{skill_id}/versions",
                headers=self._headers(),
                json_body=deepcopy(body),
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.json()

    async def retrieve_version(
        self,
        skill_id: str,
        version_id: str,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/skills/{skill_id}/versions/{version_id}",
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

    async def retrieve_version_content(
        self,
        skill_id: str,
        version_id: str,
        options: RetryOptions | None = None,
    ) -> bytes:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/skills/{skill_id}/versions/{version_id}/content",
                method="GET",
                headers=self._headers(json_content=False),
                json_body=None,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.read()

    async def delete_version(
        self,
        skill_id: str,
        version_id: str,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/skills/{skill_id}/versions/{version_id}",
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

    async def list_versions(
        self,
        skill_id: str,
        *,
        after: str | None = None,
        limit: int | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                _request_url(self.base_url, f"/skills/{skill_id}/versions", {"after": after, "limit": limit}),
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
class OpenAICompatibleFileSearchStoresClient(FileSearchStoresClient):
    provider: str
    api_key: str
    base_url: str
    fetch: Fetcher
    auth_header: str = "authorization"
    auth_prefix: str = "Bearer "
    default_purpose: str = "assistants"

    def _headers(self) -> dict[str, str]:
        value = self.api_key if not self.auth_prefix else f"{self.auth_prefix}{self.api_key}"
        return {
            self.auth_header: value,
            "content-type": "application/json",
        }

    def _files_client(self) -> OpenAICompatibleFilesClient:
        return OpenAICompatibleFilesClient(
            provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            fetch=self.fetch,
            auth_header=self.auth_header,
            auth_prefix=self.auth_prefix,
            default_purpose=self.default_purpose,
        )

    async def create(
        self,
        *,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FileSearchStore:
        response = await self.fetch(
            f"{self.base_url}/vector_stores",
            headers=self._headers(),
            json_body=drop_none({"name": display_name, "metadata": deepcopy(metadata)}),
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_vector_store(await response.json())

    async def list(
        self,
        *,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> FileSearchStoreListResult:
        response = await self.fetch(
            _request_url(self.base_url, "/vector_stores", {"limit": page_size, "after": page_token}),
            method="GET",
            headers=self._headers(),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        return FileSearchStoreListResult(
            stores=[_normalize_openai_vector_store(dict(item)) for item in payload.get("data") or []],
            next_page_token=(payload.get("last_id") or payload.get("next_page")) if payload.get("has_more") else None,
            raw_response=payload,
        )

    async def get(self, name: str) -> FileSearchStore:
        response = await self.fetch(
            f"{self.base_url}/vector_stores/{name}",
            method="GET",
            headers=self._headers(),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_vector_store(await response.json())

    async def update(
        self,
        name: str,
        *,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        expires_after: dict[str, Any] | None = None,
    ) -> FileSearchStore:
        response = await self.fetch(
            f"{self.base_url}/vector_stores/{name}",
            headers=self._headers(),
            json_body=drop_none(
                {
                    "name": display_name,
                    "metadata": deepcopy(metadata),
                    "expires_after": deepcopy(expires_after),
                }
            ),
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_vector_store(await response.json())

    async def delete(self, name: str, *, force: bool = False) -> bool:
        response = await self.fetch(
            f"{self.base_url}/vector_stores/{name}",
            method="DELETE",
            headers=self._headers(),
            json_body=({"force": True} if force else None),
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        return bool(payload.get("deleted"))

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
        uploaded = await self._files_client().upload(
            data=_normalize_binary(data),
            filename=filename,
            media_type=media_type or "application/octet-stream",
            purpose=self.default_purpose,
        )
        return await self.import_file(
            file_search_store_name=file_search_store_name,
            file_name=uploaded.id,
            custom_metadata=custom_metadata,
            chunking_config=deepcopy(chunking_config),
        )

    async def import_file(
        self,
        *,
        file_search_store_name: str,
        file_name: str,
        custom_metadata: list[dict[str, Any]] | None = None,
        chunking_config: dict[str, Any] | None = None,
    ) -> FileSearchOperation:
        response = await self.fetch(
            f"{self.base_url}/vector_stores/{file_search_store_name}/files",
            headers=self._headers(),
            json_body=drop_none(
                {
                    "file_id": file_name,
                    "attributes": _normalize_openai_vector_store_attributes(custom_metadata),
                    "chunking_strategy": deepcopy(chunking_config),
                }
            ),
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_vector_store_operation(await response.json(), store_id=file_search_store_name)

    async def list_documents(
        self,
        *,
        file_search_store_name: str,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> FileSearchDocumentListResult:
        response = await self.fetch(
            _request_url(
                self.base_url,
                f"/vector_stores/{file_search_store_name}/files",
                {"limit": page_size, "after": page_token},
            ),
            method="GET",
            headers=self._headers(),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        return FileSearchDocumentListResult(
            documents=[
                _normalize_openai_vector_store_file(dict(item), store_id=file_search_store_name)
                for item in payload.get("data") or []
            ],
            next_page_token=(payload.get("last_id") or payload.get("next_page")) if payload.get("has_more") else None,
            raw_response=payload,
        )

    async def get_document(self, name: str) -> FileSearchDocument:
        store_id, file_id = _parse_openai_vector_store_file_name(name)
        response = await self.fetch(
            f"{self.base_url}/vector_stores/{store_id}/files/{file_id}",
            method="GET",
            headers=self._headers(),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_vector_store_file(await response.json(), store_id=store_id)

    async def download_document(self, name: str) -> bytes:
        store_id, file_id = _parse_openai_vector_store_file_name(name)
        response = await self.fetch(
            f"{self.base_url}/vector_stores/{store_id}/files/{file_id}/content",
            method="GET",
            headers=self._headers(),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return await response.read()

    async def delete_document(self, name: str) -> bool:
        store_id, file_id = _parse_openai_vector_store_file_name(name)
        response = await self.fetch(
            f"{self.base_url}/vector_stores/{store_id}/files/{file_id}",
            method="DELETE",
            headers=self._headers(),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        return bool(payload.get("deleted"))

    async def update_document(
        self,
        name: str,
        *,
        custom_metadata: list[dict[str, Any]] | None = None,
        chunking_config: dict[str, Any] | None = None,
    ) -> FileSearchDocument:
        store_id, file_id = _parse_openai_vector_store_file_name(name)
        response = await self.fetch(
            f"{self.base_url}/vector_stores/{store_id}/files/{file_id}",
            headers=self._headers(),
            json_body=drop_none(
                {
                    "attributes": _normalize_openai_vector_store_attributes(custom_metadata),
                    "chunking_strategy": deepcopy(chunking_config),
                }
            ),
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_vector_store_file(await response.json(), store_id=store_id)

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
        response = await self.fetch(
            f"{self.base_url}/vector_stores/{file_search_store_name}/search",
            headers=self._headers(),
            json_body=drop_none(
                {
                    "query": deepcopy(query),
                    "filters": deepcopy(filters),
                    "max_num_results": max_num_results,
                    "ranking_options": deepcopy(ranking_options),
                    "rewrite_query": rewrite_query,
                }
            ),
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        return FileSearchSearchResult(results=list(payload.get("data") or []), raw_response=payload)

    async def create_batch(
        self,
        *,
        file_search_store_name: str,
        file_names: list[str],
        custom_metadata: list[dict[str, Any]] | None = None,
        chunking_config: dict[str, Any] | None = None,
    ) -> FileSearchBatch:
        response = await self.fetch(
            f"{self.base_url}/vector_stores/{file_search_store_name}/file_batches",
            headers=self._headers(),
            json_body=drop_none(
                {
                    "file_ids": list(file_names),
                    "attributes": _normalize_openai_vector_store_attributes(custom_metadata),
                    "chunking_strategy": deepcopy(chunking_config),
                }
            ),
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_vector_store_batch(await response.json(), store_id=file_search_store_name)

    async def get_batch(self, name: str) -> FileSearchBatch:
        store_id, batch_id = _parse_openai_vector_store_file_batch_name(name)
        response = await self.fetch(
            f"{self.base_url}/vector_stores/{store_id}/file_batches/{batch_id}",
            method="GET",
            headers=self._headers(),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_vector_store_batch(await response.json(), store_id=store_id)

    async def cancel_batch(self, name: str) -> FileSearchBatch:
        store_id, batch_id = _parse_openai_vector_store_file_batch_name(name)
        response = await self.fetch(
            f"{self.base_url}/vector_stores/{store_id}/file_batches/{batch_id}/cancel",
            headers=self._headers(),
            json_body={},
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_vector_store_batch(await response.json(), store_id=store_id)

    async def list_batch_documents(
        self,
        *,
        name: str,
        page_size: int | None = None,
        page_token: str | None = None,
        state_filter: str | None = None,
    ) -> FileSearchDocumentListResult:
        store_id, batch_id = _parse_openai_vector_store_file_batch_name(name)
        response = await self.fetch(
            _request_url(
                self.base_url,
                f"/vector_stores/{store_id}/file_batches/{batch_id}/files",
                {"limit": page_size, "after": page_token, "filter": state_filter},
            ),
            method="GET",
            headers=self._headers(),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        payload = await response.json()
        return FileSearchDocumentListResult(
            documents=[_normalize_openai_vector_store_file(dict(item), store_id=store_id) for item in payload.get("data") or []],
            next_page_token=(payload.get("last_id") or payload.get("next_page")) if payload.get("has_more") else None,
            raw_response=payload,
        )

    async def wait_batch(
        self,
        name: str,
        *,
        poll_interval_ms: int = 500,
        timeout_ms: int | None = None,
    ) -> FileSearchBatch:
        deadline = None if timeout_ms is None else time.monotonic() + max(0, timeout_ms) / 1000
        while True:
            batch = await self.get_batch(name)
            state = str(batch.state or "").lower()
            if state in {"completed", "failed", "cancelled"}:
                return batch
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'OpenAI vector store batch "{name}" did not finish before timeout.')
                await asyncio.sleep(min(max(poll_interval_ms, 0) / 1000, remaining))
            else:
                await asyncio.sleep(max(poll_interval_ms, 0) / 1000)

    async def get_operation(self, name: str) -> FileSearchOperation:
        store_id, file_id = _parse_openai_vector_store_file_name(name)
        response = await self.fetch(
            f"{self.base_url}/vector_stores/{store_id}/files/{file_id}",
            method="GET",
            headers=self._headers(),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error(self.provider, response.status_code, await response.text())
        return _normalize_openai_vector_store_operation(await response.json(), store_id=store_id)

    async def wait_operation(
        self,
        name: str,
        *,
        poll_interval_ms: int = 500,
        timeout_ms: int | None = None,
    ) -> FileSearchOperation:
        deadline = None if timeout_ms is None else time.monotonic() + max(0, timeout_ms) / 1000
        while True:
            operation = await self.get_operation(name)
            if operation.done:
                return operation
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'OpenAI vector store operation "{name}" did not finish before timeout.')
                await asyncio.sleep(min(max(poll_interval_ms, 0) / 1000, remaining))
            else:
                await asyncio.sleep(max(poll_interval_ms, 0) / 1000)


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
    headers = {auth_header: value}
    headers.update(dict(extra_headers or {}))
    return headers


def _openai_realtime_tools(config: RealtimeSessionConfig) -> list[dict[str, Any]] | None:
    return _map_tools(config.tools, provider_name="openai")


def _openai_realtime_provider_options(provider_options: dict[str, Any] | None) -> dict[str, Any]:
    if not provider_options:
        return {}
    return {
        key: value
        for key, value in provider_options.items()
        if key not in {"headers", "realtime_url", "realtime_query", "expires_after"}
    }


def _openai_realtime_audio_format(media_type: str | None, sample_rate_hz: int | None) -> dict[str, Any] | None:
    if media_type is None and sample_rate_hz is None:
        return None
    return drop_none({"type": media_type, "rate": sample_rate_hz})


def _openai_realtime_session_config(config: RealtimeSessionConfig, *, model_id: str | None = None) -> dict[str, Any]:
    audio = drop_none(
        {
            "input": drop_none(
                {
                    "format": _openai_realtime_audio_format(config.input_audio_media_type, config.input_sample_rate_hz),
                    "turn_detection": config.turn_detection,
                }
            ),
            "output": drop_none(
                {
                    "format": _openai_realtime_audio_format(config.output_audio_media_type, config.output_sample_rate_hz),
                    "voice": config.voice,
                }
            ),
        }
    )
    session: dict[str, Any] = {
        "type": "realtime",
        "model": model_id,
        "instructions": config.instructions,
        "output_modalities": ["audio"] if config.output_audio_media_type or config.voice else ["text"],
        "tools": _openai_realtime_tools(config),
        "tool_choice": _map_tool_choice(config.tool_choice, provider_name="openai") if config.tool_choice is not None else None,
        "audio": audio or None,
        **_openai_realtime_provider_options(config.provider_options),
    }
    return drop_none(session)


def _openai_realtime_session_payload(config: RealtimeSessionConfig) -> dict[str, Any]:
    return {"type": "session.update", "session": _openai_realtime_session_config(config)}


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
    if event_type in {"response.audio.delta", "response.output_audio.delta"}:
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
    if event_type in {
        "response.audio_transcript.delta",
        "response.audio_transcription.delta",
        "response.output_audio_transcript.delta",
    }:
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
    if event_type in {
        "response.audio_transcript.done",
        "response.audio_transcription.done",
        "response.output_audio_transcript.done",
        "response.output_text.done",
    }:
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
    if event_type in {"response.done", "response.completed", "response.incomplete", "response.failed"}:
        return [RealtimeResponseCompletedEvent(reason=event_type, provider_metadata=payload)]
    if event_type == "session.closed":
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
        assistant_message = _parse_responses_message(payload, self.provider)
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
                if payload.get("type") in {"response.output_item.added", "response.output_item.done"}:
                    item = payload.get("item") or {}
                    provider_data_part = _parse_provider_data_output_item(item, self.provider) if isinstance(item, dict) else None
                    if provider_data_part is not None:
                        yield StreamProviderDataEvent(provider=provider_data_part.provider, data=provider_data_part.data)
                    elif item.get("type") == "function_call":
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
                    response_reference = _response_reference_part(response_payload, self.provider)
                    if response_reference is not None:
                        yield StreamProviderDataEvent(provider=response_reference.provider, data=response_reference.data)
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
                    "input": _to_responses_input(input.messages, self.provider),
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
        provider_options = dict(resolved_config.provider_options or {})
        expires_after = provider_options.pop("expires_after", None)
        url = self.browser_token_url or f"{self.base_url}/realtime/client_secrets"
        response = await with_retry(
            lambda: self.fetch(
                url,
                headers=self._headers(),
                json_body=drop_none({
                    "expires_after": expires_after,
                    "session": _openai_realtime_session_config(
                        replace(resolved_config, provider_options=provider_options or None),
                        model_id=self.model_id,
                    ),
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
            if expires_at_ms is None and secret.get("expires_at") is not None:
                expires_at_ms = int(secret.get("expires_at")) * 1000
        else:
            value = str(payload.get("token") or payload.get("value") or "")
            expires_at_ms = payload.get("expires_at_ms")
            if expires_at_ms is None and payload.get("expires_at") is not None:
                expires_at_ms = int(payload.get("expires_at")) * 1000
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
    images_client_factory: Callable[[], ImagesClient] | None = None,
    uploads_client_factory: Callable[[], UploadsClient] | None = None,
    moderations_client_factory: Callable[[], ModerationsClient] | None = None,
    batches_client_factory: Callable[[], BatchesClient] | None = None,
    containers_client_factory: Callable[[], ContainersClient] | None = None,
    skills_client_factory: Callable[[], SkillsClient] | None = None,
    file_search_stores_client_factory: Callable[[], FileSearchStoresClient] | None = None,
    responses_client_factory: Callable[[], ResponsesClient] | None = None,
    conversations_client_factory: Callable[[], ConversationsClient] | None = None,
) -> ProviderAdapter:
    resolved_key = api_key or os.getenv(env_var)
    if not resolved_key:
        raise ConfigurationError(f"Missing {provider_name} API key.")
    requester = fetch or default_fetch
    base = base_url.rstrip("/")
    provider_agent_capabilities = _openai_compat_agent_capabilities(provider_name)
    shared_capabilities = _with_agent_capabilities(capabilities or OPENAI_COMPAT_CAPABILITIES, provider_agent_capabilities)
    grounded_capabilities = _with_agent_capabilities(OPENAI_COMPAT_GROUNDED_CAPABILITIES, provider_agent_capabilities)
    realtime_capabilities = _with_agent_capabilities(OPENAI_COMPAT_REALTIME_CAPABILITIES, provider_agent_capabilities)
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
                capabilities=grounded_capabilities,
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
                capabilities=realtime_capabilities,
            ))
            if supports_realtime
            else None
        ),
        files_client_factory=files_client_factory,
        images_client_factory=images_client_factory,
        uploads_client_factory=uploads_client_factory,
        moderations_client_factory=moderations_client_factory,
        batches_client_factory=batches_client_factory,
        containers_client_factory=containers_client_factory,
        skills_client_factory=skills_client_factory,
        file_search_stores_client_factory=file_search_stores_client_factory,
        responses_client_factory=responses_client_factory,
        conversations_client_factory=conversations_client_factory,
    )
