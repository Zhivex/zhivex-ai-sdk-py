"""Internal Responses output normalization; no HTTP or adapter dependencies.

Owned by the OpenAI-compatible provider layer. Shared request serialization
uses the same provider-data decoder; adapters own transport and streaming state.
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from ..messages import (
    normalize_finish_reason,
)
from ..types import (
    AzureOpenAIMcpApprovalRequest,
    AzureOpenAIMcpApprovalResponse,
    AzureOpenAIMcpCall,
    AzureOpenAIMcpListTools,
    AzureOpenAIResponseReference,
    CodeExecutionResultPart,
    FilePart,
    FinishReason,
    GeneratedCodePart,
    ImagePart,
    ModelMessage,
    OpenAIMcpApprovalRequest,
    OpenAIMcpApprovalResponse,
    OpenAIMcpCall,
    OpenAIMcpListTools,
    OpenAIResponseReference,
    ProviderDataPart,
    TextPart,
    TokenUsage,
    ToolCall,
    ToolCallPart,
)
from ._payload import drop_none

_PROVIDER_MANAGED_TOOL_NAMES = {
    "apply_patch_call": "apply_patch",
    "code_interpreter_call": "code_interpreter",
    "computer_call": "computer_use",
    "computer_call_output": "computer_use",
    "file_search_call": "file_search",
    "image_search_call": "image_search",
    "image_generation_call": "image_generation",
    "local_shell_call": "local_shell",
    "mcp_call": "mcp",
    "shell_call": "shell",
    "tool_search_call": "tool_search",
    "web_extractor_call": "web_extractor",
    "web_search_call": "web_search",
    "web_search_image_call": "web_search_image",
}

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

def _parse_responses_usage(payload: dict[str, Any]) -> TokenUsage | None:
    usage = payload.get("usage") or {}
    if not usage:
        return None
    return TokenUsage(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
    )

def _parse_response_finish_reason(payload: dict[str, Any]) -> tuple[FinishReason | None, str | None]:
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
    elif item_type == "reasoning" or item_type in {"program", "program_output"}:
        parts.append(_provider_data_part_for(provider_name, deepcopy(item)))
    elif item_type == "function_call":
        parts.append(
            ToolCallPart(
                tool_call=ToolCall(
                    id=item.get("call_id") or item.get("id", ""),
                    name=item.get("name", ""),
                    input=_normalize_tool_call_input(item.get("arguments")),
                    provider_metadata=drop_none(
                        {
                            "provider": provider_name,
                            "response_item_id": item.get("id"),
                            "caller": deepcopy(item.get("caller")) if isinstance(item.get("caller"), dict) else None,
                        }
                    ),
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
