from __future__ import annotations

from copy import deepcopy
import json
import os
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any

from .._http import Fetcher, default_fetch
from .._sse import parse_sse
from ..errors import ConfigurationError, ProviderHTTPError, UnsupportedFeatureError, ValidationError
from ..messages import normalize_finish_reason, serialize_json_value, validate_file_part, validate_message_parts
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..types import (
    CountTokensClient,
    CountTokensResult,
    FilePart,
    FilesClient,
    GenerateResult,
    GroundedGenerateResult,
    GroundedLanguageModel,
    GroundingSource,
    GroundedModelGenerateInput,
    LanguageModel,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    ProviderFile,
    StreamEvent,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    TextPart,
    TokenCountDetail,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolChoiceName,
    ToolExecutionResult,
    ToolResultPart,
    PortableSupport,
)
from ._payload import drop_none
from .base import ProviderAdapter, create_provider_bundle

ANTHROPIC_CAPABILITIES = ModelCapabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    parallel_tool_calls=True,
    vision=True,
    files=True,
    audio_input=False,
    audio_output=False,
    embeddings=False,
    reasoning=True,
    web_search=False,
)

ANTHROPIC_GROUNDED_CAPABILITIES = ModelCapabilities(
    streaming=False,
    tools=True,
    structured_output=False,
    json_mode=False,
    tool_choice=False,
    parallel_tool_calls=False,
    vision=True,
    files=True,
    audio_input=False,
    audio_output=False,
    embeddings=False,
    reasoning=True,
    web_search=True,
)

_ANTHROPIC_THINKING_BLOCK_TYPES = {"thinking", "redacted_thinking"}
_ANTHROPIC_FILES_BETA = "files-api-2025-04-14"
_ANTHROPIC_MCP_BETA = "mcp-client-2025-11-20"
_ANTHROPIC_DEFAULT_WEB_SEARCH_TYPE = "web_search_20250305"
_ANTHROPIC_UNSUPPORTED_SCHEMA_KEYS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minProperties",
    "maxProperties",
    "pattern",
    "format",
}


def _coerce_beta_headers(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _merge_beta_headers(*values: str | list[str] | tuple[str, ...] | None) -> str | None:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in _coerce_beta_headers(value):
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return ",".join(merged) if merged else None


def anthropic_web_search_tool(
    *,
    name: str = "web_search",
    max_uses: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
    user_location: dict[str, Any] | None = None,
    tool_type: str = _ANTHROPIC_DEFAULT_WEB_SEARCH_TYPE,
    **extra: Any,
) -> dict[str, Any]:
    return drop_none(
        {
            "type": tool_type,
            "name": name,
            "max_uses": max_uses,
            "allowed_domains": list(allowed_domains) if allowed_domains else None,
            "blocked_domains": list(blocked_domains) if blocked_domains else None,
            "user_location": deepcopy(user_location) if user_location is not None else None,
            **deepcopy(extra),
        }
    )


def anthropic_mcp_server(
    *,
    url: str,
    name: str,
    authorization_token: str | None = None,
    enabled: bool | None = None,
    allowed_tools: list[str] | None = None,
    tool_configuration: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    config = deepcopy(tool_configuration) if tool_configuration is not None else {}
    if enabled is not None:
        config["enabled"] = enabled
    if allowed_tools is not None:
        config["allowed_tools"] = list(allowed_tools)
    return drop_none(
        {
            "type": "url",
            "server_name": name,
            "url": url,
            "authorization_token": authorization_token,
            "tool_configuration": config or None,
            **deepcopy(extra),
        }
    )


def anthropic_code_execution_tool(
    *,
    name: str = "code_execution",
    tool_type: str = "code_execution_20250825",
    **extra: Any,
) -> dict[str, Any]:
    return drop_none({"type": tool_type, "name": name, **deepcopy(extra)})


def _parse_response_error(provider: str, response: Any) -> ProviderHTTPError:
    return ProviderHTTPError(
        f'Provider "{provider}" request failed with status {response.status_code}.',
        response.status_code,
        response_body="",
    )


def _is_image_media_type(media_type: str | None) -> bool:
    return bool(media_type and media_type.lower().startswith("image/"))


def _has_uploaded_file_reference(messages: list[ModelMessage]) -> bool:
    for message in messages:
        for part in message.parts:
            if part.type != "file":
                continue
            if part.file_id is not None:
                return True
            if part.provider_metadata.get("anthropic_block_type") == "container_upload":
                return True
    return False


def _extract_provider_options(provider_options: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    options = dict(provider_options or {})
    request_betas = _coerce_beta_headers(options.pop("anthropic_beta", None))
    return options, request_betas


def _normalize_anthropic_file(payload: dict[str, Any]) -> ProviderFile:
    return ProviderFile(
        provider="anthropic",
        id=str(payload.get("id") or ""),
        filename=payload.get("filename"),
        media_type=payload.get("mime_type"),
        size_bytes=payload.get("size_bytes"),
        status=payload.get("status"),
        created_at=payload.get("created_at"),
        downloadable=payload.get("downloadable"),
        metadata=dict(payload),
    )


def _normalize_binary(data: bytes | bytearray | memoryview) -> bytes:
    if isinstance(data, memoryview):
        return data.tobytes()
    return bytes(data)


def _anthropic_schema(schema: Any) -> dict[str, Any]:
    def sanitize(node: Any) -> Any:
        if isinstance(node, list):
            return [sanitize(item) for item in node]
        if not isinstance(node, dict):
            return node
        cleaned = {key: sanitize(value) for key, value in node.items() if key not in _ANTHROPIC_UNSUPPORTED_SCHEMA_KEYS}
        if cleaned.get("type") == "object":
            cleaned.setdefault("additionalProperties", False)
            properties = cleaned.get("properties")
            if isinstance(properties, dict):
                cleaned["properties"] = {str(key): sanitize(value) for key, value in properties.items()}
        return cleaned

    return sanitize(create_schema_adapter(schema).json_schema())


def _tool_block_type(part: FilePart) -> str:
    block_type = str(part.provider_metadata.get("anthropic_block_type") or "")
    if block_type:
        return block_type
    if _is_image_media_type(part.media_type):
        return "image"
    return "document"


def _document_source(part: FilePart) -> dict[str, Any]:
    if part.file_uri is not None:
        raise ValidationError('Provider "anthropic" does not support "file_uri". Use "file_id" instead.')
    if part.document_content is not None:
        return {"type": "content", "content": serialize_json_value(part.document_content)}
    if part.text is not None:
        return {"type": "text", "media_type": "text/plain", "data": part.text}
    if part.url is not None:
        return {"type": "url", "url": part.url}
    if part.data is not None:
        if (part.media_type or "").lower() == "text/plain":
            return {"type": "text", "media_type": "text/plain", "data": part.data}
        return {
            "type": "base64",
            "media_type": part.media_type or "application/pdf",
            "data": part.data,
        }
    return {"type": "file", "file_id": part.file_id}


def _image_source(part: FilePart) -> dict[str, Any]:
    if part.file_uri is not None:
        raise ValidationError('Provider "anthropic" does not support "file_uri". Use "file_id" instead.')
    if part.text is not None or part.document_content is not None:
        raise ValidationError('Provider "anthropic" image file blocks do not support "text" or "document_content" sources.')
    if part.url is not None:
        return {"type": "url", "url": part.url}
    if part.data is not None:
        return {
            "type": "base64",
            "media_type": part.media_type or "image/png",
            "data": part.data,
        }
    return {"type": "file", "file_id": part.file_id}


def _anthropic_file_block(part: FilePart) -> dict[str, Any]:
    validate_file_part(part)
    block_type = _tool_block_type(part)
    extra_fields = dict(part.provider_metadata.get("anthropic_block_fields") or {})
    if block_type == "container_upload":
        if part.file_id is None:
            raise ValidationError('Provider "anthropic" container uploads require "file_id".')
        return {"type": "container_upload", "file_id": part.file_id, **extra_fields}
    if block_type == "image":
        return {
            "type": "image",
            "source": _image_source(part),
            **({"cache_control": serialize_json_value(part.cache_control)} if part.cache_control is not None else {}),
            **extra_fields,
        }

    block = {
        "type": "document",
        "source": _document_source(part),
        **({"title": part.title or part.filename} if (part.title or part.filename) else {}),
        **({"context": part.context} if part.context else {}),
        **({"citations": {"enabled": bool(part.citations_enabled)}} if part.citations_enabled is not None else {}),
        **({"cache_control": serialize_json_value(part.cache_control)} if part.cache_control is not None else {}),
    }
    block.update(extra_fields)
    return block


def _map_block_parts(message: ModelMessage) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in message.parts:
        if part.type == "tool-call":
            thinking_blocks = part.tool_call.provider_metadata.get("anthropic_thinking_blocks")
            if isinstance(thinking_blocks, list) and thinking_blocks:
                blocks.extend(deepcopy(thinking_blocks))
                break

    for part in message.parts:
        if part.type == "text":
            raw_block = part.provider_metadata.get("anthropic_raw_block")
            if isinstance(raw_block, dict):
                blocks.append(deepcopy(raw_block))
                continue
            block = {"type": "text", "text": part.text}
            if part.provider_metadata.get("cache_control") is not None:
                block["cache_control"] = serialize_json_value(part.provider_metadata.get("cache_control"))
            blocks.append(block)
        elif part.type == "image":
            block: dict[str, Any]
            if part.image.startswith("data:") and ";base64," in part.image:
                header, body = part.image[len("data:"):].split(";base64,", 1)
                block = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.media_type or header.lower(),
                        "data": body,
                    },
                }
            else:
                block = {"type": "image", "source": {"type": "url", "url": part.image}}
            if part.provider_metadata.get("cache_control") is not None:
                block["cache_control"] = serialize_json_value(part.provider_metadata.get("cache_control"))
            blocks.append(block)
        elif part.type == "file":
            blocks.append(_anthropic_file_block(part))
        elif part.type == "tool-call":
            block_type = part.tool_call.provider_metadata.get("anthropic_tool_block_type") or "tool_use"
            blocks.append(
                {
                    "type": block_type,
                    "id": part.tool_call.id,
                    "name": part.tool_call.name,
                    "input": serialize_json_value(part.tool_call.input),
                }
            )
        elif part.type == "tool-result":
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": part.tool_result.tool_call_id,
                    "content": json.dumps(
                        part.tool_result.error.__dict__ if part.tool_result.is_error else part.tool_result.output
                    ),
                    "is_error": part.tool_result.is_error,
                }
            )
    return blocks


def _system_prompt_from_messages(messages: list[ModelMessage]) -> str:
    return "\n".join(
        part.text for message in messages if message.role == "system" for part in message.parts if part.type == "text"
    )


def _map_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    mapped = []
    for message in messages:
        if message.role == "system":
            continue
        role = "assistant" if message.role == "assistant" else "user"
        if message.role == "tool":
            role = "user"
        mapped.append({"role": role, "content": _map_block_parts(message)})
    return mapped


def _map_tools(tools: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    mapped: list[dict[str, Any]] = []
    for tool in tools.values():
        payload = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": _anthropic_schema(tool.schema),
        }
        if tool.input_examples:
            payload["input_examples"] = [serialize_json_value(item) for item in tool.input_examples]
        if tool.strict is not None:
            payload["strict"] = tool.strict
        if tool.defer_loading is not None:
            payload["defer_loading"] = tool.defer_loading
        if tool.eager_input_streaming is not None:
            payload["eager_input_streaming"] = tool.eager_input_streaming
        if tool.allowed_callers:
            payload["allowed_callers"] = list(tool.allowed_callers)
        if tool.cache_control is not None:
            payload["cache_control"] = serialize_json_value(tool.cache_control)
        mapped.append(payload)
    return mapped


def _merge_tool_payloads(
    mapped_tools: list[dict[str, Any]] | None,
    provider_options: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    options = dict(provider_options)
    extra_tools = options.pop("tools", None)
    if extra_tools is None:
        return mapped_tools, options
    if not isinstance(extra_tools, list):
        raise ValidationError('Provider "anthropic" expects "provider_options[\'tools\']" to be a list when provided.')
    merged = [*list(mapped_tools or []), *(serialize_json_value(item) for item in extra_tools)]
    return merged or None, options


def _map_tool_choice(
    tool_choice: str | ToolChoiceName | None,
    *,
    extended_thinking: bool = False,
) -> dict[str, Any] | None:
    if tool_choice is None or tool_choice == "auto":
        return None
    if tool_choice == "none":
        return {"type": "none"}
    if extended_thinking:
        raise UnsupportedFeatureError(
            'Provider "anthropic" only supports "tool_choice=auto" or "tool_choice=none" when extended thinking is enabled.'
        )
    if tool_choice == "required":
        return {"type": "any"}
    return {"type": "tool", "name": tool_choice.tool_name}


def _map_reasoning(input: ModelGenerateInput | GroundedModelGenerateInput) -> dict[str, Any] | None:
    if input.reasoning is None:
        return None
    if input.reasoning.effort is not None:
        raise UnsupportedFeatureError('Provider "anthropic" does not support "reasoning.effort" through this SDK yet.')
    if input.reasoning.budget_tokens is None:
        return None
    return {"type": "enabled", "budget_tokens": input.reasoning.budget_tokens}


def _map_structured_output(input: ModelGenerateInput) -> dict[str, Any] | None:
    if input.structured_output is None or input.structured_output.mode != "native":
        return None
    return {
        "format": {
            "type": "json_schema",
            "name": input.structured_output.name or "response",
            "schema": _anthropic_schema(input.structured_output.schema),
        }
    }


def _parse_tool_input(raw_input: str) -> tuple[Any, dict[str, Any]]:
    if not raw_input:
        return {}, {}
    try:
        return json.loads(raw_input), {}
    except Exception:
        return {"INVALID_JSON": raw_input}, {"invalid_tool_json": True, "raw_tool_json": raw_input}


def _parse_token_usage(payload: dict[str, Any]) -> TokenUsage | None:
    usage = payload.get("usage") or {}
    if not usage:
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _text_part_from_block(block: dict[str, Any]) -> TextPart:
    provider_metadata = {}
    if block.get("citations") is not None:
        provider_metadata["citations"] = deepcopy(block["citations"])
    if block.get("cache_control") is not None:
        provider_metadata["cache_control"] = deepcopy(block["cache_control"])
    return TextPart(text=str(block.get("text") or ""), provider_metadata=provider_metadata)


def _parse_assistant_message(payload: dict[str, Any]) -> ModelMessage:
    parts: list[Any] = []
    thinking_blocks: list[dict[str, Any]] = []
    attached_thinking = False
    for block in payload.get("content") or []:
        block_type = str(block.get("type") or "")
        if block_type == "text":
            parts.append(_text_part_from_block(block))
        elif block_type in _ANTHROPIC_THINKING_BLOCK_TYPES:
            thinking_blocks.append(deepcopy(block))
        elif block_type in {"tool_use", "server_tool_use", "mcp_tool_use"}:
            provider_metadata = {
                "anthropic_tool_block_type": block_type,
                "provider_managed": block_type in {"server_tool_use", "mcp_tool_use"},
                "anthropic_raw_block": deepcopy(block),
            }
            if block.get("server_name") is not None:
                provider_metadata["server_name"] = block.get("server_name")
            if thinking_blocks and not attached_thinking:
                provider_metadata["anthropic_thinking_blocks"] = deepcopy(thinking_blocks)
                attached_thinking = True
            parts.append(
                ToolCallPart(
                    tool_call=ToolCall(
                        id=str(block.get("id") or ""),
                        name=str(block.get("name") or ""),
                        input=serialize_json_value(block.get("input") or {}),
                        provider_metadata=provider_metadata,
                        )
                    )
                )
        elif block_type in {"mcp_tool_result", "web_search_tool_result"}:
            parts.append(
                ToolResultPart(
                    tool_result=ToolExecutionResult(
                        tool_call_id=str(block.get("tool_use_id") or block.get("id") or ""),
                        tool_name=str(block.get("name") or block_type),
                        output=serialize_json_value(block.get("content") or block),
                        is_error=False,
                    )
                )
            )
        elif block_type:
            parts.append(TextPart(text="", provider_metadata={"anthropic_raw_block": deepcopy(block)}))
    return ModelMessage(role="assistant", parts=parts)


def _extract_web_search_sources(payload: dict[str, Any]) -> list[GroundingSource]:
    sources: list[GroundingSource] = []
    seen: set[tuple[str, str | None]] = set()
    results_by_url: dict[str, dict[str, Any]] = {}

    for block in payload.get("content") or []:
        if block.get("type") != "web_search_tool_result":
            continue
        content = block.get("content") or []
        if isinstance(content, dict):
            content = [content]
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "web_search_result":
                continue
            url = str(item.get("url") or "")
            if not url:
                continue
            results_by_url[url] = dict(item)
            key = (url, None)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                GroundingSource(
                    url=url,
                    title=item.get("title"),
                    provider_metadata={"anthropic_web_search_result": dict(item)},
                )
            )

    for block in payload.get("content") or []:
        if block.get("type") != "text":
            continue
        for citation in block.get("citations") or []:
            if not isinstance(citation, dict) or citation.get("type") != "web_search_result_location":
                continue
            url = str(citation.get("url") or "")
            if not url:
                continue
            result = results_by_url.get(url, {})
            snippet = citation.get("cited_text") or result.get("cited_text")
            key = (url, str(snippet) if snippet is not None else None)
            if key in seen:
                continue
            seen.add(key)
            provider_metadata = {
                "anthropic_web_search_citation": dict(citation),
                **({"anthropic_web_search_result": result} if result else {}),
            }
            sources.append(
                GroundingSource(
                    url=url,
                    title=citation.get("title") or result.get("title"),
                    snippet=snippet,
                    provider_metadata=provider_metadata,
                )
            )

    return sources


def _build_web_search_tool(provider_options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    remaining = dict(provider_options)
    web_search = remaining.pop("web_search", None)
    if web_search is False:
        raise ValidationError('Provider "anthropic" grounded generation requires web search to remain enabled.')
    if web_search is None or web_search is True:
        config: dict[str, Any] = {"type": _ANTHROPIC_DEFAULT_WEB_SEARCH_TYPE, "name": "web_search"}
    elif isinstance(web_search, dict):
        config = dict(web_search)
    else:
        raise ValidationError('Provider "anthropic" expects "provider_options[\'web_search\']" to be a dict or boolean.')
    config.setdefault("type", _ANTHROPIC_DEFAULT_WEB_SEARCH_TYPE)
    config.setdefault("name", "web_search")
    return config, remaining


@dataclass(slots=True)
class AnthropicFilesClient(FilesClient):
    api_key: str
    base_url: str
    anthropic_version: str
    fetch: Fetcher
    beta_headers: list[str] = field(default_factory=list)

    def _headers(self, *, json_content: bool = True) -> dict[str, str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "anthropic-beta": _merge_beta_headers(self.beta_headers, _ANTHROPIC_FILES_BETA) or _ANTHROPIC_FILES_BETA,
        }
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
        form_data = {"purpose": purpose or "user_data"}
        if metadata:
            form_data["metadata"] = json.dumps(metadata)
        response = await self.fetch(
            f"{self.base_url}/files",
            headers=self._headers(json_content=False),
            body={
                "data": form_data,
                "files": {"file": (filename, _normalize_binary(data), media_type)},
            },
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                'Provider "anthropic" request failed.',
                response.status_code,
                response_body=await response.text(),
            )
        return _normalize_anthropic_file(await response.json())

    async def list(self) -> list[ProviderFile]:
        response = await self.fetch(
            f"{self.base_url}/files",
            method="GET",
            headers=self._headers(),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                'Provider "anthropic" request failed.',
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        return [_normalize_anthropic_file(dict(item)) for item in payload.get("data") or []]

    async def get(self, file_id: str) -> ProviderFile:
        response = await self.fetch(
            f"{self.base_url}/files/{file_id}",
            method="GET",
            headers=self._headers(),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                'Provider "anthropic" request failed.',
                response.status_code,
                response_body=await response.text(),
            )
        return _normalize_anthropic_file(await response.json())

    async def download(self, file_id: str) -> bytes:
        response = await self.fetch(
            f"{self.base_url}/files/{file_id}/content",
            method="GET",
            headers=self._headers(json_content=False),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                'Provider "anthropic" request failed.',
                response.status_code,
                response_body=await response.text(),
            )
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
            raise ProviderHTTPError(
                'Provider "anthropic" request failed.',
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        deleted_type = str(payload.get("type") or "")
        return bool(payload.get("deleted")) or deleted_type == "file_deleted"


@dataclass(slots=True)
class _AnthropicBase:
    provider: str
    model_id: str
    api_key: str
    base_url: str
    anthropic_version: str
    fetch: Fetcher
    beta_headers: list[str] = field(default_factory=list)

    def _headers(self, extra_beta_headers: list[str] | None = None) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
        }
        merged = _merge_beta_headers(self.beta_headers, extra_beta_headers)
        if merged:
            headers["anthropic-beta"] = merged
        return headers

    def _message_beta_headers(
        self,
        *,
        messages: list[ModelMessage],
        provider_options: dict[str, Any],
        request_betas: list[str],
    ) -> list[str]:
        extra = list(request_betas)
        if _has_uploaded_file_reference(messages):
            extra.append(_ANTHROPIC_FILES_BETA)
        if provider_options.get("mcp_servers") is not None:
            extra.append(_ANTHROPIC_MCP_BETA)
        return extra


@dataclass(slots=True)
class AnthropicCountTokensClient(_AnthropicBase, CountTokensClient):
    async def count(
        self,
        *,
        model_id: str,
        prompt: str | None = None,
        messages: list[ModelMessage] | None = None,
        system: str | None = None,
        tools: dict[str, Any] | None = None,
        provider_options: dict[str, Any] | None = None,
        options: Any = None,
    ) -> CountTokensResult:
        built_messages = messages
        if built_messages is None:
            content = prompt or ""
            built_messages = [ModelMessage(role="user", parts=[TextPart(text=content)])]
            if system:
                built_messages.insert(0, ModelMessage(role="system", parts=[TextPart(text=system)]))
        elif system:
            built_messages = [ModelMessage(role="system", parts=[TextPart(text=system)]), *built_messages]
        validate_message_parts(self, built_messages)
        extracted_options, request_betas = _extract_provider_options(provider_options)
        body_tools, extracted_options = _merge_tool_payloads(_map_tools(tools), extracted_options)
        body = drop_none(
            {
                "model": model_id,
                "system": _system_prompt_from_messages(built_messages),
                "messages": _map_messages(built_messages),
                "tools": body_tools,
                **extracted_options,
            }
        )
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/messages/count_tokens",
                headers=self._headers(
                    self._message_beta_headers(
                        messages=built_messages,
                        provider_options=extracted_options,
                        request_betas=request_betas,
                    )
                ),
                json_body=body,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                'Provider "anthropic" request failed.',
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        return CountTokensResult(
            total_tokens=payload.get("input_tokens"),
            details=[
                TokenCountDetail(
                    modality="input",
                    token_count=payload.get("input_tokens"),
                    provider_metadata=dict(payload),
                )
            ]
            if payload.get("input_tokens") is not None
            else [],
            raw_response=payload,
        )


@dataclass(slots=True)
class AnthropicLanguageModel(_AnthropicBase, LanguageModel):
    capabilities: ModelCapabilities = field(default_factory=lambda: ANTHROPIC_CAPABILITIES)

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        validate_message_parts(self, input.messages)
        extended_thinking = bool(input.reasoning is not None and input.reasoning.budget_tokens is not None)
        provider_options, request_betas = _extract_provider_options(input.provider_options)
        body_tools, provider_options = _merge_tool_payloads(_map_tools(input.tools), provider_options)
        body = drop_none(
            {
                "model": self.model_id,
                "system": _system_prompt_from_messages(input.messages),
                "messages": _map_messages(input.messages),
                "tools": body_tools,
                "tool_choice": _map_tool_choice(input.tool_choice, extended_thinking=extended_thinking),
                "temperature": input.temperature,
                "max_tokens": input.max_tokens or 1024,
                "output_config": _map_structured_output(input),
                **provider_options,
                "thinking": _map_reasoning(input),
            }
        )
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/messages",
                headers=self._headers(
                    self._message_beta_headers(
                        messages=input.messages,
                        provider_options=provider_options,
                        request_betas=request_betas,
                    )
                ),
                json_body=body,
                timeout_ms=input.timeout_ms,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                "Anthropic request failed.",
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        assistant_message = _parse_assistant_message(payload)
        return GenerateResult(
            messages=[assistant_message],
            text="".join(part.text for part in assistant_message.parts if part.type == "text"),
            finish_reason=normalize_finish_reason(payload.get("stop_reason")),
            provider_finish_reason=payload.get("stop_reason"),
            usage=_parse_token_usage(payload),
            raw_response=payload,
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        validate_message_parts(self, input.messages)
        extended_thinking = bool(input.reasoning is not None and input.reasoning.budget_tokens is not None)
        provider_options, request_betas = _extract_provider_options(input.provider_options)
        body_tools, provider_options = _merge_tool_payloads(_map_tools(input.tools), provider_options)
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/messages",
                headers=self._headers(
                    self._message_beta_headers(
                        messages=input.messages,
                        provider_options=provider_options,
                        request_betas=request_betas,
                    )
                ),
                json_body=drop_none(
                    {
                        "model": self.model_id,
                        "system": _system_prompt_from_messages(input.messages),
                        "messages": _map_messages(input.messages),
                        "tools": body_tools,
                        "tool_choice": _map_tool_choice(input.tool_choice, extended_thinking=extended_thinking),
                        "temperature": input.temperature,
                        "max_tokens": input.max_tokens or 1024,
                        "stream": True,
                        "output_config": _map_structured_output(input),
                        **provider_options,
                        "thinking": _map_reasoning(input),
                    }
                ),
                timeout_ms=input.timeout_ms,
                stream=True,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                "Anthropic request failed.",
                response.status_code,
                response_body=await response.text(),
            )

        async def generator() -> AsyncIterable[StreamEvent]:
            tool_buffers: dict[int, dict[str, Any]] = {}
            usage: TokenUsage | None = None
            finish_reason: str | None = None
            async for event in parse_sse(response.iter_lines()):
                if not event.data:
                    continue
                payload = json.loads(event.data)
                if event.event == "content_block_delta" and payload.get("delta", {}).get("type") == "text_delta":
                    yield StreamTextDeltaEvent(text_delta=str(payload["delta"]["text"]))
                elif event.event == "content_block_start" and payload.get("content_block", {}).get("type") in {
                    "tool_use",
                    "server_tool_use",
                    "mcp_tool_use",
                }:
                    block = payload["content_block"]
                    tool_buffers[payload["index"]] = {
                        "id": str(block.get("id") or ""),
                        "name": str(block.get("name") or ""),
                        "input": "",
                        "provider_metadata": {
                            "anthropic_tool_block_type": block.get("type"),
                            "provider_managed": block.get("type") in {"server_tool_use", "mcp_tool_use"},
                            "anthropic_raw_block": deepcopy(block),
                            **({"server_name": block.get("server_name")} if block.get("server_name") is not None else {}),
                        },
                    }
                elif event.event == "content_block_delta" and payload.get("delta", {}).get("type") == "input_json_delta":
                    current = tool_buffers.get(payload["index"])
                    if current is not None:
                        current["input"] += str(payload["delta"]["partial_json"])
                elif event.event == "content_block_stop":
                    current = tool_buffers.get(payload["index"])
                    if current is not None:
                        parsed_input, metadata = _parse_tool_input(str(current["input"]))
                        provider_metadata = dict(current["provider_metadata"])
                        provider_metadata.update(metadata)
                        yield StreamToolCallEvent(
                            tool_call=ToolCall(
                                id=current["id"],
                                name=current["name"],
                                input=parsed_input,
                                provider_metadata=provider_metadata,
                            )
                        )
                elif event.event == "message_delta":
                    finish_reason = payload.get("delta", {}).get("stop_reason") or finish_reason
                    delta_usage = payload.get("usage")
                    if isinstance(delta_usage, dict):
                        usage = _parse_token_usage({"usage": delta_usage})
                elif event.event == "message_stop":
                    usage = usage or _parse_token_usage(payload)
                    yield StreamFinishEvent(
                        finish_reason=normalize_finish_reason(finish_reason or payload.get("stop_reason")),
                        provider_finish_reason=finish_reason or payload.get("stop_reason"),
                        usage=usage,
                    )

        return generator()


@dataclass(slots=True)
class AnthropicGroundedLanguageModel(_AnthropicBase, GroundedLanguageModel):
    capabilities: ModelCapabilities = field(default_factory=lambda: ANTHROPIC_GROUNDED_CAPABILITIES)

    async def generate(self, input: GroundedModelGenerateInput) -> GroundedGenerateResult:
        validate_message_parts(self, input.messages)
        provider_options, request_betas = _extract_provider_options(input.provider_options)
        web_search_tool, remaining_options = _build_web_search_tool(provider_options)
        body_tools, remaining_options = _merge_tool_payloads([web_search_tool], remaining_options)
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/messages",
                headers=self._headers(
                    self._message_beta_headers(
                        messages=input.messages,
                        provider_options=remaining_options,
                        request_betas=request_betas,
                    )
                ),
                json_body=drop_none(
                    {
                        "model": self.model_id,
                        "system": _system_prompt_from_messages(input.messages),
                        "messages": _map_messages(input.messages),
                        "tools": body_tools,
                        "temperature": input.temperature,
                        "max_tokens": input.max_tokens or 1024,
                        **remaining_options,
                        "thinking": _map_reasoning(input),
                    }
                ),
                timeout_ms=input.timeout_ms,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                "Anthropic request failed.",
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        assistant_message = _parse_assistant_message(payload)
        return GroundedGenerateResult(
            messages=[assistant_message],
            text="".join(part.text for part in assistant_message.parts if part.type == "text"),
            sources=_extract_web_search_sources(payload),
            finish_reason=normalize_finish_reason(payload.get("stop_reason")),
            provider_finish_reason=payload.get("stop_reason"),
            usage=_parse_token_usage(payload),
            raw_response=payload,
        )


def create_anthropic(
    *,
    api_key: str | None = None,
    base_url: str = "https://api.anthropic.com/v1",
    anthropic_version: str = "2023-06-01",
    beta_headers: str | list[str] | tuple[str, ...] | None = None,
    fetch: Fetcher | None = None,
):
    resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise ConfigurationError("Missing Anthropic API key.")
    requester = fetch or default_fetch
    resolved_betas = _coerce_beta_headers(beta_headers)
    native = ProviderAdapter(
        name="anthropic",
        language_model_factory=lambda model_id: AnthropicLanguageModel(
            provider="anthropic",
            model_id=model_id,
            api_key=resolved_key,
            base_url=base_url.rstrip("/"),
            anthropic_version=anthropic_version,
            fetch=requester,
            beta_headers=list(resolved_betas),
        ),
        grounded_language_model_factory=lambda model_id: AnthropicGroundedLanguageModel(
            provider="anthropic",
            model_id=model_id,
            api_key=resolved_key,
            base_url=base_url.rstrip("/"),
            anthropic_version=anthropic_version,
            fetch=requester,
            beta_headers=list(resolved_betas),
        ),
        files_client_factory=lambda: AnthropicFilesClient(
            api_key=resolved_key,
            base_url=base_url.rstrip("/"),
            anthropic_version=anthropic_version,
            fetch=requester,
            beta_headers=list(resolved_betas),
        ),
        count_tokens_client_factory=lambda: AnthropicCountTokensClient(
            provider="anthropic",
            model_id="",
            api_key=resolved_key,
            base_url=base_url.rstrip("/"),
            anthropic_version=anthropic_version,
            fetch=requester,
            beta_headers=list(resolved_betas),
        ),
    )
    return create_provider_bundle(
        name="anthropic",
        native=native,
        portable_support=PortableSupport(
            text_generation=True,
            streaming=True,
            structured_output=True,
            tools=True,
            embeddings=False,
            grounding=True,
            retrieval=True,
            transcription=False,
            speech=False,
            portable_badge=False,
            tier="native-only",
        ),
    )
