from __future__ import annotations

import base64
import json
import os
import re
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ..errors import ConfigurationError, UnsupportedFeatureError, ValidationError
from ..messages import normalize_finish_reason
from ..realtime import CallbackRealtimeSession, RealtimeConnectionFactory, RealtimeSessionCallbacks, encode_audio_frame, tool_result_payload, unsupported_browser_token
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..types import (
    AgentCapabilities,
    AudioFrame,
    GenerateResult,
    LanguageModel,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
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
    StreamEvent,
    StreamFinishEvent,
    StreamProviderDataEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    ProviderDataPart,
    TextPart,
    TokenUsage,
    ToolCall,
    ToolCallPart,
    ToolChoiceName,
    ToolExecutionResult,
    ToolResultPart,
    PortableSupport,
)
from .base import ProviderAdapter, create_provider_bundle
from ._payload import drop_none

BEDROCK_CAPABILITIES = ModelCapabilities(
    streaming=True,
    tools=True,
    structured_output=False,
    json_mode=False,
    tool_choice=True,
    parallel_tool_calls=False,
    vision=True,
    files=False,
    audio_input=False,
    audio_output=False,
    embeddings=False,
    reasoning=False,
    web_search=False,
    agent_capabilities=AgentCapabilities(
        support_tier="tier-b",
        tool_choice_none=True,
    ),
)

BEDROCK_REALTIME_CAPABILITIES = ModelCapabilities(
    streaming=False,
    tools=False,
    structured_output=False,
    json_mode=False,
    tool_choice=False,
    parallel_tool_calls=False,
    vision=False,
    files=False,
    audio_input=True,
    audio_output=True,
    embeddings=False,
    reasoning=False,
    web_search=False,
    agent_capabilities=AgentCapabilities(
        support_tier="tier-c",
        tool_choice_none=False,
    ),
    realtime=True,
    realtime_audio_input=True,
    realtime_audio_output=True,
    realtime_tools=True,
    realtime_browser_tokens=False,
)

# Portable reasoning is limited to these Converse model families. Every other
# Bedrock id keeps BEDROCK_CAPABILITIES, including reasoning=False.
_ReasoningFamily = Literal["sonnet-5-5", "haiku-4-5", "nova-2-lite"]
_GEO_PREFIX = re.compile(r"^(?:global|us|eu|apac|jp|au|in)\.")
_ARN_MODEL = re.compile(r"(?:foundation-model|inference-profile)/([^/]+)$")
_REASONING_FAMILIES: dict[str, _ReasoningFamily] = {
    "anthropic.claude-sonnet-5-5": "sonnet-5-5",
    "anthropic.claude-haiku-4-5-20251001-v1:0": "haiku-4-5",
    "amazon.nova-2-lite-v1:0": "nova-2-lite",
}
_SONNET_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
_NOVA_EFFORTS = frozenset({"low", "medium", "high"})
_HAIKU_MIN_BUDGET = 1024
_HAIKU_STREAM_ABOVE = 21333
_HAIKU_INTERLEAVED_BETA = "interleaved-thinking-2025-05-14"
_BEDROCK_REASONING_TYPE = "reasoning_content"


def _strip_bedrock_model_id(model_id: str) -> str:
    arn = _ARN_MODEL.search(model_id)
    raw = arn.group(1) if arn else model_id
    return _GEO_PREFIX.sub("", raw)


def _reasoning_family(model_id: str) -> _ReasoningFamily | None:
    return _REASONING_FAMILIES.get(_strip_bedrock_model_id(model_id))


def _capabilities_for(model_id: str) -> ModelCapabilities:
    if _reasoning_family(model_id) is None:
        return BEDROCK_CAPABILITIES
    return ModelCapabilities(
        streaming=True,
        tools=True,
        structured_output=False,
        json_mode=False,
        tool_choice=True,
        parallel_tool_calls=False,
        vision=True,
        files=False,
        audio_input=False,
        audio_output=False,
        embeddings=False,
        reasoning=True,
        web_search=False,
        agent_capabilities=BEDROCK_CAPABILITIES.agent_capabilities,
    )


class BedrockClient(Protocol):
    async def converse(self, payload: dict[str, Any]) -> dict[str, Any]: ...


def _bedrock_realtime_build_audio(frame: AudioFrame, _config: RealtimeSessionConfig) -> list[dict[str, Any]]:
    return [{"type": "audio_input", "audio": encode_audio_frame(frame), "media_type": frame.media_type, "is_final": frame.is_final}]


def _bedrock_realtime_build_text(text: str, _config: RealtimeSessionConfig) -> list[dict[str, Any]]:
    return [{"type": "text_input", "text": text}]


def _bedrock_realtime_build_tool_result(result: ToolExecutionResult, _config: RealtimeSessionConfig) -> list[dict[str, Any]]:
    return [{"type": "tool_result", "tool_call_id": result.tool_call_id, "tool_name": result.tool_name, "output": tool_result_payload(result)}]


def _bedrock_realtime_build_update(config: RealtimeSessionConfig) -> list[dict[str, Any]]:
    return [{"type": "session.start", "config": drop_none({"voice": config.voice, "instructions": config.instructions, **(config.provider_options or {})})}]


def _bedrock_realtime_parse_event(payload: dict[str, Any]) -> list[Any]:
    event_type = str(payload.get("type") or "")
    if event_type == "text_output":
        return [RealtimeTextDeltaEvent(text_delta=str(payload.get("text") or ""), provider_metadata=payload)]
    if event_type == "audio_output":
        chunk = payload.get("audio")
        audio = base64.b64decode(chunk) if isinstance(chunk, str) and chunk else b""
        return [RealtimeAudioOutputEvent(audio=audio, media_type=str(payload.get("media_type") or "audio/pcm"), provider_metadata=payload)]
    if event_type == "transcript":
        return [
            RealtimeTranscriptEvent(
                text=str(payload.get("text") or ""),
                role="user" if payload.get("role") == "user" else "assistant",
                is_final=bool(payload.get("is_final")),
                provider_metadata=payload,
            )
        ]
    if event_type == "tool_call":
        return [
            RealtimeToolCallEvent(
                tool_call=ToolCall(
                    id=str(payload.get("id") or ""),
                    name=str(payload.get("name") or ""),
                    input=payload.get("input") or {},
                )
            )
        ]
    if event_type == "turn.end":
        return [RealtimeResponseCompletedEvent(reason=event_type, provider_metadata=payload)]
    if event_type == "session.ended":
        return [RealtimeSessionEndedEvent(reason=event_type, provider_metadata=payload)]
    return []


def _parse_data_url(value: str) -> tuple[str, bytes]:
    prefix = "data:"
    if not value.startswith(prefix) or ";base64," not in value:
        raise ValidationError("Bedrock image inputs must be provided as data URLs.")
    header, body = value[len(prefix):].split(";base64,", 1)
    return header.lower(), base64.b64decode(body)


def _to_bedrock_image_format(media_type: str) -> str:
    subtype = media_type.split("/", 1)[1].lower() if "/" in media_type else media_type.lower()
    if subtype not in {"png", "jpeg", "gif", "webp"}:
        raise ValidationError(f'Unsupported Bedrock image media type "{media_type}".')
    return subtype


def _cache_control_of(part: Any) -> Any:
    metadata = getattr(part, "provider_metadata", None)
    if isinstance(metadata, dict) and metadata.get("cache_control") is not None:
        return metadata.get("cache_control")
    return getattr(part, "cache_control", None)


def _cache_point_block(cache_control: Any, *, family: _ReasoningFamily | None, location: str) -> dict[str, Any]:
    if not isinstance(cache_control, dict):
        raise ValidationError("Bedrock cache_control must be an object.")
    if family == "nova-2-lite" and location == "tools":
        raise ValidationError("Amazon Nova 2 Lite prompt caching does not support cache checkpoints on tools.")
    ttl = cache_control.get("ttl")
    if ttl in (None, "5m"):
        # Nova 2 Lite rejects an explicit ttl. Omitting it is the 5-minute default.
        return {"cachePoint": {"type": "default"}}
    if ttl == "1h":
        if family == "nova-2-lite":
            raise ValidationError(
                "Amazon Nova 2 Lite prompt caching supports a 5-minute TTL only. Omit ttl; do not send 1h."
            )
        return {"cachePoint": {"type": "default", "ttl": "1h"}}
    raise ValidationError(f'Unsupported Bedrock cache_control ttl "{ttl}".')


def _with_cache_point(blocks: list[dict[str, Any]], cache_control: Any, *, family: _ReasoningFamily | None, location: str) -> list[dict[str, Any]]:
    if cache_control is None or not blocks:
        return blocks
    return [*blocks, _cache_point_block(cache_control, family=family, location=location)]


def _reasoning_provider_data(block: dict[str, Any]) -> dict[str, Any] | None:
    reasoning_text = block.get("reasoningText") if isinstance(block.get("reasoningText"), dict) else None
    text = reasoning_text.get("text") if reasoning_text else None
    signature = reasoning_text.get("signature") if reasoning_text else None
    redacted = _as_base64(block.get("redactedContent"))
    if not isinstance(text, str) and not isinstance(signature, str) and redacted is None:
        return None
    payload: dict[str, Any] = {"type": _BEDROCK_REASONING_TYPE, "replay": True}
    if isinstance(text, str) or isinstance(signature, str):
        reasoning: dict[str, str] = {}
        if isinstance(text, str) and text:
            reasoning["text"] = text
        if isinstance(signature, str) and signature:
            reasoning["signature"] = signature
        if reasoning:
            payload["reasoningText"] = reasoning
    if redacted is not None:
        payload["redactedContent"] = redacted
    return payload


def _replay_reasoning_block(part: Any) -> list[dict[str, Any]]:
    if not isinstance(part, ProviderDataPart) and getattr(part, "type", None) != "provider-data":
        return []
    if getattr(part, "provider", None) != "bedrock" or not isinstance(getattr(part, "data", None), dict):
        return []
    data = part.data
    if data.get("type") != _BEDROCK_REASONING_TYPE or data.get("replay") is not True:
        return []
    reasoning_text = data.get("reasoningText") if isinstance(data.get("reasoningText"), dict) else None
    text = reasoning_text.get("text") if reasoning_text else None
    signature = reasoning_text.get("signature") if reasoning_text else None
    redacted = data.get("redactedContent") if isinstance(data.get("redactedContent"), str) else None
    if not isinstance(text, str) and not isinstance(signature, str) and not redacted:
        return []
    block: dict[str, Any] = {}
    if isinstance(text, str) or isinstance(signature, str):
        mapped: dict[str, str] = {"text": text if isinstance(text, str) else ""}
        if isinstance(signature, str):
            mapped["signature"] = signature
        block["reasoningText"] = mapped
    if redacted:
        try:
            block["redactedContent"] = base64.b64decode(redacted)
        except (ValueError, TypeError) as error:
            raise ValidationError("Bedrock redacted reasoning content must be base64.") from error
    return [{"reasoningContent": block}]


def _as_base64(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, (bytes, bytearray)) and value:
        return base64.b64encode(bytes(value)).decode("ascii")
    return None


def _parse_reasoning_block(block: dict[str, Any]) -> ProviderDataPart | None:
    value = block.get("reasoningContent")
    if not isinstance(value, dict):
        return None
    data = _reasoning_provider_data(value)
    if data is None:
        return None
    return ProviderDataPart(provider="bedrock", data=data)


def _map_message_part(part: Any, model_id: str) -> list[dict[str, Any]]:
    family = _reasoning_family(model_id)
    if getattr(part, "type", None) == "provider-data" or isinstance(part, ProviderDataPart):
        return _replay_reasoning_block(part)
    if part.type == "text":
        blocks = [{"text": part.text}] if part.text else []
        return _with_cache_point(blocks, _cache_control_of(part), family=family, location="messages")
    if part.type == "image":
        media_type, raw = _parse_data_url(part.image)
        blocks = [{"image": {"format": _to_bedrock_image_format(part.media_type or media_type), "source": {"bytes": raw}}}]
        return _with_cache_point(blocks, _cache_control_of(part), family=family, location="messages")
    if isinstance(part, ToolCallPart):
        return [
            {
                "toolUse": {
                    "toolUseId": part.tool_call.id,
                    "name": part.tool_call.name,
                    "input": part.tool_call.input,
                }
            }
        ]
    if isinstance(part, ToolResultPart):
        result = part.tool_result
        content = [{"text": result.error.message}] if result.is_error and result.error is not None else [{"json": result.output}]
        payload: dict[str, Any] = {"toolUseId": result.tool_call_id, "content": content}
        if result.is_error:
            payload["status"] = "error"
        return [{"toolResult": payload}]
    return []


def _system_blocks(messages: list[ModelMessage], model_id: str) -> list[dict[str, Any]] | None:
    family = _reasoning_family(model_id)
    pieces: list[tuple[str, Any]] = []
    for message in messages:
        if message.role != "system":
            continue
        for part in message.parts:
            if part.type != "text" or not part.text:
                continue
            pieces.append((part.text, _cache_control_of(part)))
    if not pieces:
        return None
    if all(cache is None for _, cache in pieces):
        text = "\n".join(text for text, _ in pieces)
        return [{"text": text}] if text else None
    blocks: list[dict[str, Any]] = []
    for text, cache in pieces:
        blocks.extend(_with_cache_point([{"text": text}], cache, family=family, location="system"))
    return blocks


def _map_messages(messages: list[ModelMessage], model_id: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant" if message.role == "assistant" else "user",
            "content": [block for part in message.parts for block in _map_message_part(part, model_id)],
        }
        for message in messages
        if message.role != "system"
    ]


def _map_tools(tools: dict[str, Any] | None, model_id: str) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    family = _reasoning_family(model_id)
    mapped: list[dict[str, Any]] = []
    for definition in tools.values():
        if getattr(definition, "kind", None) == "hosted":
            raise UnsupportedFeatureError('Provider "bedrock" does not support hosted tools in this SDK.')
        mapped.append(
            {
                "toolSpec": {
                    "name": definition.name,
                    "description": definition.description or definition.name,
                    "inputSchema": {"json": create_schema_adapter(definition.schema).json_schema()},
                }
            }
        )
        cache_control = getattr(definition, "cache_control", None)
        if cache_control is not None:
            mapped.append(_cache_point_block(cache_control, family=family, location="tools"))
    return mapped


def _map_tool_choice(tool_choice: str | ToolChoiceName | None) -> dict[str, Any] | None:
    if tool_choice is None or tool_choice == "auto":
        return None
    if tool_choice == "none":
        return {}
    if tool_choice == "required":
        return {"any": {}}
    if isinstance(tool_choice, ToolChoiceName):
        return {"tool": {"name": tool_choice.tool_name}}
    raise ValidationError('Provider "bedrock" tool_choice must be "auto", "none", "required", or ToolChoiceName(...).')


def _map_tool_config(input: ModelGenerateInput, model_id: str) -> dict[str, Any] | None:
    tools = _map_tools(input.tools, model_id)
    tool_choice = _map_tool_choice(input.tool_choice)
    if not tools:
        return None
    return drop_none({"tools": tools, "toolChoice": tool_choice or None})


def _has_tools(input: ModelGenerateInput) -> bool:
    if input.tools:
        return True
    return any(getattr(part, "type", None) in {"tool-call", "tool-result"} for message in input.messages for part in message.parts)


def _forced_tool_choice(input: ModelGenerateInput) -> bool:
    return input.tool_choice == "required" or isinstance(input.tool_choice, ToolChoiceName)


def _reject_sonnet_tool_choice(input: ModelGenerateInput) -> None:
    if _forced_tool_choice(input):
        raise ValidationError('Bedrock Claude Sonnet 5.5 accepts tool_choice "auto" or "none" only.')


def _provider_mapping(input: ModelGenerateInput, key: str) -> dict[str, Any] | None:
    options = input.provider_options or {}
    value = options.get(key)
    return value if isinstance(value, dict) else None


def _reject_sampling(input: ModelGenerateInput, message: str, *, include_max_tokens: bool) -> None:
    found = False
    if input.temperature is not None or (include_max_tokens and input.max_tokens is not None):
        found = True
    inference = _provider_mapping(input, "inferenceConfig") or {}
    if inference.get("temperature") is not None or inference.get("topP") is not None or inference.get("top_p") is not None:
        found = True
    if include_max_tokens and inference.get("maxTokens") is not None:
        found = True
    fields = _provider_mapping(input, "additionalModelRequestFields") or {}
    for key in ("top_k", "topK", "top_p", "topP"):
        if fields.get(key) is not None:
            found = True
    options = input.provider_options or {}
    for key in ("top_p", "topP", "top_k", "topK"):
        if options.get(key) is not None:
            found = True
    if found:
        raise ValidationError(message)


def _union_betas(left: Any, right: Any) -> list[str]:
    values: list[str] = []
    for source in (left, right):
        items = source if isinstance(source, list) else [source]
        for item in items:
            if not isinstance(item, str) or not item:
                raise ValidationError('Bedrock "anthropic_beta" values must be strings.')
            if item not in values:
                values.append(item)
    return values


def _merge_request_fields(generated: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    merged = {**generated, **user}
    if "anthropic_beta" in generated and "anthropic_beta" in user:
        merged["anthropic_beta"] = _union_betas(generated["anthropic_beta"], user["anthropic_beta"])
    return merged


def _sonnet_thinking_fields(input: ModelGenerateInput) -> dict[str, Any]:
    _reject_sonnet_tool_choice(input)
    reasoning = input.reasoning
    if reasoning is None:
        return {}
    if reasoning.budget_tokens is not None:
        raise UnsupportedFeatureError(
            'Provider "bedrock" does not support reasoning.budget_tokens for Claude Sonnet 5.5; use reasoning.effort.'
        )
    effort = reasoning.effort
    if effort is None:
        raise ValidationError('Claude Sonnet 5.5 reasoning requires reasoning.effort.')
    if effort not in _SONNET_EFFORTS:
        raise UnsupportedFeatureError(
            f'Provider "bedrock" does not support reasoning effort "{effort}" for Claude Sonnet 5.5. '
            'Supported values are "low", "medium", "high", "xhigh", and "max".'
        )
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}


def _haiku_thinking_fields(input: ModelGenerateInput) -> dict[str, Any]:
    reasoning = input.reasoning
    if reasoning is None:
        return {}
    if reasoning.effort not in (None, "none"):
        raise ValidationError(
            "Bedrock Claude Haiku 4.5 extended thinking uses reasoning.budget_tokens. "
            "reasoning.effort is not a control for this model."
        )
    if reasoning.effort == "none":
        if reasoning.budget_tokens is not None:
            raise ValidationError(
                'Bedrock Claude Haiku 4.5 reasoning.effort "none" cannot be combined with reasoning.budget_tokens.'
            )
        return {}
    budget = reasoning.budget_tokens
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < _HAIKU_MIN_BUDGET:
        raise ValidationError(
            "Bedrock Claude Haiku 4.5 extended thinking requires reasoning.budget_tokens of at least 1024."
        )
    if input.max_tokens is not None and budget >= input.max_tokens and not _has_tools(input):
        raise ValidationError(
            "Bedrock Claude Haiku 4.5 extended thinking requires reasoning.budget_tokens to be less than max_tokens "
            "unless tools are present for interleaved thinking."
        )
    _reject_sampling(
        input,
        "Bedrock Claude Haiku 4.5 extended thinking cannot be combined with temperature, top_p, or top_k.",
        include_max_tokens=False,
    )
    if _forced_tool_choice(input):
        raise ValidationError(
            'Bedrock Claude Haiku 4.5 extended thinking accepts tool_choice "auto" or "none" only.'
        )
    return {
        "thinking": {"type": "enabled", "budget_tokens": budget},
        "anthropic_beta": [_HAIKU_INTERLEAVED_BETA],
    }


def _nova_reasoning_fields(input: ModelGenerateInput) -> dict[str, Any]:
    reasoning = input.reasoning
    if reasoning is None:
        return {}
    if reasoning.budget_tokens is not None:
        raise UnsupportedFeatureError(
            'Provider "bedrock" does not support reasoning.budget_tokens for Amazon Nova 2 Lite.'
        )
    effort = reasoning.effort
    if effort is None:
        raise ValidationError('Amazon Nova 2 Lite reasoning requires effort "none", "low", "medium", or "high".')
    if effort == "none":
        return {"reasoningConfig": {"type": "disabled"}}
    if effort not in _NOVA_EFFORTS:
        raise UnsupportedFeatureError(
            f'Provider "bedrock" does not support reasoning effort "{effort}" for Amazon Nova 2 Lite.'
        )
    if effort == "high":
        _reject_sampling(
            input,
            'Amazon Nova 2 Lite reasoning effort "high" cannot be combined with temperature, top_p, top_k, or max_tokens.',
            include_max_tokens=True,
        )
    return {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": effort}}


def _reasoning_fields(model_id: str, input: ModelGenerateInput) -> dict[str, Any]:
    family = _reasoning_family(model_id)
    if family == "sonnet-5-5":
        return _sonnet_thinking_fields(input)
    if family == "haiku-4-5":
        return _haiku_thinking_fields(input)
    if family == "nova-2-lite":
        return _nova_reasoning_fields(input)
    return {}


def _usage_from_mapping(usage: dict[str, Any]) -> TokenUsage:
    # cacheReadInputTokens and cacheWriteInputTokens stay out of total_tokens.
    # AWS inputTokens is the non-cached portion; callers sum the three when they need a full input count.
    return TokenUsage(
        input_tokens=usage.get("inputTokens"),
        output_tokens=usage.get("outputTokens"),
        total_tokens=usage.get("totalTokens") or ((usage.get("inputTokens") or 0) + (usage.get("outputTokens") or 0)),
        cached_input_tokens=usage.get("cacheReadInputTokens"),
        cache_write_tokens=usage.get("cacheWriteInputTokens"),
    )


def _parse_tool_use(block: dict[str, Any]) -> ToolCallPart | None:
    value = block.get("toolUse")
    if not isinstance(value, dict):
        return None
    return ToolCallPart(
        tool_call=ToolCall(
            id=str(value.get("toolUseId") or ""),
            name=str(value.get("name") or ""),
            input=value.get("input") or {},
            provider_metadata={"provider": "bedrock", "raw_tool_use": value},
        )
    )


def _request_payload(model_id: str, input: ModelGenerateInput) -> dict[str, Any]:
    provider_options = dict(input.provider_options or {})
    user_fields = provider_options.pop("additionalModelRequestFields", None)
    if user_fields is None:
        user_fields = {}
    elif not isinstance(user_fields, dict):
        raise ValidationError('Bedrock "additionalModelRequestFields" must be an object.')
    merged = _merge_request_fields(_reasoning_fields(model_id, input), user_fields)
    inference_config = drop_none({"temperature": input.temperature, "maxTokens": input.max_tokens}) or None
    payload = drop_none({
        "modelId": model_id,
        "messages": _map_messages(input.messages, model_id),
        "system": _system_blocks(input.messages, model_id),
        "inferenceConfig": inference_config,
        "toolConfig": _map_tool_config(input, model_id),
        **provider_options,
    })
    if merged:
        payload["additionalModelRequestFields"] = drop_none(merged)
    return payload


def _parse_stream_usage(payload: dict[str, Any]) -> TokenUsage | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        metadata = payload.get("metadata")
        usage = metadata.get("usage") if isinstance(metadata, dict) else None
    if not isinstance(usage, dict):
        return None
    return _usage_from_mapping(usage)


def _assistant_parts(content: Any) -> tuple[str, list[Any]]:
    if not isinstance(content, list):
        return "", []
    parts: list[Any] = []
    text_chunks: list[str] = []

    def flush_text() -> None:
        if text_chunks:
            parts.append(TextPart(text="".join(text_chunks)))
            text_chunks.clear()

    for chunk in content:
        if not isinstance(chunk, dict):
            continue
        if chunk.get("text"):
            text_chunks.append(str(chunk.get("text") or ""))
            continue
        flush_text()
        reasoning = _parse_reasoning_block(chunk)
        if reasoning is not None:
            parts.append(reasoning)
        tool_part = _parse_tool_use(chunk)
        if tool_part is not None:
            parts.append(tool_part)
    flush_text()
    text = "".join(part.text for part in parts if isinstance(part, TextPart))
    return text, parts


def _haiku_stream_required(payload: dict[str, Any], input: ModelGenerateInput) -> bool:
    fields = payload.get("additionalModelRequestFields")
    thinking = fields.get("thinking") if isinstance(fields, dict) else None
    enabled = isinstance(thinking, dict) and thinking.get("type") == "enabled"
    return bool(enabled and input.max_tokens is not None and input.max_tokens > _HAIKU_STREAM_ABOVE)


@dataclass(slots=True)
class BedrockLanguageModel(LanguageModel):
    provider: str
    model_id: str
    client: BedrockClient
    capabilities: ModelCapabilities = field(default_factory=lambda: BEDROCK_CAPABILITIES)

    def __post_init__(self) -> None:
        if self.capabilities is BEDROCK_CAPABILITIES:
            resolved = _capabilities_for(self.model_id)
            if resolved is not BEDROCK_CAPABILITIES:
                self.capabilities = resolved

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        if input.reasoning is not None and _reasoning_family(self.model_id) is None:
            raise UnsupportedFeatureError('Provider "bedrock" does not support "reasoning".')
        payload = _request_payload(self.model_id, input)
        if _reasoning_family(self.model_id) == "haiku-4-5" and _haiku_stream_required(payload, input):
            raise ValidationError(
                "Bedrock Claude Haiku 4.5 extended thinking requires streaming when max_tokens is greater than 21333."
            )
        response = await with_retry(
            lambda: self.client.converse(payload),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        content = ((response.get("output") or {}).get("message") or {}).get("content", [])
        text, parts = _assistant_parts(content)
        messages = [ModelMessage(role="assistant", parts=parts)] if parts else []
        raw_usage = response.get("usage")
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        return GenerateResult(
            messages=messages,
            text=text,
            finish_reason=normalize_finish_reason(response.get("stopReason")),
            provider_finish_reason=response.get("stopReason"),
            usage=_usage_from_mapping(usage),
            raw_response=response,
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        if input.reasoning is not None and _reasoning_family(self.model_id) is None:
            raise UnsupportedFeatureError('Provider "bedrock" does not support "reasoning".')
        converse_stream = getattr(self.client, "converse_stream", None)
        if converse_stream is None:
            raise ConfigurationError("Bedrock streaming requires a client with async converse_stream(payload).")
        payload = _request_payload(self.model_id, input)
        stream = await with_retry(
            lambda: converse_stream(payload),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )

        async def generator() -> AsyncIterable[StreamEvent]:
            tool_buffers: dict[int, dict[str, Any]] = {}
            reasoning_buffers: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None
            usage: TokenUsage | None = None
            async for event in stream:
                if "contentBlockStart" in event:
                    start = event["contentBlockStart"]
                    block_index = int(start.get("contentBlockIndex") or start.get("index") or 0)
                    tool_use = (start.get("start") or {}).get("toolUse") or start.get("toolUse")
                    if isinstance(tool_use, dict):
                        tool_buffers[block_index] = {
                            "id": str(tool_use.get("toolUseId") or ""),
                            "name": str(tool_use.get("name") or ""),
                            "input": "",
                        }
                if "contentBlockDelta" in event:
                    delta_event = event["contentBlockDelta"]
                    block_index = int(delta_event.get("contentBlockIndex") or delta_event.get("index") or 0)
                    delta = delta_event.get("delta") or {}
                    if delta.get("text"):
                        yield StreamTextDeltaEvent(text_delta=str(delta.get("text") or ""))
                    reasoning_delta = delta.get("reasoningContent")
                    if isinstance(reasoning_delta, dict):
                        raw_nested = reasoning_delta.get("reasoningText")
                        nested: dict[str, Any] = raw_nested if isinstance(raw_nested, dict) else {}
                        text_delta = reasoning_delta.get("text")
                        if not isinstance(text_delta, str):
                            text_delta = nested.get("text")
                        signature = reasoning_delta.get("signature")
                        if not isinstance(signature, str):
                            signature = nested.get("signature")
                        redacted = _as_base64(reasoning_delta.get("redactedContent") or nested.get("redactedContent"))
                        current = reasoning_buffers.setdefault(block_index, {"text": "", "signature": None, "redacted": None})
                        if isinstance(text_delta, str) and text_delta:
                            current["text"] += text_delta
                            yield StreamProviderDataEvent(
                                provider="bedrock",
                                data={"type": "reasoning_delta", "text": text_delta},
                            )
                        if isinstance(signature, str) and signature:
                            current["signature"] = signature
                        if redacted:
                            current["redacted"] = redacted
                        reasoning_buffers[block_index] = current
                    tool_delta = delta.get("toolUse")
                    if isinstance(tool_delta, dict):
                        current = tool_buffers.setdefault(block_index, {"id": "", "name": "", "input": ""})
                        if tool_delta.get("toolUseId"):
                            current["id"] = str(tool_delta.get("toolUseId") or "")
                        if tool_delta.get("name"):
                            current["name"] = str(tool_delta.get("name") or "")
                        if tool_delta.get("input"):
                            current["input"] += str(tool_delta.get("input") or "")
                if "contentBlockStop" in event:
                    stop = event["contentBlockStop"]
                    block_index = int(stop.get("contentBlockIndex") or stop.get("index") or 0)
                    completed_tool = tool_buffers.pop(block_index) if block_index in tool_buffers else None
                    if completed_tool is not None:
                        try:
                            parsed_input = json.loads(completed_tool["input"]) if completed_tool["input"] else {}
                        except json.JSONDecodeError:
                            parsed_input = completed_tool["input"]
                        yield StreamToolCallEvent(
                            tool_call=ToolCall(
                                id=completed_tool["id"],
                                name=completed_tool["name"],
                                input=parsed_input,
                                provider_metadata={"provider": "bedrock"},
                            )
                        )
                    completed_reasoning = reasoning_buffers.pop(block_index, None)
                    if completed_reasoning and (
                        completed_reasoning.get("text") or completed_reasoning.get("signature") or completed_reasoning.get("redacted")
                    ):
                        reasoning_text: dict[str, str] = {}
                        if completed_reasoning.get("text") or completed_reasoning.get("signature"):
                            reasoning_text["text"] = str(completed_reasoning.get("text") or "")
                            if completed_reasoning.get("signature"):
                                reasoning_text["signature"] = str(completed_reasoning["signature"])
                        data: dict[str, Any] = {"type": _BEDROCK_REASONING_TYPE, "replay": True}
                        if reasoning_text:
                            data["reasoningText"] = reasoning_text
                        if completed_reasoning.get("redacted"):
                            data["redactedContent"] = completed_reasoning["redacted"]
                        yield StreamProviderDataEvent(provider="bedrock", data=data)
                if "messageStop" in event:
                    finish_reason = event["messageStop"].get("stopReason") or finish_reason
                if "metadata" in event:
                    usage = _parse_stream_usage(event)
            yield StreamFinishEvent(
                finish_reason=normalize_finish_reason(finish_reason),
                provider_finish_reason=finish_reason,
                usage=usage,
            )

        return generator()


@dataclass(slots=True)
class BedrockRealtimeModel(RealtimeModel):
    provider: str
    model_id: str
    connection_factory: RealtimeConnectionFactory | None = None
    capabilities: ModelCapabilities = field(default_factory=lambda: BEDROCK_REALTIME_CAPABILITIES)

    async def connect(
        self,
        config: RealtimeSessionConfig | None = None,
        options: RealtimeConnectOptions | None = None,
    ) -> RealtimeSession:
        if self.connection_factory is None:
            raise ConfigurationError("No Bedrock realtime transport configured. Inject a realtime_connection_factory.")
        resolved_config = config or RealtimeSessionConfig()
        url = str((resolved_config.provider_options or {}).get("realtime_url") or "wss://bedrock-runtime.amazonaws.com/realtime")
        headers = dict((resolved_config.provider_options or {}).get("headers") or {})
        connection = await self.connection_factory(url, headers, options)
        session = CallbackRealtimeSession(
            provider=self.provider,
            model_id=self.model_id,
            capabilities=self.capabilities,
            config=resolved_config,
            connection=connection,
            callbacks=RealtimeSessionCallbacks(
                parse_event=_bedrock_realtime_parse_event,
                build_audio_payloads=_bedrock_realtime_build_audio,
                build_text_payloads=_bedrock_realtime_build_text,
                build_tool_result_payloads=_bedrock_realtime_build_tool_result,
                build_update_payloads=_bedrock_realtime_build_update,
                build_initial_payloads=_bedrock_realtime_build_update,
            ),
        )
        await session.initialize()
        return session

    async def create_browser_token(
        self,
        config: RealtimeSessionConfig | None = None,
        options: RealtimeConnectOptions | None = None,
    ) -> RealtimeTokenResult:
        return await unsupported_browser_token(config=config, options=options)


def create_bedrock(
    *,
    client: BedrockClient | None = None,
    region: str | None = None,
    realtime_connection_factory: RealtimeConnectionFactory | None = None,
):
    resolved_client = client
    if resolved_client is None:
        resolved_region = region or os.getenv("AWS_REGION")
        if not resolved_region:
            raise ConfigurationError("Missing AWS region for Bedrock.")

        class MissingSDKClient:
            async def converse(self, payload: dict[str, Any]) -> dict[str, Any]:
                raise ConfigurationError(
                    "No Bedrock client provided. Inject a client with an async converse(payload) method."
                )

        resolved_client = MissingSDKClient()
    native = ProviderAdapter(
        name="bedrock",
        language_model_factory=lambda model_id: BedrockLanguageModel(provider="bedrock", model_id=model_id, client=resolved_client),
        realtime_model_factory=lambda model_id: BedrockRealtimeModel(
            provider="bedrock",
            model_id=model_id,
            connection_factory=realtime_connection_factory,
        ),
    )
    return create_provider_bundle(
        name="bedrock",
        native=native,
        agent_capabilities=BEDROCK_CAPABILITIES.agent_capabilities or AgentCapabilities(),
        portable_support=PortableSupport(
            text_generation=True,
            streaming=False,
            structured_output=False,
            tools=False,
            embeddings=False,
            grounding=False,
            retrieval=True,
            transcription=False,
            speech=False,
            portable_badge=False,
            tier="native-only",
        ),
    )
