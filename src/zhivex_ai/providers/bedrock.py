from __future__ import annotations

import base64
import json
import os
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any, Protocol

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
    StreamTextDeltaEvent,
    StreamToolCallEvent,
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


def _map_message_part(part: Any) -> list[dict[str, Any]]:
    if part.type == "text":
        return [{"text": part.text}] if part.text else []
    if part.type == "image":
        media_type, raw = _parse_data_url(part.image)
        return [{"image": {"format": _to_bedrock_image_format(part.media_type or media_type), "source": {"bytes": raw}}}]
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


def _system_blocks(messages: list[ModelMessage]) -> list[dict[str, Any]] | None:
    text = "\n".join(part.text for message in messages if message.role == "system" for part in message.parts if part.type == "text")
    return [{"text": text}] if text else None


def _map_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant" if message.role == "assistant" else "user",
            "content": [block for part in message.parts for block in _map_message_part(part)],
        }
        for message in messages
        if message.role != "system"
    ]


def _map_tools(tools: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
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


def _map_tool_config(input: ModelGenerateInput) -> dict[str, Any] | None:
    tools = _map_tools(input.tools)
    tool_choice = _map_tool_choice(input.tool_choice)
    if not tools:
        return None
    return drop_none({"tools": tools, "toolChoice": tool_choice or None})


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
    inference_config = drop_none({"temperature": input.temperature, "maxTokens": input.max_tokens}) or None
    return drop_none({
        "modelId": model_id,
        "messages": _map_messages(input.messages),
        "system": _system_blocks(input.messages),
        "inferenceConfig": inference_config,
        "toolConfig": _map_tool_config(input),
        **(input.provider_options or {}),
    })


def _parse_stream_usage(payload: dict[str, Any]) -> TokenUsage | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        metadata = payload.get("metadata")
        usage = metadata.get("usage") if isinstance(metadata, dict) else None
    if not isinstance(usage, dict):
        return None
    return TokenUsage(
        input_tokens=usage.get("inputTokens"),
        output_tokens=usage.get("outputTokens"),
        total_tokens=usage.get("totalTokens") or ((usage.get("inputTokens") or 0) + (usage.get("outputTokens") or 0)),
    )


@dataclass(slots=True)
class BedrockLanguageModel(LanguageModel):
    provider: str
    model_id: str
    client: BedrockClient
    capabilities: ModelCapabilities = field(default_factory=lambda: BEDROCK_CAPABILITIES)

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        if input.reasoning is not None:
            raise UnsupportedFeatureError('Provider "bedrock" does not support "reasoning".')
        payload = _request_payload(self.model_id, input)
        response = await with_retry(
            lambda: self.client.converse(payload),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        content = ((response.get("output") or {}).get("message") or {}).get("content", [])
        text = "".join(chunk.get("text", "") for chunk in content)
        parts: list[Any] = [TextPart(text=text)] if text else []
        parts.extend(part for chunk in content if (part := _parse_tool_use(chunk)) is not None)
        messages = [ModelMessage(role="assistant", parts=parts)] if parts else []
        usage = response.get("usage") or {}
        return GenerateResult(
            messages=messages,
            text=text,
            finish_reason=normalize_finish_reason(response.get("stopReason")),
            provider_finish_reason=response.get("stopReason"),
            usage=TokenUsage(
                input_tokens=usage.get("inputTokens"),
                output_tokens=usage.get("outputTokens"),
                total_tokens=usage.get("totalTokens") or ((usage.get("inputTokens") or 0) + (usage.get("outputTokens") or 0)),
            ),
            raw_response=response,
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        if input.reasoning is not None:
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
                    current = tool_buffers.pop(block_index, None)
                    if current is not None:
                        try:
                            parsed_input = json.loads(current["input"]) if current["input"] else {}
                        except json.JSONDecodeError:
                            parsed_input = current["input"]
                        yield StreamToolCallEvent(
                            tool_call=ToolCall(
                                id=current["id"],
                                name=current["name"],
                                input=parsed_input,
                                provider_metadata={"provider": "bedrock"},
                            )
                        )
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
