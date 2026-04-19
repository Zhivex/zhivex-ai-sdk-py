from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..errors import ConfigurationError, UnsupportedFeatureError, ValidationError
from ..messages import normalize_finish_reason
from ..realtime import CallbackRealtimeSession, RealtimeConnectionFactory, RealtimeSessionCallbacks, encode_audio_frame, tool_result_payload, unsupported_browser_token
from ..runtime import with_retry
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
    TextPart,
    TokenUsage,
    ToolCall,
    ToolExecutionResult,
    PortableSupport,
)
from .base import ProviderAdapter, create_provider_bundle
from ._payload import drop_none

BEDROCK_CAPABILITIES = ModelCapabilities(
    streaming=False,
    tools=False,
    structured_output=False,
    json_mode=False,
    tool_choice=False,
    parallel_tool_calls=False,
    vision=True,
    files=False,
    audio_input=False,
    audio_output=False,
    embeddings=False,
    reasoning=False,
    web_search=False,
    agent_capabilities=AgentCapabilities(
        support_tier="tier-c",
        tool_choice_none=False,
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
    return []


def _system_blocks(messages: list[ModelMessage]) -> list[dict[str, Any]] | None:
    text = "\n".join(part.text for message in messages if message.role == "system" for part in message.parts if part.type == "text")
    return [{"text": text}] if text else None


def _map_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    return [
        {"role": "assistant" if message.role == "assistant" else "user", "content": [block for part in message.parts for block in _map_message_part(part)]}
        for message in messages
        if message.role != "system"
    ]


@dataclass(slots=True)
class BedrockLanguageModel(LanguageModel):
    provider: str
    model_id: str
    client: BedrockClient
    capabilities: ModelCapabilities = field(default_factory=lambda: BEDROCK_CAPABILITIES)

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        if input.reasoning is not None:
            raise UnsupportedFeatureError('Provider "bedrock" does not support "reasoning".')
        inference_config = drop_none({"temperature": input.temperature, "maxTokens": input.max_tokens}) or None
        payload = drop_none({
            "modelId": self.model_id,
            "messages": _map_messages(input.messages),
            "system": _system_blocks(input.messages),
            "inferenceConfig": inference_config,
            **(input.provider_options or {}),
        })
        response = await with_retry(
            lambda: self.client.converse(payload),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        text = "".join(chunk.get("text", "") for chunk in ((response.get("output") or {}).get("message") or {}).get("content", []))
        messages = [ModelMessage(role="assistant", parts=[TextPart(text=text)])] if text else []
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
