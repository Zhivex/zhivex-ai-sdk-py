from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..errors import ConfigurationError, UnsupportedFeatureError, ValidationError
from ..messages import normalize_finish_reason
from ..runtime import with_retry
from ..types import GenerateResult, LanguageModel, ModelCapabilities, ModelGenerateInput, ModelMessage, TextPart, TokenUsage
from .base import ProviderAdapter
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
)


class BedrockClient(Protocol):
    async def converse(self, payload: dict[str, Any]) -> dict[str, Any]: ...


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


def create_bedrock(*, client: BedrockClient | None = None, region: str | None = None):
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
    return ProviderAdapter(
        name="bedrock",
        language_model_factory=lambda model_id: BedrockLanguageModel(provider="bedrock", model_id=model_id, client=resolved_client),
    )
