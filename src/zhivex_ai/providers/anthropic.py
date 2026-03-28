from __future__ import annotations

import json
import os
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any

from .._http import Fetcher, default_fetch
from .._sse import parse_sse
from ..errors import ConfigurationError, ProviderHTTPError, UnsupportedFeatureError
from ..messages import normalize_finish_reason
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..types import (
    GenerateResult,
    LanguageModel,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    StreamEvent,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    TextPart,
    TokenUsage,
    ToolCall,
    ToolCallPart,
)
from .base import ProviderAdapter

ANTHROPIC_CAPABILITIES = ModelCapabilities(
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
)


def _map_block_parts(message: ModelMessage) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in message.parts:
        if part.type == "text":
            blocks.append({"type": "text", "text": part.text})
        elif part.type == "image":
            blocks.append({"type": "image", "source": {"type": "url", "url": part.image}})
        elif part.type == "tool-call":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": part.tool_call.id,
                    "name": part.tool_call.name,
                    "input": part.tool_call.input,
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
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": create_schema_adapter(tool.schema).json_schema(),
        }
        for tool in tools.values()
    ]


def _map_reasoning(input: ModelGenerateInput) -> dict[str, Any] | None:
    if input.reasoning is None:
        return None
    if input.reasoning.effort is not None:
        raise UnsupportedFeatureError('Provider "anthropic" does not support "reasoning.effort".')
    if input.reasoning.budget_tokens is None:
        return None
    return {"type": "enabled", "budget_tokens": input.reasoning.budget_tokens}


def _parse_assistant_message(payload: dict[str, Any]) -> ModelMessage:
    parts = []
    for block in payload.get("content") or []:
        if block.get("type") == "text":
            parts.append(TextPart(text=block["text"]))
        elif block.get("type") == "tool_use":
            parts.append(
                ToolCallPart(
                    tool_call=ToolCall(id=block["id"], name=block["name"], input=block.get("input") or {})
                )
            )
    return ModelMessage(role="assistant", parts=parts)


@dataclass(slots=True)
class AnthropicLanguageModel(LanguageModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    anthropic_version: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: ANTHROPIC_CAPABILITIES)

    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
        }

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        body = {
            "model": self.model_id,
            "system": _system_prompt_from_messages(input.messages),
            "messages": _map_messages(input.messages),
            "tools": _map_tools(input.tools),
            "temperature": input.temperature,
            "max_tokens": input.max_tokens or 1024,
            **(input.provider_options or {}),
            "thinking": _map_reasoning(input),
        }
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/messages",
                headers=self._headers(),
                json_body=body,
                timeout_ms=input.timeout_ms,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Anthropic request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        assistant_message = _parse_assistant_message(payload)
        usage = payload.get("usage") or {}
        return GenerateResult(
            messages=[assistant_message],
            text="".join(part.text for part in assistant_message.parts if part.type == "text"),
            finish_reason=normalize_finish_reason(payload.get("stop_reason")),
            provider_finish_reason=payload.get("stop_reason"),
            usage=TokenUsage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=(usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
            ),
            raw_response=payload,
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/messages",
                headers=self._headers(),
                json_body={
                    "model": self.model_id,
                    "system": _system_prompt_from_messages(input.messages),
                    "messages": _map_messages(input.messages),
                    "tools": _map_tools(input.tools),
                    "temperature": input.temperature,
                    "max_tokens": input.max_tokens or 1024,
                    "stream": True,
                    **(input.provider_options or {}),
                },
                timeout_ms=input.timeout_ms,
                stream=True,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Anthropic request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )

        async def generator() -> AsyncIterable[StreamEvent]:
            tool_buffers: dict[int, dict[str, str]] = {}
            async for event in parse_sse(response.iter_lines()):
                payload = json.loads(event.data)
                if event.event == "content_block_delta" and payload.get("delta", {}).get("type") == "text_delta":
                    yield StreamTextDeltaEvent(text_delta=payload["delta"]["text"])
                elif event.event == "content_block_start" and payload.get("content_block", {}).get("type") == "tool_use":
                    block = payload["content_block"]
                    tool_buffers[payload["index"]] = {"id": block["id"], "name": block["name"], "input": ""}
                elif event.event == "content_block_delta" and payload.get("delta", {}).get("type") == "input_json_delta":
                    current = tool_buffers.get(payload["index"])
                    if current is not None:
                        current["input"] += payload["delta"]["partial_json"]
                elif event.event == "content_block_stop":
                    current = tool_buffers.get(payload["index"])
                    if current is not None:
                        yield StreamToolCallEvent(
                            tool_call=ToolCall(
                                id=current["id"],
                                name=current["name"],
                                input=json.loads(current["input"] or "{}"),
                            )
                        )
                elif event.event == "message_stop":
                    yield StreamFinishEvent(
                        finish_reason=normalize_finish_reason(payload.get("stop_reason")),
                        provider_finish_reason=payload.get("stop_reason"),
                    )

        return generator()


def create_anthropic(
    *,
    api_key: str | None = None,
    base_url: str = "https://api.anthropic.com/v1",
    anthropic_version: str = "2023-06-01",
    fetch: Fetcher | None = None,
):
    resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise ConfigurationError("Missing Anthropic API key.")
    requester = fetch or default_fetch
    return ProviderAdapter(
        name="anthropic",
        language_model_factory=lambda model_id: AnthropicLanguageModel(
            provider="anthropic",
            model_id=model_id,
            api_key=resolved_key,
            base_url=base_url.rstrip("/"),
            anthropic_version=anthropic_version,
            fetch=requester,
        ),
    )
