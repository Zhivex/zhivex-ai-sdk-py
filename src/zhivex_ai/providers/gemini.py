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
    EmbedResult,
    EmbeddingModel,
    GenerateResult,
    GroundedGenerateResult,
    GroundedLanguageModel,
    GroundedModelGenerateInput,
    GroundingSource,
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
    ToolChoiceName,
    ToolCallPart,
)
from .base import ProviderAdapter
from ._payload import drop_none

GEMINI_CAPABILITIES = ModelCapabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    parallel_tool_calls=False,
    vision=True,
    files=False,
    audio_input=False,
    audio_output=False,
    embeddings=True,
    reasoning=True,
    web_search=False,
)

GEMINI_GROUNDED_CAPABILITIES = ModelCapabilities(
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
    reasoning=True,
    web_search=True,
)

GOOGLE_SEARCH_PROVIDER_OPTION = "google_search"


def _system_instruction(messages: list[ModelMessage]) -> dict[str, Any] | None:
    text = "\n".join(
        part.text for message in messages if message.role == "system" for part in message.parts if part.type == "text"
    )
    return {"parts": [{"text": text}]} if text else None


def _map_part(part: Any) -> dict[str, Any]:
    if part.type == "text":
        return {"text": part.text}
    if part.type == "image":
        data = part.image
        media_type = part.media_type or "image/jpeg"
        if data.startswith("data:") and ";base64," in data:
            header, body = data[len("data:"):].split(";base64,", 1)
            media_type = part.media_type or header.lower()
            data = body
        return {"inlineData": {"mimeType": media_type, "data": data}}
    if part.type == "tool-call":
        return {"functionCall": {"name": part.tool_call.name, "args": part.tool_call.input}}
    if part.type == "tool-result":
        return {
            "functionResponse": {
                "name": part.tool_result.tool_name,
                "response": {
                    "name": part.tool_result.tool_name,
                    "content": part.tool_result.error.__dict__ if part.tool_result.is_error else part.tool_result.output,
                },
            }
        }
    return {"text": json.dumps(str(part))}


def _map_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    return [
        {
            "role": "model" if message.role == "assistant" else "user",
            "parts": [_map_part(part) for part in message.parts],
        }
        for message in messages
        if message.role != "system"
    ]


def _google_search_enabled(provider_options: dict[str, Any] | None) -> bool:
    return bool((provider_options or {}).get(GOOGLE_SEARCH_PROVIDER_OPTION))


def _google_search_tool() -> dict[str, Any]:
    return {"googleSearch": {}}


def _map_tools(tools: dict[str, Any] | None, provider_options: dict[str, Any] | None = None) -> list[dict[str, Any]] | None:
    mapped: list[dict[str, Any]] = []
    if tools:
        mapped.append(
            {
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": create_schema_adapter(tool.schema).json_schema(),
                    }
                    for tool in tools.values()
                ]
            }
        )
    if _google_search_enabled(provider_options):
        mapped.append(_google_search_tool())
    return mapped or None


def _map_tool_config(tools: dict[str, Any] | None, tool_choice: str | ToolChoiceName | None) -> dict[str, Any] | None:
    if not tools or tool_choice is None or tool_choice == "auto":
        return None
    if tool_choice == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if tool_choice == "required":
        return {"functionCallingConfig": {"mode": "ANY"}}
    return {
        "functionCallingConfig": {
            "mode": "ANY",
            "allowedFunctionNames": [tool_choice.tool_name],
        }
    }


def _provider_options_without_google_search(provider_options: dict[str, Any] | None) -> dict[str, Any] | None:
    if not provider_options:
        return None
    remaining = {key: value for key, value in provider_options.items() if key != GOOGLE_SEARCH_PROVIDER_OPTION}
    return remaining or None


def _is_gemini_3_model(model_id: str) -> bool:
    return model_id.startswith("gemini-3")


def _is_gemini_3_pro_model(model_id: str) -> bool:
    return _is_gemini_3_model(model_id) and "pro" in model_id


def _map_reasoning(model_id: str, input: ModelGenerateInput) -> dict[str, Any] | None:
    if input.reasoning is None:
        return None
    if _is_gemini_3_model(model_id):
        if input.reasoning.budget_tokens is not None:
            raise UnsupportedFeatureError(
                'Provider "gemini" uses "reasoning.effort" for Gemini 3 models and does not support "reasoning.budgetTokens".'
            )
        if input.reasoning.effort == "none":
            raise UnsupportedFeatureError('Provider "gemini" does not support "reasoning.effort=none" for Gemini 3 models.')
        if input.reasoning.effort == "xhigh":
            raise UnsupportedFeatureError('Provider "gemini" does not support "reasoning.effort=xhigh".')
        if input.reasoning.effort == "minimal" and _is_gemini_3_pro_model(model_id):
            raise UnsupportedFeatureError(
                'Provider "gemini" does not support "reasoning.effort=minimal" for Gemini 3 Pro models.'
            )
        return {"thinkingLevel": input.reasoning.effort} if input.reasoning.effort is not None else None
    if input.reasoning.effort is not None:
        raise UnsupportedFeatureError(
            'Provider "gemini" does not support "reasoning.effort" for models earlier than Gemini 3.'
        )
    return {"thinkingBudget": input.reasoning.budget_tokens} if input.reasoning.budget_tokens is not None else None


def _generation_config(model_id: str, input: ModelGenerateInput) -> dict[str, Any]:
    config: dict[str, Any] = {"temperature": input.temperature, "maxOutputTokens": input.max_tokens}
    if input.reasoning is not None:
        config["thinkingConfig"] = _map_reasoning(model_id, input)
    if input.structured_output is not None and input.structured_output.mode == "native":
        config["responseMimeType"] = "application/json"
        config["responseJsonSchema"] = create_schema_adapter(input.structured_output.schema).json_schema()
    return drop_none(config)


def _parse_assistant_message(candidate: dict[str, Any] | None) -> ModelMessage:
    parts = []
    for part in ((candidate or {}).get("content") or {}).get("parts", []):
        if part.get("text"):
            parts.append(TextPart(text=part["text"]))
        elif part.get("functionCall"):
            call = part["functionCall"]
            parts.append(ToolCallPart(tool_call=ToolCall(id=f'{call["name"]}-0', name=call["name"], input=call.get("args") or {})))
    return ModelMessage(role="assistant", parts=parts)


def _extract_grounding_sources(payload: dict[str, Any]) -> list[GroundingSource]:
    candidate = (payload.get("candidates") or [None])[0] or {}
    grounding_metadata = candidate.get("groundingMetadata") or {}
    sources: list[GroundingSource] = []
    seen: set[str] = set()
    for chunk in grounding_metadata.get("groundingChunks") or []:
        web = chunk.get("web")
        if not isinstance(web, dict):
            continue
        url = web.get("uri")
        if not isinstance(url, str) or not url or url in seen:
            continue
        seen.add(url)
        sources.append(
            GroundingSource(
                url=url,
                title=web.get("title"),
                snippet=web.get("text"),
                provider_metadata=web,
            )
        )
    return sources


@dataclass(slots=True)
class GeminiLanguageModel(LanguageModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: GEMINI_CAPABILITIES)

    def _url(self, action: str) -> str:
        separator = "&" if "?" in action else "?"
        return f"{self.base_url}/models/{self.model_id}:{action}{separator}key={self.api_key}"

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        generation_config = _generation_config(self.model_id, input) or None
        response = await with_retry(
            lambda: self.fetch(
                self._url("generateContent"),
                headers={"content-type": "application/json"},
                json_body=drop_none({
                    "contents": _map_messages(input.messages),
                    "systemInstruction": _system_instruction(input.messages),
                    "tools": _map_tools(input.tools, input.provider_options),
                    "toolConfig": _map_tool_config(input.tools, input.tool_choice),
                    **(_provider_options_without_google_search(input.provider_options) or {}),
                    "generationConfig": generation_config,
                }),
                timeout_ms=input.timeout_ms,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(f"Gemini request failed with status {response.status_code}.", response.status_code, response_body=await response.text())
        payload = await response.json()
        candidate = (payload.get("candidates") or [None])[0]
        assistant_message = _parse_assistant_message(candidate)
        usage = payload.get("usageMetadata") or {}
        return GenerateResult(
            messages=[assistant_message],
            text="".join(part.text for part in assistant_message.parts if part.type == "text"),
            finish_reason=normalize_finish_reason(candidate.get("finishReason") if candidate else None),
            provider_finish_reason=candidate.get("finishReason") if candidate else None,
            usage=TokenUsage(
                input_tokens=usage.get("promptTokenCount"),
                output_tokens=usage.get("candidatesTokenCount"),
                total_tokens=usage.get("totalTokenCount")
                or ((usage.get("promptTokenCount") or 0) + (usage.get("candidatesTokenCount") or 0)),
            )
            if usage
            else None,
            raw_response=payload,
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        generation_config = _generation_config(self.model_id, input) or None
        response = await with_retry(
            lambda: self.fetch(
                self._url("streamGenerateContent?alt=sse"),
                headers={"content-type": "application/json"},
                json_body=drop_none({
                    "contents": _map_messages(input.messages),
                    "systemInstruction": _system_instruction(input.messages),
                    "tools": _map_tools(input.tools, input.provider_options),
                    "toolConfig": _map_tool_config(input.tools, input.tool_choice),
                    **(_provider_options_without_google_search(input.provider_options) or {}),
                    "generationConfig": generation_config,
                }),
                timeout_ms=input.timeout_ms,
                stream=True,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(f"Gemini request failed with status {response.status_code}.", response.status_code, response_body=await response.text())

        async def generator() -> AsyncIterable[StreamEvent]:
            async for event in parse_sse(response.iter_lines()):
                payload = json.loads(event.data)
                candidate = (payload.get("candidates") or [None])[0]
                parts = ((candidate or {}).get("content") or {}).get("parts", [])
                for part in parts:
                    if part.get("text"):
                        yield StreamTextDeltaEvent(text_delta=part["text"])
                    if part.get("functionCall"):
                        call = part["functionCall"]
                        yield StreamToolCallEvent(
                            tool_call=ToolCall(id=f'{call["name"]}-0', name=call["name"], input=call.get("args") or {})
                        )
                if candidate and candidate.get("finishReason"):
                    yield StreamFinishEvent(
                        finish_reason=normalize_finish_reason(candidate["finishReason"]),
                        provider_finish_reason=candidate["finishReason"],
                    )

        return generator()


@dataclass(slots=True)
class GeminiGroundedLanguageModel(GroundedLanguageModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: GEMINI_GROUNDED_CAPABILITIES)

    def _url(self, action: str) -> str:
        separator = "&" if "?" in action else "?"
        return f"{self.base_url}/models/{self.model_id}:{action}{separator}key={self.api_key}"

    async def generate(self, input: GroundedModelGenerateInput) -> GroundedGenerateResult:
        response = await with_retry(
            lambda: self.fetch(
                self._url("generateContent"),
                headers={"content-type": "application/json"},
                json_body=drop_none({
                    "contents": _map_messages(input.messages),
                    "systemInstruction": _system_instruction(input.messages),
                    "tools": [_google_search_tool()],
                    **(input.provider_options or {}),
                    "generationConfig": drop_none({
                        "temperature": input.temperature,
                        "maxOutputTokens": input.max_tokens,
                        "thinkingConfig": _map_reasoning(
                            self.model_id,
                            ModelGenerateInput(messages=input.messages, reasoning=input.reasoning),
                        )
                        if input.reasoning is not None
                        else None,
                    }),
                }),
                timeout_ms=input.timeout_ms,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(f"Gemini request failed with status {response.status_code}.", response.status_code, response_body=await response.text())
        payload = await response.json()
        candidate = (payload.get("candidates") or [None])[0]
        assistant_message = _parse_assistant_message(candidate)
        usage = payload.get("usageMetadata") or {}
        return GroundedGenerateResult(
            messages=[assistant_message],
            text="".join(part.text for part in assistant_message.parts if part.type == "text"),
            finish_reason=normalize_finish_reason(candidate.get("finishReason") if candidate else None),
            provider_finish_reason=candidate.get("finishReason") if candidate else None,
            usage=TokenUsage(
                input_tokens=usage.get("promptTokenCount"),
                output_tokens=usage.get("candidatesTokenCount"),
                total_tokens=usage.get("totalTokenCount")
                or ((usage.get("promptTokenCount") or 0) + (usage.get("candidatesTokenCount") or 0)),
            )
            if usage
            else None,
            raw_response=payload,
            sources=_extract_grounding_sources(payload),
        )


@dataclass(slots=True)
class GeminiEmbeddingModel(EmbeddingModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: GEMINI_CAPABILITIES)

    async def embed(self, values: list[str], options: Any = None) -> EmbedResult:
        embeddings: list[list[float]] = []
        for value in values:
            response = await with_retry(
                lambda value=value: self.fetch(
                    f"{self.base_url}/models/{self.model_id}:embedContent?key={self.api_key}",
                    headers={"content-type": "application/json"},
                    json_body={"content": {"parts": [{"text": value}]}},
                    timeout_ms=getattr(options, "timeout_ms", None),
                ),
                max_retries=getattr(options, "max_retries", 0) or 0,
                retry_backoff_ms=getattr(options, "retry_backoff_ms", 250) or 250,
            )
            if response.status_code >= 400:
                raise ProviderHTTPError(f"Gemini request failed with status {response.status_code}.", response.status_code, response_body=await response.text())
            payload = await response.json()
            embeddings.append(payload["embedding"]["values"])
        return EmbedResult(embeddings=embeddings)


def create_gemini(
    *,
    api_key: str | None = None,
    base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    fetch: Fetcher | None = None,
):
    resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
    if not resolved_key:
        raise ConfigurationError("Missing Gemini API key.")
    requester = fetch or default_fetch
    base = base_url.rstrip("/")
    return ProviderAdapter(
        name="gemini",
        language_model_factory=lambda model_id: GeminiLanguageModel(
            provider="gemini", model_id=model_id, api_key=resolved_key, base_url=base, fetch=requester
        ),
        embedding_model_factory=lambda model_id: GeminiEmbeddingModel(
            provider="gemini", model_id=model_id, api_key=resolved_key, base_url=base, fetch=requester
        ),
        grounded_language_model_factory=lambda model_id: GeminiGroundedLanguageModel(
            provider="gemini", model_id=model_id, api_key=resolved_key, base_url=base, fetch=requester
        ),
    )
