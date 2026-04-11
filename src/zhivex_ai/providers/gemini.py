from __future__ import annotations

import base64
import json
from copy import deepcopy
import os
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

from .._http import Fetcher, default_fetch
from .._sse import parse_sse
from ..errors import ConfigurationError, ProviderHTTPError, UnsupportedFeatureError
from ..messages import normalize_finish_reason
from ..realtime import (
    CallbackRealtimeSession,
    RealtimeConnectionFactory,
    RealtimeSessionCallbacks,
    encode_audio_frame,
    open_websocket_connection,
    tool_result_payload,
    unsupported_browser_token,
)
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..types import (
    AudioFrame,
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
    RealtimeAudioOutputEvent,
    RealtimeConnectOptions,
    RealtimeModel,
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
    ToolChoiceName,
    ToolCallPart,
    ToolExecutionResult,
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

GEMINI_REALTIME_CAPABILITIES = ModelCapabilities(
    streaming=False,
    tools=False,
    structured_output=False,
    json_mode=False,
    tool_choice=False,
    parallel_tool_calls=False,
    vision=True,
    files=False,
    audio_input=True,
    audio_output=True,
    embeddings=False,
    reasoning=True,
    web_search=False,
    realtime=True,
    realtime_audio_input=True,
    realtime_audio_output=True,
    realtime_tools=True,
    realtime_browser_tokens=False,
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
        function_call = {"name": part.tool_call.name, "args": part.tool_call.input}
        thought_signature = part.tool_call.provider_metadata.get("thought_signature")
        payload = {"functionCall": function_call}
        if thought_signature is not None:
            payload["thought_signature"] = thought_signature
        return payload
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


_GEMINI_SUPPORTED_SCHEMA_KEYS = {
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "properties",
    "required",
    "items",
    "anyOf",
    "oneOf",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
}


def _normalize_gemini_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    def visit(node: Any) -> Any:
        if isinstance(node, list):
            return [visit(item) for item in node]
        if not isinstance(node, dict):
            return node

        normalized: dict[str, Any] = {}
        for key, value in node.items():
            if key not in _GEMINI_SUPPORTED_SCHEMA_KEYS:
                continue
            if key == "properties" and isinstance(value, dict):
                normalized[key] = {str(name): visit(property_schema) for name, property_schema in value.items()}
                continue
            normalized[key] = visit(value)
        return normalized

    return visit(deepcopy(schema))


def _map_tools(tools: dict[str, Any] | None, provider_options: dict[str, Any] | None = None) -> list[dict[str, Any]] | None:
    mapped: list[dict[str, Any]] = []
    if tools:
        mapped.append(
            {
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": (
                            _normalize_gemini_tool_schema(create_schema_adapter(tool.schema).json_schema())
                            if getattr(tool, "source", None) == "mcp"
                            else create_schema_adapter(tool.schema).json_schema()
                        ),
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


def _gemini_realtime_url(base_url: str, api_key: str, provider_options: dict[str, Any] | None = None) -> str:
    override = (provider_options or {}).get("realtime_url")
    if isinstance(override, str) and override:
        return override
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = dict((provider_options or {}).get("realtime_query") or {})
    query.setdefault("key", api_key)
    return urlunparse((scheme, parsed.netloc, "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent", "", urlencode(query), ""))


def _gemini_realtime_headers(provider_options: dict[str, Any] | None = None) -> dict[str, str]:
    return dict((provider_options or {}).get("headers") or {})


def _gemini_realtime_tools(config: RealtimeSessionConfig) -> list[dict[str, Any]] | None:
    return _map_tools(config.tools, config.provider_options)


def _gemini_realtime_setup(config: RealtimeSessionConfig, model_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": f"models/{model_id}",
        "generation_config": drop_none({
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": config.voice}}} if config.voice else None,
            "response_modalities": ["AUDIO"] if config.output_audio_media_type else ["TEXT"],
        }),
        "tools": _gemini_realtime_tools(config),
        "system_instruction": {"parts": [{"text": config.instructions}]} if config.instructions else None,
        **(_provider_options_without_google_search(config.provider_options) or {}),
    }
    return {"setup": drop_none(payload)}


def _gemini_realtime_build_audio(frame: AudioFrame, _config: RealtimeSessionConfig) -> list[dict[str, Any]]:
    return [
        {
            "realtime_input": {
                "media_chunks": [
                    {
                        "mime_type": frame.media_type,
                        "data": encode_audio_frame(frame),
                    }
                ]
            }
        }
    ]


def _gemini_realtime_build_text(text: str, _config: RealtimeSessionConfig) -> list[dict[str, Any]]:
    return [
        {
            "client_content": {
                "turns": [{"role": "user", "parts": [{"text": text}]}],
                "turn_complete": True,
            }
        }
    ]


def _gemini_realtime_build_tool_result(result: ToolExecutionResult, _config: RealtimeSessionConfig) -> list[dict[str, Any]]:
    return [
        {
            "tool_response": {
                "function_responses": [
                    {
                        "id": result.tool_call_id,
                        "name": result.tool_name,
                        "response": tool_result_payload(result),
                    }
                ]
            }
        }
    ]


def _gemini_realtime_build_update(config: RealtimeSessionConfig, model_id: str) -> list[dict[str, Any]]:
    return [_gemini_realtime_setup(config, model_id)]


def _gemini_realtime_parse_event(payload: dict[str, Any]) -> list[Any]:
    if "setupComplete" in payload:
        return []
    if isinstance(payload.get("server_content"), dict):
        content = payload["server_content"]
        model_turn = dict(content.get("model_turn") or {})
        parts = list(model_turn.get("parts") or [])
        events: list[Any] = []
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                events.append(
                    RealtimeTextDeltaEvent(
                        text_delta=str(part.get("text") or ""),
                        provider_metadata=payload,
                    )
                )
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and inline.get("data"):
                audio = base64.b64decode(str(inline.get("data")))
                events.append(
                    RealtimeAudioOutputEvent(
                        audio=audio,
                        media_type=str(inline.get("mimeType") or inline.get("mime_type") or "audio/pcm"),
                        provider_metadata=payload,
                    )
                )
            if isinstance(part, dict) and part.get("functionCall"):
                call = dict(part.get("functionCall") or {})
                events.append(
                    RealtimeToolCallEvent(
                        tool_call=ToolCall(
                            id=str(call.get("id") or f'{call.get("name", "")}-0'),
                            name=str(call.get("name") or ""),
                            input=call.get("args") or {},
                        )
                    )
                )
        input_transcription = content.get("input_transcription")
        if isinstance(input_transcription, dict) and input_transcription.get("text"):
            events.append(
                RealtimeTranscriptEvent(
                    text=str(input_transcription.get("text") or ""),
                    role="user",
                    is_final=bool(content.get("turn_complete")),
                    provider_metadata=payload,
                )
            )
        output_transcription = content.get("output_transcription")
        if isinstance(output_transcription, dict) and output_transcription.get("text"):
            events.append(
                RealtimeTranscriptEvent(
                    text=str(output_transcription.get("text") or ""),
                    role="assistant",
                    is_final=bool(content.get("turn_complete")),
                    provider_metadata=payload,
                )
            )
        if content.get("turn_complete"):
            events.append(RealtimeSessionEndedEvent(reason="turn-complete", provider_metadata=payload))
        return events
    if isinstance(payload.get("tool_call"), dict):
        call = dict(payload.get("tool_call") or {})
        return [
            RealtimeToolCallEvent(
                tool_call=ToolCall(
                    id=str(call.get("id") or f'{call.get("name", "")}-0'),
                    name=str(call.get("name") or ""),
                    input=call.get("args") or {},
                )
            )
        ]
    if isinstance(payload.get("error"), dict):
        return [RealtimeSessionEndedEvent(reason="error", provider_metadata=payload)]
    return []


def _parse_assistant_message(candidate: dict[str, Any] | None) -> ModelMessage:
    parts = []
    for part in ((candidate or {}).get("content") or {}).get("parts", []):
        if part.get("text"):
            parts.append(TextPart(text=part["text"]))
        elif part.get("functionCall"):
            call = part["functionCall"]
            provider_metadata: dict[str, Any] = {}
            if part.get("thoughtSignature") is not None:
                provider_metadata["thought_signature"] = part["thoughtSignature"]
            parts.append(
                ToolCallPart(
                    tool_call=ToolCall(
                        id=f'{call["name"]}-0',
                        name=call["name"],
                        input=call.get("args") or {},
                        provider_metadata=provider_metadata,
                    )
                )
            )
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
                        provider_metadata: dict[str, Any] = {}
                        if part.get("thoughtSignature") is not None:
                            provider_metadata["thought_signature"] = part["thoughtSignature"]
                        yield StreamToolCallEvent(
                            tool_call=ToolCall(
                                id=f'{call["name"]}-0',
                                name=call["name"],
                                input=call.get("args") or {},
                                provider_metadata=provider_metadata,
                            )
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


@dataclass(slots=True)
class GeminiRealtimeModel(RealtimeModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    realtime_url: str | None = None
    connection_factory: RealtimeConnectionFactory | None = None
    capabilities: ModelCapabilities = field(default_factory=lambda: GEMINI_REALTIME_CAPABILITIES)

    async def connect(
        self,
        config: RealtimeSessionConfig | None = None,
        options: RealtimeConnectOptions | None = None,
    ) -> RealtimeSession:
        resolved_config = config or RealtimeSessionConfig()
        url = self.realtime_url or _gemini_realtime_url(self.base_url, self.api_key, resolved_config.provider_options)
        headers = _gemini_realtime_headers(resolved_config.provider_options)
        factory = self.connection_factory or (lambda u, h, o: open_websocket_connection(u, headers=h, options=o))
        connection = await factory(url, headers, options)
        session = CallbackRealtimeSession(
            provider=self.provider,
            model_id=self.model_id,
            capabilities=self.capabilities,
            config=resolved_config,
            connection=connection,
            callbacks=RealtimeSessionCallbacks(
                parse_event=_gemini_realtime_parse_event,
                build_audio_payloads=_gemini_realtime_build_audio,
                build_text_payloads=_gemini_realtime_build_text,
                build_tool_result_payloads=_gemini_realtime_build_tool_result,
                build_update_payloads=lambda session_config: _gemini_realtime_build_update(session_config, self.model_id),
                build_initial_payloads=lambda session_config: _gemini_realtime_build_update(session_config, self.model_id),
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


def create_gemini(
    *,
    api_key: str | None = None,
    base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    fetch: Fetcher | None = None,
    realtime_url: str | None = None,
    realtime_connection_factory: RealtimeConnectionFactory | None = None,
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
        realtime_model_factory=lambda model_id: GeminiRealtimeModel(
            provider="gemini",
            model_id=model_id,
            api_key=resolved_key,
            base_url=base,
            fetch=requester,
            realtime_url=realtime_url,
            connection_factory=realtime_connection_factory,
        ),
    )
