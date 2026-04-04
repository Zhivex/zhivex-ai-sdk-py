from __future__ import annotations

import base64
import json
import os
from collections.abc import AsyncIterable
from dataclasses import dataclass, field, replace
from typing import Any

from .._http import Fetcher, default_fetch
from .._sse import parse_sse
from ..errors import ConfigurationError, ProviderHTTPError, UnsupportedFeatureError
from ..messages import normalize_finish_reason
from ..runtime import with_retry
from ..schema import create_schema_adapter
from ..types import (
    AudioInput,
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
    RetryOptions,
    SpeechModel,
    SpeechOutput,
    StreamEvent,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    TokenUsage,
    ToolCall,
    ToolChoiceName,
    TranscriptionModel,
    TranscriptionOutput,
)
from .base import ProviderAdapter

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


def _parse_json_error(provider_name: str, status_code: int, body: str) -> ProviderHTTPError:
    return ProviderHTTPError(f"{provider_name} request failed with status {status_code}.", status_code, response_body=body)


def _map_content_parts(message: ModelMessage) -> str | list[dict[str, Any]]:
    text_parts = [part for part in message.parts if part.type == "text"]
    image_parts = [part for part in message.parts if part.type == "image"]
    if not image_parts:
        return "".join(part.text for part in text_parts)
    return [
        *({"type": "text", "text": part.text} for part in text_parts),
        *({"type": "image_url", "image_url": {"url": part.image}} for part in image_parts),
    ]


def _map_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            tool_result = next((part.tool_result for part in message.parts if part.type == "tool-result"), None)
            payload.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_result.tool_call_id if tool_result else None,
                    "content": json.dumps(tool_result.error.__dict__ if tool_result and tool_result.is_error else tool_result.output),
                }
            )
            continue

        tool_calls = [
            {
                "id": part.tool_call.id,
                "type": "function",
                "function": {
                    "name": part.tool_call.name,
                    "arguments": json.dumps(part.tool_call.input),
                },
            }
            for part in message.parts
            if part.type == "tool-call"
        ]
        item: dict[str, Any] = {
            "role": message.role,
            "content": _map_content_parts(message),
        }
        if tool_calls:
            item["tool_calls"] = tool_calls
        payload.append(item)
    return payload


def _map_tools(tools: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    mapped = []
    for tool in tools.values():
        mapped.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": create_schema_adapter(tool.schema).json_schema(),
                },
            }
        )
    return mapped


def _map_tool_choice(tool_choice: str | ToolChoiceName | None) -> str | dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    return {
        "type": "function",
        "function": {
            "name": tool_choice.tool_name,
        },
    }


def _map_structured_output(input: ModelGenerateInput) -> dict[str, Any] | None:
    if input.structured_output is None or input.structured_output.mode != "native":
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": input.structured_output.name or "response",
            "strict": True,
            "schema": create_schema_adapter(input.structured_output.schema).json_schema(),
        },
    }


def _map_reasoning(input: ModelGenerateInput, provider_name: str) -> dict[str, Any]:
    if input.reasoning is None:
        return {}
    if input.reasoning.budget_tokens is not None:
        raise UnsupportedFeatureError(f'Provider "{provider_name}" does not support "reasoning.budgetTokens".')
    return {
        "reasoning_effort": input.reasoning.effort,
        "max_completion_tokens": input.max_tokens,
    }


def _parse_assistant_message(message: dict[str, Any]) -> ModelMessage:
    parts = []
    content = message.get("content")
    if isinstance(content, str) and content:
        from ..types import TextPart, ToolCallPart

        parts.append(TextPart(text=content))
    for call in message.get("tool_calls", []) or []:
        from ..types import ToolCallPart

        parts.append(
            ToolCallPart(
                tool_call=ToolCall(
                    id=call["id"],
                    name=call["function"]["name"],
                    input=json.loads(call["function"].get("arguments") or "{}"),
                )
            )
        )
    return ModelMessage(role="assistant", parts=parts)


def _to_responses_input(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        content = [
            {"type": "input_text", "text": part.text}
            for part in message.parts
            if part.type == "text"
        ]
        items.append({"role": message.role, "content": content})
    return items


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
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": _map_messages(input.messages),
            "tools": _map_tools(input.tools),
            "tool_choice": _map_tool_choice(input.tool_choice),
            "response_format": _map_structured_output(input),
            "temperature": input.temperature,
            **(input.provider_options or {}),
            **_map_reasoning(input, self.provider),
            "stream": False,
        }
        if input.reasoning is None:
            body["max_tokens"] = input.max_tokens

        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/chat/completions",
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
        choice = (payload.get("choices") or [{}])[0]
        assistant_message = _parse_assistant_message(choice.get("message") or {})
        return GenerateResult(
            messages=[assistant_message],
            text="".join(part.text for part in assistant_message.parts if part.type == "text"),
            finish_reason=normalize_finish_reason(choice.get("finish_reason")),
            provider_finish_reason=choice.get("finish_reason"),
            usage=TokenUsage(
                input_tokens=payload.get("usage", {}).get("prompt_tokens"),
                output_tokens=payload.get("usage", {}).get("completion_tokens"),
                total_tokens=payload.get("usage", {}).get("total_tokens"),
            ),
            raw_response=payload,
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": _map_messages(input.messages),
            "tools": _map_tools(input.tools),
            "tool_choice": _map_tool_choice(input.tool_choice),
            "response_format": _map_structured_output(input),
            "temperature": input.temperature,
            **(input.provider_options or {}),
            **_map_reasoning(input, self.provider),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if input.reasoning is None:
            body["max_tokens"] = input.max_tokens

        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/chat/completions",
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
            tool_buffers: dict[str, dict[str, str]] = {}
            async for event in parse_sse(response.iter_lines()):
                if event.data == "[DONE]":
                    return
                payload = json.loads(event.data)
                choice = (payload.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    yield StreamTextDeltaEvent(text_delta=delta["content"])
                for tool_call in delta.get("tool_calls") or []:
                    tool_id = tool_call.get("id") or str(tool_call.get("index"))
                    existing = tool_buffers.setdefault(tool_id, {"name": "", "args": ""})
                    existing["name"] = existing["name"] or tool_call.get("function", {}).get("name", "")
                    existing["args"] += tool_call.get("function", {}).get("arguments", "")
                    if choice.get("finish_reason") == "tool_calls":
                        yield StreamToolCallEvent(
                            tool_call=ToolCall(
                                id=tool_id,
                                name=existing["name"],
                                input=json.loads(existing["args"] or "{}"),
                            )
                        )
                if choice.get("finish_reason"):
                    usage = payload.get("usage") or {}
                    yield StreamFinishEvent(
                        finish_reason=normalize_finish_reason(choice.get("finish_reason")),
                        provider_finish_reason=choice.get("finish_reason"),
                        usage=TokenUsage(
                            input_tokens=usage.get("prompt_tokens"),
                            output_tokens=usage.get("completion_tokens"),
                            total_tokens=usage.get("total_tokens"),
                        )
                        if usage
                        else None,
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


@dataclass(slots=True)
class OpenAICompatibleGroundedLanguageModel(_BaseOpenAICompatible, GroundedLanguageModel):
    capabilities: ModelCapabilities = field(default_factory=lambda: OPENAI_COMPAT_GROUNDED_CAPABILITIES)

    async def generate(self, input: GroundedModelGenerateInput) -> GroundedGenerateResult:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/responses",
                headers=self._headers(),
                json_body={
                    "model": self.model_id,
                    "input": _to_responses_input(input.messages),
                    "tools": [{"type": "web_search_preview"}],
                    "temperature": input.temperature,
                    "max_output_tokens": input.max_tokens,
                    **(input.provider_options or {}),
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
    supports_grounding: bool = False,
) -> ProviderAdapter:
    resolved_key = api_key or os.getenv(env_var)
    if not resolved_key:
        raise ConfigurationError(f"Missing {provider_name} API key.")
    requester = fetch or default_fetch
    base = base_url.rstrip("/")
    shared_capabilities = capabilities or OPENAI_COMPAT_CAPABILITIES
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
            if supports_audio
            else None
        ),
        speech_model_factory=(
            (lambda model_id: OpenAICompatibleSpeechModel(
                provider=provider_name,
                model_id=model_id,
                api_key=resolved_key,
                base_url=base,
                fetch=requester,
                auth_header=auth_header,
                auth_prefix=auth_prefix,
            ))
            if supports_audio
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
            ))
            if supports_grounding
            else None
        ),
    )
