"""xAI (Grok) adapter.

Language calls use the Responses API by default. ``api_mode="chat"`` is the
legacy Chat Completions path and is rejected for ``grok-4.7``. Image, video,
speech, and transcription use the dedicated xAI media routes.

Official references:
https://docs.x.ai/developers/grok-4-7
https://docs.x.ai/developers/model-capabilities/text/reasoning
https://docs.x.ai/docs/models
https://docs.x.ai/developers/model-capabilities/images/generation
https://docs.x.ai/developers/migration/imagine-image-quality-nov-2
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import quote, urlparse

from .._http import Fetcher, default_fetch
from .._sse import parse_sse
from ..errors import ConfigurationError, ProviderHTTPError, UnsupportedFeatureError, ValidationError
from ..messages import hosted_tool, is_hosted_tool_definition, normalize_finish_reason
from ..runtime import with_retry
from ..types import (
    AgentCapabilities,
    AudioInput,
    GenerateResult,
    GroundedGenerateResult,
    GroundedLanguageModel,
    GroundedModelGenerateInput,
    ImagesClient,
    ImagesResult,
    LanguageModel,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    PortableSupport,
    ProviderFile,
    ProviderImage,
    ProviderOptions,
    RetryOptions,
    SpeechModel,
    SpeechOutput,
    StreamEvent,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    TextPart,
    TokenUsage,
    ToolCallPart,
    ToolResultPart,
    TranscriptionModel,
    TranscriptionOutput,
    VideoOperation,
    VideosClient,
)
from .base import ProviderAdapter, create_provider_bundle
from .openai_compat import OpenAICompatibleLanguageModel, _extract_sources, _parse_json_error


XAI_GLOBAL_BASE_URL = "https://api.x.ai/v1"
XAI_US_BASE_URL = "https://us.api.x.ai/v1"
XAI_IMAGE_MODELS = (
    "grok-imagine-image",
    "grok-imagine-image-2.0",
    "grok-imagine-image-quality",
)
XAI_VIDEO_MODELS = (
    "grok-imagine-video",
    "grok-imagine-video-1.5",
    "grok-imagine-video-1.5-lite",
)
XAI_TEXT_TO_SPEECH_MODEL = "grok-tts"
XAI_SPEECH_TO_TEXT_MODEL = "grok-voice-transcribe-2.0"
XAI_SPEECH_TO_SPEECH_MODEL = "grok-voice-3.0"
XAI_DEPRECATED_MODELS = {
    "grok-voice-2.0": 'Deprecated speech-to-speech model. Use "grok-voice-3.0".',
    "grok-imagine-image-quality": (
        'Deprecated on 2026-09-02 and retired on 2026-11-02. Migrate to '
        '"grok-imagine-image-2.0" and set quality to "low", "medium", or "auto".'
    ),
}
_IMAGE_ASPECT_RATIOS = frozenset({
    "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "2:1", "1:2",
    "19.5:9", "9:19.5", "20:9", "9:20", "21:9", "5:2", "auto",
})
_VIDEO_ASPECT_RATIOS = frozenset({"1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "5:2"})
_IMAGE_RESOLUTIONS = frozenset({"1k", "2k"})
_VIDEO_RESOLUTIONS = frozenset({"480p", "720p", "1080p"})
_FORBIDDEN_HEADERS = frozenset({
    "authorization", "content-type", "content-length", "host", "connection", "transfer-encoding",
})
_XAI_AGENT = AgentCapabilities(
    support_tier="tier-b",
    tool_choice_none=True,
    hosted_web_search=True,
    hosted_file_search=True,
    code_execution=True,
)


def _normalized(model_id: str) -> str:
    return model_id.strip().lower()


def _matches(model_id: str, pattern: str) -> bool:
    return re.match(pattern, _normalized(model_id)) is not None


def _is_grok_43(model_id: str) -> bool:
    return _matches(model_id, r"grok-4\.3(?:[-@]|$)")


def _is_grok_45(model_id: str) -> bool:
    return _matches(model_id, r"grok-4\.5(?:[-@]|$)")


def _is_grok_46(model_id: str) -> bool:
    return _matches(model_id, r"grok-4\.6(?:[-@]|$)")


def _is_grok_47(model_id: str) -> bool:
    return _matches(model_id, r"grok-4\.7(?:[-@]|$)")


def _is_grok_420_multi_agent(model_id: str) -> bool:
    return _matches(model_id, r"grok-4\.20(?:-\d{4})?-multi-agent(?:[-@]|$)")


def _is_non_reasoning(model_id: str) -> bool:
    return "non-reasoning" in _normalized(model_id)


def _is_always_reasoning(model_id: str) -> bool:
    return _is_grok_45(model_id) or _is_grok_46(model_id) or _is_grok_47(model_id)


def _reasoning_efforts(model_id: str) -> tuple[str, ...] | None:
    if _is_non_reasoning(model_id):
        return None
    if _is_grok_46(model_id) or _is_grok_47(model_id) or _is_grok_45(model_id) or _is_grok_420_multi_agent(model_id):
        return ("low", "medium", "high", "xhigh")
    if _is_grok_43(model_id):
        return ("none", "low", "medium", "high", "xhigh")
    return ("none", "low", "medium", "high")


def _language_capabilities(model_id: str) -> ModelCapabilities:
    file_search = not _is_grok_47(model_id)
    return ModelCapabilities(
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
        reasoning=not _is_non_reasoning(model_id),
        web_search=True,
        agent_capabilities=replace(_XAI_AGENT, hosted_file_search=file_search),
    )


def _media_capabilities(kind: str) -> ModelCapabilities:
    return ModelCapabilities(
        streaming=False,
        tools=False,
        structured_output=False,
        json_mode=False,
        tool_choice=False,
        parallel_tool_calls=False,
        vision=kind == "image",
        files=False,
        audio_input=kind == "transcription",
        audio_output=kind == "speech",
        embeddings=False,
        reasoning=False,
        web_search=False,
    )


def _us_host(base_url: str) -> bool:
    return urlparse(base_url).hostname == "us.api.x.ai"


def _assert_media_region(base_url: str, feature: str) -> None:
    if _us_host(base_url):
        raise UnsupportedFeatureError(
            "Provider \"xai\" US regional endpoint currently serves grok-4.7 and grok-4.6 text models only. "
            f"{feature} is not available there."
        )


def _assert_known_model(model_id: str, allowed: tuple[str, ...], label: str) -> None:
    if model_id not in allowed:
        supported = ", ".join(allowed)
        raise UnsupportedFeatureError(
            f'Provider "xai" {label} model "{model_id}" is not supported. Supported models: {supported}.'
        )


def _resolve_base_url(
    *,
    base_url: str | None,
    region: str | None,
    allow_unsafe_endpoints: bool,
) -> str:
    if region is not None and region not in {"global", "us"}:
        raise ConfigurationError('Provider "xai" region must be "global" or "us".')
    if base_url is not None and region is not None:
        raise ConfigurationError('Provider "xai" accepts base_url or region, not both.')
    raw = base_url or (XAI_US_BASE_URL if region == "us" else XAI_GLOBAL_BASE_URL)
    parsed = urlparse(raw)
    if parsed.scheme != "https" and not allow_unsafe_endpoints:
        raise ConfigurationError('Provider "xai" base_url must use https.')
    if not parsed.netloc:
        raise ConfigurationError('Provider "xai" base_url must include a host.')
    return raw.rstrip("/")


def _filter_headers(headers: dict[str, str] | None) -> dict[str, str]:
    return {
        key: value
        for key, value in (headers or {}).items()
        if key.lower() not in _FORBIDDEN_HEADERS
    }


def _validate_language(model_id: str, input: ModelGenerateInput, base_url: str) -> None:
    if _us_host(base_url) and not _is_grok_46(model_id) and not _is_grok_47(model_id):
        raise UnsupportedFeatureError(
            "Provider \"xai\" US regional endpoint currently serves grok-4.7 and grok-4.6 only. "
            f'Model "{model_id}" is not available at https://us.api.x.ai/v1.'
        )
    reasoning = input.reasoning
    if reasoning is not None and reasoning.budget_tokens is not None:
        raise UnsupportedFeatureError('Provider "xai" does not support "reasoning.budgetTokens".')
    if reasoning is not None and (
        getattr(reasoning, "mode", None) is not None or getattr(reasoning, "context", None) is not None
    ):
        raise UnsupportedFeatureError('Provider "xai" does not support "reasoning.mode" or "reasoning.context".')
    if reasoning is not None and getattr(reasoning, "include_thoughts", None):
        raise UnsupportedFeatureError(
            'Provider "xai" does not expose reasoning through "includeThoughts"; '
            "use provider_options.include with encrypted reasoning when required."
        )
    if _is_non_reasoning(model_id) and reasoning is not None:
        raise UnsupportedFeatureError(f'Provider "xai" model "{model_id}" does not support reasoning.')
    effort = reasoning.effort if reasoning is not None else None
    supported = _reasoning_efforts(model_id)
    if effort and supported and effort not in supported:
        joined = ", ".join(supported)
        raise UnsupportedFeatureError(
            f'Provider "xai" model "{model_id}" does not support reasoning effort "{effort}". '
            f"Supported efforts: {joined}."
        )
    options = input.provider_options or {}
    api_mode = options.get("api_mode", "responses")
    if api_mode not in {"chat", "responses"}:
        raise ValidationError('Provider "xai" api_mode must be "chat" or "responses".')
    if _is_grok_47(model_id) and api_mode == "chat":
        raise UnsupportedFeatureError(
            f'Provider "xai" model "{model_id}" accepts the Responses API only. '
            'Omit api_mode or set it to "responses". See https://docs.x.ai/developers/grok-4-7.'
        )
    if api_mode == "chat" and any(is_hosted_tool_definition(tool) for tool in (input.tools or {}).values()):
        raise UnsupportedFeatureError('Provider "xai" supports hosted tools through the Responses API only.')
    reasoning_active = reasoning is not None or _is_always_reasoning(model_id)
    if reasoning_active and any(options.get(key) is not None for key in ("presence_penalty", "frequency_penalty", "stop")):
        raise UnsupportedFeatureError(
            f'Provider "xai" model "{model_id}" does not support presence_penalty, frequency_penalty, '
            "or stop while reasoning is active."
        )


def _split_transport_options(input: ModelGenerateInput) -> tuple[str, dict[str, Any], dict[str, str], str | None]:
    options = dict(input.provider_options or {})
    api_mode = str(options.pop("api_mode", "responses"))
    headers = _filter_headers(options.pop("headers", None) if isinstance(options.get("headers"), dict) else None)
    conversation_id = options.pop("conversation_id", None)
    prompt_cache_key = options.pop("prompt_cache_key", None)
    cache_key = prompt_cache_key or conversation_id
    if api_mode == "responses":
        if options.get("store") is False:
            include = options.get("include")
            items = list(include) if isinstance(include, list) else []
            if "reasoning.encrypted_content" not in items:
                items.append("reasoning.encrypted_content")
            options["include"] = items
        if cache_key:
            options["prompt_cache_key"] = cache_key
        return api_mode, options, headers, None
    return api_mode, options, headers, cache_key if isinstance(cache_key, str) else None


def _chat_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            for part in message.parts:
                if isinstance(part, ToolResultPart):
                    output = part.tool_result.error.message if part.tool_result.is_error and part.tool_result.error else part.tool_result.output
                    mapped.append({
                        "role": "tool",
                        "tool_call_id": part.tool_result.tool_call_id,
                        "content": output if isinstance(output, str) else json.dumps(output),
                    })
            continue
        text = "".join(part.text for part in message.parts if isinstance(part, TextPart))
        payload: dict[str, Any] = {"role": message.role, "content": text}
        tool_calls = [part.tool_call for part in message.parts if isinstance(part, ToolCallPart)]
        if tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.input)},
                }
                for call in tool_calls
            ]
        mapped.append(payload)
    return mapped


class _HeaderInjectingFetch:
    def __init__(self, fetch: Fetcher, default_headers: dict[str, str]) -> None:
        self._fetch = fetch
        self._default_headers = default_headers
        self.request_headers: dict[str, str] = {}

    async def __call__(self, url: str, *, method: str = "POST", headers: dict[str, str], **kwargs: Any):
        merged = {**self._default_headers, **self.request_headers, **headers}
        return await self._fetch(url, method=method, headers=merged, **kwargs)


class XAILanguageModel(LanguageModel):
    provider = "xai"

    def __init__(self, model_id: str, delegate: OpenAICompatibleLanguageModel, base_url: str, fetch_wrapper: _HeaderInjectingFetch) -> None:
        self.model_id = model_id
        self.capabilities = _language_capabilities(model_id)
        self._delegate = delegate
        self._base_url = base_url
        self._fetch = fetch_wrapper

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[StreamEvent]:
        _validate_language(self.model_id, input, self._base_url)
        mode, options, headers, chat_cache_key = _split_transport_options(input)
        self._fetch.request_headers = headers
        if mode == "chat":
            return self._chat_stream(input, options, chat_cache_key)
        prepared = replace(input, provider_options=options or None)
        return await self._delegate.stream(prepared)

    async def _chat(
        self,
        input: ModelGenerateInput,
        options: dict[str, Any],
        cache_key: str | None,
        *,
        stream: bool,
    ):
        headers = {"authorization": f"Bearer {self._delegate.api_key}", "content-type": "application/json"}
        if cache_key:
            headers["x-grok-conv-id"] = cache_key
        headers.update(self._fetch.request_headers)
        body = {
            **options,
            "model": self.model_id,
            "messages": _chat_messages(input.messages),
            "temperature": input.temperature,
            "max_tokens": input.max_tokens,
            "stream": True if stream else None,
        }
        if input.reasoning is not None and input.reasoning.effort is not None:
            body["reasoning_effort"] = input.reasoning.effort
        body = {key: value for key, value in body.items() if value is not None}
        response = await with_retry(
            lambda: self._fetch(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json_body=body,
                timeout_ms=input.timeout_ms,
                stream=stream,
            ),
            max_retries=input.max_retries or 0,
            retry_backoff_ms=input.retry_backoff_ms or 250,
        )
        if response.status_code >= 400:
            raise _parse_json_error("xai", response.status_code, await response.text())
        return response

    async def _chat_result(self, input: ModelGenerateInput, options: dict[str, Any], cache_key: str | None) -> GenerateResult:
        response = await self._chat(input, options, cache_key, stream=False)
        payload = await response.json()
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        usage = payload.get("usage") or {}
        return GenerateResult(
            messages=[ModelMessage(role="assistant", parts=[TextPart(text=text)] if text else [])],
            text=text,
            finish_reason=normalize_finish_reason(choice.get("finish_reason")),
            provider_finish_reason=choice.get("finish_reason"),
            usage=TokenUsage(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
            ),
            raw_response=payload,
        )

    def _chat_stream(self, input: ModelGenerateInput, options: dict[str, Any], cache_key: str | None):
        async def generator() -> AsyncIterable[StreamEvent]:
            response = await self._chat(input, options, cache_key, stream=True)
            finish_reason = None
            async for event in parse_sse(response.iter_lines()):
                if event.data == "[DONE]":
                    break
                payload = json.loads(event.data)
                choice = (payload.get("choices") or [{}])[0]
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    yield StreamTextDeltaEvent(text_delta=delta)
                if choice.get("finish_reason"):
                    finish_reason = choice.get("finish_reason")
            yield StreamFinishEvent(
                finish_reason=normalize_finish_reason(finish_reason),
                provider_finish_reason=finish_reason,
            )
        return generator()

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:  # type: ignore[no-redef]
        _validate_language(self.model_id, input, self._base_url)
        mode, options, headers, chat_cache_key = _split_transport_options(input)
        self._fetch.request_headers = headers
        if mode == "chat":
            return await self._chat_result(input, options, chat_cache_key)
        prepared = replace(input, provider_options=options or None)
        return await self._delegate.generate(prepared)


class XAIGroundedLanguageModel(GroundedLanguageModel):
    provider = "xai"

    def __init__(self, model_id: str, language_model: XAILanguageModel) -> None:
        self.model_id = model_id
        self.capabilities = language_model.capabilities
        self._language_model = language_model

    async def generate(self, input: GroundedModelGenerateInput) -> GroundedGenerateResult:
        result = await self._language_model.generate(ModelGenerateInput(
            messages=input.messages,
            tools={"web_search": xai_web_search_tool()},
            temperature=input.temperature,
            max_tokens=input.max_tokens,
            reasoning=input.reasoning,
            provider_options=input.provider_options,
            timeout_ms=input.timeout_ms,
            max_retries=input.max_retries,
            retry_backoff_ms=input.retry_backoff_ms,
        ))
        return GroundedGenerateResult(
            text=result.text or "",
            messages=result.messages,
            sources=_extract_sources(result.raw_response),
            finish_reason=result.finish_reason,
            provider_finish_reason=result.provider_finish_reason,
            usage=result.usage,
            raw_response=result.raw_response,
        )


def _sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if value is not None}


def xai_web_search_tool(
    *,
    allowed_domains: list[str] | None = None,
    excluded_domains: list[str] | None = None,
    enable_image_understanding: bool | None = None,
    enable_image_search: bool | None = None,
) -> Any:
    if allowed_domains and excluded_domains:
        raise UnsupportedFeatureError(
            'Provider "xai" Web Search cannot combine allowed_domains with excluded_domains.'
        )
    config = _sanitize_config({
        "enable_image_understanding": enable_image_understanding,
        "enable_image_search": enable_image_search,
    })
    if allowed_domains:
        config["filters"] = {"allowed_domains": list(allowed_domains)}
    if excluded_domains:
        config["filters"] = {"excluded_domains": list(excluded_domains)}
    return hosted_tool(
        name="web_search",
        provider="xai",
        type="web_search",
        tool_class="web-search",
        config=config or None,
    )


def xai_x_search_tool(
    *,
    allowed_x_handles: list[str] | None = None,
    excluded_x_handles: list[str] | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    enable_image_understanding: bool | None = None,
) -> Any:
    return hosted_tool(
        name="x_search",
        provider="xai",
        type="x_search",
        tool_class="web-search",
        config=_sanitize_config({
            "allowed_x_handles": allowed_x_handles,
            "excluded_x_handles": excluded_x_handles,
            "from_date": from_date,
            "to_date": to_date,
            "enable_image_understanding": enable_image_understanding,
        }) or None,
    )


def xai_code_execution_tool(**config: Any) -> Any:
    return hosted_tool(
        name="code_execution",
        provider="xai",
        type="code_interpreter",
        tool_class="code-execution",
        config=_sanitize_config(config) or None,
    )


def xai_file_search_tool(
    *,
    vector_store_ids: list[str],
    max_num_results: int | None = None,
    filters: dict[str, Any] | None = None,
) -> Any:
    return hosted_tool(
        name="file_search",
        provider="xai",
        type="file_search",
        tool_class="file-search",
        config=_sanitize_config({
            "vector_store_ids": list(vector_store_ids),
            "max_num_results": max_num_results,
            "filters": filters,
        }),
    )


def _encode_resource_id(value: str) -> str:
    if not value or value in {".", ".."} or re.search(r"[\\/?#\s]", value):
        raise ConfigurationError("xAI file ID must be a non-empty opaque identifier without path separators.")
    return quote(value, safe="")


@dataclass(slots=True)
class _XAIRequest:
    api_key: str
    base_url: str
    fetch: Fetcher

    def _headers(self, *, json_content: bool = True) -> dict[str, str]:
        headers = {"authorization": f"Bearer {self.api_key}"}
        if json_content:
            headers["content-type"] = "application/json"
        return headers

    async def _json(self, method: str, path: str, *, json_body: dict[str, Any] | None = None, timeout_ms: int | None = None, max_retries: int | None = None) -> dict[str, Any]:
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/{path.lstrip('/')}",
                method=method,
                headers=self._headers(),
                json_body=json_body,
                timeout_ms=timeout_ms,
            ),
            max_retries=max_retries or 0,
            retry_backoff_ms=250,
        )
        if response.status_code >= 400:
            raise _parse_json_error("xai", response.status_code, await response.text())
        payload = await response.json()
        if not isinstance(payload, dict):
            raise ValidationError('Provider "xai" returned a non-object JSON payload.')
        return payload


class XAIFilesClient(_XAIRequest):
    async def upload(
        self,
        *,
        data: bytes | bytearray | memoryview,
        filename: str,
        media_type: str = "application/pdf",
        purpose: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderFile:
        options = dict(metadata or {})
        expires_after = options.get("expires_after")
        if expires_after is not None and (
            isinstance(expires_after, bool) or not isinstance(expires_after, int) or not 3_600 <= expires_after <= 2_592_000
        ):
            raise UnsupportedFeatureError(
                'Provider "xai" requires files.expires_after to be an integer between 3600 and 2592000 seconds.'
            )
        form_data = {"purpose": purpose or "assistants"}
        if expires_after is not None:
            form_data["expires_after"] = str(expires_after)
        raw = bytes(data)
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url}/files",
                headers={"authorization": f"Bearer {self.api_key}"},
                body={"data": form_data, "files": {"file": (filename, raw, media_type)}},
                json_body=None,
                timeout_ms=None,
            ),
            max_retries=0,
            retry_backoff_ms=250,
        )
        if response.status_code >= 400:
            raise _parse_json_error("xai", response.status_code, await response.text())
        return _file_from_payload(await response.json())

    async def list(self) -> list[ProviderFile]:
        payload = await self._json("GET", "files")
        return [_file_from_payload(item) for item in payload.get("data") or payload.get("files") or []]

    async def get(self, file_id: str) -> ProviderFile:
        return _file_from_payload(await self._json("GET", f"files/{_encode_resource_id(file_id)}"))

    async def download(self, file_id: str) -> bytes:
        response = await self.fetch(
            f"{self.base_url}/files/{_encode_resource_id(file_id)}/content",
            method="GET",
            headers=self._headers(json_content=False),
            json_body=None,
            timeout_ms=None,
        )
        if response.status_code >= 400:
            raise _parse_json_error("xai", response.status_code, await response.text())
        return await response.read()

    async def delete(self, file_id: str) -> bool:
        await self._json("DELETE", f"files/{_encode_resource_id(file_id)}")
        return True


def _file_from_payload(payload: dict[str, Any]) -> ProviderFile:
    return ProviderFile(
        provider="xai",
        id=str(payload.get("id") or ""),
        filename=payload.get("filename"),
        media_type=payload.get("mime_type"),
        size_bytes=payload.get("bytes"),
        status=payload.get("status"),
        metadata=dict(payload),
    )


class XAIImagesClient(_XAIRequest, ImagesClient):
    async def generate(
        self,
        *,
        prompt: str,
        model: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        moderation: str | None = None,
        user: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ImagesResult:
        del background, moderation, user
        model_id = model or ""
        _assert_known_model(model_id, XAI_IMAGE_MODELS, "image")
        _assert_media_region(self.base_url, "Image generation")
        options = dict(extra_body or {})
        count = options.pop("n", None)
        aspect_ratio = options.pop("aspect_ratio", None)
        resolution = options.pop("resolution", None)
        if size in {"1k", "2k"} and resolution is None:
            resolution = size
        elif size is not None and resolution is None:
            raise UnsupportedFeatureError('Provider "xai" image size must be "1k" or "2k". Use resolution for output size.')
        if resolution is not None and resolution not in _IMAGE_RESOLUTIONS:
            raise UnsupportedFeatureError('Provider "xai" image resolution must be "1k" or "2k".')
        if aspect_ratio is not None and aspect_ratio not in _IMAGE_ASPECT_RATIOS:
            raise UnsupportedFeatureError(f'Provider "xai" does not support image aspect ratio "{aspect_ratio}".')
        selected_quality = quality if quality is not None else options.pop("quality", None)
        if selected_quality is not None and model_id != "grok-imagine-image-2.0":
            raise UnsupportedFeatureError('Provider "xai" image quality is supported only for "grok-imagine-image-2.0".')
        if selected_quality is not None and selected_quality not in {"low", "medium", "auto"}:
            raise UnsupportedFeatureError('Provider "xai" image quality must be "low", "medium", or "auto".')
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10):
            raise UnsupportedFeatureError('Provider "xai" image generation count must be an integer from 1 to 10.')
        response_format = output_format or options.pop("response_format", None)
        poll_interval_ms = options.pop("poll_interval_ms", 0)
        body = {
            key: value for key, value in options.items()
            if key not in {"model", "prompt", "image", "images"}
        }
        body.update({
            "model": model_id,
            "prompt": prompt,
            "n": count,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "quality": selected_quality,
            "response_format": response_format,
        })
        body = {key: value for key, value in body.items() if value is not None}
        payload = await self._json("POST", "images/generations", json_body=body)
        if not isinstance(payload.get("data"), list) and isinstance(payload.get("request_id"), str):
            payload = await self._poll_image(payload["request_id"], poll_interval_ms)
        return _images_result(payload)

    async def edit(self, **kwargs: Any) -> ImagesResult:
        raise UnsupportedFeatureError(
            'Provider "xai" image edits use images/edits. Pass image references through extra_body on generate '
            "or use the native images client with an image URL in extra_body['image']."
        )

    async def variation(self, **kwargs: Any) -> ImagesResult:
        raise UnsupportedFeatureError('Provider "xai" does not support image variations.')

    async def _poll_image(self, request_id: str, poll_interval_ms: int) -> dict[str, Any]:
        deadline = time.monotonic() + 120
        payload: dict[str, Any] = {"status": "pending"}
        while payload.get("status") not in {"done", "failed"}:
            if time.monotonic() >= deadline:
                raise ProviderHTTPError("xAI image generation timed out after 120000ms.", 408)
            await asyncio.sleep(max(poll_interval_ms, 0) / 1000)
            payload = await self._json("GET", f"images/{quote(request_id, safe='')}")
        if payload.get("status") == "failed":
            message = ((payload.get("error") or {}) if isinstance(payload.get("error"), dict) else {}).get("message")
            raise ProviderHTTPError(f"xAI image generation failed{': ' + message if message else '.'}", 502, response_body=json.dumps(payload))
        return payload


def _images_result(payload: dict[str, Any]) -> ImagesResult:
    data = payload.get("data") if isinstance(payload.get("data"), list) else []
    if not data:
        raise ProviderHTTPError("xAI image generation returned no images.", 502, response_body=json.dumps(payload))
    images = [
        ProviderImage(
            provider="xai",
            url=item.get("url") if isinstance(item, dict) else None,
            b64_json=item.get("b64_json") if isinstance(item, dict) else None,
            media_type=(item.get("mime_type") if isinstance(item, dict) else None) or "image/jpeg",
            metadata=item if isinstance(item, dict) else {},
        )
        for item in data
    ]
    return ImagesResult(images=images, raw_response=payload)


class XAIVideosClient(_XAIRequest, VideosClient):
    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        config: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        options: RetryOptions | None = None,
    ) -> VideoOperation:
        _assert_known_model(model, XAI_VIDEO_MODELS, "video")
        _assert_media_region(self.base_url, "Video generation")
        settings = {**(config or {}), **(extra_body or {})}
        duration = settings.pop("duration", None)
        aspect_ratio = settings.pop("aspect_ratio", None)
        resolution = settings.pop("resolution", None)
        if duration is not None and (isinstance(duration, bool) or not isinstance(duration, int) or not 1 <= duration <= 15):
            raise UnsupportedFeatureError('Provider "xai" video duration must be an integer from 1 to 15.')
        if aspect_ratio is not None and aspect_ratio not in _VIDEO_ASPECT_RATIOS:
            raise UnsupportedFeatureError(f'Provider "xai" does not support video aspect ratio "{aspect_ratio}".')
        if resolution is not None and resolution not in _VIDEO_RESOLUTIONS:
            raise UnsupportedFeatureError('Provider "xai" video resolution must be "480p", "720p", or "1080p".')
        if model == "grok-imagine-video" and resolution == "1080p":
            raise UnsupportedFeatureError('Provider "xai" model "grok-imagine-video" supports video resolutions up to 720p.')
        body = {key: value for key, value in settings.items() if key not in {"model", "prompt"}}
        body.update({"model": model, "prompt": prompt, "duration": duration, "aspect_ratio": aspect_ratio, "resolution": resolution})
        body = {key: value for key, value in body.items() if value is not None}
        timeout_ms = options.timeout_ms if options and options.timeout_ms is not None else 600_000
        payload = await self._json("POST", "videos/generations", json_body=body, timeout_ms=timeout_ms, max_retries=0)
        request_id = payload.get("request_id")
        deadline = time.monotonic() + timeout_ms / 1000
        while payload.get("status") not in {"done", "failed"}:
            if not isinstance(request_id, str):
                raise ProviderHTTPError("xAI video generation returned no request id.", 502, response_body=json.dumps(payload))
            if time.monotonic() >= deadline:
                raise ProviderHTTPError(f"xAI video generation timed out after {timeout_ms}ms.", 408)
            await asyncio.sleep(0)
            payload = await self._json("GET", f"videos/{quote(request_id, safe='')}", timeout_ms=timeout_ms)
        if payload.get("status") == "failed" or not isinstance((payload.get("video") or {}).get("url"), str):
            message = ((payload.get("error") or {}) if isinstance(payload.get("error"), dict) else {}).get("message")
            raise ProviderHTTPError(f"xAI video generation failed{': ' + message if message else '.'}", 502, response_body=json.dumps(payload))
        return VideoOperation(name=str(request_id or ""), done=True, response=payload.get("video"), raw_response=payload)

    async def get_operation(self, name: str, options: RetryOptions | None = None) -> VideoOperation:
        payload = await self._json("GET", f"videos/{quote(name, safe='')}", timeout_ms=options.timeout_ms if options else None)
        return VideoOperation(name=name, done=payload.get("status") in {"done", "failed"}, response=payload.get("video"), raw_response=payload)

    async def wait_operation(self, name: str, *, poll_interval_ms: int = 10_000, timeout_ms: int | None = None) -> VideoOperation:
        deadline = time.monotonic() + (timeout_ms or 600_000) / 1000
        operation = await self.get_operation(name)
        while not operation.done:
            if time.monotonic() >= deadline:
                raise ProviderHTTPError("xAI video generation timed out.", 408)
            await asyncio.sleep(poll_interval_ms / 1000)
            operation = await self.get_operation(name)
        return operation

    async def download(self, uri: str, options: RetryOptions | None = None) -> bytes:
        response = await self.fetch(
            uri,
            method="GET",
            headers=self._headers(json_content=False),
            json_body=None,
            timeout_ms=options.timeout_ms if options else None,
        )
        if response.status_code >= 400:
            raise _parse_json_error("xai", response.status_code, await response.text())
        return await response.read()


class XAISpeechModel(SpeechModel):
    provider = "xai"

    def __init__(self, model_id: str, api_key: str, base_url: str, fetch: Fetcher) -> None:
        self.model_id = model_id
        self.capabilities = _media_capabilities("speech")
        self._request = _XAIRequest(api_key=api_key, base_url=base_url, fetch=fetch)
        _assert_voice("speech", model_id)
        _assert_media_region(base_url, "Text to speech")

    async def generate_speech(
        self,
        *,
        input: str,
        voice: str | None = None,
        provider_options: ProviderOptions | None = None,
        options: RetryOptions | None = None,
    ) -> SpeechOutput:
        if voice is not None and not isinstance(voice, str):
            raise UnsupportedFeatureError('Provider "xai" accepts string voice names only.')
        if not input.strip():
            raise ConfigurationError('Provider "xai" text-to-speech requires text.')
        if len(input) > 60_000:
            raise UnsupportedFeatureError('Provider "xai" text-to-speech accepts at most 60000 characters.')
        settings = dict(provider_options or {})
        language = settings.pop("language", None) or "en"
        body = {
            key: value for key, value in settings.items() if key not in {"model", "text", "voice_id", "language"}
        }
        body.update({"text": input, "voice_id": voice or "eve", "language": language})
        response = await with_retry(
            lambda: self._request.fetch(
                f"{self._request.base_url}/tts",
                headers=self._request._headers(),
                json_body=body,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=250,
        )
        if response.status_code >= 400:
            raise _parse_json_error("xai", response.status_code, await response.text())
        content_type = response.headers.get("content-type", "")
        if "json" in content_type or settings.get("with_timestamps") is True:
            payload = await response.json()
            audio = payload.get("audio")
            if not isinstance(audio, str):
                raise ProviderHTTPError("xAI text-to-speech response did not contain audio.", 502)
            return SpeechOutput(audio=base64.b64decode(audio), media_type="audio/mpeg", raw_response={key: value for key, value in payload.items() if key != "audio"})
        return SpeechOutput(audio=await response.read(), media_type=content_type or "audio/mpeg")


class XAITranscriptionModel(TranscriptionModel):
    provider = "xai"

    def __init__(self, model_id: str, api_key: str, base_url: str, fetch: Fetcher) -> None:
        self.model_id = model_id
        self.capabilities = _media_capabilities("transcription")
        self._api_key = api_key
        self._base_url = base_url
        self._fetch = fetch
        _assert_voice("transcription", model_id)
        _assert_media_region(base_url, "Speech to text")

    async def transcribe(
        self,
        *,
        audio: AudioInput,
        prompt: str | None = None,
        language: str | None = None,
        provider_options: ProviderOptions | None = None,
        options: RetryOptions | None = None,
    ) -> TranscriptionOutput:
        if prompt is not None:
            raise UnsupportedFeatureError(
                'Provider "xai" speech-to-text does not support prompt. Pass key terms through provider_options["keyterm"].'
            )
        options_map = dict(provider_options or {})
        remote_url = options_map.pop("url", None)
        data = {"model": self.model_id}
        if language:
            data["language"] = language
        files = None
        if isinstance(remote_url, str):
            data["url"] = remote_url
        else:
            raw = audio.data
            if isinstance(raw, str):
                raw = base64.b64decode(raw)
            elif isinstance(raw, memoryview):
                raw = raw.tobytes()
            files = {"file": (audio.filename or "audio", bytes(raw), audio.media_type)}
        for key, value in options_map.items():
            if key in {"model", "file"}:
                continue
            data[key] = value if isinstance(value, str) else json.dumps(value)
        response = await with_retry(
            lambda: self._fetch(
                f"{self._base_url}/stt",
                headers={"authorization": f"Bearer {self._api_key}"},
                body={"data": data, "files": files},
                json_body=None,
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=250,
        )
        if response.status_code >= 400:
            raise _parse_json_error("xai", response.status_code, await response.text())
        payload = await response.json()
        return TranscriptionOutput(text=payload.get("text") if isinstance(payload.get("text"), str) else "", raw_response=payload)


def _assert_voice(kind: str, model_id: str) -> None:
    normalized = _normalized(model_id)
    if normalized == "grok-voice-2.0":
        raise UnsupportedFeatureError(
            'Provider "xai" model "grok-voice-2.0" is deprecated. Speech-to-speech uses "grok-voice-3.0". '
            'Text-to-speech uses "grok-tts", and speech-to-text uses "grok-voice-transcribe-2.0".'
        )
    if kind == "speech" and normalized != XAI_TEXT_TO_SPEECH_MODEL:
        hint = ' "grok-voice-3.0" is the speech-to-speech model, not text-to-speech.' if normalized == XAI_SPEECH_TO_SPEECH_MODEL else ""
        raise UnsupportedFeatureError(f'Provider "xai" text-to-speech model is "{XAI_TEXT_TO_SPEECH_MODEL}".{hint}')
    if kind == "transcription" and normalized != XAI_SPEECH_TO_TEXT_MODEL:
        raise UnsupportedFeatureError(f'Provider "xai" speech-to-text model is "{XAI_SPEECH_TO_TEXT_MODEL}".')


def create_xai(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    region: str | None = None,
    headers: dict[str, str] | None = None,
    fetch: Fetcher | None = None,
    allow_unsafe_endpoints: bool = False,
):
    resolved_key = api_key or os.getenv("XAI_API_KEY")
    if not resolved_key:
        raise ConfigurationError("Missing xAI API key.")
    resolved_base = _resolve_base_url(base_url=base_url, region=region, allow_unsafe_endpoints=allow_unsafe_endpoints)
    requester = fetch or default_fetch
    default_headers = _filter_headers(headers)

    def language_model_factory(model_id: str) -> XAILanguageModel:
        wrapper = _HeaderInjectingFetch(requester, default_headers)
        delegate = OpenAICompatibleLanguageModel(
            provider="xai",
            model_id=model_id,
            api_key=resolved_key,
            base_url=resolved_base,
            fetch=wrapper,
            capabilities=_language_capabilities(model_id),
        )
        return XAILanguageModel(model_id, delegate, resolved_base, wrapper)

    native = ProviderAdapter(
        name="xai",
        language_model_factory=language_model_factory,
        transcription_model_factory=lambda model_id: XAITranscriptionModel(model_id, resolved_key, resolved_base, requester),
        speech_model_factory=lambda model_id: XAISpeechModel(model_id, resolved_key, resolved_base, requester),
        grounded_language_model_factory=lambda model_id: XAIGroundedLanguageModel(model_id, language_model_factory(model_id)),
        files_client_factory=lambda: XAIFilesClient(api_key=resolved_key, base_url=resolved_base, fetch=requester),
        images_client_factory=lambda: XAIImagesClient(api_key=resolved_key, base_url=resolved_base, fetch=requester),
        videos_client_factory=lambda: XAIVideosClient(api_key=resolved_key, base_url=resolved_base, fetch=requester),
    )
    return create_provider_bundle(
        name="xai",
        native=native,
        agent_capabilities=_XAI_AGENT,
        portable_support=PortableSupport(
            text_generation=True,
            streaming=True,
            structured_output=True,
            tools=True,
            embeddings=False,
            grounding=True,
            retrieval=True,
            transcription=True,
            speech=True,
            portable_badge=False,
            tier="native-only",
        ),
    )
