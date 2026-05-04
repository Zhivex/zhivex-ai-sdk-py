from __future__ import annotations

import base64
import os
from dataclasses import replace
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

from .._http import Fetcher, default_fetch
from ..errors import ConfigurationError, ProviderHTTPError, ValidationError
from ..messages import hosted_tool
from ..runtime import with_retry
from ..types import (
    AgentCapabilities,
    AudioInput,
    HostedToolClass,
    HostedToolDefinition,
    ModelCapabilities,
    PortableSupport,
    RetryOptions,
    SpeechModel,
    SpeechOutput,
    TranscriptionModel,
    TranscriptionOutput,
)
from .base import create_provider_bundle
from .openai_compat import (
    OPENAI_COMPAT_CAPABILITIES,
    OPENAI_COMPAT_SPEECH_CAPABILITIES,
    OPENAI_COMPAT_TRANSCRIPTION_CAPABILITIES,
    OpenAICompatibleResponsesClient,
    create_openai_compatible_provider,
)

QwenRegion = Literal["intl", "us", "cn"]

QWEN_REGION_BASE_URLS: dict[QwenRegion, str] = {
    "intl": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


def _qwen_base_url(region: QwenRegion) -> str:
    try:
        return QWEN_REGION_BASE_URLS[region]
    except KeyError as exc:
        supported = ", ".join(sorted(QWEN_REGION_BASE_URLS))
        raise ConfigurationError(f'Unsupported qwen region "{region}". Supported regions: {supported}.') from exc


def _qwen_responses_base_url(base_url: str) -> str:
    parsed = urlparse(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/api/v2/apps/protocols/compatible-mode/v1"):
        return base_url.rstrip("/")
    for suffix in ("/compatible-mode/v1", "/compatible-mode"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            f"{path}/api/v2/apps/protocols/compatible-mode/v1",
            "",
            "",
            "",
        )
    )


def _qwen_speech_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    for suffix in ("/api/v2/apps/protocols/compatible-mode/v1", "/compatible-mode/v1", "/compatible-mode", "/api/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            f"{path}/api/v1/services/aigc/multimodal-generation/generation",
            "",
            "",
            "",
        )
    )


def _infer_qwen_media_type(url: str | None) -> str:
    normalized = (url or "").lower()
    if normalized.endswith(".mp3"):
        return "audio/mpeg"
    if normalized.endswith(".ogg") or normalized.endswith(".opus"):
        return "audio/ogg"
    if normalized.endswith(".pcm"):
        return "audio/pcm"
    return "audio/wav"


def qwen_hosted_tool(
    type: str,
    *,
    name: str | None = None,
    tool_class: HostedToolClass | None = None,
    **config: Any,
) -> HostedToolDefinition:
    return hosted_tool(
        name=name or type,
        provider="qwen",
        type=type,
        config=config or None,
        tool_class=tool_class,
    )


def qwen_web_search_tool(**config: Any) -> HostedToolDefinition:
    return qwen_hosted_tool("web_search", tool_class="web-search", **config)


def qwen_web_extractor_tool(**config: Any) -> HostedToolDefinition:
    return qwen_hosted_tool("web_extractor", tool_class="web-search", **config)


def qwen_code_interpreter_tool(**config: Any) -> HostedToolDefinition:
    return qwen_hosted_tool("code_interpreter", tool_class="code-execution", **config)


def qwen_file_search_tool(*, vector_store_ids: list[str], **config: Any) -> HostedToolDefinition:
    return qwen_hosted_tool(
        "file_search",
        tool_class="file-search",
        vector_store_ids=list(vector_store_ids),
        **config,
    )


def qwen_web_search_image_tool(**config: Any) -> HostedToolDefinition:
    return qwen_hosted_tool("web_search_image", tool_class="web-search", **config)


def qwen_image_search_tool(**config: Any) -> HostedToolDefinition:
    return qwen_hosted_tool("image_search", **config)


def qwen_mcp_tool(
    *,
    server_label: str,
    server_url: str,
    server_protocol: str = "sse",
    server_description: str | None = None,
    headers: dict[str, str] | None = None,
    **config: Any,
) -> HostedToolDefinition:
    return qwen_hosted_tool(
        "mcp",
        tool_class="remote-mcp",
        server_label=server_label,
        server_url=server_url,
        server_protocol=server_protocol,
        server_description=server_description,
        headers=headers,
        **config,
    )


def _qwen_asr_audio_data(audio: AudioInput) -> str:
    if isinstance(audio.data, str):
        if audio.data.startswith(("http://", "https://", "data:")):
            return audio.data
        return f"data:{audio.media_type};base64,{audio.data}"
    if isinstance(audio.data, memoryview):
        raw = audio.data.tobytes()
    else:
        raw = bytes(audio.data)
    return f"data:{audio.media_type};base64,{base64.b64encode(raw).decode('ascii')}"


@dataclass(slots=True)
class QwenTranscriptionModel(TranscriptionModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: OPENAI_COMPAT_TRANSCRIPTION_CAPABILITIES)

    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }

    async def transcribe(
        self,
        *,
        audio: AudioInput,
        prompt: str | None = None,
        language: str | None = None,
        provider_options: dict[str, Any] | None = None,
        options: RetryOptions | None = None,
    ) -> TranscriptionOutput:
        input_options = dict(provider_options or {})
        asr_options = dict(input_options.pop("asr_options", {}))
        if language:
            asr_options["language"] = language
        messages: list[dict[str, Any]] = []
        if prompt:
            messages.append({"role": "system", "content": [{"text": prompt}]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": _qwen_asr_audio_data(audio)},
                    }
                ],
            }
        )
        response = await with_retry(
            lambda: self.fetch(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json_body={
                    "model": self.model_id,
                    "messages": messages,
                    "asr_options": asr_options or None,
                    **input_options,
                },
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f'Provider "{self.provider}" transcription request failed with status {response.status_code}.',
                response.status_code,
                response_body=await response.text(),
            )
        payload = await response.json()
        message = (((payload.get("choices") or [{}])[0]).get("message") or {})
        return TranscriptionOutput(text=str(message.get("content") or ""), audio=audio, raw_response=payload)


@dataclass(slots=True)
class QwenSpeechModel(SpeechModel):
    provider: str
    model_id: str
    api_key: str
    base_url: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: OPENAI_COMPAT_SPEECH_CAPABILITIES)

    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }

    async def generate_speech(
        self,
        *,
        input: str,
        voice: str | None = None,
        provider_options: dict[str, Any] | None = None,
        options: RetryOptions | None = None,
    ) -> SpeechOutput:
        input_options = dict(provider_options or {})
        response = await with_retry(
            lambda: self.fetch(
                _qwen_speech_url(self.base_url),
                headers=self._headers(),
                json_body={
                    "model": self.model_id,
                    "input": {
                        **input_options,
                        "text": input,
                        "voice": voice or input_options.get("voice") or "Cherry",
                    },
                },
                timeout_ms=options.timeout_ms if options else None,
            ),
            max_retries=options.max_retries if options and options.max_retries is not None else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f'Provider "{self.provider}" request failed with status {response.status_code}.',
                response.status_code,
                response_body=await response.text(),
            )

        payload = await response.json()
        audio_info = ((payload.get("output") or {}).get("audio") or {})
        if isinstance(audio_info.get("url"), str) and audio_info.get("url"):
            audio_response = await with_retry(
                lambda: self.fetch(
                    str(audio_info["url"]),
                    method="GET",
                    headers={},
                    json_body=None,
                    body=None,
                    timeout_ms=options.timeout_ms if options else None,
                ),
                max_retries=options.max_retries if options and options.max_retries is not None else 0,
                retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
            )
            if audio_response.status_code >= 400:
                raise ProviderHTTPError(
                    f'Provider "{self.provider}" audio download failed with status {audio_response.status_code}.',
                    audio_response.status_code,
                    response_body=await audio_response.text(),
                )
            return SpeechOutput(
                audio=await audio_response.read(),
                media_type=audio_response.headers.get("content-type", _infer_qwen_media_type(str(audio_info.get("url")))),
                raw_response=payload,
            )

        if isinstance(audio_info.get("data"), str) and audio_info.get("data"):
            return SpeechOutput(
                audio=base64.b64decode(str(audio_info.get("data"))),
                media_type=_infer_qwen_media_type(str(audio_info.get("url"))),
                raw_response=payload,
            )

        raise ValidationError('Provider "qwen" did not return audio data for speech generation.')


def create_qwen(
    *,
    api_key: str | None = None,
    region: QwenRegion = "intl",
    base_url: str | None = None,
    responses_base_url: str | None = None,
    fetch: Fetcher | None = None,
):
    resolved_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not resolved_key:
        raise ConfigurationError("Missing qwen API key.")
    requester = fetch or default_fetch
    resolved_base_url = (base_url or _qwen_base_url(region)).rstrip("/")
    resolved_responses_base_url = (responses_base_url or _qwen_responses_base_url(resolved_base_url)).rstrip("/")
    capabilities = replace(
        OPENAI_COMPAT_CAPABILITIES,
        tools=True,
        tool_choice=True,
        parallel_tool_calls=False,
        web_search=True,
    )
    native = create_openai_compatible_provider(
        provider_name="qwen",
        env_var="QWEN_API_KEY",
        api_key=resolved_key,
        base_url=resolved_base_url,
        responses_base_url=resolved_responses_base_url,
        fetch=requester,
        capabilities=capabilities,
        supports_grounding=True,
        default_grounding_tool={"type": "web_search"},
        responses_client_factory=lambda: OpenAICompatibleResponsesClient(
            provider="qwen",
            model_id="",
            api_key=resolved_key,
            base_url=resolved_responses_base_url,
            fetch=requester,
        ),
    )
    native = replace(
        native,
        transcription_model_factory=lambda model_id: QwenTranscriptionModel(
            provider="qwen",
            model_id=model_id,
            api_key=resolved_key,
            base_url=resolved_base_url,
            fetch=requester,
        ),
        speech_model_factory=lambda model_id: QwenSpeechModel(
            provider="qwen",
            model_id=model_id,
            api_key=resolved_key,
            base_url=resolved_base_url,
            fetch=requester,
        ),
    )
    return create_provider_bundle(
        name="qwen",
        native=native,
        agent_capabilities=native.language_model("").capabilities.agent_capabilities or AgentCapabilities(),
        portable_support=PortableSupport(
            text_generation=True,
            streaming=True,
            structured_output=True,
            tools=True,
            embeddings=True,
            grounding=True,
            retrieval=True,
            transcription=True,
            speech=True,
            portable_badge=False,
            tier="compatibility",
        ),
    )
