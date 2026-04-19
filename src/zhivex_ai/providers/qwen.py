from __future__ import annotations

import base64
import os
from dataclasses import replace
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

from .._http import Fetcher, default_fetch
from ..errors import ConfigurationError, ProviderHTTPError, ValidationError
from ..runtime import with_retry
from ..types import AgentCapabilities, ModelCapabilities, PortableSupport, RetryOptions, SpeechModel, SpeechOutput
from .base import create_provider_bundle
from .openai_compat import (
    OPENAI_COMPAT_CAPABILITIES,
    OPENAI_COMPAT_SPEECH_CAPABILITIES,
    create_openai_compatible_provider,
)


def _qwen_speech_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    for suffix in ("/compatible-mode/v1", "/compatible-mode", "/api/v1"):
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
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    fetch: Fetcher | None = None,
):
    resolved_key = api_key or os.getenv("QWEN_API_KEY")
    if not resolved_key:
        raise ConfigurationError("Missing qwen API key.")
    requester = fetch or default_fetch
    capabilities = replace(
        OPENAI_COMPAT_CAPABILITIES,
        tools=False,
        tool_choice=False,
        parallel_tool_calls=False,
    )
    native = create_openai_compatible_provider(
        provider_name="qwen",
        env_var="QWEN_API_KEY",
        api_key=resolved_key,
        base_url=base_url,
        fetch=requester,
        capabilities=capabilities,
    )
    native = replace(
        native,
        speech_model_factory=lambda model_id: QwenSpeechModel(
            provider="qwen",
            model_id=model_id,
            api_key=resolved_key,
            base_url=base_url,
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
            tools=False,
            embeddings=True,
            grounding=False,
            retrieval=True,
            transcription=False,
            speech=True,
            portable_badge=False,
            tier="compatibility",
        ),
    )
