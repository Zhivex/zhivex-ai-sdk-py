from __future__ import annotations

from .errors import UnsupportedFeatureError, ValidationError
from .types import (
    AudioInput,
    PortableSpeechConfig,
    PortableTranscriptionConfig,
    RetryOptions,
    SpeechModel,
    SpeechOutput,
    TranscriptionModel,
    TranscriptionOutput,
)


def _validate_audio_input(audio: AudioInput) -> None:
    if not audio.media_type.strip():
        raise ValidationError('The "audio.media_type" field is required.')


async def transcribe_audio(
    *,
    model: TranscriptionModel,
    audio: AudioInput,
    prompt: str | None = None,
    language: str | None = None,
    config: PortableTranscriptionConfig | None = None,
    provider_options: dict[str, object] | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
) -> TranscriptionOutput:
    if getattr(model, "portable", False) and provider_options is not None:
        raise ValidationError(
            "Portable transcription does not accept provider_options. "
            "Use `provider.native.transcription_model(...)` when you need provider-specific configuration."
        )
    if not model.capabilities.audio_input:
        raise UnsupportedFeatureError(
            f'Model "{model.provider}/{model.model_id}" does not support audio input.'
        )

    _validate_audio_input(audio)
    resolved_config = config or PortableTranscriptionConfig(prompt=prompt, language=language)
    result = await model.transcribe(
        audio=audio,
        prompt=resolved_config.prompt,
        language=resolved_config.language,
        provider_options=provider_options,
        options=RetryOptions(
            timeout_ms=timeout_ms,
            max_retries=max_retries,
            retry_backoff_ms=retry_backoff_ms,
        ),
    )
    result.audio = audio
    return result


async def generate_speech(
    *,
    model: SpeechModel,
    input: str,
    voice: str | None = None,
    config: PortableSpeechConfig | None = None,
    provider_options: dict[str, object] | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
) -> SpeechOutput:
    if getattr(model, "portable", False) and provider_options is not None:
        raise ValidationError(
            "Portable speech generation does not accept provider_options. "
            "Use `provider.native.speech_model(...)` when you need provider-specific configuration."
        )
    if not model.capabilities.audio_output:
        raise UnsupportedFeatureError(
            f'Model "{model.provider}/{model.model_id}" does not support audio output.'
        )

    resolved_config = config or PortableSpeechConfig(voice=voice)
    resolved_provider_options = dict(provider_options or {})
    if resolved_config.audio_format is not None:
        if getattr(model, "provider", "") in {"openai", "azure-openai"}:
            resolved_provider_options["format"] = resolved_config.audio_format
        elif getattr(model, "provider", "") == "openrouter":
            resolved_provider_options.setdefault("audio", {})
            audio_options = dict(resolved_provider_options["audio"])
            audio_options["format"] = resolved_config.audio_format
            resolved_provider_options["audio"] = audio_options
    result = await model.generate_speech(
        input=input,
        voice=resolved_config.voice,
        provider_options=resolved_provider_options or None,
        options=RetryOptions(
            timeout_ms=timeout_ms,
            max_retries=max_retries,
            retry_backoff_ms=retry_backoff_ms,
        ),
    )
    result.input = input
    return result
