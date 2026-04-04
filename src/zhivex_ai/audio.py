from __future__ import annotations

from .errors import UnsupportedFeatureError, ValidationError
from .types import AudioInput, RetryOptions, SpeechModel, SpeechOutput, TranscriptionModel, TranscriptionOutput


def _validate_audio_input(audio: AudioInput) -> None:
    if not audio.media_type.strip():
        raise ValidationError('The "audio.media_type" field is required.')


async def transcribe_audio(
    *,
    model: TranscriptionModel,
    audio: AudioInput,
    prompt: str | None = None,
    language: str | None = None,
    provider_options: dict[str, object] | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
) -> TranscriptionOutput:
    if not model.capabilities.audio_input:
        raise UnsupportedFeatureError(
            f'Model "{model.provider}/{model.model_id}" does not support audio input.'
        )

    _validate_audio_input(audio)
    result = await model.transcribe(
        audio=audio,
        prompt=prompt,
        language=language,
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
    provider_options: dict[str, object] | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
) -> SpeechOutput:
    if not model.capabilities.audio_output:
        raise UnsupportedFeatureError(
            f'Model "{model.provider}/{model.model_id}" does not support audio output.'
        )

    result = await model.generate_speech(
        input=input,
        voice=voice,
        provider_options=provider_options,
        options=RetryOptions(
            timeout_ms=timeout_ms,
            max_retries=max_retries,
            retry_backoff_ms=retry_backoff_ms,
        ),
    )
    result.input = input
    return result
