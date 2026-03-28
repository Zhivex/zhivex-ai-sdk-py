from __future__ import annotations

from .errors import UnsupportedFeatureError
from .types import EmbedOutput, EmbeddingModel, RetryOptions


async def embed(
    *,
    model: EmbeddingModel,
    value: str | list[str],
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
) -> EmbedOutput:
    values = [value] if isinstance(value, str) else value
    if not model.capabilities.embeddings:
        raise UnsupportedFeatureError(f'Model "{model.provider}/{model.model_id}" does not support embeddings.')
    result = await model.embed(
        values,
        RetryOptions(timeout_ms=timeout_ms, max_retries=max_retries, retry_backoff_ms=retry_backoff_ms),
    )
    return EmbedOutput(embeddings=result.embeddings, usage=result.usage, raw_response=result.raw_response, values=values)


async def embed_many(
    *,
    model: EmbeddingModel,
    values: list[str],
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
) -> EmbedOutput:
    return await embed(
        model=model,
        value=values,
        timeout_ms=timeout_ms,
        max_retries=max_retries,
        retry_backoff_ms=retry_backoff_ms,
    )
