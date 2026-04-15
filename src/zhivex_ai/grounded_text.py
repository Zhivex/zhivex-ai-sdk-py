from __future__ import annotations

from .errors import UnsupportedFeatureError, ValidationError
from .generate_text import _apply_retrieval
from .messages import create_text_message
from .types import (
    GenerateGroundedTextOutput,
    GroundedLanguageModel,
    GroundedModelGenerateInput,
    ModelMessage,
    PortableGroundingConfig,
    PortableRetrievalConfig,
    ReasoningConfig,
)


def _build_messages(
    *,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    system: str | None = None,
) -> list[ModelMessage]:
    if prompt is not None and messages is not None:
        raise ValidationError('Pass either "prompt" or "messages", but not both.')

    built = list(messages or [])
    if system:
        built.insert(0, create_text_message("system", system))
    if prompt:
        built.append(create_text_message("user", prompt))
    return built


async def generate_grounded_text(
    *,
    model: GroundedLanguageModel,
    prompt: str | None = None,
    messages: list[ModelMessage] | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning: ReasoningConfig | None = None,
    config: PortableGroundingConfig | None = None,
    retrieval: PortableRetrievalConfig | None = None,
    provider_options: dict[str, object] | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    retry_backoff_ms: int | None = None,
) -> GenerateGroundedTextOutput:
    if getattr(model, "portable", False) and provider_options is not None:
        raise ValidationError(
            "Portable grounded generation does not accept provider_options. "
            "Use `provider.native.grounded_language_model(...)` when you need provider-specific configuration."
        )
    if not model.capabilities.web_search:
        raise UnsupportedFeatureError(
            f'Model "{model.provider}/{model.model_id}" does not support web search.'
        )

    built_messages = _build_messages(prompt=prompt, messages=messages, system=system)
    built_messages = _apply_retrieval(built_messages, retrieval)
    result = await model.generate(
        GroundedModelGenerateInput(
            messages=built_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
            provider_options=provider_options,
            timeout_ms=timeout_ms,
            max_retries=max_retries,
            retry_backoff_ms=retry_backoff_ms,
        )
    )
    sources = result.sources[: config.max_sources] if config and config.max_sources is not None else result.sources
    return GenerateGroundedTextOutput(
        text=result.text or "",
        sources=sources,
        queries=result.queries,
        supports=result.supports,
        search_entry_point=result.search_entry_point,
        finish_reason=result.finish_reason,
        provider_finish_reason=result.provider_finish_reason,
        usage=result.usage,
        messages=built_messages,
        raw_response=result.raw_response,
    )
