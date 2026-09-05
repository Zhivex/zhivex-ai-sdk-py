from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace, dataclass
from datetime import date
from typing import Iterable, Literal, Sequence

from .errors import ValidationError
from .types import ModelCapabilities

CatalogProviderId = str
ModelApiSurface = Literal[
    "language",
    "image",
    "realtime",
    "embedding",
    "transcription",
    "speech",
    "video",
    "media",
    "rerank",
    "interactions",
]
ModelAvailability = Literal["stable", "preview", "limited", "deprecated", "retired"]
ModelSupportEvidence = Literal["catalog-only", "offline-contract", "live-smoke"]
RecommendedUse = Literal[
    "chat",
    "reasoning",
    "speed",
    "vision",
    "tools",
    "embedding",
    "retrieval",
    "audio",
    "translation",
    "realtime",
]
_MODEL_API_SURFACES = {
    "language",
    "image",
    "realtime",
    "embedding",
    "transcription",
    "speech",
    "video",
    "media",
    "rerank",
    "interactions",
}
_MODEL_AVAILABILITIES = {"stable", "preview", "limited", "deprecated", "retired"}
_MODEL_SUPPORT_EVIDENCE = {"catalog-only", "offline-contract", "live-smoke"}
_RECOMMENDED_USES = {
    "chat",
    "reasoning",
    "speed",
    "vision",
    "tools",
    "embedding",
    "retrieval",
    "audio",
    "translation",
    "realtime",
}


def _validate_rate(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a finite, non-negative number.")
    if not math.isfinite(float(value)) or value < 0:
        raise ValidationError(f"{name} must be a finite, non-negative number.")


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Source-backed token pricing used to derive a conservative routing rate."""

    currency: str
    source_url: str
    input_per_1m_tokens: float | None = None
    output_per_1m_tokens: float | None = None
    cached_input_per_1m_tokens: float | None = None
    effective_from: str | None = None
    effective_until: str | None = None

    def __post_init__(self) -> None:
        currency = self.currency.strip().upper()
        if len(currency) != 3 or not currency.isalpha():
            raise ValidationError(
                "ModelPricing.currency must be a three-letter currency code."
            )
        if not self.source_url.startswith("https://"):
            raise ValidationError("ModelPricing.source_url must be an HTTPS URL.")
        object.__setattr__(self, "currency", currency)
        for name in (
            "input_per_1m_tokens",
            "output_per_1m_tokens",
            "cached_input_per_1m_tokens",
        ):
            _validate_rate(f"ModelPricing.{name}", getattr(self, name))
        if self.input_per_1m_tokens is None and self.output_per_1m_tokens is None:
            raise ValidationError(
                "ModelPricing requires an input or output token rate."
            )
        try:
            start = (
                date.fromisoformat(self.effective_from) if self.effective_from else None
            )
            end = (
                date.fromisoformat(self.effective_until)
                if self.effective_until
                else None
            )
        except ValueError as error:
            raise ValidationError(
                "ModelPricing effective dates must use ISO YYYY-MM-DD format."
            ) from error
        if start is not None and end is not None and end < start:
            raise ValidationError(
                "ModelPricing.effective_until cannot precede effective_from."
            )

    def conservative_cost_per_1k_tokens(
        self, *, as_of: date | None = None
    ) -> float | None:
        effective_date = as_of or date.today()
        if self.effective_from and effective_date < date.fromisoformat(
            self.effective_from
        ):
            return None
        if self.effective_until and effective_date > date.fromisoformat(
            self.effective_until
        ):
            return None
        rates = (self.input_per_1m_tokens, self.output_per_1m_tokens)
        known_rates = [float(rate) for rate in rates if rate is not None]
        return max(known_rates) / 1_000 if known_rates else None


@dataclass(frozen=True, slots=True)
class ModelCatalogEntry:
    provider: CatalogProviderId
    model_id: str
    aliases: Sequence[str] = ()
    cost_per_1k_tokens: float | None = None
    recommended_for: Sequence[RecommendedUse] = ()
    api_surface: ModelApiSurface = "language"
    availability: ModelAvailability = "stable"
    regions: Sequence[str] = ()
    support_evidence: ModelSupportEvidence = "catalog-only"
    source_urls: Sequence[str] = ()
    max_tool_calls_per_turn: int | None = None
    parallel_tool_calls: bool | None = None
    structured_output: bool | None = None
    capabilities: ModelCapabilities | None = None
    pricing: ModelPricing | None = None
    verified_at: str | None = None
    replacement_model_id: str | None = None

    def __post_init__(self) -> None:
        provider = self.provider.strip()
        model_id = self.model_id.strip()
        if not provider or not model_id:
            raise ValidationError(
                "Model catalog provider and model_id must be non-empty."
            )
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model_id", model_id)
        for field_name in ("aliases", "recommended_for", "regions", "source_urls"):
            original = getattr(self, field_name)
            if isinstance(original, str):
                raise ValidationError(
                    f"ModelCatalogEntry.{field_name} must be a sequence, not a string."
                )
            values = tuple(original)
            if len(values) != len(set(values)):
                raise ValidationError(
                    f"ModelCatalogEntry.{field_name} cannot contain duplicates."
                )
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValidationError(
                    f"ModelCatalogEntry.{field_name} must contain non-empty strings."
                )
            object.__setattr__(self, field_name, values)
        if model_id in self.aliases:
            raise ValidationError("A canonical model_id cannot also be its own alias.")
        if self.api_surface not in _MODEL_API_SURFACES:
            raise ValidationError(
                f'Unsupported model api_surface "{self.api_surface}".'
            )
        if self.availability not in _MODEL_AVAILABILITIES:
            raise ValidationError(
                f'Unsupported model availability "{self.availability}".'
            )
        if self.support_evidence not in _MODEL_SUPPORT_EVIDENCE:
            raise ValidationError(
                f'Unsupported model support_evidence "{self.support_evidence}".'
            )
        invalid_recommendations = set(self.recommended_for) - _RECOMMENDED_USES
        if invalid_recommendations:
            raise ValidationError(
                f"Unsupported recommended_for values: {sorted(invalid_recommendations)}."
            )
        _validate_rate("ModelCatalogEntry.cost_per_1k_tokens", self.cost_per_1k_tokens)
        if self.cost_per_1k_tokens is not None and self.pricing is not None:
            raise ValidationError(
                "Use either legacy cost_per_1k_tokens or typed pricing, not both."
            )
        if (
            self.max_tool_calls_per_turn is not None
            and self.max_tool_calls_per_turn < 1
        ):
            raise ValidationError("max_tool_calls_per_turn must be positive when set.")
        if self.verified_at is not None:
            try:
                date.fromisoformat(self.verified_at)
            except ValueError as error:
                raise ValidationError(
                    "verified_at must use ISO YYYY-MM-DD format."
                ) from error
        if self.replacement_model_id == model_id:
            raise ValidationError("replacement_model_id must differ from model_id.")
        if (
            self.capabilities is not None
            and self.structured_output is not None
            and self.capabilities.structured_output != self.structured_output
        ):
            raise ValidationError(
                "structured_output conflicts with capabilities.structured_output."
            )
        if (
            self.capabilities is not None
            and self.parallel_tool_calls is not None
            and self.capabilities.parallel_tool_calls != self.parallel_tool_calls
        ):
            raise ValidationError(
                "parallel_tool_calls conflicts with capabilities.parallel_tool_calls."
            )


class ModelCatalog:
    def __init__(self, entries: Iterable[ModelCatalogEntry]) -> None:
        copied_entries = tuple(deepcopy(tuple(entries)))
        index: dict[tuple[str, str], ModelCatalogEntry] = {}
        for entry in copied_entries:
            if not isinstance(entry, ModelCatalogEntry):
                raise ValidationError(
                    "ModelCatalog entries must be ModelCatalogEntry instances."
                )
            for identifier in (entry.model_id, *entry.aliases):
                key = (entry.provider, identifier)
                if key in index:
                    existing = index[key]
                    raise ValidationError(
                        f'Catalog identifier collision for "{entry.provider}/{identifier}" '
                        f'between "{existing.model_id}" and "{entry.model_id}".'
                    )
                index[key] = entry
        self._entries = copied_entries
        self._index = index

    def find(
        self, provider: CatalogProviderId, model_id: str
    ) -> ModelCatalogEntry | None:
        entry = self._index.get((provider, model_id))
        return deepcopy(entry) if entry is not None else None

    def list(self) -> list[ModelCatalogEntry]:
        return list(deepcopy(self._entries))


def create_model_catalog(entries: Iterable[ModelCatalogEntry]) -> ModelCatalog:
    return ModelCatalog(entries)


_VERIFIED_AT = "2026-08-29"
_OPENAI_MODELS = ("https://developers.openai.com/api/docs/models",)
_AZURE_MODELS = (
    "https://learn.microsoft.com/en-us/azure/foundry/foundry-models/concepts/models-sold-directly-by-azure",
)
_ANTHROPIC_MODELS = ("https://platform.claude.com/docs/en/models/overview",)
_ANTHROPIC_PRICING = "https://platform.claude.com/docs/en/about-claude/pricing"
_GEMINI_MODELS = ("https://ai.google.dev/gemini-api/docs/models",)
_GEMINI_LIFECYCLE = (
    "https://ai.google.dev/gemini-api/docs/models",
    "https://ai.google.dev/gemini-api/docs/deprecations",
)
_GEMINI_PRICING = "https://ai.google.dev/gemini-api/docs/pricing"
_VERTEX_MODELS = ("https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models",)
_QWEN_MODELS = ("https://help.aliyun.com/en/model-studio/models",)
_QWEN_EMBEDDINGS = ("https://help.aliyun.com/en/model-studio/embedding",)
_QWEN_RERANK = ("https://help.aliyun.com/en/model-studio/embedding-rerank-model/",)
_KIMI_MODELS = ("https://platform.moonshot.ai/docs/guide/start-using-kimi-api",)
_DEEPSEEK_MODELS = ("https://api-docs.deepseek.com/quick_start/pricing",)
_META_MODELS = ("https://dev.meta.ai/docs/models",)
_BEDROCK_MODELS = (
    "https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-anthropic.html",
)


def _capabilities(
    *,
    streaming: bool = False,
    tools: bool = False,
    structured_output: bool = False,
    json_mode: bool = False,
    tool_choice: bool = False,
    parallel_tool_calls: bool = False,
    vision: bool = False,
    files: bool = False,
    audio_input: bool = False,
    audio_output: bool = False,
    embeddings: bool = False,
    reasoning: bool = False,
    web_search: bool = False,
    realtime: bool = False,
    realtime_audio_input: bool = False,
    realtime_audio_output: bool = False,
    realtime_tools: bool = False,
    realtime_browser_tokens: bool = False,
) -> ModelCapabilities:
    return ModelCapabilities(
        streaming=streaming,
        tools=tools,
        structured_output=structured_output,
        json_mode=json_mode,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        vision=vision,
        files=files,
        audio_input=audio_input,
        audio_output=audio_output,
        embeddings=embeddings,
        reasoning=reasoning,
        web_search=web_search,
        realtime=realtime,
        realtime_audio_input=realtime_audio_input,
        realtime_audio_output=realtime_audio_output,
        realtime_tools=realtime_tools,
        realtime_browser_tokens=realtime_browser_tokens,
    )


_OPENAI_LANGUAGE = _capabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    parallel_tool_calls=True,
    vision=True,
    files=True,
    reasoning=True,
)
_ANTHROPIC_LANGUAGE = _capabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    parallel_tool_calls=True,
    vision=True,
    files=True,
    reasoning=True,
)
_GEMINI_LANGUAGE = _capabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    vision=True,
    files=True,
    audio_input=True,
    embeddings=True,
    reasoning=True,
)
_QWEN_LANGUAGE = _capabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    vision=True,
    reasoning=True,
)
_KIMI_LANGUAGE = _capabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    vision=True,
    reasoning=True,
)
_DEEPSEEK_LANGUAGE = _capabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    reasoning=True,
)
_META_LANGUAGE = _capabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    parallel_tool_calls=True,
    vision=True,
    files=True,
    reasoning=True,
)
_LOCAL_LANGUAGE = _capabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    vision=True,
    reasoning=True,
)
_IMAGE = _capabilities(vision=True)
_EMBEDDING = _capabilities(embeddings=True)
_TRANSCRIPTION = _capabilities(audio_input=True)
_SPEECH = _capabilities(audio_output=True)
_REALTIME = _capabilities(
    streaming=True,
    tools=True,
    audio_input=True,
    audio_output=True,
    realtime=True,
    realtime_audio_input=True,
    realtime_audio_output=True,
    realtime_tools=True,
)
_VIDEO = _capabilities(vision=True)


def _entry(
    provider: CatalogProviderId,
    model_id: str,
    *,
    aliases: Sequence[str] = (),
    recommended_for: Sequence[RecommendedUse] = (),
    api_surface: ModelApiSurface = "language",
    availability: ModelAvailability = "stable",
    regions: Sequence[str] = (),
    support_evidence: ModelSupportEvidence = "catalog-only",
    source_urls: Sequence[str],
    capabilities: ModelCapabilities,
    pricing: ModelPricing | None = None,
    replacement_model_id: str | None = None,
    max_tool_calls_per_turn: int | None = None,
    parallel_tool_calls: bool | None = None,
    structured_output: bool | None = None,
    verified_at: str = _VERIFIED_AT,
) -> ModelCatalogEntry:
    return ModelCatalogEntry(
        provider=provider,
        model_id=model_id,
        aliases=aliases,
        recommended_for=recommended_for,
        api_surface=api_surface,
        availability=availability,
        regions=regions,
        support_evidence=support_evidence,
        source_urls=source_urls,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        parallel_tool_calls=parallel_tool_calls,
        structured_output=structured_output,
        capabilities=capabilities,
        pricing=pricing,
        verified_at=verified_at,
        replacement_model_id=replacement_model_id,
    )


def _usd(
    input_rate: float,
    output_rate: float,
    source_url: str,
    *,
    effective_from: str | None = None,
    effective_until: str | None = None,
) -> ModelPricing:
    return ModelPricing(
        currency="USD",
        source_url=source_url,
        input_per_1m_tokens=input_rate,
        output_per_1m_tokens=output_rate,
        effective_from=effective_from,
        effective_until=effective_until,
    )


default_model_catalog = create_model_catalog(
    [
        _entry(
            "deepseek", "deepseek-v4-flash-vision-exp",
            recommended_for=("reasoning", "tools", "vision"),
            availability="preview",
            source_urls=("https://api-docs.deepseek.com/guides/vision/",),
            capabilities=replace(_DEEPSEEK_LANGUAGE, vision=True),
            verified_at="2026-09-05",
        ),
        _entry(
            "openai",
            "gpt-6-astra",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            capabilities=_OPENAI_LANGUAGE,
            availability="stable",
            source_urls=("https://developers.openai.com/api/docs/models/gpt-6-astra",),
            verified_at="2026-09-05",
        ),
        _entry(
            "azure-openai",
            "gpt-6-astra",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            capabilities=_OPENAI_LANGUAGE,
            availability="stable",
            source_urls=("https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses",),
            verified_at="2026-09-05",
        ),
        _entry(
            "anthropic",
            "claude-fable-5-1",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            capabilities=_ANTHROPIC_LANGUAGE,
            availability="stable",
            source_urls=("https://platform.claude.com/docs/en/models/fable-5-1/overview",),
            verified_at="2026-09-05",
            pricing=_usd(10, 50, "https://platform.claude.com/docs/en/models/fable-5-1/overview"),
        ),
        _entry(
            "anthropic",
            "claude-mythos-5-1",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            capabilities=_ANTHROPIC_LANGUAGE,
            availability="limited",
            source_urls=("https://platform.claude.com/docs/en/release-notes/overview",),
            verified_at="2026-09-05",
            pricing=_usd(10, 50, "https://platform.claude.com/docs/en/release-notes/overview"),
        ),
        _entry(
            "gemini",
            "gemini-3.8-flash",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            capabilities=_GEMINI_LANGUAGE,
            availability="stable",
            source_urls=("https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash",),
            verified_at="2026-09-05",
            pricing=_usd(0.75, 3.75, "https://ai.google.dev/gemini-api/docs/latest-model", effective_from="2026-09-02", effective_until="2026-12-31"),
        ),
        _entry(
            "vertex",
            "gemini-3.8-flash",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            capabilities=_GEMINI_LANGUAGE,
            availability="stable",
            source_urls=("https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-flash",),
            verified_at="2026-09-05",
        ),
        _entry(
            "qwen",
            "qwen3.8-max-0902",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            capabilities=_QWEN_LANGUAGE,
            availability="stable",
            source_urls=("https://www.alibabacloud.com/help/en/model-studio/newly-released-models",),
            verified_at="2026-09-05",
            aliases=("qwen3.8-max-2026-09-02",),
            regions=("intl",),
        ),
        # OpenAI: aliases are limited to provider-declared moving aliases.
        _entry(
            "openai",
            "gpt-5.6-sol",
            aliases=("gpt-5.6",),
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=_OPENAI_MODELS,
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "openai",
            "gpt-5.6-terra",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=_OPENAI_MODELS,
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "openai",
            "gpt-5.6-luna",
            recommended_for=("chat", "speed", "tools", "vision"),
            source_urls=_OPENAI_MODELS,
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "openai",
            "gpt-5.6-cyber",
            recommended_for=("reasoning", "tools"),
            availability="limited",
            source_urls=(
                "https://developers.openai.com/api/docs/models/gpt-5.6-cyber",
            ),
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "openai",
            "gpt-5.5",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=("https://developers.openai.com/api/docs/models/gpt-5.5",),
            capabilities=_OPENAI_LANGUAGE,
            pricing=_usd(
                5, 30, "https://developers.openai.com/api/docs/models/gpt-5.5"
            ),
        ),
        _entry(
            "openai",
            "gpt-5.4",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=_OPENAI_MODELS,
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "openai",
            "gpt-5.4-mini",
            recommended_for=("chat", "tools", "speed"),
            source_urls=_OPENAI_MODELS,
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "openai",
            "gpt-5.4-nano",
            recommended_for=("speed", "tools"),
            source_urls=_OPENAI_MODELS,
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "openai",
            "gpt-4o-mini",
            recommended_for=("chat", "speed", "tools"),
            source_urls=("https://developers.openai.com/api/docs/models/gpt-4o-mini",),
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "openai",
            "gpt-image-2",
            recommended_for=("vision",),
            api_surface="image",
            source_urls=_OPENAI_MODELS,
            capabilities=_IMAGE,
        ),
        _entry(
            "openai",
            "gpt-image-1.5",
            recommended_for=("vision",),
            api_surface="image",
            availability="deprecated",
            source_urls=(
                "https://developers.openai.com/api/docs/models/gpt-image-1.5",
            ),
            capabilities=_IMAGE,
            replacement_model_id="gpt-image-2",
        ),
        _entry(
            "openai",
            "gpt-image-1",
            recommended_for=("vision",),
            api_surface="image",
            availability="deprecated",
            source_urls=_OPENAI_MODELS,
            capabilities=_IMAGE,
            replacement_model_id="gpt-image-2",
        ),
        _entry(
            "openai",
            "gpt-realtime-2.1",
            recommended_for=("speed", "audio", "realtime"),
            api_surface="realtime",
            source_urls=_OPENAI_MODELS,
            capabilities=_REALTIME,
        ),
        _entry(
            "openai",
            "gpt-realtime-2.1-mini",
            recommended_for=("speed", "audio", "realtime"),
            api_surface="realtime",
            source_urls=_OPENAI_MODELS,
            capabilities=_REALTIME,
        ),
        _entry(
            "openai",
            "gpt-realtime-2",
            recommended_for=("audio", "realtime"),
            api_surface="realtime",
            source_urls=(
                "https://developers.openai.com/api/docs/models/gpt-realtime-2",
            ),
            capabilities=_REALTIME,
        ),
        _entry(
            "openai",
            "gpt-realtime",
            recommended_for=("audio", "realtime"),
            api_surface="realtime",
            source_urls=_OPENAI_MODELS,
            capabilities=_REALTIME,
        ),
        _entry(
            "openai",
            "gpt-realtime-translate",
            recommended_for=("audio", "translation", "realtime"),
            api_surface="realtime",
            source_urls=_OPENAI_MODELS,
            capabilities=_REALTIME,
        ),
        _entry(
            "openai",
            "gpt-realtime-whisper",
            recommended_for=("audio",),
            api_surface="transcription",
            source_urls=_OPENAI_MODELS,
            capabilities=_TRANSCRIPTION,
        ),
        # Azure deployments are catalog references; regional prices stay application-owned.
        _entry(
            "azure-openai",
            "gpt-5.6-sol",
            aliases=("gpt-5.6",),
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=_AZURE_MODELS,
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "azure-openai",
            "gpt-5.6-terra",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=_AZURE_MODELS,
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "azure-openai",
            "gpt-5.6-luna",
            recommended_for=("chat", "speed", "tools", "vision"),
            source_urls=_AZURE_MODELS,
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "azure-openai",
            "gpt-chat-latest",
            aliases=("chat-latest",),
            recommended_for=("chat", "tools"),
            availability="preview",
            source_urls=_AZURE_MODELS,
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "azure-openai",
            "gpt-5.5",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=_AZURE_MODELS,
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "azure-openai",
            "gpt-5.4",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=_AZURE_MODELS,
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "azure-openai",
            "gpt-5.4-mini",
            recommended_for=("chat", "tools", "speed"),
            source_urls=_AZURE_MODELS,
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "azure-openai",
            "gpt-5.4-nano",
            recommended_for=("speed", "tools"),
            source_urls=_AZURE_MODELS,
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "azure-openai",
            "gpt-image-2",
            recommended_for=("vision",),
            api_surface="image",
            source_urls=_AZURE_MODELS,
            capabilities=_IMAGE,
        ),
        _entry(
            "azure-openai",
            "gpt-image-1.5",
            recommended_for=("vision",),
            api_surface="image",
            availability="deprecated",
            source_urls=_AZURE_MODELS,
            capabilities=_IMAGE,
            replacement_model_id="gpt-image-2",
        ),
        _entry(
            "azure-openai",
            "gpt-realtime-2.1",
            recommended_for=("speed", "audio", "realtime"),
            api_surface="realtime",
            source_urls=_AZURE_MODELS,
            capabilities=_REALTIME,
        ),
        _entry(
            "azure-openai",
            "gpt-realtime-2",
            recommended_for=("audio", "realtime"),
            api_surface="realtime",
            source_urls=_AZURE_MODELS,
            capabilities=_REALTIME,
        ),
        _entry(
            "azure-openai",
            "text-embedding-3-large",
            recommended_for=("embedding", "retrieval"),
            api_surface="embedding",
            source_urls=_AZURE_MODELS,
            capabilities=_EMBEDDING,
        ),
        _entry(
            "azure-openai",
            "text-embedding-3-small",
            recommended_for=("embedding", "retrieval"),
            api_surface="embedding",
            source_urls=_AZURE_MODELS,
            capabilities=_EMBEDDING,
        ),
        # Anthropic versions remain distinct models; prices retain input/output semantics.
        _entry(
            "anthropic",
            "claude-fable-5",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=_ANTHROPIC_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
            pricing=_usd(10, 50, _ANTHROPIC_PRICING),
        ),
        _entry(
            "anthropic",
            "claude-mythos-5",
            recommended_for=("reasoning", "tools", "vision"),
            availability="limited",
            source_urls=_ANTHROPIC_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
            pricing=_usd(10, 50, _ANTHROPIC_PRICING),
        ),
        _entry(
            "anthropic",
            "claude-opus-5",
            recommended_for=("reasoning", "tools", "vision"),
            source_urls=_ANTHROPIC_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
        ),
        _entry(
            "anthropic",
            "claude-opus-4-8",
            recommended_for=("reasoning", "tools", "vision"),
            source_urls=_ANTHROPIC_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
            pricing=_usd(5, 25, _ANTHROPIC_PRICING),
        ),
        _entry(
            "anthropic",
            "claude-opus-4-7",
            recommended_for=("reasoning", "tools", "vision"),
            source_urls=_ANTHROPIC_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
            pricing=_usd(5, 25, _ANTHROPIC_PRICING),
        ),
        _entry(
            "anthropic",
            "claude-opus-4-6",
            recommended_for=("reasoning", "tools", "vision"),
            source_urls=_ANTHROPIC_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
            pricing=_usd(5, 25, _ANTHROPIC_PRICING),
        ),
        _entry(
            "anthropic",
            "claude-sonnet-5",
            verified_at="2026-09-05",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=_ANTHROPIC_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
            pricing=_usd(
                2,
                10,
                "https://platform.claude.com/docs/en/models/sonnet-5/overview",
                effective_from="2026-09-05",
            ),
        ),
        _entry(
            "anthropic",
            "claude-sonnet-4-6",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=_ANTHROPIC_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
            pricing=_usd(3, 15, _ANTHROPIC_PRICING),
        ),
        _entry(
            "anthropic",
            "claude-haiku-4-5-20251001",
            aliases=("claude-haiku-4-5",),
            recommended_for=("speed", "tools", "vision"),
            source_urls=_ANTHROPIC_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
            pricing=_usd(1, 5, _ANTHROPIC_PRICING),
        ),
        # Gemini Developer API: retired preview IDs are lifecycle records, never aliases.
        _entry(
            "gemini",
            "gemini-3.7-flash",
            aliases=("gemini-flash-latest",),
            recommended_for=("chat", "reasoning", "speed", "tools", "vision"),
            source_urls=_GEMINI_MODELS,
            capabilities=_GEMINI_LANGUAGE,
            pricing=_usd(
                0.75,
                3.75,
                _GEMINI_PRICING,
                effective_from="2026-08-01",
                effective_until="2026-12-31",
            ),
        ),
        _entry(
            "gemini",
            "gemini-3.6-flash",
            recommended_for=("chat", "reasoning", "speed", "tools", "vision"),
            support_evidence="offline-contract",
            source_urls=_GEMINI_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        _entry(
            "gemini",
            "gemini-3.5-flash",
            recommended_for=("chat", "speed", "tools", "vision"),
            source_urls=_GEMINI_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        _entry(
            "gemini",
            "gemini-3.5-flash-lite",
            recommended_for=("chat", "reasoning", "speed", "tools", "vision"),
            support_evidence="offline-contract",
            source_urls=_GEMINI_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        _entry(
            "gemini",
            "gemini-3.1-pro-preview",
            aliases=("gemini-pro-latest",),
            recommended_for=("reasoning", "tools", "vision"),
            availability="preview",
            source_urls=_GEMINI_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        _entry(
            "gemini",
            "gemini-3.1-flash-lite",
            recommended_for=("speed", "tools", "vision"),
            source_urls=_GEMINI_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        _entry(
            "gemini",
            "gemini-3.1-flash-lite-preview",
            recommended_for=("speed", "tools", "vision"),
            availability="retired",
            source_urls=_GEMINI_LIFECYCLE,
            capabilities=_GEMINI_LANGUAGE,
            replacement_model_id="gemini-3.1-flash-lite",
        ),
        _entry(
            "gemini",
            "gemini-omni-1.1-flash",
            recommended_for=("chat", "speed", "tools", "vision", "audio"),
            api_surface="interactions",
            source_urls=_GEMINI_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        _entry(
            "gemini",
            "gemini-omni-flash-preview",
            recommended_for=("chat", "speed", "tools", "vision", "audio"),
            api_surface="interactions",
            availability="deprecated",
            source_urls=_GEMINI_LIFECYCLE,
            capabilities=_GEMINI_LANGUAGE,
            replacement_model_id="gemini-omni-1.1-flash",
        ),
        _entry(
            "gemini",
            "gemini-3.5-transcribe",
            recommended_for=("audio",),
            api_surface="transcription",
            source_urls=_GEMINI_MODELS,
            capabilities=_TRANSCRIPTION,
        ),
        _entry(
            "gemini",
            "gemini-3.5-transcribe-live",
            recommended_for=("audio", "realtime"),
            api_surface="transcription",
            source_urls=_GEMINI_MODELS,
            capabilities=_TRANSCRIPTION,
        ),
        _entry(
            "gemini",
            "gemini-3.5-live-translate-preview",
            recommended_for=("audio", "translation", "realtime"),
            api_surface="realtime",
            availability="preview",
            source_urls=_GEMINI_MODELS,
            capabilities=_REALTIME,
        ),
        _entry(
            "gemini",
            "gemini-3.1-flash-live-preview",
            recommended_for=("speed", "audio", "vision", "realtime"),
            api_surface="realtime",
            availability="preview",
            source_urls=_GEMINI_MODELS,
            capabilities=_REALTIME,
        ),
        _entry(
            "gemini",
            "gemini-3.1-flash-tts-preview",
            recommended_for=("audio",),
            api_surface="speech",
            availability="preview",
            source_urls=_GEMINI_MODELS,
            capabilities=_SPEECH,
        ),
        _entry(
            "gemini",
            "gemini-3.1-flash-image",
            recommended_for=("vision",),
            api_surface="image",
            source_urls=_GEMINI_MODELS,
            capabilities=_IMAGE,
        ),
        _entry(
            "gemini",
            "gemini-3.1-flash-image-preview",
            recommended_for=("vision",),
            api_surface="image",
            availability="retired",
            source_urls=_GEMINI_LIFECYCLE,
            capabilities=_IMAGE,
            replacement_model_id="gemini-3.1-flash-image",
        ),
        _entry(
            "gemini",
            "gemini-3.1-flash-lite-image",
            recommended_for=("speed", "vision"),
            api_surface="image",
            source_urls=_GEMINI_MODELS,
            capabilities=_IMAGE,
        ),
        _entry(
            "gemini",
            "gemini-3-pro-image",
            recommended_for=("reasoning", "vision"),
            api_surface="image",
            source_urls=_GEMINI_MODELS,
            capabilities=_IMAGE,
        ),
        _entry(
            "gemini",
            "gemini-3-pro-image-preview",
            recommended_for=("reasoning", "vision"),
            api_surface="image",
            availability="retired",
            source_urls=_GEMINI_LIFECYCLE,
            capabilities=_IMAGE,
            replacement_model_id="gemini-3-pro-image",
        ),
        _entry(
            "gemini",
            "veo-3.1-generate-preview",
            recommended_for=("vision",),
            api_surface="video",
            availability="preview",
            source_urls=_GEMINI_MODELS,
            capabilities=_VIDEO,
        ),
        _entry(
            "gemini",
            "veo-3.1-fast-generate-preview",
            recommended_for=("speed", "vision"),
            api_surface="video",
            availability="preview",
            source_urls=_GEMINI_MODELS,
            capabilities=_VIDEO,
        ),
        _entry(
            "gemini",
            "lyria-3-pro-preview",
            recommended_for=("audio",),
            api_surface="media",
            availability="preview",
            source_urls=_GEMINI_MODELS,
            capabilities=_SPEECH,
        ),
        _entry(
            "gemini",
            "lyria-3-clip-preview",
            recommended_for=("audio",),
            api_surface="media",
            availability="preview",
            source_urls=_GEMINI_MODELS,
            capabilities=_SPEECH,
        ),
        _entry(
            "gemini",
            "gemini-2.5-flash",
            recommended_for=("speed", "tools", "vision"),
            source_urls=_GEMINI_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        _entry(
            "gemini",
            "gemini-2.5-pro",
            recommended_for=("reasoning", "tools", "vision"),
            source_urls=_GEMINI_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        # Vertex mirrors only models documented on that host; Omni remains Gemini-only.
        _entry(
            "vertex",
            "gemini-3.7-flash",
            aliases=("gemini-flash-latest",),
            recommended_for=("chat", "reasoning", "speed", "tools", "vision"),
            regions=("global", "us", "eu"),
            source_urls=(
                "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-7-flash",
            ),
            capabilities=_GEMINI_LANGUAGE,
        ),
        _entry(
            "vertex",
            "gemini-3.6-flash",
            recommended_for=("chat", "reasoning", "speed", "tools", "vision"),
            regions=("global",),
            support_evidence="offline-contract",
            source_urls=_VERTEX_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        _entry(
            "vertex",
            "gemini-3.5-flash",
            recommended_for=("chat", "speed", "tools", "vision"),
            regions=("global", "us", "eu"),
            source_urls=_VERTEX_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        _entry(
            "vertex",
            "gemini-3.5-flash-lite",
            recommended_for=("chat", "reasoning", "speed", "tools", "vision"),
            regions=("global", "us", "eu"),
            support_evidence="offline-contract",
            source_urls=_VERTEX_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        _entry(
            "vertex",
            "gemini-3.1-pro-preview",
            aliases=("gemini-pro-latest",),
            recommended_for=("reasoning", "tools", "vision"),
            availability="preview",
            regions=("global",),
            source_urls=_VERTEX_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        _entry(
            "vertex",
            "gemini-3.1-flash-lite",
            recommended_for=("speed", "tools", "vision"),
            regions=("global", "us", "eu"),
            source_urls=_VERTEX_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        _entry(
            "vertex",
            "gemini-3.1-flash-lite-preview",
            recommended_for=("speed", "tools", "vision"),
            availability="retired",
            regions=("global",),
            source_urls=_GEMINI_LIFECYCLE,
            capabilities=_GEMINI_LANGUAGE,
            replacement_model_id="gemini-3.1-flash-lite",
        ),
        _entry(
            "vertex",
            "gemini-3.1-flash-live-preview",
            recommended_for=("speed", "audio", "vision", "realtime"),
            api_surface="realtime",
            availability="preview",
            regions=("global",),
            source_urls=_VERTEX_MODELS,
            capabilities=_REALTIME,
        ),
        _entry(
            "vertex",
            "gemini-3.1-flash-tts-preview",
            recommended_for=("audio",),
            api_surface="speech",
            availability="preview",
            regions=("global",),
            source_urls=_VERTEX_MODELS,
            capabilities=_SPEECH,
        ),
        _entry(
            "vertex",
            "gemini-3.1-flash-image",
            recommended_for=("vision",),
            api_surface="image",
            regions=("global",),
            source_urls=_VERTEX_MODELS,
            capabilities=_IMAGE,
        ),
        _entry(
            "vertex",
            "gemini-3.1-flash-image-preview",
            recommended_for=("vision",),
            api_surface="image",
            availability="retired",
            regions=("global",),
            source_urls=_GEMINI_LIFECYCLE,
            capabilities=_IMAGE,
            replacement_model_id="gemini-3.1-flash-image",
        ),
        _entry(
            "vertex",
            "gemini-3-pro-image",
            recommended_for=("reasoning", "vision"),
            api_surface="image",
            regions=("global",),
            source_urls=_VERTEX_MODELS,
            capabilities=_IMAGE,
        ),
        _entry(
            "vertex",
            "gemini-3-pro-image-preview",
            recommended_for=("reasoning", "vision"),
            api_surface="image",
            availability="retired",
            regions=("global",),
            source_urls=_GEMINI_LIFECYCLE,
            capabilities=_IMAGE,
            replacement_model_id="gemini-3-pro-image",
        ),
        _entry(
            "vertex",
            "veo-3.1-generate-preview",
            recommended_for=("vision",),
            api_surface="video",
            availability="preview",
            regions=("global",),
            source_urls=_VERTEX_MODELS,
            capabilities=_VIDEO,
        ),
        _entry(
            "vertex",
            "veo-3.1-fast-generate-preview",
            recommended_for=("speed", "vision"),
            api_surface="video",
            availability="preview",
            regions=("global",),
            source_urls=_VERTEX_MODELS,
            capabilities=_VIDEO,
        ),
        _entry(
            "vertex",
            "lyria-3-pro-preview",
            recommended_for=("audio",),
            api_surface="media",
            availability="preview",
            regions=("global",),
            source_urls=_VERTEX_MODELS,
            capabilities=_SPEECH,
        ),
        _entry(
            "vertex",
            "lyria-3-clip-preview",
            recommended_for=("audio",),
            api_surface="media",
            availability="preview",
            regions=("global",),
            source_urls=_VERTEX_MODELS,
            capabilities=_SPEECH,
        ),
        _entry(
            "vertex",
            "gemini-2.5-flash",
            recommended_for=("speed", "tools", "vision"),
            regions=("global", "us", "eu"),
            source_urls=_VERTEX_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        _entry(
            "vertex",
            "gemini-2.5-pro",
            recommended_for=("reasoning", "tools", "vision"),
            regions=("global", "us", "eu"),
            source_urls=_VERTEX_MODELS,
            capabilities=_GEMINI_LANGUAGE,
        ),
        # Qwen pricing is regional and tiered, so the default catalog never invents one scalar.
        _entry(
            "qwen",
            "qwen3.8-max",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            regions=("cn", "intl", "us"),
            support_evidence="offline-contract",
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3.8-flash",
            recommended_for=("chat", "speed", "tools", "vision"),
            regions=("cn", "intl", "us"),
            source_urls=("https://help.aliyun.com/en/model-studio/qwen3-8-flash",),
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3.8-max-preview",
            recommended_for=("reasoning", "tools", "vision"),
            availability="preview",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3.7-max",
            recommended_for=("reasoning", "tools"),
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3.7-max-preview",
            recommended_for=("reasoning", "tools"),
            availability="preview",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3.7-max-2026-05-20",
            recommended_for=("reasoning", "tools"),
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3.7-max-2026-06-08",
            recommended_for=("reasoning", "tools", "vision"),
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3.7-plus",
            recommended_for=("chat", "tools", "reasoning", "vision"),
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3.7-plus-2026-05-26",
            recommended_for=("chat", "tools", "reasoning", "vision"),
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3.6-plus",
            recommended_for=("chat", "tools", "reasoning", "vision"),
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3.6-plus-2026-04-02",
            recommended_for=("chat", "tools", "reasoning", "vision"),
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3.6-flash",
            recommended_for=("speed", "tools", "vision"),
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3.5-plus",
            recommended_for=("chat", "tools", "reasoning", "vision"),
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3.5-plus-2026-04-20",
            recommended_for=("chat", "tools", "reasoning", "vision"),
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen-plus",
            recommended_for=("chat", "tools", "reasoning"),
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen-flash",
            recommended_for=("speed", "tools"),
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3-coder-plus",
            recommended_for=("tools", "reasoning"),
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3-coder-flash",
            recommended_for=("speed", "tools"),
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_QWEN_LANGUAGE,
        ),
        _entry(
            "qwen",
            "qwen3.7-text-embedding",
            recommended_for=("embedding", "retrieval"),
            api_surface="embedding",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_EMBEDDINGS,
            capabilities=_EMBEDDING,
        ),
        _entry(
            "qwen",
            "qwen3.7-text-embedding-flash",
            recommended_for=("embedding", "retrieval", "speed"),
            api_surface="embedding",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_EMBEDDINGS,
            capabilities=_EMBEDDING,
        ),
        _entry(
            "qwen",
            "text-embedding-v4",
            recommended_for=("embedding", "retrieval"),
            api_surface="embedding",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_EMBEDDINGS,
            capabilities=_EMBEDDING,
        ),
        _entry(
            "qwen",
            "text-embedding-v3",
            recommended_for=("embedding", "retrieval"),
            api_surface="embedding",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_EMBEDDINGS,
            capabilities=_EMBEDDING,
        ),
        _entry(
            "qwen",
            "tongyi-embedding-vision-plus",
            recommended_for=("embedding", "retrieval", "vision"),
            api_surface="embedding",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_EMBEDDINGS,
            capabilities=_EMBEDDING,
        ),
        _entry(
            "qwen",
            "tongyi-embedding-vision-flash",
            recommended_for=("embedding", "retrieval", "vision", "speed"),
            api_surface="embedding",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_EMBEDDINGS,
            capabilities=_EMBEDDING,
        ),
        _entry(
            "qwen",
            "qwen3-rerank",
            recommended_for=("retrieval",),
            api_surface="rerank",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_RERANK,
            capabilities=_EMBEDDING,
        ),
        _entry(
            "qwen",
            "qwen3-asr-flash",
            recommended_for=("speed", "audio"),
            api_surface="transcription",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_TRANSCRIPTION,
        ),
        _entry(
            "qwen",
            "qwen3-asr-flash-2026-02-10",
            recommended_for=("speed", "audio"),
            api_surface="transcription",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_TRANSCRIPTION,
        ),
        _entry(
            "qwen",
            "qwen3-asr-flash-realtime",
            recommended_for=("speed", "audio", "realtime"),
            api_surface="transcription",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_TRANSCRIPTION,
        ),
        _entry(
            "qwen",
            "qwen3-tts-flash",
            recommended_for=("speed", "audio"),
            api_surface="speech",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_SPEECH,
        ),
        _entry(
            "qwen",
            "qwen3-tts-instruct-flash",
            recommended_for=("audio",),
            api_surface="speech",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_SPEECH,
        ),
        _entry(
            "qwen",
            "qwen3-tts-instruct-flash-2026-01-26",
            recommended_for=("audio",),
            api_surface="speech",
            regions=("cn", "intl", "us"),
            source_urls=_QWEN_MODELS,
            capabilities=_SPEECH,
        ),
        # Kimi and DeepSeek lifecycle entries preserve exact IDs without unsafe remapping.
        _entry(
            "kimi",
            "kimi-k3",
            recommended_for=("reasoning", "tools", "vision"),
            source_urls=(
                "https://forum.moonshot.ai/t/kimi-k3-is-here-our-most-capable-model/480",
            ),
            capabilities=_KIMI_LANGUAGE,
        ),
        _entry(
            "kimi",
            "kimi-k2.6",
            recommended_for=("reasoning", "tools", "vision"),
            source_urls=(
                "https://forum.moonshot.ai/t/meet-kimi-k2-6-advancing-open-source-coding/369",
            ),
            capabilities=_KIMI_LANGUAGE,
        ),
        _entry(
            "kimi",
            "kimi-k2.5",
            recommended_for=("reasoning", "tools", "vision"),
            source_urls=_KIMI_MODELS,
            capabilities=_KIMI_LANGUAGE,
        ),
        _entry(
            "kimi",
            "kimi-k2",
            recommended_for=("reasoning", "tools"),
            source_urls=_KIMI_MODELS,
            capabilities=_KIMI_LANGUAGE,
        ),
        _entry(
            "kimi",
            "moonshot-v1-8k",
            recommended_for=("chat",),
            availability="deprecated",
            source_urls=_KIMI_MODELS,
            capabilities=_KIMI_LANGUAGE,
            replacement_model_id="kimi-k2.6",
        ),
        _entry(
            "kimi",
            "moonshot-v1-32k",
            recommended_for=("chat",),
            availability="deprecated",
            source_urls=_KIMI_MODELS,
            capabilities=_KIMI_LANGUAGE,
            replacement_model_id="kimi-k2.6",
        ),
        _entry(
            "kimi",
            "moonshot-v1-128k",
            recommended_for=("chat",),
            availability="deprecated",
            source_urls=_KIMI_MODELS,
            capabilities=_KIMI_LANGUAGE,
            replacement_model_id="kimi-k2.6",
        ),
        _entry(
            "deepseek",
            "deepseek-v4-pro",
            recommended_for=("chat", "reasoning", "tools"),
            source_urls=_DEEPSEEK_MODELS,
            capabilities=_DEEPSEEK_LANGUAGE,
        ),
        _entry(
            "deepseek",
            "deepseek-v4-flash",
            recommended_for=("chat", "speed", "reasoning", "tools"),
            source_urls=_DEEPSEEK_MODELS,
            capabilities=_DEEPSEEK_LANGUAGE,
        ),
        _entry(
            "deepseek",
            "deepseek-chat",
            recommended_for=("chat",),
            availability="retired",
            source_urls=_DEEPSEEK_MODELS,
            capabilities=_DEEPSEEK_LANGUAGE,
            replacement_model_id="deepseek-v4-flash",
        ),
        _entry(
            "deepseek",
            "deepseek-reasoner",
            recommended_for=("reasoning",),
            availability="retired",
            source_urls=_DEEPSEEK_MODELS,
            capabilities=_DEEPSEEK_LANGUAGE,
            replacement_model_id="deepseek-v4-pro",
        ),
        # Model author, API host and support evidence remain separate dimensions.
        _entry(
            "meta",
            "muse-spark-1.2",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            support_evidence="offline-contract",
            source_urls=(
                "https://dev.meta.ai/docs/models",
                "https://developer.meta.com/ai/resources/blog/build-with-muse-code/",
            ),
            capabilities=_META_LANGUAGE,
            parallel_tool_calls=True,
            structured_output=True,
        ),
        _entry(
            "meta",
            "muse-spark-1.2-contributor",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            availability="preview",
            source_urls=(
                "https://dev.meta.ai/docs/models",
                "https://dev.meta.ai/docs/pricing-rate-limits",
            ),
            capabilities=_META_LANGUAGE,
            parallel_tool_calls=True,
            structured_output=True,
        ),
        _entry(
            "meta",
            "muse-spark-1.1",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            availability="preview",
            source_urls=_META_MODELS,
            capabilities=_META_LANGUAGE,
            parallel_tool_calls=True,
            structured_output=True,
        ),
        _entry(
            "openrouter",
            "openai/gpt-5.4-mini",
            recommended_for=("chat", "tools", "speed"),
            source_urls=("https://openrouter.ai/openai/gpt-5.4-mini",),
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "openrouter",
            "openai/gpt-4o-mini",
            recommended_for=("chat", "tools", "speed"),
            source_urls=("https://openrouter.ai/openai/gpt-4o-mini",),
            capabilities=_OPENAI_LANGUAGE,
        ),
        _entry(
            "openrouter",
            "meta/muse-spark-1.2",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            availability="preview",
            source_urls=(
                "https://developer.meta.com/ai/resources/blog/build-with-muse-code/",
                "https://openrouter.ai/meta/muse-spark-1.2",
            ),
            capabilities=_META_LANGUAGE,
        ),
        _entry(
            "openrouter",
            "meta/muse-glimmer-30b",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=(
                "https://developer.meta.com/ai/models/muse-glimmer/",
                "https://openrouter.ai/meta/muse-glimmer-30b",
            ),
            capabilities=_LOCAL_LANGUAGE,
            max_tool_calls_per_turn=1,
            parallel_tool_calls=False,
        ),
        _entry(
            "bedrock",
            "anthropic.claude-fable-5",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=_BEDROCK_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
        ),
        _entry(
            "bedrock",
            "anthropic.claude-mythos-5",
            recommended_for=("reasoning", "tools", "vision"),
            availability="limited",
            source_urls=_BEDROCK_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
        ),
        _entry(
            "bedrock",
            "anthropic.claude-opus-5",
            recommended_for=("reasoning", "tools", "vision"),
            source_urls=_BEDROCK_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
        ),
        _entry(
            "bedrock",
            "anthropic.claude-opus-4-8",
            recommended_for=("reasoning", "tools", "vision"),
            source_urls=_BEDROCK_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
        ),
        _entry(
            "bedrock",
            "anthropic.claude-opus-4-7",
            recommended_for=("reasoning", "tools", "vision"),
            source_urls=_BEDROCK_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
        ),
        _entry(
            "bedrock",
            "anthropic.claude-sonnet-5",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=_BEDROCK_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
        ),
        _entry(
            "bedrock",
            "anthropic.claude-sonnet-4-6",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=_BEDROCK_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
        ),
        _entry(
            "bedrock",
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            recommended_for=("speed", "tools", "vision"),
            source_urls=_BEDROCK_MODELS,
            capabilities=_ANTHROPIC_LANGUAGE,
        ),
        _entry(
            "bedrock",
            "amazon.nova-premier-v1:0",
            recommended_for=("reasoning", "tools", "vision"),
            source_urls=(
                "https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html",
            ),
            capabilities=_LOCAL_LANGUAGE,
        ),
        _entry(
            "bedrock",
            "amazon.nova-pro-v1:0",
            recommended_for=("chat", "tools", "vision"),
            source_urls=(
                "https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html",
            ),
            capabilities=_LOCAL_LANGUAGE,
        ),
        # Self-hosted routes omit token prices because local compute is not free or portable.
        _entry(
            "ollama",
            "llama3.2",
            recommended_for=("chat", "speed"),
            source_urls=("https://ollama.com/library/llama3.2",),
            capabilities=_LOCAL_LANGUAGE,
        ),
        _entry(
            "ollama",
            "muse-glimmer:30b",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=("https://ollama.com/library/muse-glimmer",),
            capabilities=_LOCAL_LANGUAGE,
            max_tool_calls_per_turn=1,
            parallel_tool_calls=False,
        ),
        _entry(
            "ollama",
            "muse-glimmer:30b-mlx",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=("https://ollama.com/library/muse-glimmer",),
            capabilities=_LOCAL_LANGUAGE,
            max_tool_calls_per_turn=1,
            parallel_tool_calls=False,
        ),
        _entry(
            "vllm",
            "meta-models/Muse-Glimmer-30B",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=(
                "https://dev.meta.ai/docs/muse-glimmer/get-the-model",
                "https://dev.meta.ai/docs/muse-glimmer/vllm",
            ),
            capabilities=_LOCAL_LANGUAGE,
            max_tool_calls_per_turn=1,
            parallel_tool_calls=False,
        ),
        _entry(
            "vllm",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            recommended_for=("chat", "tools", "vision"),
            source_urls=(
                "https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md",
            ),
            capabilities=_LOCAL_LANGUAGE,
        ),
        _entry(
            "vllm",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            recommended_for=("chat", "reasoning", "tools", "vision"),
            source_urls=(
                "https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md",
            ),
            capabilities=_LOCAL_LANGUAGE,
        ),
    ]
)
