from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CatalogProviderId = str
RecommendedUse = Literal["chat", "reasoning", "speed", "vision", "tools"]


@dataclass(slots=True)
class ModelCatalogEntry:
    provider: CatalogProviderId
    model_id: str
    aliases: list[str] = field(default_factory=list)
    cost_per_1k_tokens: float | None = None
    recommended_for: list[RecommendedUse] = field(default_factory=list)


class ModelCatalog:
    def __init__(self, entries: list[ModelCatalogEntry]) -> None:
        self._entries = entries

    def find(self, provider: CatalogProviderId, model_id: str) -> ModelCatalogEntry | None:
        for entry in self._entries:
            if entry.provider == provider and (entry.model_id == model_id or model_id in entry.aliases):
                return entry
        return None

    def list(self) -> list[ModelCatalogEntry]:
        return list(self._entries)


def create_model_catalog(entries: list[ModelCatalogEntry]) -> ModelCatalog:
    return ModelCatalog(entries)


default_model_catalog = create_model_catalog(
    [
        ModelCatalogEntry("openai", "gpt-4o-mini", cost_per_1k_tokens=0.6, recommended_for=["chat", "tools", "speed"]),
        ModelCatalogEntry("azure-openai", "gpt-4o-mini", cost_per_1k_tokens=0.6, recommended_for=["chat", "tools"]),
        ModelCatalogEntry("anthropic", "claude-3-5-sonnet", cost_per_1k_tokens=3, recommended_for=["reasoning", "tools"]),
        ModelCatalogEntry("gemini", "gemini-2.0-flash", cost_per_1k_tokens=0.35, recommended_for=["speed", "vision"]),
        ModelCatalogEntry("vertex", "gemini-2.0-flash", cost_per_1k_tokens=0.35, recommended_for=["speed", "vision"]),
        ModelCatalogEntry("qwen", "qwen-plus", cost_per_1k_tokens=0.8, recommended_for=["chat", "tools", "reasoning"]),
        ModelCatalogEntry("kimi", "kimi-k2-0905-preview", cost_per_1k_tokens=2, recommended_for=["reasoning", "tools"]),
        ModelCatalogEntry("openrouter", "openai/gpt-4o-mini", cost_per_1k_tokens=0.7, recommended_for=["chat", "tools"]),
        ModelCatalogEntry("bedrock", "anthropic.claude-3-5-sonnet", cost_per_1k_tokens=3, recommended_for=["reasoning"]),
        ModelCatalogEntry("ollama", "llama3.2", cost_per_1k_tokens=0, recommended_for=["chat", "speed"]),
    ]
)
