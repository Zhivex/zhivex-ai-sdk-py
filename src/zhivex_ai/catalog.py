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
        ModelCatalogEntry("openai", "gpt-5.5", cost_per_1k_tokens=5, recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("openai", "gpt-5.4", cost_per_1k_tokens=2.5, recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("openai", "gpt-5.4-mini", aliases=["gpt-4o-mini"], recommended_for=["chat", "tools", "speed"]),
        ModelCatalogEntry("openai", "gpt-5.4-nano", recommended_for=["speed", "tools"]),
        ModelCatalogEntry("openai", "gpt-realtime-1.5", recommended_for=["speed"]),
        ModelCatalogEntry("azure-openai", "gpt-5.4-mini", aliases=["gpt-4o-mini"], recommended_for=["chat", "tools"]),
        ModelCatalogEntry("azure-openai", "gpt-5.4", recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("anthropic", "claude-opus-4-7", cost_per_1k_tokens=5, recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry("anthropic", "claude-sonnet-4-6", cost_per_1k_tokens=3, recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("anthropic", "claude-haiku-4-5-20251001", aliases=["claude-haiku-4-5"], cost_per_1k_tokens=1, recommended_for=["speed", "tools", "vision"]),
        ModelCatalogEntry("anthropic", "claude-sonnet-4-20250514", aliases=["claude-sonnet-4"], cost_per_1k_tokens=3, recommended_for=["reasoning", "tools"]),
        ModelCatalogEntry("gemini", "gemini-3.1-pro-preview", aliases=["gemini-pro-latest"], recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry("gemini", "gemini-3-flash-preview", aliases=["gemini-flash-latest"], recommended_for=["chat", "speed", "tools", "vision"]),
        ModelCatalogEntry("gemini", "gemini-3.1-flash-lite-preview", recommended_for=["speed", "tools", "vision"]),
        ModelCatalogEntry("gemini", "gemini-2.5-flash", cost_per_1k_tokens=0.35, recommended_for=["speed", "tools", "vision"]),
        ModelCatalogEntry("gemini", "gemini-2.5-pro", recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry("vertex", "gemini-3.1-pro-preview", aliases=["gemini-pro-latest"], recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry("vertex", "gemini-3-flash-preview", aliases=["gemini-flash-latest"], recommended_for=["chat", "speed", "tools", "vision"]),
        ModelCatalogEntry("vertex", "gemini-2.5-flash", cost_per_1k_tokens=0.35, recommended_for=["speed", "tools", "vision"]),
        ModelCatalogEntry("vertex", "gemini-2.5-pro", recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry("qwen", "qwen3.5-plus", aliases=["qwen3.5-plus-2026-02-15"], cost_per_1k_tokens=0.4, recommended_for=["chat", "tools", "reasoning", "vision"]),
        ModelCatalogEntry("qwen", "qwen3.5-flash", aliases=["qwen3.5-flash-2026-02-23"], cost_per_1k_tokens=0.1, recommended_for=["speed", "tools", "vision"]),
        ModelCatalogEntry("qwen", "qwen3-max", aliases=["qwen3-max-2026-01-23", "qwen3-max-preview"], cost_per_1k_tokens=1.2, recommended_for=["reasoning", "tools"]),
        ModelCatalogEntry("qwen", "qwen-plus", cost_per_1k_tokens=0.4, recommended_for=["chat", "tools", "reasoning"]),
        ModelCatalogEntry("qwen", "qwen-flash", cost_per_1k_tokens=0.05, recommended_for=["speed", "tools"]),
        ModelCatalogEntry("qwen", "qwen-max-latest", aliases=["qwen-max", "qwen-max-2025-01-25"], cost_per_1k_tokens=1.6, recommended_for=["reasoning", "tools"]),
        ModelCatalogEntry("qwen", "qwen3-coder-plus", aliases=["qwen3-coder-plus-2025-09-23", "qwen3-coder-plus-2025-07-22"], cost_per_1k_tokens=0.574, recommended_for=["tools", "reasoning"]),
        ModelCatalogEntry("qwen", "qwen3-coder-flash", aliases=["qwen3-coder-flash-2025-07-28"], cost_per_1k_tokens=0.144, recommended_for=["speed", "tools"]),
        ModelCatalogEntry("qwen", "qwen3-asr-flash", aliases=["qwen3-asr-flash-2025-09-08", "qwen3-asr-flash-us", "qwen3-asr-flash-2025-09-08-us"], recommended_for=["speed"]),
        ModelCatalogEntry("qwen", "qwen3-tts-flash", aliases=["qwen3-tts-flash-2025-11-27", "qwen3-tts-flash-2025-09-18"], recommended_for=["speed"]),
        ModelCatalogEntry(
            "kimi",
            "kimi-k2.6",
            aliases=[
                "kimi-k2.5",
                "kimi-k2",
                "kimi-k2-0905-preview",
                "kimi-k2-thinking",
                "kimi-k2-thinking-turbo",
                "moonshot-v1-8k",
                "moonshot-v1-32k",
                "moonshot-v1-128k",
            ],
            cost_per_1k_tokens=4,
            recommended_for=["reasoning", "tools", "vision"],
        ),
        ModelCatalogEntry("openrouter", "openai/gpt-5.4-mini", aliases=["openai/gpt-4o-mini"], recommended_for=["chat", "tools"]),
        ModelCatalogEntry("bedrock", "anthropic.claude-opus-4-6", recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry("bedrock", "anthropic.claude-sonnet-4-6", cost_per_1k_tokens=3, recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("bedrock", "anthropic.claude-haiku-4-5-20251001-v1:0", cost_per_1k_tokens=1, recommended_for=["speed", "tools", "vision"]),
        ModelCatalogEntry("bedrock", "amazon.nova-premier-v1:0", recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry("bedrock", "amazon.nova-pro-v1:0", recommended_for=["chat", "tools", "vision"]),
        ModelCatalogEntry("bedrock", "anthropic.claude-3-5-sonnet", cost_per_1k_tokens=3, recommended_for=["reasoning"]),
        ModelCatalogEntry("ollama", "llama3.2", cost_per_1k_tokens=0, recommended_for=["chat", "speed"]),
    ]
)
