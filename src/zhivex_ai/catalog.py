from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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
ModelAvailability = Literal["stable", "preview"]
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


@dataclass(slots=True)
class ModelCatalogEntry:
    provider: CatalogProviderId
    model_id: str
    aliases: list[str] = field(default_factory=list)
    cost_per_1k_tokens: float | None = None
    recommended_for: list[RecommendedUse] = field(default_factory=list)
    api_surface: ModelApiSurface = "language"
    availability: ModelAvailability = "stable"
    regions: list[str] = field(default_factory=list)
    support_evidence: ModelSupportEvidence = "catalog-only"
    source_urls: list[str] = field(default_factory=list)
    max_tool_calls_per_turn: int | None = None
    parallel_tool_calls: bool | None = None
    structured_output: bool | None = None


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
        ModelCatalogEntry("openai", "gpt-5.6-sol", aliases=["gpt-5.6"], recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("openai", "gpt-5.6-terra", recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("openai", "gpt-5.6-luna", recommended_for=["chat", "speed", "tools", "vision"]),
        ModelCatalogEntry("openai", "gpt-5.5", cost_per_1k_tokens=5, recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("openai", "gpt-5.4", cost_per_1k_tokens=2.5, recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("openai", "gpt-5.4-mini", aliases=["gpt-4o-mini"], recommended_for=["chat", "tools", "speed"]),
        ModelCatalogEntry("openai", "gpt-5.4-nano", recommended_for=["speed", "tools"]),
        ModelCatalogEntry("openai", "gpt-image-2", aliases=["gpt-image-1.5", "gpt-image-1"], recommended_for=["vision"], api_surface="image"),
        ModelCatalogEntry("openai", "gpt-realtime-2.1", aliases=["gpt-realtime-2", "gpt-realtime-1.5", "gpt-realtime"], recommended_for=["speed", "audio"], api_surface="realtime"),
        ModelCatalogEntry("openai", "gpt-realtime-2.1-mini", recommended_for=["speed", "audio"], api_surface="realtime"),
        ModelCatalogEntry("openai", "gpt-realtime-translate", recommended_for=["audio"], api_surface="realtime"),
        ModelCatalogEntry("openai", "gpt-realtime-whisper", recommended_for=["audio"], api_surface="transcription"),
        ModelCatalogEntry("azure-openai", "gpt-5.6-sol", aliases=["gpt-5.6"], recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("azure-openai", "gpt-5.6-terra", recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("azure-openai", "gpt-5.6-luna", recommended_for=["chat", "speed", "tools", "vision"]),
        ModelCatalogEntry("azure-openai", "gpt-5.5", cost_per_1k_tokens=5, recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("azure-openai", "gpt-chat-latest", aliases=["chat-latest", "gpt-5.5-instant"], recommended_for=["chat", "tools"]),
        ModelCatalogEntry("azure-openai", "gpt-5.4-mini", aliases=["gpt-4o-mini"], recommended_for=["chat", "tools"]),
        ModelCatalogEntry("azure-openai", "gpt-5.4", recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("azure-openai", "gpt-image-2", aliases=["gpt-image-1.5", "gpt-image-1"], recommended_for=["vision"], api_surface="image"),
        ModelCatalogEntry("azure-openai", "gpt-realtime-2.1", aliases=["gpt-realtime-2", "gpt-realtime-1.5", "gpt-realtime"], recommended_for=["speed", "audio"], api_surface="realtime"),
        ModelCatalogEntry("azure-openai", "gpt-realtime-2.1-mini", recommended_for=["speed", "audio"], api_surface="realtime"),
        ModelCatalogEntry("azure-openai", "text-embedding-3-large", recommended_for=["embedding", "retrieval"], api_surface="embedding"),
        ModelCatalogEntry("azure-openai", "text-embedding-3-small", recommended_for=["embedding", "retrieval"], api_surface="embedding"),
        ModelCatalogEntry("anthropic", "claude-fable-5", cost_per_1k_tokens=5, recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("anthropic", "claude-mythos-5", recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry("anthropic", "claude-opus-5", cost_per_1k_tokens=5, recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry("anthropic", "claude-opus-4-8", aliases=["claude-opus-4-7"], cost_per_1k_tokens=5, recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry("anthropic", "claude-sonnet-5", cost_per_1k_tokens=3, recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("anthropic", "claude-sonnet-4-6", cost_per_1k_tokens=3, recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("anthropic", "claude-haiku-4-5-20251001", aliases=["claude-haiku-4-5"], cost_per_1k_tokens=1, recommended_for=["speed", "tools", "vision"]),
        ModelCatalogEntry(
            "gemini",
            "gemini-3.6-flash",
            recommended_for=["chat", "reasoning", "speed", "tools", "vision"],
            support_evidence="offline-contract",
            source_urls=["https://ai.google.dev/gemini-api/docs/latest-model"],
        ),
        ModelCatalogEntry(
            "gemini",
            "gemini-3.5-flash-lite",
            recommended_for=["chat", "reasoning", "speed", "tools", "vision"],
            support_evidence="offline-contract",
            source_urls=["https://ai.google.dev/gemini-api/docs/latest-model"],
        ),
        ModelCatalogEntry("gemini", "gemini-3.1-pro-preview", aliases=["gemini-pro-latest"], recommended_for=["reasoning", "tools", "vision"], availability="preview"),
        ModelCatalogEntry("gemini", "gemini-3.5-flash", aliases=["gemini-flash-latest"], recommended_for=["chat", "speed", "tools", "vision"]),
        ModelCatalogEntry("gemini", "gemini-omni-flash-preview", recommended_for=["chat", "speed", "tools", "vision", "audio"], api_surface="interactions", availability="preview"),
        ModelCatalogEntry("gemini", "gemini-3-flash-preview", recommended_for=["chat", "speed", "tools", "vision"], availability="preview"),
        ModelCatalogEntry("gemini", "gemini-3.1-flash-lite", aliases=["gemini-3.1-flash-lite-preview"], recommended_for=["speed", "tools", "vision"]),
        ModelCatalogEntry("gemini", "gemini-3.5-live-translate-preview", recommended_for=["audio", "translation", "realtime"], api_surface="realtime", availability="preview"),
        ModelCatalogEntry("gemini", "gemini-3.1-flash-live-preview", recommended_for=["speed", "audio", "vision"], api_surface="realtime", availability="preview"),
        ModelCatalogEntry("gemini", "gemini-3.1-flash-tts-preview", recommended_for=["audio"], api_surface="speech", availability="preview"),
        ModelCatalogEntry("gemini", "gemini-3.1-flash-image", aliases=["gemini-3.1-flash-image-preview"], recommended_for=["vision"], api_surface="image"),
        ModelCatalogEntry("gemini", "gemini-3.1-flash-lite-image", recommended_for=["speed", "vision"], api_surface="image"),
        ModelCatalogEntry("gemini", "gemini-3-pro-image", aliases=["gemini-3-pro-image-preview"], recommended_for=["reasoning", "vision"], api_surface="image"),
        ModelCatalogEntry("gemini", "veo-3.1-generate-preview", recommended_for=["vision"], api_surface="video", availability="preview"),
        ModelCatalogEntry("gemini", "veo-3.1-fast-generate-preview", recommended_for=["speed", "vision"], api_surface="video", availability="preview"),
        ModelCatalogEntry("gemini", "lyria-3-pro-preview", recommended_for=["audio"], api_surface="media", availability="preview"),
        ModelCatalogEntry("gemini", "lyria-3-clip-preview", recommended_for=["audio"], api_surface="media", availability="preview"),
        ModelCatalogEntry("gemini", "gemini-2.5-flash", cost_per_1k_tokens=0.35, recommended_for=["speed", "tools", "vision"]),
        ModelCatalogEntry("gemini", "gemini-2.5-pro", recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry(
            "vertex",
            "gemini-3.6-flash",
            recommended_for=["chat", "reasoning", "speed", "tools", "vision"],
            regions=["global"],
            support_evidence="offline-contract",
            source_urls=["https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-6-flash"],
        ),
        ModelCatalogEntry(
            "vertex",
            "gemini-3.5-flash-lite",
            recommended_for=["chat", "reasoning", "speed", "tools", "vision"],
            regions=["global", "us", "eu"],
            support_evidence="offline-contract",
            source_urls=["https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-flash-lite"],
        ),
        ModelCatalogEntry("vertex", "gemini-3.1-pro-preview", aliases=["gemini-pro-latest"], recommended_for=["reasoning", "tools", "vision"], availability="preview"),
        ModelCatalogEntry("vertex", "gemini-3.5-flash", aliases=["gemini-flash-latest"], recommended_for=["chat", "speed", "tools", "vision"]),
        ModelCatalogEntry("vertex", "gemini-3-flash-preview", recommended_for=["chat", "speed", "tools", "vision"], availability="preview"),
        ModelCatalogEntry("vertex", "gemini-3.1-flash-lite", aliases=["gemini-3.1-flash-lite-preview"], recommended_for=["speed", "tools", "vision"]),
        ModelCatalogEntry("vertex", "gemini-3.1-flash-live-preview", recommended_for=["speed", "audio", "vision"], api_surface="realtime", availability="preview"),
        ModelCatalogEntry("vertex", "gemini-3.1-flash-tts-preview", recommended_for=["audio"], api_surface="speech", availability="preview"),
        ModelCatalogEntry("vertex", "gemini-3.1-flash-image", aliases=["gemini-3.1-flash-image-preview"], recommended_for=["vision"], api_surface="image"),
        ModelCatalogEntry("vertex", "gemini-3-pro-image", aliases=["gemini-3-pro-image-preview"], recommended_for=["reasoning", "vision"], api_surface="image"),
        ModelCatalogEntry("vertex", "veo-3.1-generate-preview", recommended_for=["vision"], api_surface="video", availability="preview"),
        ModelCatalogEntry("vertex", "veo-3.1-fast-generate-preview", recommended_for=["speed", "vision"], api_surface="video", availability="preview"),
        ModelCatalogEntry("vertex", "lyria-3-pro-preview", recommended_for=["audio"], api_surface="media", availability="preview"),
        ModelCatalogEntry("vertex", "lyria-3-clip-preview", recommended_for=["audio"], api_surface="media", availability="preview"),
        ModelCatalogEntry("vertex", "gemini-2.5-flash", cost_per_1k_tokens=0.35, recommended_for=["speed", "tools", "vision"]),
        ModelCatalogEntry("vertex", "gemini-2.5-pro", recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry(
            "qwen",
            "qwen3.8-max",
            recommended_for=["chat", "reasoning", "tools", "vision"],
            regions=["cn", "intl", "us"],
            support_evidence="offline-contract",
            source_urls=["https://help.aliyun.com/en/model-studio/models"],
        ),
        ModelCatalogEntry("qwen", "qwen3.7-max", aliases=["qwen3.7-max-2026-05-20", "qwen3.7-max-preview"], cost_per_1k_tokens=1.2, recommended_for=["reasoning", "tools"]),
        ModelCatalogEntry("qwen", "qwen3.7-max-2026-06-08", recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry("qwen", "qwen3.7-plus", aliases=["qwen3.7-plus-2026-05-26"], cost_per_1k_tokens=0.4, recommended_for=["chat", "tools", "reasoning", "vision"]),
        ModelCatalogEntry("qwen", "qwen3.6-plus", aliases=["qwen3.6-plus-2026-04-02"], cost_per_1k_tokens=0.4, recommended_for=["chat", "tools", "reasoning", "vision"]),
        ModelCatalogEntry("qwen", "qwen3.6-flash", aliases=["qwen3.6-flash-2026-04-16"], cost_per_1k_tokens=0.05, recommended_for=["speed", "tools", "vision"]),
        ModelCatalogEntry("qwen", "qwen3.6-max-preview", cost_per_1k_tokens=1.2, recommended_for=["reasoning", "tools"], availability="preview"),
        ModelCatalogEntry("qwen", "qwen3.6-35b-a3b", recommended_for=["chat", "tools", "reasoning", "vision"]),
        ModelCatalogEntry("qwen", "qwen3.6-27b", recommended_for=["chat", "tools", "reasoning", "vision"]),
        ModelCatalogEntry("qwen", "qwen3.5-plus", aliases=["qwen3.5-plus-2026-04-20", "qwen3.5-plus-2026-02-15"], cost_per_1k_tokens=0.4, recommended_for=["chat", "tools", "reasoning", "vision"]),
        ModelCatalogEntry("qwen", "qwen3.5-flash", aliases=["qwen3.5-flash-2026-02-23"], cost_per_1k_tokens=0.1, recommended_for=["speed", "tools", "vision"]),
        ModelCatalogEntry("qwen", "qwen3-max", aliases=["qwen3-max-2026-01-23", "qwen3-max-preview", "qwen3-max-2025-09-23"], cost_per_1k_tokens=1.2, recommended_for=["reasoning", "tools"]),
        ModelCatalogEntry("qwen", "qwen-plus", cost_per_1k_tokens=0.4, recommended_for=["chat", "tools", "reasoning"]),
        ModelCatalogEntry("qwen", "qwen-flash", cost_per_1k_tokens=0.05, recommended_for=["speed", "tools"]),
        ModelCatalogEntry("qwen", "qwen-max-latest", aliases=["qwen-max", "qwen-max-2025-01-25"], cost_per_1k_tokens=1.6, recommended_for=["reasoning", "tools"]),
        ModelCatalogEntry("qwen", "qwen3-coder-plus", aliases=["qwen3-coder-plus-2025-09-23", "qwen3-coder-plus-2025-07-22"], cost_per_1k_tokens=0.574, recommended_for=["tools", "reasoning"]),
        ModelCatalogEntry("qwen", "qwen3-coder-flash", aliases=["qwen3-coder-flash-2025-07-28"], cost_per_1k_tokens=0.144, recommended_for=["speed", "tools"]),
        ModelCatalogEntry("qwen", "text-embedding-v4", aliases=["text-embedding-v3"], cost_per_1k_tokens=0.07, recommended_for=["embedding", "retrieval"], api_surface="embedding"),
        ModelCatalogEntry("qwen", "tongyi-embedding-vision-plus", aliases=["tongyi-embedding-vision-flash"], recommended_for=["embedding", "retrieval", "vision"], api_surface="embedding"),
        ModelCatalogEntry("qwen", "qwen3-rerank", recommended_for=["retrieval"], api_surface="rerank"),
        ModelCatalogEntry("qwen", "qwen3-asr-flash", aliases=["qwen3-asr-flash-2026-02-10", "qwen3-asr-flash-2025-09-08", "qwen3-asr-flash-us", "qwen3-asr-flash-2025-09-08-us"], recommended_for=["speed", "audio"], api_surface="transcription"),
        ModelCatalogEntry("qwen", "qwen3-asr-flash-realtime", aliases=["qwen3-asr-flash-realtime-2026-02-10", "qwen3-asr-flash-realtime-2025-10-27"], recommended_for=["speed", "audio"], api_surface="transcription"),
        ModelCatalogEntry("qwen", "qwen3-tts-flash", aliases=["qwen3-tts-flash-2025-11-27", "qwen3-tts-flash-2025-09-18"], recommended_for=["speed", "audio"], api_surface="speech"),
        ModelCatalogEntry("qwen", "qwen3-tts-instruct-flash", aliases=["qwen3-tts-instruct-flash-2026-01-26"], recommended_for=["audio"], api_surface="speech"),
        ModelCatalogEntry(
            "kimi",
            "kimi-k3",
            recommended_for=["reasoning", "tools", "vision"],
        ),
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
        ModelCatalogEntry(
            "deepseek",
            "deepseek-v4-pro",
            recommended_for=["chat", "reasoning", "tools"],
        ),
        ModelCatalogEntry(
            "deepseek",
            "deepseek-v4-flash",
            recommended_for=["chat", "speed", "reasoning", "tools"],
        ),
        ModelCatalogEntry(
            "meta",
            "muse-spark-1.2",
            recommended_for=["chat", "reasoning", "tools", "vision"],
            support_evidence="offline-contract",
            source_urls=[
                "https://dev.meta.ai/docs/models",
                "https://developer.meta.com/ai/resources/blog/build-with-muse-code/",
            ],
            parallel_tool_calls=True,
            structured_output=True,
        ),
        ModelCatalogEntry(
            "meta",
            "muse-spark-1.2-contributor",
            recommended_for=["chat", "reasoning", "tools", "vision"],
            availability="preview",
            source_urls=[
                "https://dev.meta.ai/docs/models",
                "https://dev.meta.ai/docs/pricing-rate-limits",
            ],
            parallel_tool_calls=True,
            structured_output=True,
        ),
        ModelCatalogEntry(
            "meta",
            "muse-spark-1.1",
            recommended_for=["chat", "reasoning", "tools", "vision"],
            availability="preview",
            source_urls=["https://dev.meta.ai/docs/models"],
            parallel_tool_calls=True,
            structured_output=True,
        ),
        ModelCatalogEntry("openrouter", "openai/gpt-5.4-mini", aliases=["openai/gpt-4o-mini"], recommended_for=["chat", "tools"]),
        ModelCatalogEntry(
            "openrouter",
            "meta/muse-spark-1.2",
            recommended_for=["chat", "reasoning", "tools", "vision"],
            availability="preview",
            source_urls=[
                "https://developer.meta.com/ai/resources/blog/build-with-muse-code/",
                "https://openrouter.ai/meta/muse-spark-1.2",
            ],
        ),
        ModelCatalogEntry(
            "openrouter",
            "meta/muse-glimmer-30b",
            recommended_for=["chat", "reasoning", "tools", "vision"],
            source_urls=[
                "https://developer.meta.com/ai/models/muse-glimmer/",
                "https://openrouter.ai/meta/muse-glimmer-30b",
            ],
            max_tool_calls_per_turn=1,
            parallel_tool_calls=False,
        ),
        ModelCatalogEntry("bedrock", "anthropic.claude-opus-4-8", aliases=["anthropic.claude-opus-4-6"], recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry("bedrock", "anthropic.claude-sonnet-5", recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("bedrock", "anthropic.claude-sonnet-4-6", cost_per_1k_tokens=3, recommended_for=["chat", "reasoning", "tools", "vision"]),
        ModelCatalogEntry("bedrock", "anthropic.claude-haiku-4-5-20251001-v1:0", cost_per_1k_tokens=1, recommended_for=["speed", "tools", "vision"]),
        ModelCatalogEntry("bedrock", "amazon.nova-premier-v1:0", recommended_for=["reasoning", "tools", "vision"]),
        ModelCatalogEntry("bedrock", "amazon.nova-pro-v1:0", recommended_for=["chat", "tools", "vision"]),
        ModelCatalogEntry("bedrock", "anthropic.claude-3-5-sonnet", cost_per_1k_tokens=3, recommended_for=["reasoning"]),
        ModelCatalogEntry("ollama", "llama3.2", cost_per_1k_tokens=0, recommended_for=["chat", "speed"]),
        ModelCatalogEntry(
            "ollama",
            "muse-glimmer:30b",
            cost_per_1k_tokens=0,
            recommended_for=["chat", "reasoning", "tools", "vision"],
            source_urls=["https://ollama.com/library/muse-glimmer"],
            max_tool_calls_per_turn=1,
            parallel_tool_calls=False,
        ),
        ModelCatalogEntry(
            "ollama",
            "muse-glimmer:30b-mlx",
            cost_per_1k_tokens=0,
            recommended_for=["chat", "reasoning", "tools", "vision"],
            source_urls=["https://ollama.com/library/muse-glimmer"],
            max_tool_calls_per_turn=1,
            parallel_tool_calls=False,
        ),
        ModelCatalogEntry(
            "vllm",
            "meta-models/Muse-Glimmer-30B",
            cost_per_1k_tokens=0,
            recommended_for=["chat", "reasoning", "tools", "vision"],
            source_urls=[
                "https://dev.meta.ai/docs/muse-glimmer/get-the-model",
                "https://dev.meta.ai/docs/muse-glimmer/vllm",
            ],
            max_tool_calls_per_turn=1,
            parallel_tool_calls=False,
        ),
        ModelCatalogEntry(
            "vllm",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            cost_per_1k_tokens=0,
            recommended_for=["chat", "tools", "vision"],
            source_urls=[
                "https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md",
            ],
        ),
        ModelCatalogEntry(
            "vllm",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            cost_per_1k_tokens=0,
            recommended_for=["chat", "reasoning", "tools", "vision"],
            source_urls=[
                "https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md",
            ],
        ),
    ]
)
