from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, TypeVar, cast

if TYPE_CHECKING:
    from ._gemini_voices import GeminiVoicesClient
    from ._qwen_rerank import QwenRerankClient
    from ._vertex_platform import VertexAgentPlatformClient, VertexModelGardenClient, VertexRagClient
    from ._native_sessions import AnthropicMessagesClient, OpenAIAgentSessionsClient, OpenAILiveClient

_ClientT = TypeVar("_ClientT")


@dataclass(slots=True)
class NativeExtensions:
    """Provider-owned native resources, independent of the portable adapter.

    Each provider supplies its own lazy, typed factory. Unsuccessful factories
    are not cached, so transient initialization errors can be retried.
    """

    _clients: dict[str, object] = field(default_factory=dict, init=False, repr=False)

    def resolve(self, resource: str, factory: Callable[[], _ClientT]) -> _ClientT:
        if resource not in self._clients:
            self._clients[resource] = factory()
        return cast(_ClientT, self._clients[resource])


class NativeServiceAccess:
    """Compatibility facade for documented native resources.

    New resources should live on provider-owned extension namespaces. These
    accessors preserve existing concrete return types and constructor factories.
    """

    __slots__ = ()

    name: str
    extensions: NativeExtensions
    messages_client_factory: Callable[[], AnthropicMessagesClient] | None
    agent_sessions_client_factory: Callable[[], OpenAIAgentSessionsClient] | None
    live_client_factory: Callable[[], OpenAILiveClient] | None
    rag_client_factory: Callable[[], VertexRagClient] | None
    model_garden_client_factory: Callable[[], VertexModelGardenClient] | None
    agent_platform_client_factory: Callable[[], VertexAgentPlatformClient] | None
    voices_client_factory: Callable[[], GeminiVoicesClient] | None
    rerank_client_factory: Callable[[], QwenRerankClient] | None

    def voices(self) -> GeminiVoicesClient:
        """Beta Gemini voice design, replication and library resources."""
        if self.voices_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose native voices.')
        return self.extensions.resolve("voices", self.voices_client_factory)

    def rerank(self) -> QwenRerankClient:
        """Beta Qwen relevance ranking; separate from portable embeddings."""
        if self.rerank_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose native reranking.')
        return self.extensions.resolve("rerank", self.rerank_client_factory)

    def rag(self) -> VertexRagClient:
        """Beta Google-managed RAG Engine; unavailable in Express Mode."""
        if self.rag_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose RAG Engine.')
        return self.extensions.resolve("rag", self.rag_client_factory)

    def model_garden(self) -> VertexModelGardenClient:
        """Beta publisher-native prediction and model discovery on Google Cloud."""
        if self.model_garden_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose Model Garden.')
        return self.extensions.resolve("model_garden", self.model_garden_client_factory)

    def agent_platform(self) -> VertexAgentPlatformClient:
        """Beta Google-managed Agent Runtime, Sessions, and Memory Bank."""
        if self.agent_platform_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose Google Agent Platform.')
        return self.extensions.resolve("agent_platform", self.agent_platform_client_factory)

    def messages(self) -> AnthropicMessagesClient:
        if self.messages_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose native Messages.')
        return self.extensions.resolve("messages", self.messages_client_factory)

    def agent_sessions(self) -> OpenAIAgentSessionsClient:
        if self.agent_sessions_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose managed agent sessions.')
        return self.extensions.resolve("agent_sessions", self.agent_sessions_client_factory)

    def live(self) -> OpenAILiveClient:
        if self.live_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose GPT-Live.')
        return self.extensions.resolve("live", self.live_client_factory)

