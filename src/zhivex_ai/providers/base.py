from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..types import EmbeddingModel, LanguageModel


@dataclass(slots=True)
class ProviderAdapter:
    name: str
    language_model_factory: Callable[[str], LanguageModel]
    embedding_model_factory: Callable[[str], EmbeddingModel] | None = None

    def __call__(self, model_id: str) -> LanguageModel:
        return self.language_model_factory(model_id)

    def language_model(self, model_id: str) -> LanguageModel:
        return self.language_model_factory(model_id)

    def embedding_model(self, model_id: str) -> EmbeddingModel:
        if self.embedding_model_factory is None:
            raise AttributeError("This provider does not expose embedding models.")
        return self.embedding_model_factory(model_id)
