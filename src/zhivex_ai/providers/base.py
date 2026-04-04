from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..types import EmbeddingModel, GroundedLanguageModel, LanguageModel, SpeechModel, TranscriptionModel


@dataclass(slots=True)
class ProviderAdapter:
    name: str
    language_model_factory: Callable[[str], LanguageModel]
    embedding_model_factory: Callable[[str], EmbeddingModel] | None = None
    transcription_model_factory: Callable[[str], TranscriptionModel] | None = None
    speech_model_factory: Callable[[str], SpeechModel] | None = None
    grounded_language_model_factory: Callable[[str], GroundedLanguageModel] | None = None

    def __call__(self, model_id: str) -> LanguageModel:
        return self.language_model_factory(model_id)

    def language_model(self, model_id: str) -> LanguageModel:
        return self.language_model_factory(model_id)

    def embedding_model(self, model_id: str) -> EmbeddingModel:
        if self.embedding_model_factory is None:
            raise AttributeError("This provider does not expose embedding models.")
        return self.embedding_model_factory(model_id)

    def transcription_model(self, model_id: str) -> TranscriptionModel:
        if self.transcription_model_factory is None:
            raise AttributeError("This provider does not expose transcription models.")
        return self.transcription_model_factory(model_id)

    def speech_model(self, model_id: str) -> SpeechModel:
        if self.speech_model_factory is None:
            raise AttributeError("This provider does not expose speech models.")
        return self.speech_model_factory(model_id)

    def grounded_language_model(self, model_id: str) -> GroundedLanguageModel:
        if self.grounded_language_model_factory is None:
            raise AttributeError("This provider does not expose grounded language models.")
        return self.grounded_language_model_factory(model_id)
