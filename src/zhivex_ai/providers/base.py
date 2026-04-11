from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..types import EmbeddingModel, GroundedLanguageModel, LanguageModel, RealtimeModel, SpeechModel, TranscriptionModel


@dataclass(slots=True)
class ProviderAdapter:
    name: str
    language_model_factory: Callable[[str], LanguageModel]
    embedding_model_factory: Callable[[str], EmbeddingModel] | None = None
    transcription_model_factory: Callable[[str], TranscriptionModel] | None = None
    speech_model_factory: Callable[[str], SpeechModel] | None = None
    grounded_language_model_factory: Callable[[str], GroundedLanguageModel] | None = None
    realtime_model_factory: Callable[[str], RealtimeModel] | None = None
    _language_model_cache: dict[str, LanguageModel] = field(default_factory=dict, init=False, repr=False)
    _embedding_model_cache: dict[str, EmbeddingModel] = field(default_factory=dict, init=False, repr=False)
    _transcription_model_cache: dict[str, TranscriptionModel] = field(default_factory=dict, init=False, repr=False)
    _speech_model_cache: dict[str, SpeechModel] = field(default_factory=dict, init=False, repr=False)
    _grounded_language_model_cache: dict[str, GroundedLanguageModel] = field(default_factory=dict, init=False, repr=False)
    _realtime_model_cache: dict[str, RealtimeModel] = field(default_factory=dict, init=False, repr=False)

    def __call__(self, model_id: str) -> LanguageModel:
        return self.language_model(model_id)

    def language_model(self, model_id: str) -> LanguageModel:
        return self._cached_model(self._language_model_cache, self.language_model_factory, model_id)

    def embedding_model(self, model_id: str) -> EmbeddingModel:
        if self.embedding_model_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose embedding models.')
        return self._cached_model(self._embedding_model_cache, self.embedding_model_factory, model_id)

    def transcription_model(self, model_id: str) -> TranscriptionModel:
        if self.transcription_model_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose transcription models.')
        return self._cached_model(self._transcription_model_cache, self.transcription_model_factory, model_id)

    def speech_model(self, model_id: str) -> SpeechModel:
        if self.speech_model_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose speech models.')
        return self._cached_model(self._speech_model_cache, self.speech_model_factory, model_id)

    def grounded_language_model(self, model_id: str) -> GroundedLanguageModel:
        if self.grounded_language_model_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose grounded language models.')
        return self._cached_model(self._grounded_language_model_cache, self.grounded_language_model_factory, model_id)

    def realtime_model(self, model_id: str) -> RealtimeModel:
        if self.realtime_model_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose realtime models.')
        return self._cached_model(self._realtime_model_cache, self.realtime_model_factory, model_id)

    @staticmethod
    def _cached_model(cache: dict[str, object], factory: Callable[[str], object], model_id: str):
        model = cache.get(model_id)
        if model is None:
            model = factory(model_id)
            cache[model_id] = model
        return model
