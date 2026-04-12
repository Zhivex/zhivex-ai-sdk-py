from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..errors import UnsupportedFeatureError, ValidationError
from ..types import (
    BatchesClient,
    CountTokensClient,
    ConversationsClient,
    EmbeddingModel,
    FileSearchStoresClient,
    FilesClient,
    GroundedLanguageModel,
    GroundedModelGenerateInput,
    GroundedGenerateResult,
    LanguageModel,
    GenerateResult,
    ImagesClient,
    ModelGenerateInput,
    ModerationsClient,
    NativeSupport,
    PortableSupport,
    RealtimeModel,
    ResponsesClient,
    SpeechModel,
    SpeechOutput,
    TranscriptionModel,
    TranscriptionOutput,
    UploadsClient,
)


@dataclass(slots=True)
class ProviderAdapter:
    name: str
    language_model_factory: Callable[[str], LanguageModel]
    embedding_model_factory: Callable[[str], EmbeddingModel] | None = None
    transcription_model_factory: Callable[[str], TranscriptionModel] | None = None
    speech_model_factory: Callable[[str], SpeechModel] | None = None
    grounded_language_model_factory: Callable[[str], GroundedLanguageModel] | None = None
    realtime_model_factory: Callable[[str], RealtimeModel] | None = None
    files_client_factory: Callable[[], FilesClient] | None = None
    images_client_factory: Callable[[], ImagesClient] | None = None
    uploads_client_factory: Callable[[], UploadsClient] | None = None
    moderations_client_factory: Callable[[], ModerationsClient] | None = None
    batches_client_factory: Callable[[], BatchesClient] | None = None
    count_tokens_client_factory: Callable[[], CountTokensClient] | None = None
    file_search_stores_client_factory: Callable[[], FileSearchStoresClient] | None = None
    responses_client_factory: Callable[[], ResponsesClient] | None = None
    conversations_client_factory: Callable[[], ConversationsClient] | None = None
    _language_model_cache: dict[str, LanguageModel] = field(default_factory=dict, init=False, repr=False)
    _embedding_model_cache: dict[str, EmbeddingModel] = field(default_factory=dict, init=False, repr=False)
    _transcription_model_cache: dict[str, TranscriptionModel] = field(default_factory=dict, init=False, repr=False)
    _speech_model_cache: dict[str, SpeechModel] = field(default_factory=dict, init=False, repr=False)
    _grounded_language_model_cache: dict[str, GroundedLanguageModel] = field(default_factory=dict, init=False, repr=False)
    _realtime_model_cache: dict[str, RealtimeModel] = field(default_factory=dict, init=False, repr=False)
    _files_client: FilesClient | None = field(default=None, init=False, repr=False)
    _images_client: ImagesClient | None = field(default=None, init=False, repr=False)
    _uploads_client: UploadsClient | None = field(default=None, init=False, repr=False)
    _moderations_client: ModerationsClient | None = field(default=None, init=False, repr=False)
    _batches_client: BatchesClient | None = field(default=None, init=False, repr=False)
    _count_tokens_client: CountTokensClient | None = field(default=None, init=False, repr=False)
    _file_search_stores_client: FileSearchStoresClient | None = field(default=None, init=False, repr=False)
    _responses_client: ResponsesClient | None = field(default=None, init=False, repr=False)
    _conversations_client: ConversationsClient | None = field(default=None, init=False, repr=False)

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

    def files(self) -> FilesClient:
        if self.files_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose a files client.')
        if self._files_client is None:
            self._files_client = self.files_client_factory()
        return self._files_client

    def images(self) -> ImagesClient:
        if self.images_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose an images client.')
        if self._images_client is None:
            self._images_client = self.images_client_factory()
        return self._images_client

    def uploads(self) -> UploadsClient:
        if self.uploads_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose an uploads client.')
        if self._uploads_client is None:
            self._uploads_client = self.uploads_client_factory()
        return self._uploads_client

    def moderations(self) -> ModerationsClient:
        if self.moderations_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose a moderations client.')
        if self._moderations_client is None:
            self._moderations_client = self.moderations_client_factory()
        return self._moderations_client

    def batches(self) -> BatchesClient:
        if self.batches_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose a batches client.')
        if self._batches_client is None:
            self._batches_client = self.batches_client_factory()
        return self._batches_client

    def tokens(self) -> CountTokensClient:
        if self.count_tokens_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose a token counting client.')
        if self._count_tokens_client is None:
            self._count_tokens_client = self.count_tokens_client_factory()
        return self._count_tokens_client

    def file_search_stores(self) -> FileSearchStoresClient:
        if self.file_search_stores_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose file search stores.')
        if self._file_search_stores_client is None:
            self._file_search_stores_client = self.file_search_stores_client_factory()
        return self._file_search_stores_client

    def responses(self) -> ResponsesClient:
        if self.responses_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose a responses client.')
        if self._responses_client is None:
            self._responses_client = self.responses_client_factory()
        return self._responses_client

    def conversations(self) -> ConversationsClient:
        if self.conversations_client_factory is None:
            raise AttributeError(f'Provider "{self.name}" does not expose a conversations client.')
        if self._conversations_client is None:
            self._conversations_client = self.conversations_client_factory()
        return self._conversations_client

    @staticmethod
    def _cached_model(cache: dict[str, object], factory: Callable[[str], object], model_id: str):
        model = cache.get(model_id)
        if model is None:
            model = factory(model_id)
            cache[model_id] = model
        return model


@dataclass(slots=True)
class PortableLanguageModel:
    native_model: LanguageModel
    provider: str
    model_id: str
    capabilities: object
    portable: bool = True

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        if input.provider_options is not None:
            raise ValidationError(
                f'Portable model "{self.provider}/{self.model_id}" does not accept provider_options. '
                "Use provider.native.language_model(...) when you need provider-specific configuration."
            )
        return await self.native_model.generate(input)

    async def stream(self, input: ModelGenerateInput):
        if input.provider_options is not None:
            raise ValidationError(
                f'Portable model "{self.provider}/{self.model_id}" does not accept provider_options. '
                "Use provider.native.language_model(...) when you need provider-specific configuration."
            )
        return await self.native_model.stream(input)


@dataclass(slots=True)
class PortableGroundedLanguageModel:
    native_model: GroundedLanguageModel
    provider: str
    model_id: str
    capabilities: object
    portable: bool = True

    async def generate(self, input: GroundedModelGenerateInput) -> GroundedGenerateResult:
        if input.provider_options is not None:
            raise ValidationError(
                f'Portable grounded model "{self.provider}/{self.model_id}" does not accept provider_options. '
                "Use provider.native.grounded_language_model(...) when you need provider-specific configuration."
            )
        return await self.native_model.generate(input)


@dataclass(slots=True)
class PortableEmbeddingModel:
    native_model: EmbeddingModel
    provider: str
    model_id: str
    capabilities: object
    portable: bool = True

    async def embed(self, values: list[str], options=None):
        return await self.native_model.embed(values, options)


@dataclass(slots=True)
class PortableTranscriptionModel:
    native_model: TranscriptionModel
    provider: str
    model_id: str
    capabilities: object
    portable: bool = True

    async def transcribe(
        self,
        *,
        audio,
        prompt: str | None = None,
        language: str | None = None,
        provider_options=None,
        options=None,
    ) -> TranscriptionOutput:
        if provider_options is not None:
            raise ValidationError(
                f'Portable transcription model "{self.provider}/{self.model_id}" does not accept provider_options. '
                "Use provider.native.transcription_model(...) when you need provider-specific configuration."
            )
        return await self.native_model.transcribe(audio=audio, prompt=prompt, language=language, provider_options=None, options=options)


@dataclass(slots=True)
class PortableSpeechModel:
    native_model: SpeechModel
    provider: str
    model_id: str
    capabilities: object
    portable: bool = True

    async def generate_speech(
        self,
        *,
        input: str,
        voice: str | None = None,
        provider_options=None,
        options=None,
    ) -> SpeechOutput:
        if provider_options is not None:
            raise ValidationError(
                f'Portable speech model "{self.provider}/{self.model_id}" does not accept provider_options. '
                "Use provider.native.speech_model(...) when you need provider-specific configuration."
            )
        return await self.native_model.generate_speech(input=input, voice=voice, provider_options=None, options=options)


@dataclass(slots=True)
class PortableProviderNamespace:
    name: str
    native_adapter: ProviderAdapter
    portable_support: PortableSupport

    def _unsupported(self) -> UnsupportedFeatureError:
        return UnsupportedFeatureError(
            f'Provider "{self.name}" does not satisfy the portable contract. '
            'Use the explicit native namespace (`provider.native`) for provider-specific access.'
        )

    def _ensure_portable(self) -> None:
        if not self.portable_support.portable_badge:
            raise self._unsupported()

    def __call__(self, model_id: str) -> PortableLanguageModel:
        return self.language_model(model_id)

    def language_model(self, model_id: str) -> PortableLanguageModel:
        self._ensure_portable()
        model = self.native_adapter.language_model(model_id)
        return PortableLanguageModel(native_model=model, provider=model.provider, model_id=model.model_id, capabilities=model.capabilities)

    def embedding_model(self, model_id: str) -> PortableEmbeddingModel:
        self._ensure_portable()
        model = self.native_adapter.embedding_model(model_id)
        return PortableEmbeddingModel(native_model=model, provider=model.provider, model_id=model.model_id, capabilities=model.capabilities)

    def grounded_language_model(self, model_id: str) -> PortableGroundedLanguageModel:
        self._ensure_portable()
        model = self.native_adapter.grounded_language_model(model_id)
        return PortableGroundedLanguageModel(native_model=model, provider=model.provider, model_id=model.model_id, capabilities=model.capabilities)

    def transcription_model(self, model_id: str) -> PortableTranscriptionModel:
        self._ensure_portable()
        model = self.native_adapter.transcription_model(model_id)
        return PortableTranscriptionModel(native_model=model, provider=model.provider, model_id=model.model_id, capabilities=model.capabilities)

    def speech_model(self, model_id: str) -> PortableSpeechModel:
        self._ensure_portable()
        model = self.native_adapter.speech_model(model_id)
        return PortableSpeechModel(native_model=model, provider=model.provider, model_id=model.model_id, capabilities=model.capabilities)


@dataclass(slots=True)
class ProviderBundle:
    name: str
    portable: PortableProviderNamespace
    native: ProviderAdapter
    portable_support: PortableSupport
    native_support: NativeSupport
    tier: str

    def __call__(self, model_id: str) -> PortableLanguageModel:
        return self.portable(model_id)

    def language_model(self, model_id: str) -> PortableLanguageModel:
        return self.portable.language_model(model_id)

    def embedding_model(self, model_id: str) -> PortableEmbeddingModel:
        return self.portable.embedding_model(model_id)

    def grounded_language_model(self, model_id: str) -> PortableGroundedLanguageModel:
        return self.portable.grounded_language_model(model_id)

    def transcription_model(self, model_id: str) -> PortableTranscriptionModel:
        return self.portable.transcription_model(model_id)

    def speech_model(self, model_id: str) -> PortableSpeechModel:
        return self.portable.speech_model(model_id)

    def realtime_model(self, model_id: str) -> RealtimeModel:
        return self.native.realtime_model(model_id)

    def files(self) -> FilesClient:
        return self.native.files()

    def images(self) -> ImagesClient:
        return self.native.images()

    def uploads(self) -> UploadsClient:
        return self.native.uploads()

    def moderations(self) -> ModerationsClient:
        return self.native.moderations()

    def batches(self) -> BatchesClient:
        return self.native.batches()

    def tokens(self) -> CountTokensClient:
        return self.native.tokens()

    def file_search_stores(self) -> FileSearchStoresClient:
        return self.native.file_search_stores()

    def responses(self) -> ResponsesClient:
        return self.native.responses()

    def conversations(self) -> ConversationsClient:
        return self.native.conversations()


def build_native_support(adapter: ProviderAdapter) -> NativeSupport:
    language_model = adapter.language_model_factory("")
    return NativeSupport(
        text_generation=True,
        streaming=bool(language_model.capabilities.streaming),
        tools=bool(language_model.capabilities.tools),
        structured_output=bool(language_model.capabilities.structured_output),
        embeddings=adapter.embedding_model_factory is not None,
        grounding=adapter.grounded_language_model_factory is not None,
        transcription=adapter.transcription_model_factory is not None,
        speech=adapter.speech_model_factory is not None,
        realtime=adapter.realtime_model_factory is not None,
        files=adapter.files_client_factory is not None,
        file_search=adapter.file_search_stores_client_factory is not None,
        images=adapter.images_client_factory is not None,
        uploads=adapter.uploads_client_factory is not None,
        moderations=adapter.moderations_client_factory is not None,
        batches=adapter.batches_client_factory is not None,
        responses=adapter.responses_client_factory is not None,
        conversations=adapter.conversations_client_factory is not None,
    )


def create_provider_bundle(
    *,
    name: str,
    native: ProviderAdapter,
    portable_support: PortableSupport,
) -> ProviderBundle:
    return ProviderBundle(
        name=name,
        portable=PortableProviderNamespace(name=name, native_adapter=native, portable_support=portable_support),
        native=native,
        portable_support=portable_support,
        native_support=build_native_support(native),
        tier=portable_support.tier,
    )


def resolve_provider_adapter(provider: ProviderAdapter | ProviderBundle) -> ProviderAdapter:
    return provider.native if isinstance(provider, ProviderBundle) else provider
