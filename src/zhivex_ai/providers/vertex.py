from __future__ import annotations

import os
import math
import base64
from dataclasses import dataclass, field, replace
from copy import deepcopy
from typing import Any, cast
from urllib.parse import urlsplit

from .._http import Fetcher, default_fetch
from ..errors import ConfigurationError, ProviderHTTPError, ValidationError, UnsupportedFeatureError
from ..messages import normalize_finish_reason
from ..realtime import CallbackRealtimeSession, RealtimeConnectionFactory, RealtimeSessionCallbacks, open_websocket_connection, unsupported_browser_token
from ..runtime import with_retry
from ..types import AudioInput, TranscriptionOutput, RealtimeTranscriptEvent
from ..types import AgentCapabilities, CountTokensResult, EmbedResult, EmbeddingContent, EmbeddingModel, GroundedGenerateResult, GroundedModelGenerateInput, HostedToolDefinition, ModelCapabilities, ModelGenerateInput, PortableSupport, RealtimeConnectOptions, RealtimeSession, RealtimeSessionConfig, RealtimeTokenResult, RetryOptions, TokenCountDetail, TokenUsage
from ._vertex_platform import VertexAgentPlatformClient, VertexModelGardenClient, VertexRagClient
from ._vertex_native import VertexBatchesClient, VertexCachedContentsClient, VertexInteractionsClient, VertexVideosClient, VertexMediaClient
from ._vertex_native import _request
from ._vertex_auth import VertexAuth, default_credentials
from ._vertex_chat import (
    GEMMA_MAAS_MODEL, MISTRAL_VERTEX_MODELS, VertexChatLanguageModel,
    VERTEX_MAAS_TEXT_MODELS, resolve_maas_model, maas_capabilities,
)
from .base import ProviderAdapter, ProviderBundle, create_provider_bundle
from ._payload import drop_none
from .gemini import (
    GEMINI_CAPABILITIES,
    GEMINI_GROUNDED_CAPABILITIES,
    GEMINI_REALTIME_CAPABILITIES,
    GeminiCountTokensClient,
    GeminiImagesClient,
    GeminiGroundedLanguageModel,
    GeminiLanguageModel,
    GeminiSpeechModel,
    GeminiTranscriptionModel,
    gemini_code_execution_tool,
    gemini_computer_use_tool,
    gemini_google_maps_tool,
    gemini_google_search_tool,
    gemini_url_context_tool,
    _extract_grounding_queries,
    _extract_grounding_sources,
    _extract_grounding_supports,
    _embedding_request_options,
    _embedding_content_parts,
    _extract_search_entry_point,
    _gemini_realtime_build_audio,
    _gemini_realtime_build_text,
    _gemini_realtime_build_tool_result,
    _gemini_realtime_build_update,
    _gemini_realtime_setup,
    _gemini_realtime_parse_event,
    _map_messages,
    _map_reasoning,
    _map_tools,
    _parse_assistant_message,
    _provider_options_without_mapped_tools,
    _system_instruction,
    _provider_option_value,
)

VERTEX_REALTIME_CAPABILITIES = replace(GEMINI_REALTIME_CAPABILITIES, realtime_browser_tokens=False)
_VERTEX_VERTEX_AI_SEARCH_PROVIDER_OPTIONS = ("vertex_ai_search", "vertexAiSearch")
_VERTEX_EXTERNAL_SEARCH_PROVIDER_OPTIONS = ("external_search", "externalSearch")


def vertex_google_search_tool(*, exclude_domains: list[str] | None = None, **extra: Any) -> HostedToolDefinition:
    return replace(gemini_google_search_tool(exclude_domains=exclude_domains, **extra), provider="vertex")


def vertex_google_maps_tool(**config: Any) -> HostedToolDefinition:
    return replace(gemini_google_maps_tool(**config), provider="vertex")


def vertex_url_context_tool(**config: Any) -> HostedToolDefinition:
    return replace(gemini_url_context_tool(**config), provider="vertex")


def vertex_code_execution_tool(**config: Any) -> HostedToolDefinition:
    return replace(gemini_code_execution_tool(**config), provider="vertex")


def vertex_computer_use_tool(**config: Any) -> HostedToolDefinition:
    return replace(gemini_computer_use_tool(**config), provider="vertex")


def vertex_vertex_ai_search_tool(*, datastore: str, **extra: Any) -> dict[str, Any]:
    return {"retrieval": {"vertexAiSearch": {"datastore": datastore, **deepcopy(extra)}}}


def vertex_external_search_tool(
    *,
    endpoint: str,
    api_key: str,
    api_spec: str = "SIMPLE_SEARCH",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "retrieval": {
            "externalApi": {
                "apiSpec": api_spec,
                "endpoint": endpoint,
                "apiAuth": {"apiKeyConfig": {"apiKeyString": api_key}},
                **deepcopy(extra),
            }
        }
    }


def _vertex_extract_provider_option(provider_options: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in provider_options:
            return provider_options.pop(name)
    return None


def _vertex_has_explicit_grounding_tools(provider_options: dict[str, Any] | None) -> bool:
    if not provider_options:
        return False
    if any(name in provider_options for name in _VERTEX_VERTEX_AI_SEARCH_PROVIDER_OPTIONS + _VERTEX_EXTERNAL_SEARCH_PROVIDER_OPTIONS):
        return True
    if provider_options.get("tools"):
        return True
    if provider_options.get("built_in_tools") or provider_options.get("builtInTools"):
        return True
    return any(
        name in provider_options
        for name in ("google_search", "googleSearch", "google_maps", "googleMaps", "url_context", "urlContext")
    )


def _normalize_vertex_provider_options(provider_options: dict[str, Any] | None) -> dict[str, Any] | None:
    if not provider_options:
        return provider_options
    normalized = deepcopy(provider_options)
    raw_tools = normalized.get("tools")
    if raw_tools is None:
        resolved_raw_tools: list[dict[str, Any]] = []
    elif isinstance(raw_tools, list):
        resolved_raw_tools = [deepcopy(item) for item in raw_tools]
    else:
        raise ValidationError('Vertex provider_options["tools"] must be a list when provided.')

    vertex_ai_search = _vertex_extract_provider_option(normalized, *_VERTEX_VERTEX_AI_SEARCH_PROVIDER_OPTIONS)
    if vertex_ai_search not in (None, False):
        if not isinstance(vertex_ai_search, dict):
            raise ValidationError('Vertex provider_options["vertex_ai_search"] must be a dictionary.')
        resolved_raw_tools.append(vertex_vertex_ai_search_tool(**vertex_ai_search))

    external_search = _vertex_extract_provider_option(normalized, *_VERTEX_EXTERNAL_SEARCH_PROVIDER_OPTIONS)
    if external_search not in (None, False):
        if not isinstance(external_search, dict):
            raise ValidationError('Vertex provider_options["external_search"] must be a dictionary.')
        resolved_raw_tools.append(vertex_external_search_tool(**external_search))

    if resolved_raw_tools:
        normalized["tools"] = resolved_raw_tools
    elif "tools" in normalized:
        normalized.pop("tools", None)
    return normalized


@dataclass(slots=True)
class VertexEmbeddingModel(EmbeddingModel):
    provider: str
    model_id: str
    base_url: str
    access_token: str
    fetch: Fetcher
    capabilities: ModelCapabilities = field(default_factory=lambda: GEMINI_CAPABILITIES)

    def _url(self) -> str:
        return f"{self.base_url}/publishers/google/models/{self.model_id}:predict"

    def _headers(self) -> dict[str, str]:
        return {"content-type": "application/json"}

    @staticmethod
    def _vector(value: Any) -> list[float]:
        if not isinstance(value, list) or not value or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            or not math.isfinite(item) for item in value
        ):
            raise ValidationError("Vertex returned an invalid or missing embedding vector.")
        return value

    async def _embed_content(self, values: list[EmbeddingContent], options: Any) -> EmbedResult:
        config = _embedding_request_options(options)
        if config.get("auto_truncate") is not None:
            raise ValidationError("auto_truncate is not supported by the Vertex embedContent contract.")
        for key in ("task_types", "titles"):
            if key in config and (not isinstance(config[key], list) or len(config[key]) != len(values)):
                raise ValidationError(f"{key} must contain one item per embedding input.")
        retry = RetryOptions(
            timeout_ms=_provider_option_value(options, "timeout_ms", "timeoutMs"),
            max_retries=_provider_option_value(options, "max_retries", "maxRetries"),
            retry_backoff_ms=_provider_option_value(options, "retry_backoff_ms", "retryBackoffMs"),
        )
        embeddings = []
        responses = []
        # embedContent embeds one Content; list entries are independent inputs,
        # while multiple parts inside an entry form one multimodal embedding.
        for index, value in enumerate(values):
            body = drop_none({
                "content": {"parts": _embedding_content_parts(value)},
                "taskType": config["task_types"][index] if "task_types" in config else config.get("task_type"),
                "title": config["titles"][index] if "titles" in config else config.get("title"),
                "outputDimensionality": config.get("output_dimensionality"),
            })
            payload = await _request(
                self.fetch,
                f"{self.base_url}/publishers/google/models/{self.model_id}:embedContent",
                "POST", body, retry,
            )
            embeddings.append(self._vector((payload.get("embedding") or {}).get("values")))
            responses.append(payload)
        return EmbedResult(embeddings=embeddings, raw_response=responses)

    async def _embed_multimodal_predict(self, values: list[EmbeddingContent], options: Any) -> EmbedResult:
        config = _embedding_request_options(options)
        if any(config.get(key) is not None for key in ("task_type", "task_types", "title", "titles", "auto_truncate")):
            raise ValidationError("multimodalembedding@001 does not support text embedding task/title/truncation options.")
        instances = []
        fields = []
        for value in values:
            parts = _embedding_content_parts(value)
            if len(parts) != 1:
                raise ValidationError("multimodalembedding@001 requires one text or image per portable input; use native Model Garden raw_predict for combined modalities or video segments.")
            part = parts[0]
            if "text" in part:
                instances.append({"text": part["text"]})
                fields.append("textEmbedding")
            elif "inlineData" in part and str(part["inlineData"].get("mimeType", "")).startswith("image/"):
                instances.append({"image": {"bytesBase64Encoded": part["inlineData"]["data"]}})
                fields.append("imageEmbedding")
            elif "fileData" in part and str(part["fileData"].get("mimeType", "")).startswith("image/") and str(part["fileData"].get("fileUri", "")).startswith("gs://"):
                instances.append({"image": {"gcsUri": part["fileData"]["fileUri"]}})
                fields.append("imageEmbedding")
            else:
                raise ValidationError("multimodalembedding@001 portable inputs must be text, inline images, or GCS images.")
        retry = RetryOptions(
            timeout_ms=_provider_option_value(options, "timeout_ms", "timeoutMs"),
            max_retries=_provider_option_value(options, "max_retries", "maxRetries"),
            retry_backoff_ms=_provider_option_value(options, "retry_backoff_ms", "retryBackoffMs"),
        )
        embeddings = []
        responses = []
        # This model accepts one instance per request and returns distinct vectors
        # for each modality, rather than a fused vector like Embedding 2.
        for instance, response_field in zip(instances, fields, strict=True):
            payload = await _request(self.fetch, self._url(), "POST", {
                "instances": [instance],
                "parameters": drop_none({"dimension": config.get("output_dimensionality")}),
            }, retry)
            predictions = payload.get("predictions")
            if not isinstance(predictions, list) or len(predictions) != 1 or not isinstance(predictions[0], dict):
                raise ValidationError("Vertex multimodal embedding response cardinality does not match its input.")
            embeddings.append(self._vector(predictions[0].get(response_field)))
            responses.append(payload)
        return EmbedResult(embeddings=embeddings, raw_response=responses)

    async def embed(self, values: list[EmbeddingContent], options: Any = None) -> EmbedResult:
        if not values:
            return EmbedResult(embeddings=[])
        if self.model_id in {"gemini-embedding-2", "gemini-embedding-2-preview"}:
            return await self._embed_content(values, options)
        if self.model_id == "multimodalembedding@001":
            return await self._embed_multimodal_predict(values, options)
        config = _embedding_request_options(options)
        task_type = config.get("task_type")
        title = config.get("title")
        output_dimensionality = config.get("output_dimensionality")
        auto_truncate = config.get("auto_truncate")
        task_types = config.get("task_types")
        titles = config.get("titles")
        response = await with_retry(
            lambda: self.fetch(
                self._url(),
                headers=self._headers(),
                json_body={
                    "instances": [
                        {
                            "content": {"parts": _embedding_content_parts(value)} if not isinstance(value, str) else value,
                            **({"task_type": task_types[index]} if isinstance(task_types, list) and index < len(task_types) else {}),
                            **({"title": titles[index]} if isinstance(titles, list) and index < len(titles) else {}),
                            **({"task_type": task_type} if not isinstance(task_types, list) and task_type is not None else {}),
                            **({"title": title} if not isinstance(titles, list) and title is not None else {}),
                        }
                        for index, value in enumerate(values)
                    ],
                    "parameters": {
                        **({"outputDimensionality": output_dimensionality} if output_dimensionality is not None else {}),
                        **({"autoTruncate": auto_truncate} if auto_truncate is not None else {}),
                    },
                },
                timeout_ms=_provider_option_value(options, "timeout_ms", "timeoutMs"),
            ),
            max_retries=_provider_option_value(options, "max_retries", "maxRetries") or 0,
            retry_backoff_ms=_provider_option_value(options, "retry_backoff_ms", "retryBackoffMs") or 250,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(f"Vertex request failed with status {response.status_code}.", response.status_code, response_body=await response.text())
        payload = await response.json()
        predictions = payload.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(values):
            raise ValidationError("Vertex embedding response cardinality does not match its inputs.")
        return EmbedResult(embeddings=[self._vector(prediction.get("embeddings", {}).get("values")) for prediction in predictions], raw_response=payload)


def create_vertex(
    *,
    access_token: str | None = None,
    api_key: str | None = None,
    credentials: Any = None,
    express_mode: bool | None = None,
    project_id: str | None = None,
    location: str = "us-central1",
    api_version: str = "v1",
    base_url: str | None = None,
    fetch: Fetcher | None = None,
    realtime_url: str | None = None,
    realtime_connection_factory: RealtimeConnectionFactory | None = None,
) -> ProviderBundle:
    if access_token == "" or api_key == "":
        raise ConfigurationError("Vertex credentials must not be empty.")
    if sum(value is not None for value in (access_token, api_key, credentials)) > 1:
        raise ConfigurationError("Specify only one of access_token, api_key, or credentials.")
    # Explicit authentication always wins over ambient environment credentials.
    explicit_auth = any(value is not None for value in (access_token, api_key, credentials))
    resolved_token = access_token
    resolved_key = api_key
    if not explicit_auth:
        resolved_token = os.getenv("VERTEX_ACCESS_TOKEN") or os.getenv("GOOGLE_ACCESS_TOKEN")
        if not resolved_token:
            resolved_key = os.getenv("VERTEX_API_KEY") or os.getenv("GOOGLE_API_KEY")
    resolved_project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if not resolved_token and not resolved_key and credentials is None:
        credentials, adc_project = default_credentials()
        resolved_project = resolved_project or adc_project
    express = bool(resolved_key and not resolved_project) if express_mode is None else express_mode
    if express and not resolved_key:
        raise ConfigurationError("Vertex Express Mode requires an API key.")
    if not express and not resolved_project and not base_url:
        raise ConfigurationError("Missing Vertex project ID.")
    host = (
        "aiplatform.googleapis.com" if express or location == "global" else
        f"aiplatform.{location}.rep.googleapis.com" if location in {"us", "eu"} else
        f"{location}-aiplatform.googleapis.com"
    )
    resolved_base = base_url or (
        f"https://{host}/{api_version}" if express else
        f"https://{host}/{api_version}/projects/{resolved_project}/locations/{location}"
    )
    auth = VertexAuth(access_token=resolved_token, api_key=resolved_key, credentials=credentials)
    requester = fetch or default_fetch

    class VertexFetcher:
        async def __call__(
            self,
            url: str,
            *,
            method: str = "POST",
            headers: dict[str, str],
            json_body: dict[str, object] | None,
            body: Any = None,
            timeout_ms: int | None,
            stream: bool = False,
        ):
            merged = {key: value for key, value in headers.items() if key.lower() not in {"authorization", "x-goog-api-key"}}
            destination = urlsplit(url)
            origin = urlsplit(resolved_base)
            storage_download = (destination.scheme == "https" and destination.netloc == "storage.googleapis.com"
                and destination.path.startswith("/storage/v1/b/") and method == "GET" and not resolved_key)
            if storage_download or (destination.scheme, destination.netloc) == (origin.scheme, origin.netloc):
                merged.update(await auth.headers(method, url))
            request_kwargs: dict[str, Any] = {
                "headers": merged,
                "json_body": json_body,
                "timeout_ms": timeout_ms,
                "stream": stream,
            }
            if method != "POST":
                request_kwargs["method"] = method
            if body is not None:
                request_kwargs["body"] = body
            return await requester(url, **request_kwargs)

    wrapped_fetch = cast(Fetcher, VertexFetcher())

    class VertexLanguageModel(GeminiLanguageModel):
        def _url(self, action: str) -> str:  # type: ignore[override]
            return f"{self.base_url}/publishers/google/models/{self.model_id}:{action}"

        def _vertex_input(self, input: ModelGenerateInput) -> ModelGenerateInput:
            normalized = replace(input, provider_options=_normalize_vertex_provider_options(input.provider_options))
            if self.model_id == "gemini-3.8-flash-cyber" and _map_tools(normalized.tools, normalized.provider_options):
                raise UnsupportedFeatureError("Gemini 3.8 Flash Cyber does not support tools.")
            return normalized

        async def generate(self, input: ModelGenerateInput):  # type: ignore[override]
            return await super().generate(self._vertex_input(input))

        async def stream(self, input: ModelGenerateInput):  # type: ignore[override]
            return await super().stream(self._vertex_input(input))

    class VertexSpeechModel(GeminiSpeechModel):
        def _url(self, action: str) -> str:  # type: ignore[override]
            return f"{self.base_url}/publishers/google/models/{self.model_id}:{action}"

    class VertexTranscriptionModel(GeminiTranscriptionModel):
        def _url(self, action: str) -> str:  # type: ignore[override]
            return f"{self.base_url}/publishers/google/models/{self.model_id}:{action}"

        async def transcribe(
            self, *, audio: AudioInput, prompt: str | None = None,
            language: str | None = None,
            provider_options: dict[str, Any] | None = None,
            options: RetryOptions | None = None,
        ) -> TranscriptionOutput:
            if self.model_id == "gemini-3.5-transcribe-live-preview":
                raise ValidationError("Use the realtime client for streaming Transcribe.")
            if self.model_id != "gemini-3.5-transcribe-preview":
                return await super().transcribe(
                    audio=audio, prompt=prompt, language=language,
                    provider_options=provider_options, options=options,
                )
            if prompt is not None:
                raise ValidationError("Transcribe accepts audio only; use customVocabulary for speech biasing.")
            body = deepcopy(provider_options or {})
            if any(key in body for key in ("contents", "systemInstruction", "tools", "toolConfig")):
                raise ValidationError("Transcribe does not accept content overrides, system instructions or tools.")
            config = body.setdefault("generationConfig", {})
            if not isinstance(config, dict):
                raise ValidationError("generationConfig must be an object.")
            transcription = config.setdefault("audioTranscriptionConfig", {})
            if not isinstance(transcription, dict):
                raise ValidationError("audioTranscriptionConfig must be an object.")
            if language:
                transcription["languageCodes"] = [language]
            encoded = audio.data if isinstance(audio.data, str) else base64.b64encode(bytes(audio.data)).decode("ascii")
            body["contents"] = [{"role": "user", "parts": [
                {"inlineData": {"mimeType": audio.media_type, "data": encoded}},
            ]}]
            payload = await _request(self.fetch, self._url("generateContent"), "POST", body, options)
            candidate = (payload.get("candidates") or [{}])[0]
            parts = (candidate.get("content") or {}).get("parts") or []
            text = "".join(
                str(part.get("text") or (part.get("audioTranscription") or {}).get("text") or "")
                for part in parts if isinstance(part, dict) and not part.get("thought")
            )
            if not text.strip():
                raise ValidationError("Transcribe returned no transcript.")
            return TranscriptionOutput(text=text, raw_response=payload)

    class VertexGroundedLanguageModel(GeminiGroundedLanguageModel):
        capabilities = GEMINI_GROUNDED_CAPABILITIES

        def _url(self, action: str) -> str:  # type: ignore[override]
            return f"{self.base_url}/publishers/google/models/{self.model_id}:{action}"

        async def generate(self, input: GroundedModelGenerateInput):  # type: ignore[override]
            normalized_provider_options = _normalize_vertex_provider_options(input.provider_options)

            response = await with_retry(
                lambda: wrapped_fetch(
                    self._url("generateContent"),
                    headers={"content-type": "application/json"},
                    json_body=drop_none({
                        "contents": _map_messages(input.messages),
                        "systemInstruction": _system_instruction(input.messages),
                        "tools": _map_tools(
                            None,
                            normalized_provider_options,
                            force_google_search=not _vertex_has_explicit_grounding_tools(normalized_provider_options),
                        ),
                        **(_provider_options_without_mapped_tools(normalized_provider_options) or {}),
                        "generationConfig": drop_none({
                            "temperature": input.temperature,
                            "maxOutputTokens": input.max_tokens,
                            "thinkingConfig": _map_reasoning(
                                self.model_id,
                                ModelGenerateInput(messages=input.messages, reasoning=input.reasoning),
                            )
                            if input.reasoning is not None
                            else None,
                        }),
                    }),
                    timeout_ms=input.timeout_ms,
                ),
                max_retries=input.max_retries or 0,
                retry_backoff_ms=input.retry_backoff_ms or 250,
                )
            if response.status_code >= 400:
                raise ProviderHTTPError(
                    f"Vertex request failed with status {response.status_code}.",
                    response.status_code,
                    response_body=await response.text(),
                )
            payload = await response.json()
            candidate = (payload.get("candidates") or [None])[0]
            assistant_message = _parse_assistant_message(candidate)
            usage = payload.get("usageMetadata") or {}

            return GroundedGenerateResult(
                messages=[assistant_message],
                text="".join(part.text for part in assistant_message.parts if part.type == "text"),
                finish_reason=normalize_finish_reason(candidate.get("finishReason") if candidate else None),
                provider_finish_reason=candidate.get("finishReason") if candidate else None,
                usage=TokenUsage(
                    input_tokens=usage.get("promptTokenCount"),
                    output_tokens=usage.get("candidatesTokenCount"),
                    total_tokens=usage.get("totalTokenCount")
                    or ((usage.get("promptTokenCount") or 0) + (usage.get("candidatesTokenCount") or 0)),
                )
                if usage
                else None,
                raw_response=payload,
                sources=_extract_grounding_sources(payload),
                queries=_extract_grounding_queries(payload),
                supports=_extract_grounding_supports(payload),
                search_entry_point=_extract_search_entry_point(payload),
            )

    class VertexCountTokensClient(GeminiCountTokensClient):
        def _url(self, model_id: str) -> str:  # type: ignore[override]
            return f"{resolved_base.rstrip('/')}/publishers/google/models/{model_id}:countTokens"

        async def count(self, **kwargs: Any) -> CountTokensResult:  # type: ignore[override]
            model_id = kwargs["model_id"]
            prompt = kwargs.get("prompt")
            messages = kwargs.get("messages")
            system = kwargs.get("system")
            tools = kwargs.get("tools")
            provider_options = _normalize_vertex_provider_options(kwargs.get("provider_options"))
            options = kwargs.get("options")
            from .gemini import (
                _build_messages_for_request,
                _map_messages,
                _map_tools,
                _provider_options_without_mapped_tools,
                _system_instruction,
            )

            built_messages = _build_messages_for_request(prompt=prompt, messages=messages, system=system)
            request = drop_none(
                {
                    "contents": _map_messages(built_messages),
                    "systemInstruction": _system_instruction(built_messages),
                    "tools": _map_tools(tools, provider_options),
                    **(_provider_options_without_mapped_tools(provider_options) or {}),
                }
            )
            response = await with_retry(
                lambda: wrapped_fetch(
                    self._url(model_id),
                    method="POST",
                    headers={"content-type": "application/json"},
                    json_body=request,
                    timeout_ms=options.timeout_ms if options else None,
                ),
                max_retries=options.max_retries if options and options.max_retries is not None else 0,
                retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
            )
            if response.status_code >= 400:
                raise ProviderHTTPError(
                    f"Vertex request failed with status {response.status_code}.",
                    response.status_code,
                    response_body=await response.text(),
                )
            payload = await response.json()
            return CountTokensResult(
                total_tokens=payload.get("totalTokens"),
                cached_content_token_count=payload.get("cachedContentTokenCount"),
                total_billable_characters=payload.get("totalBillableCharacters"),
                details=[
                    TokenCountDetail(
                        modality=item.get("modality"),
                        token_count=item.get("tokenCount"),
                        billable_characters=item.get("billableCharacters"),
                        provider_metadata=dict(item),
                    )
                    for item in payload.get("promptTokensDetails") or []
                    if isinstance(item, dict)
                ],
                raw_response=payload,
            )

    class VertexRealtimeModel:
        provider = "vertex"
        capabilities = VERTEX_REALTIME_CAPABILITIES

        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            if model_id == "gemini-3.5-transcribe-live-preview":
                self.capabilities = replace(VERTEX_REALTIME_CAPABILITIES, realtime_audio_output=False, realtime_tools=False)
            elif model_id == "gemini-3.5-live-translate-preview":
                self.capabilities = replace(VERTEX_REALTIME_CAPABILITIES, realtime_tools=False)

        async def connect(
            self,
            config: RealtimeSessionConfig | None = None,
            options: RealtimeConnectOptions | None = None,
        ) -> RealtimeSession:
            resolved_config = config or RealtimeSessionConfig()
            if express:
                raise ConfigurationError("Vertex Live requires a standard project/location configuration.")
            setup = _gemini_realtime_setup(resolved_config, self.model_id)
            is_transcribe = self.model_id == "gemini-3.5-transcribe-live-preview"
            if is_transcribe:
                settings = setup["setup"]
                if settings.get("systemInstruction") or settings.get("tools") or resolved_config.voice:
                    raise ValidationError("Live Transcribe accepts audio without instructions, tools or a voice.")
                if settings["generationConfig"].get("responseModalities") != ["TEXT"]:
                    raise ValidationError("Live Transcribe requires TEXT output.")
                settings.setdefault("inputAudioTranscription", {})

            def parse_event(payload):
                events = _gemini_realtime_parse_event(payload)
                if not is_transcribe or not isinstance(payload, dict):
                    return events
                content = payload.get("serverContent") or payload.get("server_content") or {}
                # Transcribe finalizes utterances independently of conversation turns.
                for event in events:
                    if isinstance(event, RealtimeTranscriptEvent) and event.role == "user":
                        event.is_final = True
                interim = content.get("interimInputTranscription") or content.get("interim_input_transcription")
                if isinstance(interim, dict) and interim.get("text"):
                    events.insert(0, RealtimeTranscriptEvent(text=str(interim["text"]), role="user", is_final=False, provider_metadata=payload))
                return events

            def build_text(text, session_config):
                if is_transcribe:
                    raise ValidationError("Live Transcribe accepts audio input only.")
                return _gemini_realtime_build_text(text, session_config, self.model_id)
            setup["setup"]["model"] = (self.model_id if self.model_id.startswith("projects/") else
                f"projects/{resolved_project}/locations/{location}/publishers/google/models/{self.model_id}")
            url = realtime_url or resolved_config.provider_options.get("realtime_url") if resolved_config.provider_options else realtime_url
            if not url:
                url = f"wss://{host}/ws/google.cloud.aiplatform.{api_version}.LlmBidiService/BidiGenerateContent"
            headers = {
                **dict((resolved_config.provider_options or {}).get("headers") or {}),
                **(await auth.headers("GET", url)),
            }
            factory = realtime_connection_factory or (lambda u, h, o: open_websocket_connection(u, headers=h, options=o))
            connection = await factory(url, headers, options)
            session = CallbackRealtimeSession(
                provider="vertex",
                model_id=self.model_id,
                capabilities=self.capabilities,
                config=resolved_config,
                connection=connection,
                callbacks=RealtimeSessionCallbacks(
                    parse_event=parse_event,
                    build_audio_payloads=_gemini_realtime_build_audio,
                    build_text_payloads=build_text,
                    build_tool_result_payloads=_gemini_realtime_build_tool_result,
                    build_update_payloads=lambda session_config: _gemini_realtime_build_update(session_config, self.model_id),
                    build_initial_payloads=lambda session_config: [setup],
                ),
            )
            await session.initialize(ready_event="setupComplete", timeout_ms=options.timeout_ms if options and options.timeout_ms is not None else 10_000)
            return session

        async def create_browser_token(
            self,
            config: RealtimeSessionConfig | None = None,
            options: RealtimeConnectOptions | None = None,
        ) -> RealtimeTokenResult:
            return await unsupported_browser_token(config=config, options=options)

    def language_model(model_id: str) -> VertexChatLanguageModel | VertexLanguageModel:
        model_id = resolve_maas_model(model_id)
        if model_id in MISTRAL_VERTEX_MODELS:
            if express:
                raise ConfigurationError("Mistral requires a standard Vertex project configuration.")
            return VertexChatLanguageModel(
                model_id=model_id, base_url=resolved_base.rstrip("/"),
                fetch=wrapped_fetch, capabilities=maas_capabilities(model_id), raw_predict=True,
            )
        if model_id in VERTEX_MAAS_TEXT_MODELS | {GEMMA_MAAS_MODEL}:
            if express:
                raise ConfigurationError("MaaS requires a standard Vertex project configuration.")
            return VertexChatLanguageModel(
                model_id=model_id, base_url=f"{resolved_base.rstrip('/')}/endpoints/openapi",
                fetch=wrapped_fetch, capabilities=maas_capabilities(model_id),
            )
        return VertexLanguageModel(
            provider="vertex",
            model_id=model_id,
            api_key="unused",
            base_url=resolved_base.rstrip("/"),
            fetch=wrapped_fetch,
            capabilities=replace(GEMINI_CAPABILITIES, tools=False, tool_choice=False, embeddings=False, agent_capabilities=AgentCapabilities()) if model_id == "gemini-3.8-flash-cyber" else GEMINI_CAPABILITIES,
        )

    def grounded_language_model(model_id: str) -> VertexGroundedLanguageModel:
        if model_id == "gemini-3.8-flash-cyber":
            raise UnsupportedFeatureError("Gemini 3.8 Flash Cyber does not support grounding.")
        return VertexGroundedLanguageModel(
            provider="vertex", model_id=model_id, api_key="unused",
            base_url=resolved_base.rstrip("/"), fetch=wrapped_fetch,
        )

    native = ProviderAdapter(
        name="vertex",
        language_model_factory=language_model,
        embedding_model_factory=lambda model_id: VertexEmbeddingModel(
            provider="vertex",
            model_id=model_id,
            base_url=resolved_base.rstrip("/"),
            access_token=resolved_token or "",
            fetch=wrapped_fetch,
        ),
        speech_model_factory=lambda model_id: VertexSpeechModel(
            provider="vertex",
            model_id=model_id,
            api_key="unused",
            base_url=resolved_base.rstrip("/"),
            fetch=wrapped_fetch,
        ),
        transcription_model_factory=lambda model_id: VertexTranscriptionModel(
            provider="vertex",
            model_id=model_id,
            api_key="unused",
            base_url=resolved_base.rstrip("/"),
            fetch=wrapped_fetch,
        ),
        grounded_language_model_factory=grounded_language_model,
        count_tokens_client_factory=lambda: VertexCountTokensClient(
            provider="vertex",
            api_key="unused",
            base_url=resolved_base.rstrip("/"),
            fetch=wrapped_fetch,
        ),
        images_client_factory=lambda: GeminiImagesClient(
            provider="vertex",
            api_key=None,
            base_url=resolved_base.rstrip("/"),
            fetch=wrapped_fetch,
            vertex=True,
        ),
        videos_client_factory=lambda: VertexVideosClient(
            provider="vertex",
            api_key=None,
            base_url=resolved_base.rstrip("/"),
            fetch=wrapped_fetch,
            vertex=True,
        ),
        media_client_factory=lambda: VertexMediaClient(
            provider="vertex",
            api_key=None,
            base_url=resolved_base.rstrip("/"),
            fetch=wrapped_fetch,
            vertex=True,
        ),
        rag_client_factory=(lambda: VertexRagClient(resolved_base, wrapped_fetch, "ragCorpora")) if not express else None,
        model_garden_client_factory=(lambda: VertexModelGardenClient(resolved_base, wrapped_fetch)) if not express else None,
        agent_platform_client_factory=(lambda: VertexAgentPlatformClient(resolved_base, wrapped_fetch)) if not express else None,
        batches_client_factory=(lambda: VertexBatchesClient(provider="vertex", api_key="", base_url=resolved_base, fetch=wrapped_fetch)) if not express else None,
        caches_client_factory=(lambda: VertexCachedContentsClient(api_key="", base_url=resolved_base, fetch=wrapped_fetch)) if not express else None,
        interactions_client_factory=(lambda: VertexInteractionsClient(api_key="", base_url=resolved_base.replace(f"/{api_version}/", "/v1beta1/", 1), fetch=wrapped_fetch)) if not express else None,
        realtime_model_factory=lambda model_id: VertexRealtimeModel(model_id),
    )
    return create_provider_bundle(
        name="vertex",
        native=native,
        agent_capabilities=GEMINI_CAPABILITIES.agent_capabilities or AgentCapabilities(),
        portable_support=PortableSupport(
            text_generation=True,
            streaming=True,
            structured_output=True,
            tools=True,
            embeddings=True,
            grounding=True,
            retrieval=True,
            transcription=True,
            speech=True,
            portable_badge=True,
            tier="portable",
        ),
    )
