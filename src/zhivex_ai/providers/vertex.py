from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .._http import Fetcher, default_fetch
from ..errors import ConfigurationError, ProviderHTTPError
from ..realtime import CallbackRealtimeSession, RealtimeConnectionFactory, RealtimeSessionCallbacks, open_websocket_connection, unsupported_browser_token
from ..runtime import with_retry
from ..types import CountTokensResult, EmbedResult, EmbeddingModel, ModelCapabilities, PortableSupport, RealtimeConnectOptions, RealtimeSession, RealtimeSessionConfig, RealtimeTokenResult, TokenCountDetail
from .base import ProviderAdapter, create_provider_bundle
from ._payload import drop_none
from .gemini import (
    GEMINI_CAPABILITIES,
    GEMINI_GROUNDED_CAPABILITIES,
    GEMINI_REALTIME_CAPABILITIES,
    GeminiCountTokensClient,
    GeminiGroundedLanguageModel,
    GeminiLanguageModel,
    GeminiSpeechModel,
    GeminiTranscriptionModel,
    _embedding_request_options,
    _gemini_realtime_build_audio,
    _gemini_realtime_build_text,
    _gemini_realtime_build_tool_result,
    _gemini_realtime_build_update,
    _gemini_realtime_parse_event,
    _provider_option_value,
)


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
        return {"content-type": "application/json", "authorization": f"Bearer {self.access_token}"}

    async def embed(self, values: list[str], options: Any = None) -> EmbedResult:
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
                            "content": value,
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
        payload = await response.json()
        return EmbedResult(embeddings=[prediction.get("embeddings", {}).get("values", []) for prediction in payload.get("predictions", [])], raw_response=payload)


def create_vertex(
    *,
    access_token: str | None = None,
    project_id: str | None = None,
    location: str = "us-central1",
    api_version: str = "v1",
    base_url: str | None = None,
    fetch: Fetcher | None = None,
    realtime_url: str | None = None,
    realtime_connection_factory: RealtimeConnectionFactory | None = None,
):
    resolved_token = access_token or os.getenv("VERTEX_ACCESS_TOKEN") or os.getenv("GOOGLE_ACCESS_TOKEN")
    if not resolved_token:
        raise ConfigurationError("Missing Vertex access token.")
    resolved_project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if not resolved_project and not base_url:
        raise ConfigurationError("Missing Vertex project ID.")
    resolved_base = base_url or f"https://{location}-aiplatform.googleapis.com/{api_version}/projects/{resolved_project}/locations/{location}"
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
            merged = dict(headers)
            merged["authorization"] = f"Bearer {resolved_token}"
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

    wrapped_fetch = VertexFetcher()

    class VertexLanguageModel(GeminiLanguageModel):
        def _url(self, action: str) -> str:  # type: ignore[override]
            return f"{self.base_url}/publishers/google/models/{self.model_id}:{action}"

    class VertexSpeechModel(GeminiSpeechModel):
        def _url(self, action: str) -> str:  # type: ignore[override]
            return f"{self.base_url}/publishers/google/models/{self.model_id}:{action}"

    class VertexTranscriptionModel(GeminiTranscriptionModel):
        def _url(self, action: str) -> str:  # type: ignore[override]
            return f"{self.base_url}/publishers/google/models/{self.model_id}:{action}"

    class VertexGroundedLanguageModel(GeminiGroundedLanguageModel):
        capabilities = GEMINI_GROUNDED_CAPABILITIES

        def _url(self, action: str) -> str:  # type: ignore[override]
            return f"{self.base_url}/publishers/google/models/{self.model_id}:{action}"

    class VertexCountTokensClient(GeminiCountTokensClient):
        def _url(self, model_id: str) -> str:  # type: ignore[override]
            return f"{resolved_base.rstrip('/')}/publishers/google/models/{model_id}:countTokens"

        async def count(self, **kwargs: Any) -> CountTokensResult:  # type: ignore[override]
            model_id = kwargs["model_id"]
            prompt = kwargs.get("prompt")
            messages = kwargs.get("messages")
            system = kwargs.get("system")
            tools = kwargs.get("tools")
            provider_options = kwargs.get("provider_options")
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
        capabilities = GEMINI_REALTIME_CAPABILITIES

        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        async def connect(
            self,
            config: RealtimeSessionConfig | None = None,
            options: RealtimeConnectOptions | None = None,
        ) -> RealtimeSession:
            resolved_config = config or RealtimeSessionConfig()
            url = realtime_url or resolved_config.provider_options.get("realtime_url") if resolved_config.provider_options else realtime_url
            if not url:
                url = f"wss://{location}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.{api_version}.PredictionService.BidiGenerateContent"
            headers = {
                "authorization": f"Bearer {resolved_token}",
                **dict((resolved_config.provider_options or {}).get("headers") or {}),
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
                    parse_event=_gemini_realtime_parse_event,
                    build_audio_payloads=_gemini_realtime_build_audio,
                    build_text_payloads=_gemini_realtime_build_text,
                    build_tool_result_payloads=_gemini_realtime_build_tool_result,
                    build_update_payloads=lambda session_config: _gemini_realtime_build_update(session_config, self.model_id),
                    build_initial_payloads=lambda session_config: _gemini_realtime_build_update(session_config, self.model_id),
                ),
            )
            await session.initialize()
            return session

        async def create_browser_token(
            self,
            config: RealtimeSessionConfig | None = None,
            options: RealtimeConnectOptions | None = None,
        ) -> RealtimeTokenResult:
            return await unsupported_browser_token(config=config, options=options)

    native = ProviderAdapter(
        name="vertex",
        language_model_factory=lambda model_id: VertexLanguageModel(
            provider="vertex",
            model_id=model_id,
            api_key="unused",
            base_url=resolved_base.rstrip("/"),
            fetch=wrapped_fetch,
        ),
        embedding_model_factory=lambda model_id: VertexEmbeddingModel(
            provider="vertex",
            model_id=model_id,
            base_url=resolved_base.rstrip("/"),
            access_token=resolved_token,
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
        grounded_language_model_factory=lambda model_id: VertexGroundedLanguageModel(
            provider="vertex",
            model_id=model_id,
            api_key="unused",
            base_url=resolved_base.rstrip("/"),
            fetch=wrapped_fetch,
        ),
        count_tokens_client_factory=lambda: VertexCountTokensClient(
            provider="vertex",
            api_key="unused",
            base_url=resolved_base.rstrip("/"),
            fetch=wrapped_fetch,
        ),
        realtime_model_factory=lambda model_id: VertexRealtimeModel(model_id),
    )
    return create_provider_bundle(
        name="vertex",
        native=native,
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
