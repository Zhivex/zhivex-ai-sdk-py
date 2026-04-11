from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .._http import Fetcher, default_fetch
from ..errors import ConfigurationError
from ..realtime import CallbackRealtimeSession, RealtimeConnectionFactory, RealtimeSessionCallbacks, open_websocket_connection, unsupported_browser_token
from ..runtime import with_retry
from ..types import EmbedResult, EmbeddingModel, ModelCapabilities, RealtimeConnectOptions, RealtimeSession, RealtimeSessionConfig, RealtimeTokenResult
from .base import ProviderAdapter
from .gemini import (
    GEMINI_CAPABILITIES,
    GEMINI_REALTIME_CAPABILITIES,
    GeminiLanguageModel,
    GeminiSpeechModel,
    _gemini_realtime_build_audio,
    _gemini_realtime_build_text,
    _gemini_realtime_build_tool_result,
    _gemini_realtime_build_update,
    _gemini_realtime_parse_event,
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
        response = await with_retry(
            lambda: self.fetch(
                self._url(),
                headers=self._headers(),
                json_body={"instances": [{"content": value} for value in values]},
                timeout_ms=getattr(options, "timeout_ms", None),
            ),
            max_retries=getattr(options, "max_retries", 0) or 0,
            retry_backoff_ms=getattr(options, "retry_backoff_ms", 250) or 250,
        )
        payload = await response.json()
        return EmbedResult(embeddings=[prediction.get("embeddings", {}).get("values", []) for prediction in payload.get("predictions", [])], raw_response=payload)


def create_vertex(
    *,
    access_token: str | None = None,
    project_id: str | None = None,
    location: str = "us-central1",
    api_version: str = "v1beta1",
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
        async def __call__(self, url: str, *, headers: dict[str, str], json_body: dict[str, object], timeout_ms: int | None, stream: bool = False):
            merged = dict(headers)
            merged["authorization"] = f"Bearer {resolved_token}"
            return await requester(url, headers=merged, json_body=json_body, timeout_ms=timeout_ms, stream=stream)

    wrapped_fetch = VertexFetcher()

    class VertexLanguageModel(GeminiLanguageModel):
        def _url(self, action: str) -> str:  # type: ignore[override]
            return f"{self.base_url}/publishers/google/models/{self.model_id}:{action}"

    class VertexSpeechModel(GeminiSpeechModel):
        def _url(self, action: str) -> str:  # type: ignore[override]
            return f"{self.base_url}/publishers/google/models/{self.model_id}:{action}"

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

    return ProviderAdapter(
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
        realtime_model_factory=lambda model_id: VertexRealtimeModel(model_id),
    )
