from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .._http import Fetcher, default_fetch
from ..errors import ConfigurationError
from ..runtime import with_retry
from ..types import EmbedResult, EmbeddingModel, ModelCapabilities
from .base import ProviderAdapter
from .gemini import GEMINI_CAPABILITIES, GeminiLanguageModel


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
    )
