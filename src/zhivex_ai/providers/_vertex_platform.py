"""Beta clients for Google-managed agent resources, with raw Google payloads."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import re
from typing import Any

from .._http import Fetcher
from ..errors import ProviderHTTPError, ValidationError
from ..types import RetryOptions, ModelCapabilities
from ._vertex_chat import (
    VertexChatLanguageModel,
    MISTRAL_VERTEX_MODELS,
    VERTEX_MAAS_TEXT_CAPABILITIES,
    maas_capabilities,
    resolve_maas_model,
)
from ._vertex_native import _request, resource_url
from .openai_compat import OpenAICompatibleLanguageModel


class VertexResponsesLanguageModel(OpenAICompatibleLanguageModel):
    """Normalize Responses while leaving authentication to the Vertex fetcher."""

    _require_terminal_event = True

    def _headers(self, *, json_content: bool = True) -> dict[str, str]:
        return {"content-type": "application/json"} if json_content else {}


@dataclass
class VertexResourceClient:
    base_url: str
    fetch: Fetcher
    collection: str

    def _path(self, name: str) -> str:
        return (
            name
            if name.startswith(("projects/", self.collection + "/"))
            else f"{self.collection}/{name}"
        )

    async def create(
        self,
        body: dict[str, Any],
        *,
        params: dict[str, Any] | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        return await _request(
            self.fetch,
            resource_url(self.base_url, self.collection, params),
            "POST",
            body,
            options,
        )

    async def get(
        self, name: str, options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await _request(
            self.fetch,
            resource_url(self.base_url, self._path(name)),
            "GET",
            None,
            options,
        )

    async def list(
        self,
        *,
        page_size: int | None = None,
        page_token: str | None = None,
        filter: str | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        return await _request(
            self.fetch,
            resource_url(
                self.base_url,
                self.collection,
                {"pageSize": page_size, "pageToken": page_token, "filter": filter},
            ),
            "GET",
            None,
            options,
        )

    async def update(
        self,
        name: str,
        body: dict[str, Any],
        *,
        update_mask: str,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        return await _request(
            self.fetch,
            resource_url(self.base_url, self._path(name), {"updateMask": update_mask}),
            "PATCH",
            body,
            options,
        )

    async def delete(
        self, name: str, *, force: bool = False, options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await _request(
            self.fetch,
            resource_url(
                self.base_url, self._path(name), {"force": "true"} if force else None
            ),
            "DELETE",
            None,
            options,
        )


class VertexSessionsClient(VertexResourceClient):
    async def append_event(
        self, name: str, body: dict[str, Any], options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await _request(
            self.fetch,
            resource_url(self.base_url, self._path(name) + ":appendEvent"),
            "POST",
            body,
            options,
        )

    async def events(
        self,
        name: str,
        *,
        page_size: int | None = None,
        page_token: str | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        return await _request(
            self.fetch,
            resource_url(
                self.base_url,
                self._path(name) + "/events",
                {"pageSize": page_size, "pageToken": page_token},
            ),
            "GET",
            None,
            options,
        )


class VertexMemoryBankClient(VertexResourceClient):
    async def generate(
        self, body: dict[str, Any], options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await _request(
            self.fetch,
            resource_url(self.base_url, self.collection + ":generate"),
            "POST",
            body,
            options,
        )

    async def retrieve(
        self, body: dict[str, Any], options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await _request(
            self.fetch,
            resource_url(self.base_url, self.collection + ":retrieve"),
            "POST",
            body,
            options,
        )


class VertexAgentRuntimeClient(VertexResourceClient):
    async def query(
        self, name: str, body: dict[str, Any], options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await _request(
            self.fetch,
            resource_url(self.base_url, self._path(name) + ":query"),
            "POST",
            body,
            options,
        )

    async def stream_query(
        self, name: str, body: dict[str, Any], options: RetryOptions | None = None
    ) -> Any:
        response = await self.fetch(
            resource_url(
                self.base_url, self._path(name) + ":streamQuery", {"alt": "sse"}
            ),
            headers={"content-type": "application/json"},
            json_body=body,
            timeout_ms=options.timeout_ms if options else None,
            stream=True,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Vertex request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        return response


@dataclass
class VertexAgentPlatformClient:
    base_url: str
    fetch: Fetcher

    def agents(self) -> VertexAgentRuntimeClient:
        return VertexAgentRuntimeClient(self.base_url, self.fetch, "reasoningEngines")

    def _agent_base(self, agent: str, *, beta: bool = False) -> str:
        if (
            not agent
            or "/" in agent
            and not agent.startswith(("projects/", "reasoningEngines/"))
        ):
            raise ValidationError(
                "Expected a Vertex agent ID or reasoningEngines resource name."
            )
        path = (
            agent
            if agent.startswith(("projects/", "reasoningEngines/"))
            else f"reasoningEngines/{agent}"
        )
        base = resource_url(self.base_url, path)
        if beta:
            # Preserve the explicitly configured origin while selecting the
            # documented Memory Bank API version.
            from urllib.parse import urlsplit

            version = urlsplit(base).path.strip("/").split("/")[0]
            base = base.replace(f"/{version}/", "/v1beta1/", 1)
        return base

    def sessions(self, agent: str) -> VertexSessionsClient:
        return VertexSessionsClient(self._agent_base(agent), self.fetch, "sessions")

    def memory_bank(self, agent: str) -> VertexMemoryBankClient:
        return VertexMemoryBankClient(
            self._agent_base(agent, beta=True), self.fetch, "memories"
        )

    async def get_operation(
        self, name: str, options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await _request(
            self.fetch, resource_url(self.base_url, name), "GET", None, options
        )

    async def wait_operation(
        self, name: str, *, poll_interval_ms: int = 1000, timeout_ms: int = 60_000
    ) -> dict[str, Any]:
        async with asyncio.timeout(timeout_ms / 1000):
            while True:
                operation = await self.get_operation(name)
                if operation.get("done"):
                    return operation
                await asyncio.sleep(max(poll_interval_ms, 1) / 1000)


@dataclass
class VertexModelGardenClient:
    base_url: str
    fetch: Fetcher

    def _registered_model_path(self, model: str) -> str:
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_-]+(?:@[A-Za-z0-9_-]+)?", model):
            raise ValidationError("Expected a registered model ID, optionally with a version or alias.")
        return f"models/{model}"

    async def upload_model(self, body: dict[str, Any], *, options: RetryOptions | None = None) -> dict[str, Any]:
        """Register an artifact using Google's Model payload; return its operation."""
        return await _request(self.fetch, resource_url(self.base_url, "models:upload"), "POST", body, options)

    async def get_registered_model(self, model: str, *, options: RetryOptions | None = None) -> dict[str, Any]:
        return await _request(self.fetch, resource_url(self.base_url, self._registered_model_path(model)), "GET", None, options)

    async def list_registered_models(
        self, *, page_size: int | None = None, page_token: str | None = None,
        filter: str | None = None, order_by: str | None = None, read_mask: str | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        """One project model-registry page, separate from publisher catalog models."""
        return await _request(self.fetch, resource_url(self.base_url, "models", {
            "pageSize": page_size, "pageToken": page_token, "filter": filter,
            "orderBy": order_by, "readMask": read_mask,
        }), "GET", None, options)

    async def update_registered_model(
        self, model: str, body: dict[str, Any], *, update_mask: str,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        return await _request(self.fetch, resource_url(self.base_url, self._registered_model_path(model), {"updateMask": update_mask}), "PATCH", body, options)

    async def delete_registered_model(self, model: str, *, options: RetryOptions | None = None) -> dict[str, Any]:
        """Delete a registered model; does not automatically undeploy it."""
        return await _request(self.fetch, resource_url(self.base_url, self._registered_model_path(model)), "DELETE", None, options)

    def _endpoint_url(self, endpoint: str, action: str | None = None) -> str:
        if not isinstance(endpoint, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", endpoint):
            raise ValidationError("Expected a Vertex endpoint ID, not a URL or resource path.")
        return resource_url(self.base_url, f"endpoints/{endpoint}" + (f":{action}" if action else ""))

    async def get_endpoint(self, endpoint: str, *, options: RetryOptions | None = None) -> dict[str, Any]:
        return await _request(self.fetch, self._endpoint_url(endpoint), "GET", None, options)

    async def create_endpoint(
        self, body: dict[str, Any], *, endpoint_id: str | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        """Create an endpoint; returns the operation without polling or deploying."""
        if endpoint_id is not None:
            self._endpoint_url(endpoint_id)
        return await _request(self.fetch, resource_url(self.base_url, "endpoints", {"endpointId": endpoint_id}), "POST", body, options)

    async def update_endpoint(
        self, endpoint: str, body: dict[str, Any], *, update_mask: str,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        self._endpoint_url(endpoint)
        return await _request(self.fetch, resource_url(self.base_url, f"endpoints/{endpoint}", {"updateMask": update_mask}), "PATCH", body, options)

    async def delete_endpoint(self, endpoint: str, *, options: RetryOptions | None = None) -> dict[str, Any]:
        """Delete an empty endpoint; never implicitly undeploy its models."""
        return await _request(self.fetch, self._endpoint_url(endpoint), "DELETE", None, options)

    async def deploy_model(
        self, endpoint: str, body: dict[str, Any], *, options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        """Submit native deployedModel/trafficSplit; returns a Google operation."""
        return await _request(self.fetch, self._endpoint_url(endpoint, "deployModel"), "POST", body, options)

    async def undeploy_model(
        self, endpoint: str, body: dict[str, Any], *, options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        """Submit native deployedModelId/trafficSplit without deleting the endpoint."""
        return await _request(self.fetch, self._endpoint_url(endpoint, "undeployModel"), "POST", body, options)

    async def get_operation(self, name: str, options: RetryOptions | None = None) -> dict[str, Any]:
        if not re.fullmatch(r"(?:projects/[^/]+/locations/[^/]+/)?(?:(?:endpoints|models)/[A-Za-z0-9_-]+/)?operations/[A-Za-z0-9_-]+", name):
            raise ValidationError("Expected a Vertex model, endpoint or regional operation name.")
        return await _request(self.fetch, resource_url(self.base_url, name), "GET", None, options)

    async def wait_operation(
        self, name: str, *, poll_interval_ms: int = 1000, timeout_ms: int = 60_000,
    ) -> dict[str, Any]:
        """Wait for done, preserving terminal errors; timeout never resubmits."""
        async with asyncio.timeout(timeout_ms / 1000):
            while True:
                operation = await self.get_operation(name)
                if operation.get("done"):
                    return operation
                await asyncio.sleep(max(poll_interval_ms, 1) / 1000)

    async def list_endpoints(
        self, *, page_size: int | None = None, page_token: str | None = None,
        filter: str | None = None, order_by: str | None = None,
        read_mask: str | None = None, options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        """Return one endpoint page; callers explicitly follow nextPageToken."""
        return await _request(
            self.fetch, resource_url(self.base_url, "endpoints", {
                "pageSize": page_size, "pageToken": page_token, "filter": filter,
                "orderBy": order_by, "readMask": read_mask,
            }), "GET", None, options,
        )

    async def endpoint_predict(
        self, *, endpoint: str, body: dict[str, Any], options: RetryOptions | None = None,
    ) -> Any:
        """Native instances/parameters prediction for a deployed endpoint."""
        return await _request(self.fetch, self._endpoint_url(endpoint, "predict"), "POST", body, options)

    async def endpoint_raw_predict(
        self, *, endpoint: str, body: bytes, content_type: str = "application/json",
        stream: bool = False, options: RetryOptions | None = None,
    ) -> Any:
        """Arbitrary binary request; returns the caller-owned HTTP response."""
        from ..runtime import with_retry

        url = self._endpoint_url(endpoint, "streamRawPredict" if stream else "rawPredict")
        if not isinstance(body, bytes):
            raise ValidationError("Raw endpoint prediction requires bytes.")
        if not content_type or any(ord(c) < 32 or ord(c) == 127 for c in content_type):
            raise ValidationError("Expected a valid content type.")

        async def send():
            response = await self.fetch(
                url, method="POST", headers={"content-type": content_type}, body=body, json_body=None,
                timeout_ms=options.timeout_ms if options else None, stream=stream,
            )
            if response.status_code >= 400:
                raise ProviderHTTPError(
                    f"Vertex request failed with status {response.status_code}.",
                    response.status_code, response_body=await response.text(),
                )
            return response

        return await with_retry(
            send, max_retries=(options.max_retries or 0) if options else 0,
            retry_backoff_ms=options.retry_backoff_ms if options and options.retry_backoff_ms is not None else 250,
        )

    def responses_model(
        self, model: str, *, endpoint: str = "openapi",
        capabilities: ModelCapabilities | None = None,
    ) -> VertexResponsesLanguageModel:
        """Explicit normalized Responses route; defaults to text and streaming.

        Additional capabilities must be verified for the chosen deployment.
        This factory does not enable the stored-response lifecycle.
        """
        if not model or not endpoint or "/" in endpoint:
            raise ValidationError("Expected a model ID and a Vertex endpoint ID.")
        return VertexResponsesLanguageModel(
            provider="vertex", model_id=model, api_key="",
            base_url=resource_url(self.base_url, f"endpoints/{endpoint}"),
            fetch=self.fetch,
            capabilities=capabilities if capabilities is not None else VERTEX_MAAS_TEXT_CAPABILITIES,
        )

    def language_model(
        self,
        model: str,
        *,
        endpoint: str = "openapi",
        capabilities: ModelCapabilities | None = None,
    ) -> VertexChatLanguageModel:
        """Normalized Chat Completions for MaaS or a deployed endpoint.

        Unknown models default to text and streaming only. Callers can supply
        capabilities verified for their particular deployment.
        """
        if not model or not endpoint or "/" in endpoint:
            raise ValidationError("Expected a model ID and a Vertex endpoint ID.")
        model = resolve_maas_model(model)
        resolved = capabilities if capabilities is not None else maas_capabilities(model)
        if model in MISTRAL_VERTEX_MODELS and endpoint == "openapi":
            return VertexChatLanguageModel(
                model_id=model, base_url=self.base_url.rstrip("/"), fetch=self.fetch,
                capabilities=resolved, raw_predict=True,
            )
        return VertexChatLanguageModel(
            model_id=model,
            base_url=resource_url(self.base_url, f"endpoints/{endpoint}"),
            fetch=self.fetch,
            capabilities=resolved,
        )

    def _model(self, publisher: str, model: str) -> str:
        if "/" in publisher or "/" in model or not publisher or not model:
            raise ValidationError(
                "Expected a publisher and model ID, not a resource path."
            )
        return f"publishers/{publisher}/models/{model}"

    async def embed_content(
        self, *, model: str, body: dict[str, Any], publisher: str = "google",
        options: RetryOptions | None = None,
    ) -> Any:
        """Native embedContent payload and response, including model-specific config."""
        url = resource_url(self.base_url, self._model(publisher, model) + ":embedContent")
        return await _request(self.fetch, url, "POST", body, options)

    async def raw_predict(
        self,
        *,
        publisher: str,
        model: str,
        body: dict[str, Any],
        stream: bool = False,
        options: RetryOptions | None = None,
    ) -> Any:
        action = "streamRawPredict" if stream else "rawPredict"
        url = resource_url(self.base_url, self._model(publisher, model) + ":" + action)
        if not stream:
            return await _request(self.fetch, url, "POST", body, options)
        response = await self.fetch(
            url,
            headers={"content-type": "application/json"},
            json_body=body,
            timeout_ms=options.timeout_ms if options else None,
            stream=True,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Vertex request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        return response

    async def mistral_ocr(
        self, body: dict[str, Any], *, model: str = "mistral-ocr-2505",
        options: RetryOptions | None = None,
    ) -> Any:
        """Native unary OCR; preserves pages, images, annotations and usage."""
        from copy import deepcopy

        document = body.get("document")
        if not isinstance(document, dict):
            raise ValidationError("Mistral OCR requires a document object.")
        kind = document.get("type")
        if kind not in ("document_url", "image_url") or not isinstance(document.get(kind), str) or not document[kind]:
            raise ValidationError("Mistral OCR requires a document_url or image_url string.")
        if "stream" in body and body["stream"] is not False:
            raise ValidationError("This Mistral OCR method supports unary responses only.")
        payload = deepcopy(body)
        payload["model"] = model.split("@", 1)[0]
        return await self.raw_predict(
            publisher="mistralai", model=model, body=payload, options=options,
        )

    async def codestral_fim(
        self, body: dict[str, Any], *, model: str = "codestral-2",
        options: RetryOptions | None = None,
    ) -> Any:
        """Native Codestral completion; streaming returns a caller-owned response."""
        from copy import deepcopy

        if not isinstance(body.get("prompt"), str):
            raise ValidationError("Codestral FIM requires a string prompt.")
        if "suffix" in body and body["suffix"] is not None and not isinstance(body["suffix"], str):
            raise ValidationError("Codestral FIM suffix must be a string or null.")
        if "stream" in body and not isinstance(body["stream"], bool):
            raise ValidationError("Codestral FIM stream must be a boolean.")
        if "messages" in body:
            raise ValidationError("Codestral FIM accepts prompt/suffix, not chat messages.")
        payload = deepcopy(body)
        payload["model"] = model.split("@", 1)[0]
        return await self.raw_predict(
            publisher="mistralai", model=model, body=payload,
            stream=payload.get("stream", False), options=options,
        )

    async def anthropic_messages(
        self, *, model: str, body: dict[str, Any], options: RetryOptions | None = None
    ) -> Any:
        from copy import deepcopy

        payload = deepcopy(body)
        payload.pop("model", None)
        payload["anthropic_version"] = "vertex-2023-10-16"
        return await self.raw_predict(
            publisher="anthropic",
            model=model,
            body=payload,
            stream=bool(payload.get("stream")),
            options=options,
        )

    async def chat_completions(
        self,
        body: dict[str, Any],
        *,
        endpoint: str = "openapi",
        options: RetryOptions | None = None,
    ) -> Any:
        return await self._openapi_request("chat/completions", body, endpoint, options)

    async def responses(
        self,
        body: dict[str, Any],
        *,
        endpoint: str = "openapi",
        options: RetryOptions | None = None,
    ) -> Any:
        """Native Responses POST; streaming returns a caller-owned HTTP response.

        This preserves the provider's schema and does not imply support for
        stored-response retrieval, deletion, or every OpenAI Responses feature.
        """
        return await self._openapi_request("responses", body, endpoint, options)

    async def _openapi_request(
        self, action: str, body: dict[str, Any], endpoint: str,
        options: RetryOptions | None,
    ) -> Any:
        if "/" in endpoint or not endpoint:
            raise ValidationError("Expected a Vertex endpoint ID.")
        url = resource_url(self.base_url, f"endpoints/{endpoint}/{action}")
        if not body.get("stream"):
            return await _request(self.fetch, url, "POST", body, options)
        response = await self.fetch(
            url,
            headers={"content-type": "application/json"},
            json_body=body,
            timeout_ms=options.timeout_ms if options else None,
            stream=True,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Vertex request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        return response

    def _catalog_base(self) -> str:
        from urllib.parse import urlsplit

        parsed = urlsplit(self.base_url)
        return f"{parsed.scheme}://{parsed.netloc}/v1beta1"

    async def get_model(
        self, *, publisher: str, model: str, options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await _request(
            self.fetch,
            resource_url(self._catalog_base(), self._model(publisher, model)),
            "GET",
            None,
            options,
        )

    async def list_models(
        self,
        *,
        publisher: str = "google",
        page_size: int | None = None,
        page_token: str | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        self._model(publisher, "validation")
        return await _request(
            self.fetch,
            resource_url(
                self._catalog_base(),
                f"publishers/{publisher}/models",
                {"pageSize": page_size, "pageToken": page_token},
            ),
            "GET",
            None,
            options,
        )


class VertexRagClient(VertexResourceClient):
    """Beta RAG corpus lifecycle and retrieval using native Google JSON."""

    async def get_engine_config(self, options: RetryOptions | None = None) -> dict[str, Any]:
        return await _request(self.fetch, resource_url(self.base_url, "ragEngineConfig"), "GET", None, options)

    async def update_engine_config(self, body: dict[str, Any], options: RetryOptions | None = None) -> dict[str, Any]:
        """Update the configured region's engine; returns a raw operation."""
        return await _request(self.fetch, resource_url(self.base_url, "ragEngineConfig"), "PATCH", body, options)

    def _path(self, name: str) -> str:
        path = super()._path(name)
        if not re.fullmatch(r"(?:projects/[^/:]+/locations/[^/:]+/)?ragCorpora/[A-Za-z0-9_-]+", path):
            raise ValidationError("Expected a RAG corpus ID or resource name.")
        return path

    async def update(self, name: str, body: dict[str, Any], *, update_mask: str | None = None, options: RetryOptions | None = None) -> dict[str, Any]:
        if update_mask is not None:
            raise ValidationError("RAG corpus patch does not accept updateMask.")
        return await _request(self.fetch, resource_url(self.base_url, self._path(name)), "PATCH", body, options)

    def _files(self, corpus: str) -> str:
        return self._path(corpus) + "/ragFiles"

    def _file(self, corpus: str, file_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
            raise ValidationError("Expected a RAG file ID.")
        return self._files(corpus) + "/" + file_id

    async def import_files(self, corpus: str, body: dict[str, Any], options: RetryOptions | None = None) -> dict[str, Any]:
        return await _request(self.fetch, resource_url(self.base_url, self._files(corpus) + ":import"), "POST", body, options)

    async def list_files(self, corpus: str, *, page_size: int | None = None, page_token: str | None = None, options: RetryOptions | None = None) -> dict[str, Any]:
        return await _request(self.fetch, resource_url(self.base_url, self._files(corpus), {"pageSize": page_size, "pageToken": page_token}), "GET", None, options)

    async def get_file(self, corpus: str, file_id: str, options: RetryOptions | None = None) -> dict[str, Any]:
        return await _request(self.fetch, resource_url(self.base_url, self._file(corpus, file_id)), "GET", None, options)

    async def delete_file(self, corpus: str, file_id: str, options: RetryOptions | None = None) -> dict[str, Any]:
        return await _request(self.fetch, resource_url(self.base_url, self._file(corpus, file_id)), "DELETE", None, options)

    async def retrieve_contexts(self, body: dict[str, Any], options: RetryOptions | None = None) -> dict[str, Any]:
        return await _request(self.fetch, self.base_url.rstrip("/") + ":retrieveContexts", "POST", body, options)

    async def get_operation(self, name: str, options: RetryOptions | None = None) -> dict[str, Any]:
        if not re.fullmatch(r"(?:projects/[^/:]+/locations/[^/:]+/)?operations/[A-Za-z0-9_-]+", name):
            raise ValidationError("Expected an operation resource name.")
        return await _request(self.fetch, resource_url(self.base_url, name), "GET", None, options)

    async def wait_operation(self, name: str, *, poll_interval_ms: int = 1000, timeout_ms: int = 60_000) -> dict[str, Any]:
        async with asyncio.timeout(timeout_ms / 1000):
            while True:
                operation = await self.get_operation(name)
                if operation.get("done"):
                    return operation
                await asyncio.sleep(max(poll_interval_ms, 1) / 1000)
