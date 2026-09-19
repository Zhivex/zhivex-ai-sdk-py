"""Google Cloud native resource clients (separate from Express Mode)."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from .._http import Fetcher
from ..errors import ProviderHTTPError, ValidationError, UnsupportedFeatureError
from ..types import (
    CachedContent,
    RetryOptions,
    VideoOperation,
    GeneratedMedia,
    MediaResult,
)
from ..runtime import with_retry
from .gemini import (
    GeminiBatchesClient,
    GeminiCachedContentsClient,
    GeminiInteractionsClient,
    GeminiVideosClient,
    GeminiMediaClient,
    _map_part,
    _normalize_cached_content,
)


def resource_url(base_url: str, path: str, params: dict[str, Any] | None = None) -> str:
    """Accept resource names, never destinations supplied by a response."""
    if any(c in path for c in ("?", "#", "%", "\\")) or any(
        p in {".", "..", ""} for p in path.split("/")
    ):
        raise ValidationError("Invalid Vertex resource name.")
    if "://" in path or path.startswith("/") or any(ord(c) <= 32 for c in path):
        raise ValidationError("Invalid Vertex resource name.")
    base = base_url.rstrip("/")
    if path.startswith("projects/"):
        parsed = urlsplit(base)
        version = parsed.path.strip("/").split("/")[0]
        url = f"{parsed.scheme}://{parsed.netloc}/{version}/{path}"
    else:
        url = f"{base}/{path}"
    query = urlencode(
        {
            k: str(v).lower() if isinstance(v, bool) else v
            for k, v in (params or {}).items()
            if v is not None
        }
    )
    return url + (f"?{query}" if query else "")


class VertexCachedContentsClient(GeminiCachedContentsClient):
    def _url(self, path: str, params: dict[str, Any] | None = None) -> str:
        return resource_url(self.base_url, path, params)

    async def create(
        self, body: dict[str, Any], options: RetryOptions | None = None
    ) -> CachedContent:
        payload = deepcopy(body)
        model = payload.get("model")
        if isinstance(model, str) and not model.startswith("projects/"):
            model = (
                model
                if model.startswith("publishers/")
                else f"publishers/google/models/{model.removeprefix('models/')}"
            )
            parsed = urlsplit(resource_url(self.base_url, model))
            payload["model"] = parsed.path.split("/", 2)[2]
        return await super().create(payload, options)

    async def update(
        self, name: str, body: dict[str, Any], options: RetryOptions | None = None
    ) -> CachedContent:
        fields = [field for field in body if field != "name"]
        if not fields:
            raise ValidationError("Vertex cache update requires fields to update.")
        payload = await _request(
            self.fetch,
            self._url(name, {"updateMask": ",".join(fields)}),
            "PATCH",
            body,
            options,
        )
        return _normalize_cached_content(payload)


class VertexInteractionsClient(GeminiInteractionsClient):
    def _url(self, path: str, params: dict[str, Any] | None = None) -> str:
        return resource_url(self.base_url, path, params)

    async def create(
        self, body: dict[str, Any], options: RetryOptions | None = None
    ) -> dict[str, Any]:
        # Vertex has its own schema/version lifecycle. Do not apply Developer
        # API field removals (such as response_mime_type) to this native API.
        if not body.get("stream"):
            return await _request(
                self.fetch, self._url("interactions"), "POST", body, options
            )
        response = await self.fetch(
            self._url("interactions", {"alt": "sse"}),
            headers={"content-type": "application/json"},
            json_body=deepcopy(body),
            timeout_ms=options.timeout_ms if options else None,
            stream=True,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Vertex request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        return {"stream": response}

    async def delete(
        self, interaction_id: str, options: RetryOptions | None = None
    ) -> bool:
        await _request(
            self.fetch,
            self._url(f"interactions/{interaction_id}"),
            "DELETE",
            None,
            options,
        )
        return True

    async def list(
        self,
        *,
        page_size: int | None = None,
        page_token: str | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        return await _request(
            self.fetch,
            self._url(
                "interactions", {"page_size": page_size, "page_token": page_token}
            ),
            "GET",
            None,
            options,
        )

    async def wait(
        self,
        interaction_id: str,
        *,
        poll_interval_ms: int = 10_000,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        async def poll() -> dict[str, Any]:
            while True:
                interaction = await self.retrieve(interaction_id)
                if interaction.get("status") in {
                    "completed",
                    "failed",
                    "cancelled",
                    "expired",
                    "incomplete",
                    "requires_action",
                }:
                    return interaction
                await asyncio.sleep(max(poll_interval_ms, 1) / 1000)

        async with asyncio.timeout(None if timeout_ms is None else timeout_ms / 1000):
            return await poll()


async def _request(
    fetch: Fetcher,
    url: str,
    method: str,
    body: dict[str, Any] | None,
    options: RetryOptions | None,
) -> dict[str, Any]:
    async def send() -> dict[str, Any]:
        response = await fetch(
            url,
            method=method,
            headers={"content-type": "application/json"},
            json_body=deepcopy(body),
            timeout_ms=options.timeout_ms if options else None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError(
                f"Vertex request failed with status {response.status_code}.",
                response.status_code,
                response_body=await response.text(),
            )
        if response.status_code == 204 or not await response.text():
            return {}
        return await response.json()

    return await with_retry(
        send,
        max_retries=(options.max_retries or 0) if options else 0,
        retry_backoff_ms=(
            options.retry_backoff_ms if options.retry_backoff_ms is not None else 250
        )
        if options
        else 250,
    )


class VertexBatchesClient(GeminiBatchesClient):
    def _url(self, path: str) -> str:
        return resource_url(self.base_url, path)

    def _name(self, batch_id: str) -> str:
        if batch_id.startswith(("projects/", "batchPredictionJobs/")):
            return batch_id
        return f"batchPredictionJobs/{batch_id}"

    async def create(
        self, body: dict[str, Any], options: RetryOptions | None = None
    ) -> dict[str, Any]:
        # Vertex BatchPredictionJob has inputConfig/outputConfig, unlike the
        # Gemini Developer API's batchGenerateContent request envelope.
        model = body.get("model")
        cyber_path = "publishers/google/models/gemini-3.8-flash-cyber"
        if isinstance(model, str) and (
            model in {"gemini-3.8-flash-cyber", cyber_path}
            or (model.startswith("projects/") and model.endswith("/" + cyber_path))
        ):
            raise UnsupportedFeatureError("Gemini 3.8 Flash Cyber does not support batch inference.")
        return await _request(
            self.fetch, self._url("batchPredictionJobs"), "POST", body, options
        )

    async def create_embeddings(
        self, body: dict[str, Any], options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await self.create(body, options)

    async def retrieve(
        self, batch_id: str, options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await _request(
            self.fetch, self._url(self._name(batch_id)), "GET", None, options
        )

    async def list(
        self,
        *,
        after: str | None = None,
        limit: int | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        return await _request(
            self.fetch,
            resource_url(
                self.base_url,
                "batchPredictionJobs",
                {"pageToken": after, "pageSize": limit},
            ),
            "GET",
            None,
            options,
        )

    async def cancel(
        self, batch_id: str, options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await _request(
            self.fetch, self._url(f"{self._name(batch_id)}:cancel"), "POST", {}, options
        )

    async def delete(
        self, batch_id: str, options: RetryOptions | None = None
    ) -> dict[str, Any]:
        return await _request(
            self.fetch, self._url(self._name(batch_id)), "DELETE", None, options
        )

    async def wait(
        self,
        batch_id: str,
        *,
        poll_interval_ms: int = 10_000,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        async def poll() -> dict[str, Any]:
            while True:
                batch = await self.retrieve(batch_id)
                if batch.get("state") in {
                    "JOB_STATE_SUCCEEDED",
                    "JOB_STATE_FAILED",
                    "JOB_STATE_CANCELLED",
                    "JOB_STATE_EXPIRED",
                    "JOB_STATE_PARTIALLY_SUCCEEDED",
                }:
                    return batch
                await asyncio.sleep(max(poll_interval_ms, 1) / 1000)

        async with asyncio.timeout(None if timeout_ms is None else timeout_ms / 1000):
            return await poll()


class VertexVideosClient(GeminiVideosClient):
    def _normalize_operation(self, payload: dict[str, Any]) -> VideoOperation:
        normalized = deepcopy(payload)
        response = normalized.get("response")
        if isinstance(response, dict) and "videos" in response:
            response["generatedVideos"] = response["videos"]
        operation = super()._normalize_operation(normalized)
        operation.raw_response.update(payload)
        return operation

    async def download(self, uri: str, options: RetryOptions | None = None) -> bytes:
        if uri.startswith("gs://"):
            parsed = urlsplit(uri)
            if (
                not parsed.netloc
                or not parsed.path.lstrip("/")
                or parsed.username
                or parsed.port
            ):
                raise ValidationError("Expected a Cloud Storage bucket/object URI.")
            uri = f"https://storage.googleapis.com/storage/v1/b/{quote(parsed.netloc, safe='')}/o/{quote(parsed.path.lstrip('/'), safe='')}?alt=media"
        return await super().download(uri, options)

    async def get_operation(
        self, name: str, options: RetryOptions | None = None
    ) -> VideoOperation:
        if "/operations/" not in name:
            raise ValidationError("Expected a full Vertex Veo operation name.")
        model_name = name.rsplit("/operations/", 1)[0]
        payload = await _request(
            self.fetch,
            resource_url(self.base_url, model_name + ":fetchPredictOperation"),
            "POST",
            {"operationName": name},
            options,
        )
        return self._normalize_operation(payload)

    async def wait_operation(
        self,
        name: str,
        *,
        poll_interval_ms: int = 10_000,
        timeout_ms: int | None = None,
    ) -> VideoOperation:
        async with asyncio.timeout(None if timeout_ms is None else timeout_ms / 1000):
            while True:
                operation = await self.get_operation(name)
                if operation.done:
                    return operation
                await asyncio.sleep(max(poll_interval_ms, 1) / 1000)


class VertexMediaClient(GeminiMediaClient):
    async def generate_music(
        self,
        *,
        prompt: str,
        model: str = "lyria-3-clip-preview",
        parts: list[Any] | None = None,
        provider_options: dict[str, Any] | None = None,
        options: RetryOptions | None = None,
    ) -> MediaResult:
        if model == "lyria-002":
            if parts:
                raise ValidationError("Lyria 2 accepts text only.")
            config = deepcopy(provider_options or {})
            payload = await _request(
                self.fetch,
                resource_url(
                    self.base_url, "publishers/google/models/lyria-002:predict"
                ),
                "POST",
                {
                    "instances": [{"prompt": prompt, **config.pop("instance", {})}],
                    "parameters": config.pop("parameters", {}),
                    **config,
                },
                options,
            )
            media = [
                GeneratedMedia(
                    provider="vertex",
                    b64_data=item.get("audioContent") or item["bytesBase64Encoded"],
                    media_type=item.get("mimeType", "audio/wav"),
                    metadata=item,
                )
                for item in payload.get("predictions", [])
                if item.get("audioContent") or item.get("bytesBase64Encoded")
            ]
            return MediaResult(media=media, raw_response=payload)
        if model in {"lyria-3-clip-preview", "lyria-3-pro-preview"}:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for part in parts or []:
                mapped = _map_part(part)
                if "text" in mapped:
                    content.append({"type": "text", "text": mapped["text"]})
                    continue
                inline = mapped.get("inlineData")
                file = mapped.get("fileData")
                data = inline or file
                if not isinstance(data, dict):
                    raise ValidationError("Unsupported Lyria interaction input part.")
                mime = data.get("mimeType", "")
                kind = mime.split("/", 1)[0]
                if kind not in {"image", "audio", "video"}:
                    raise ValidationError("Unsupported Lyria interaction media type.")
                content.append(
                    {
                        "type": kind,
                        "mime_type": mime,
                        **(
                            {"data": data["data"]}
                            if inline
                            else {"uri": data["fileUri"]}
                        ),
                    }
                )
            config = deepcopy(provider_options or {})
            config.setdefault("store", False)
            if config.get("stream"):
                raise ValidationError(
                    "Use interactions().create for raw streaming music responses."
                )
            version = urlsplit(self.base_url).path.strip("/").split("/")[0]
            client = VertexInteractionsClient(
                api_key="",
                base_url=self.base_url.replace(f"/{version}/", "/v1beta1/", 1),
                fetch=self.fetch,
            )
            payload = await client.create(
                {"model": model, "input": content, **config}, options
            )
            output = [
                item
                for step in payload.get("steps", [])
                for item in step.get("content", [])
            ]
            output.extend(payload.get("outputs", []))
            media = [
                GeneratedMedia(
                    provider="vertex",
                    b64_data=item.get("data"),
                    url=item.get("uri"),
                    media_type=item.get("mime_type", "audio/mpeg"),
                    metadata=item,
                )
                for item in output
                if item.get("type") == "audio" and (item.get("data") or item.get("uri"))
            ]
            text = "".join(
                item.get("text", "") for item in output if item.get("type") == "text"
            )
            return MediaResult(media=media, text=text or None, raw_response=payload)
        return await super().generate_music(
            prompt=prompt,
            model=model,
            parts=parts,
            provider_options=provider_options,
            options=options,
        )
