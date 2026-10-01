"""Beta native text reranking; provider response shapes are preserved."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

from .._http import Fetcher
from ..errors import ConfigurationError, ParseError, ProviderHTTPError, UnsupportedFeatureError, ValidationError
from ..types import RetryOptions


def qwen_rerank_base_url(base_url: str, region: str, workspace_id: str | None) -> str:
    parsed = urlparse(base_url)
    if any(character in base_url for character in ("\r", "\n", "\t", "\x00")) or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigurationError("Qwen rerank base URL must be an HTTPS URL without credentials, query or fragment.")
    if workspace_id is not None and (not isinstance(workspace_id, str) or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", workspace_id)):
        raise ConfigurationError("Qwen workspace_id must be a single DNS label.")
    official_hosts = {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com"}
    if workspace_id and parsed.hostname in official_hosts:
        locations = {"cn": "cn-beijing", "intl": "ap-southeast-1", "us": "us-east-1"}
        if region not in locations:
            raise ConfigurationError("Unsupported Qwen rerank region.")
        parsed = parsed._replace(netloc=f"{workspace_id}.{locations[region]}.maas.aliyuncs.com")
    path = parsed.path.rstrip("/")
    for suffix in ("/api/v2/apps/protocols/compatible-mode/v1", "/compatible-mode/v1", "/compatible-api/v1", "/api/v1"):
        if path.endswith(suffix):
            path = path[:-len(suffix)]
            break
    return urlunparse(parsed._replace(path=path)).rstrip("/")


@dataclass(slots=True)
class QwenRerankClient:
    """Native text-only qwen3.7-text-rerank and qwen3-rerank requests."""

    api_key: str = field(repr=False)
    base_url: str
    fetch: Fetcher

    async def create(
        self,
        *,
        query: str,
        documents: Sequence[str],
        model: str = "qwen3.7-text-rerank",
        top_n: int | None = None,
        instruct: str | None = None,
        options: RetryOptions | None = None,
    ) -> dict[str, Any]:
        if not isinstance(model, str) or model not in {"qwen3.7-text-rerank", "qwen3-rerank"}:
            raise ValidationError("Qwen text rerank supports qwen3.7-text-rerank and qwen3-rerank.")
        host = urlparse(self.base_url).hostname or ""
        if model == "qwen3.7-text-rerank" and (
            host in {"dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com"}
            or host.endswith((".ap-southeast-1.maas.aliyuncs.com", ".us-east-1.maas.aliyuncs.com"))
        ):
            raise UnsupportedFeatureError("qwen3.7-text-rerank is documented for Beijing; use region='cn' or an explicit trusted gateway.")
        if not isinstance(query, str) or not query.strip():
            raise ValidationError("Qwen rerank query must be a non-empty string.")
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence) or not 1 <= len(documents) <= 500:
            raise ValidationError("Qwen rerank requires between 1 and 500 text documents.")
        submitted_documents = list(documents)
        if any(not isinstance(document, str) or not document.strip() for document in submitted_documents):
            raise ValidationError("Qwen rerank documents must be non-empty strings.")
        if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1):
            raise ValidationError("Qwen rerank top_n must be a positive integer.")
        if instruct is not None and (not isinstance(instruct, str) or not instruct.strip()):
            raise ValidationError("Qwen rerank instruct must be a non-empty string.")
        parameters: dict[str, Any] = {}
        if top_n is not None:
            parameters["top_n"] = top_n
        if instruct is not None:
            parameters["instruct"] = instruct
        if model == "qwen3-rerank":
            path = "/compatible-api/v1/reranks"
            body: dict[str, Any] = {"model": model, "query": query, "documents": submitted_documents, **parameters}
        else:
            path = "/api/v1/services/rerank/text-rerank/text-rerank"
            body = {"model": model, "input": {"query": query, "documents": submitted_documents}, "parameters": parameters}
        response = await self.fetch(
            self.base_url + path, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json_body=body, timeout_ms=options.timeout_ms if options else None,
        )
        if response.status_code >= 400:
            raise ProviderHTTPError("Qwen rerank request failed.", response.status_code,
                                    response_body=await response.text(), response_headers=dict(response.headers))
        try:
            payload = await response.json()
        except (ValueError, TypeError) as exc:
            raise ParseError("Qwen rerank returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise ParseError("Qwen rerank must return a JSON object.")
        if payload.get("code"):
            raise ProviderHTTPError("Qwen rerank returned a provider error.", response.status_code,
                                    response_body=await response.text(), response_headers=dict(response.headers))
        output = payload if model == "qwen3-rerank" else payload.get("output")
        results = output.get("results") if isinstance(output, dict) else None
        if not isinstance(results, list):
            raise ParseError("Qwen rerank response is missing results.")
        seen: set[int] = set()
        for result in results:
            if not isinstance(result, dict):
                raise ParseError("Qwen rerank results must be objects.")
            index, score = result.get("index"), result.get("relevance_score")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(submitted_documents) or index in seen:
                raise ParseError("Qwen rerank returned an invalid document index.")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1 or not math.isfinite(score):
                raise ParseError("Qwen rerank returned an invalid relevance score.")
            seen.add(index)
        return payload
