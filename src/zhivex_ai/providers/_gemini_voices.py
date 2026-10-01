"""Beta Gemini Voices REST client; consent verification is performed by Google."""
from __future__ import annotations

import base64
import re
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from .._http import Fetcher
from ..errors import ParseError, ProviderHTTPError, ValidationError
from ..types import RetryOptions


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f'Gemini Voices requires a non-empty "{field}".')
    return value


def _audio(value: Any, field: str) -> None:
    if not isinstance(value, dict):
        raise ValidationError(f'Gemini Voices requires "{field}" audio.')
    data = _required_text(value.get("data"), f"{field}.data")
    mime = _required_text(value.get("mime_type"), f"{field}.mime_type")
    if not mime.startswith("audio/"):
        raise ValidationError("Gemini Voices requires an audio MIME type.")
    try:
        if not base64.b64decode(data, validate=True):
            raise ValueError
    except ValueError:
        raise ValidationError("Gemini Voices requires non-empty base64 audio data.") from None


@dataclass(slots=True)
class GeminiVoicesClient:
    """Native Beta API. Returned dicts preserve keys, expiry, samples, and usage."""

    api_key: str = field(repr=False)
    base_url: str
    fetch: Fetcher

    async def _request(self, method: str, path: str, *, body: dict[str, Any] | None = None,
                       params: dict[str, Any] | None = None, options: RetryOptions | None = None) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path}"
        if params:
            url += "?" + urlencode(params, doseq=True)
        response = await self.fetch(url, method=method,
                                    headers={"content-type": "application/json", "x-goog-api-key": self.api_key},
                                    json_body=body, timeout_ms=options.timeout_ms if options else None)
        if response.status_code >= 400:
            raise ProviderHTTPError(f"Gemini Voices request failed with status {response.status_code}.",
                                    response.status_code, response_body=await response.text(),
                                    response_headers=dict(response.headers))
        if method == "DELETE" or response.status_code == 204:
            return {}
        try:
            payload = await response.json()
        except (ValueError, UnicodeError):
            raise ParseError("Gemini Voices returned invalid JSON.") from None
        if not isinstance(payload, dict):
            raise ParseError("Gemini Voices returned a non-object response.")
        return payload

    async def create(self, body: dict[str, Any], options: RetryOptions | None = None) -> dict[str, Any]:
        payload = deepcopy(body)
        voice = payload.get("voice")
        if not isinstance(voice, dict):
            raise ValidationError('Gemini Voices requires a "voice" object.')
        store = payload.setdefault("store", True)
        if not isinstance(store, bool):
            raise ValidationError('Gemini Voices "store" must be a boolean.')
        kind = voice.get("type")
        if kind == "prompted":
            if not store:
                raise ValidationError("Prompted Gemini voices require store=True.")
            prompted = voice.get("prompted")
            if not isinstance(prompted, dict):
                raise ValidationError('Prompted Gemini voices require "prompted.input".')
            _required_text(prompted.get("input"), "prompted.input")
            if "replicated" in voice:
                raise ValidationError("A prompted voice cannot include replicated audio.")
        elif kind == "replicated":
            replicated = voice.get("replicated")
            if not isinstance(replicated, dict):
                raise ValidationError("Replicated Gemini voices require source and consent audio.")
            _audio(replicated.get("source_audio"), "replicated.source_audio")
            _audio(replicated.get("consent_audio"), "replicated.consent_audio")
            if "prompted" in voice:
                raise ValidationError("A replicated voice cannot include a prompted voice.")
        else:
            raise ValidationError('Gemini Voices creation type must be "prompted" or "replicated".')
        return await self._request("POST", "voices", body=payload, options=options)

    async def design(self, *, prompt: str, model: str | None = None, display_name: str | None = None,
                     options: RetryOptions | None = None) -> dict[str, Any]:
        voice: dict[str, Any] = {"type": "prompted", "prompted": {"input": prompt}}
        if model is not None:
            voice["model"] = model
        if display_name is not None:
            voice["display_name"] = display_name
        return await self.create({"store": True, "voice": voice}, options)

    async def replicate(self, *, source_audio: bytes, consent_audio: bytes,
                        source_mime_type: str = "audio/wav", consent_mime_type: str = "audio/wav",
                        store: bool = True, model: str | None = None, display_name: str | None = None,
                        options: RetryOptions | None = None) -> dict[str, Any]:
        voice: dict[str, Any] = {"type": "replicated", "replicated": {
            "source_audio": {"mime_type": source_mime_type, "data": base64.b64encode(source_audio).decode("ascii")},
            "consent_audio": {"mime_type": consent_mime_type, "data": base64.b64encode(consent_audio).decode("ascii")},
        }}
        if model is not None:
            voice["model"] = model
        if display_name is not None:
            voice["display_name"] = display_name
        return await self.create({"store": store, "voice": voice}, options)

    async def list(self, params: dict[str, Any] | None = None, options: RetryOptions | None = None) -> dict[str, Any]:
        query = deepcopy(params or {})
        allowed = {"page_size", "page_token", "language_code", "region_code", "accent", "persona", "context", "gender", "pitch", "type", "search"}
        if query.keys() - allowed:
            raise ValidationError("Gemini Voices list received unsupported query parameters.")
        size = query.get("page_size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
            raise ValidationError("Gemini Voices page_size must be a positive integer.")
        if "search" in query and (not isinstance(query["search"], str) or len(query["search"].encode("utf-8")) > 2048):
            raise ValidationError("Gemini Voices search must be a string of at most 2048 bytes.")
        payload = await self._request("GET", "voices", params=query, options=options)
        voices = payload.get("voices", [])
        token = payload.get("next_page_token")
        if not isinstance(voices, list) or any(not isinstance(voice, dict) for voice in voices):
            raise ParseError("Gemini Voices returned an invalid voices collection.")
        if token is not None and not isinstance(token, str):
            raise ParseError("Gemini Voices returned an invalid pagination token.")
        return payload

    async def iter_voices(self, params: dict[str, Any] | None = None, options: RetryOptions | None = None) -> AsyncIterator[dict[str, Any]]:
        query = deepcopy(params or {})
        if query.get("page_token") is not None and not isinstance(query["page_token"], str):
            raise ValidationError("Gemini Voices page_token must be a string.")
        seen = {query["page_token"]} if query.get("page_token") else set()
        while True:
            page = await self.list(query, options)
            for voice in page.get("voices") or []:
                yield voice
            token = page.get("next_page_token")
            if not token:
                return
            if token in seen:
                raise ParseError("Gemini Voices returned a repeated pagination token.")
            seen.add(token)
            query["page_token"] = token

    @staticmethod
    def _resource(voice_id: str) -> str:
        name = voice_id.removeprefix("voices/")
        if not re.fullmatch(r"voice_[A-Za-z0-9_-]+", name):
            raise ValidationError("Gemini Voices get/delete require a stored custom voice ID.")
        return f"voices/{name}"

    async def get(self, voice_id: str, options: RetryOptions | None = None) -> dict[str, Any]:
        return await self._request("GET", self._resource(voice_id), options=options)

    async def delete(self, voice_id: str, options: RetryOptions | None = None) -> None:
        await self._request("DELETE", self._resource(voice_id), options=options)
