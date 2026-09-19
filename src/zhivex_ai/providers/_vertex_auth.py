"""Authentication shared by Vertex REST and WebSocket transports."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from ..errors import ConfigurationError


class VertexAuth:
    def __init__(
        self, *, access_token: str | None, api_key: str | None, credentials: Any = None
    ) -> None:
        self._access_token = access_token
        self._api_key = api_key
        self._credentials = credentials
        self._lock = threading.Lock()

    def _credential_headers(self, method: str, url: str) -> dict[str, str]:
        # The lock lives in the worker thread so cancellation cannot allow a
        # second task to mutate credentials while a refresh is still running.
        with self._lock:
            try:
                from google.auth.transport.requests import Request
            except ImportError:
                raise ConfigurationError(
                    "Install zhivex-ai-sdk[vertex] to use Google credentials."
                ) from None
            headers: dict[str, str] = {}
            request = Request()
            try:
                self._credentials.before_request(request, method, url, headers)
            except Exception:
                raise ConfigurationError("Vertex credential refresh failed.") from None
            finally:
                request.session.close()
            return headers

    async def headers(self, method: str, url: str) -> dict[str, str]:
        if self._api_key:
            return {"x-goog-api-key": self._api_key}
        if self._access_token:
            return {"authorization": f"Bearer {self._access_token}"}
        return await asyncio.to_thread(self._credential_headers, method, url)


def default_credentials() -> tuple[Any, str | None]:
    try:
        import google.auth
    except ImportError:
        raise ConfigurationError(
            "Missing Vertex credentials. Install zhivex-ai-sdk[vertex] for ADC or supply an access token/API key."
        ) from None
    try:
        return google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except Exception:
        raise ConfigurationError(
            "Missing Vertex credentials. Configure ADC or supply an access token/API key."
        ) from None
