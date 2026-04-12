from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any


class ZhivexAIError(Exception):
    pass


class ConfigurationError(ZhivexAIError):
    pass


class ValidationError(ZhivexAIError):
    pass


class UnsupportedFeatureError(ZhivexAIError):
    pass


class ParseError(ZhivexAIError):
    pass


class ProviderHTTPError(ZhivexAIError):
    def __init__(
        self,
        message: str,
        status: int,
        *,
        response_body: str | None = None,
        response_headers: dict[str, Any] | None = None,
        retry_after_ms: int | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.response_body = response_body
        self.response_headers = dict(response_headers or {})
        self.retry_after_ms = retry_after_ms if retry_after_ms is not None else _parse_retry_after_ms(self.response_headers)
        self.retryable = retryable if retryable is not None else (status in {408, 429} or status >= 500)

    def __str__(self) -> str:
        message = super().__str__()
        if not self.response_body:
            return message
        body = self.response_body.strip()
        if not body:
            return message
        compact = " ".join(body.split())
        snippet = compact if len(compact) <= 500 else f"{compact[:497]}..."
        return f"{message} Response body: {snippet}"


def _parse_retry_after_ms(headers: dict[str, Any] | None) -> int | None:
    if not headers:
        return None
    value: Any = None
    for key, header_value in headers.items():
        if str(key).lower() == "retry-after":
            value = header_value
            break
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(int(float(value) * 1000), 0)
    text = str(value).strip()
    if not text:
        return None
    try:
        return max(int(float(text) * 1000), 0)
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    delta = retry_at - datetime.now(timezone.utc)
    return max(int(delta.total_seconds() * 1000), 0)
