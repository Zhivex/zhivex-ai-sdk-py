from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any


_SECRET_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "authorization",
    "client_secret",
    "id_token",
    "password",
    "refresh_token",
    "secret",
    "token",
}
_BEARER_RE = re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_INLINE_SECRET_RE = re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|client[_-]?secret|refresh[_-]?token|secret|token)\b\s*[:=]\s*([^\s,;]+)")


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
        snippet = redact_provider_error_body(self.response_body)
        if not snippet:
            return message
        return f"{message} Response body: {snippet}"


def redact_provider_error_body(body: str, *, max_length: int = 500) -> str:
    text = str(body or "").strip()
    if not text:
        return ""
    redacted = _redact_json_body(text)
    compact = " ".join(redacted.split())
    compact = _BEARER_RE.sub(r"\1 [redacted]", compact)
    compact = _INLINE_SECRET_RE.sub(lambda match: f"{match.group(1)}=[redacted]", compact)
    if len(compact) <= max_length:
        return compact
    return f"{compact[: max_length - 3]}..."


def _redact_json_body(text: str) -> str:
    try:
        payload = json.loads(text)
    except Exception:
        return text
    return json.dumps(_redact_json_value(payload), separators=(",", ":"), sort_keys=True)


def _redact_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if str(key).lower() in _SECRET_KEYS:
                redacted[key] = "[redacted]"
            else:
                redacted[key] = _redact_json_value(item)
        return redacted
    if isinstance(value, list):
        return [_redact_json_value(item) for item in value]
    return value


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
