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


class WorkflowConflictError(ValidationError):
    """Raised when concurrent workflow state cannot be committed safely."""


class WorkflowLeaseLostError(WorkflowConflictError):
    """Raised when a workflow worker no longer owns its execution lease."""


class WorkflowDefinitionMismatchError(ValidationError):
    """Raised when persisted state belongs to a different workflow definition."""


class WorkflowRunNotFoundError(ValidationError):
    """Raised when a requested workflow run or checkpoint cannot be found."""


class WorkflowInterruptError(ValidationError):
    """Raised when a workflow interrupt cannot be resumed safely."""


class UnsupportedFeatureError(ZhivexAIError):
    pass


class ParseError(ZhivexAIError):
    pass


class ToolExecutionSuspended(ZhivexAIError):
    def __init__(
        self,
        message: str,
        *,
        pending_approval: Any,
        messages: list[Any] | None = None,
        steps: list[Any] | None = None,
        tool_results: list[Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.pending_approval = pending_approval
        self.messages = list(messages or [])
        self.steps = list(steps or [])
        self.tool_results = list(tool_results or [])


class ToolExecutionOutcomeUnknown(ZhivexAIError):
    """Raised when a timed-out tool may still have produced an external side effect."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        tool_call_id: str,
        timeout_ms: int,
        idempotency_key: str,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.timeout_ms = timeout_ms
        self.idempotency_key = idempotency_key
        self.outcome_unknown = True


class AgentRunCancelled(ZhivexAIError):
    """Raised when an atomic cancellation wins against an active agent worker."""

    def __init__(self, run_id: str, *, reason: str | None = None) -> None:
        message = f'Agent run "{run_id}" was cancelled'
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message)
        self.run_id = run_id
        self.reason = reason


class AgentEventDeliveryError(ZhivexAIError):
    """Raised when an application event callback fails during an agent run."""

    def __init__(
        self,
        run_id: str,
        *,
        event_type: str,
        durable_state_committed: bool,
    ) -> None:
        suffix = " after the durable terminal state was committed" if durable_state_committed else ""
        super().__init__(f'Agent event callback failed for "{event_type}" on run "{run_id}"{suffix}.')
        self.run_id = run_id
        self.event_type = event_type
        self.durable_state_committed = durable_state_committed


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
