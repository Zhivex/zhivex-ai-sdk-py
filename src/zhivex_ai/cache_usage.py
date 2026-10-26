"""Cache-write usage and pricing shared by providers and cost estimates.

OpenAI Responses reports 30-minute and 12-hour cache writes separately.
Twelve-hour writes are billed at 2.5× the uncached input rate. The catalog
cache-write rate remains the 30-minute / default write rate.

See https://developers.openai.com/api/docs/guides/prompt-caching
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import ValidationError
from .types import CacheWriteTokensDetails, JsonValue, TokenUsage

OPENAI_CACHE_WRITE_12H_INPUT_MULTIPLIER = 2.5
_DETAIL_FIELDS = ("ttl_30m_tokens", "ttl_12h_tokens")


def _reject_counter(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValidationError(f"Invalid usage counter {name}.")
    return value


def read_cache_write_tokens_details(value: object) -> CacheWriteTokensDetails:
    """Read a lifetime split. Absent fields are omitted; unknown fields and bad counters raise."""
    if not isinstance(value, dict):
        raise ValidationError("Invalid cache_write_tokens_details.")
    for key in value:
        if key not in _DETAIL_FIELDS:
            raise ValidationError(f"Invalid cache_write_tokens_details.{key}.")
    parsed: dict[str, int] = {}
    for key in _DETAIL_FIELDS:
        if key not in value or value[key] is None:
            continue
        parsed[key] = _reject_counter(f"cache_write_tokens_details.{key}", value[key])
    if not parsed:
        raise ValidationError("Invalid cache_write_tokens_details.")
    return CacheWriteTokensDetails(**parsed)


def detail_token_sum(details: CacheWriteTokensDetails) -> int:
    return (details.ttl_30m_tokens or 0) + (details.ttl_12h_tokens or 0)


def cache_write_tokens_details_from_provider(details: object) -> CacheWriteTokensDetails | None:
    """Map OpenAI ``cache_write_tokens_details`` when either lifetime counter is present."""
    if not isinstance(details, dict):
        return None
    parsed: dict[str, int] = {}
    for source_key, field_name in (
        ("ttl_30m_tokens", "ttl_30m_tokens"),
        ("ttl_12h_tokens", "ttl_12h_tokens"),
    ):
        count = details.get(source_key)
        if isinstance(count, bool) or not isinstance(count, int):
            continue
        parsed[field_name] = count
    if not parsed:
        return None
    return CacheWriteTokensDetails(**parsed)


def _details_consistent(usage: TokenUsage) -> bool:
    details = usage.cache_write_tokens_details
    if details is None:
        return (usage.cache_write_tokens or 0) == 0
    return usage.cache_write_tokens is not None and detail_token_sum(details) == usage.cache_write_tokens


def merge_cache_write_tokens_details(
    usages: list[TokenUsage | None],
) -> CacheWriteTokensDetails | None:
    """Keep a lifetime split only when every write in the group has a consistent split."""
    present = [usage for usage in usages if usage is not None]
    if not any(usage.cache_write_tokens_details is not None for usage in present):
        return None
    if any(not _details_consistent(usage) for usage in present):
        return None
    saw_30m = False
    saw_12h = False
    ttl_30m_tokens = 0
    ttl_12h_tokens = 0
    for usage in present:
        details = usage.cache_write_tokens_details
        if details is None:
            continue
        if details.ttl_30m_tokens is not None:
            saw_30m = True
            ttl_30m_tokens += details.ttl_30m_tokens
        if details.ttl_12h_tokens is not None:
            saw_12h = True
            ttl_12h_tokens += details.ttl_12h_tokens
    if not saw_30m and not saw_12h:
        return None
    return CacheWriteTokensDetails(
        ttl_30m_tokens=ttl_30m_tokens if saw_30m else None,
        ttl_12h_tokens=ttl_12h_tokens if saw_12h else None,
    )


def _sum_when_complete(usages: list[TokenUsage], name: str) -> int | None:
    values = [getattr(usage, name) for usage in usages]
    if any(value is None for value in values):
        return None
    return sum(values)


def merge_token_usage(usages: list[TokenUsage | None]) -> TokenUsage | None:
    present = [usage for usage in usages if usage is not None]
    if not present:
        return None
    return TokenUsage(
        input_tokens=_sum_when_complete(present, "input_tokens"),
        output_tokens=_sum_when_complete(present, "output_tokens"),
        total_tokens=_sum_when_complete(present, "total_tokens"),
        cached_input_tokens=_sum_when_complete(present, "cached_input_tokens"),
        cache_write_tokens=_sum_when_complete(present, "cache_write_tokens"),
        cache_write_tokens_details=merge_cache_write_tokens_details(present),
    )


def cache_usage_kwargs(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    kwargs: dict[str, Any] = {}
    if "cached_input_tokens" in payload:
        kwargs["cached_input_tokens"] = payload.get("cached_input_tokens")
    if "cache_write_tokens" in payload:
        kwargs["cache_write_tokens"] = payload.get("cache_write_tokens")
    if payload.get("cache_write_tokens_details") is not None:
        kwargs["cache_write_tokens_details"] = read_cache_write_tokens_details(
            payload.get("cache_write_tokens_details")
        )
    return kwargs


def serialized_cache_usage(usage: TokenUsage) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {}
    if usage.cached_input_tokens is not None:
        payload["cached_input_tokens"] = usage.cached_input_tokens
    if usage.cache_write_tokens is not None:
        payload["cache_write_tokens"] = usage.cache_write_tokens
    details = usage.cache_write_tokens_details
    if details is not None:
        split: dict[str, JsonValue] = {}
        if details.ttl_30m_tokens is not None:
            split["ttl_30m_tokens"] = details.ttl_30m_tokens
        if details.ttl_12h_tokens is not None:
            split["ttl_12h_tokens"] = details.ttl_12h_tokens
        payload["cache_write_tokens_details"] = split
    return payload


@dataclass(frozen=True, slots=True)
class CacheWriteLifetimePrice:
    tokens: int
    rate_per_1k_tokens: float | None
    amount: float


@dataclass(frozen=True, slots=True)
class PricedCacheWriteLifetimes:
    tokens: int
    rate_per_1k_tokens: float | None
    amount: float
    ttl_30m: CacheWriteLifetimePrice
    ttl_12h: CacheWriteLifetimePrice


def _lifetime_line(
    tokens: int,
    rate_per_1k_tokens: float | None,
    input_multiplier: float,
) -> CacheWriteLifetimePrice | None:
    if tokens == 0:
        return CacheWriteLifetimePrice(
            tokens=0,
            rate_per_1k_tokens=None if rate_per_1k_tokens is None else rate_per_1k_tokens * input_multiplier,
            amount=0,
        )
    if rate_per_1k_tokens is None:
        return None
    if rate_per_1k_tokens < 0:
        raise ValidationError("Invalid catalog price.")
    amount = tokens * rate_per_1k_tokens * input_multiplier / 1000
    return CacheWriteLifetimePrice(
        tokens=tokens,
        rate_per_1k_tokens=rate_per_1k_tokens * input_multiplier,
        amount=amount,
    )


def price_cache_write_lifetimes(
    *,
    cache_write_tokens: int,
    details: CacheWriteTokensDetails,
    cache_write_rate_per_1k_tokens: float | None,
    input_rate_per_1k_tokens: float | None,
    input_multiplier: float = 1,
) -> tuple[PricedCacheWriteLifetimes | None, list[str]]:
    """Price a reported lifetime split.

    Thirty-minute tokens use the catalog write rate. Twelve-hour tokens use
    2.5× the uncached input rate. ``amount`` is the full write charge.
    """
    ttl_30m_tokens = details.ttl_30m_tokens or 0
    ttl_12h_tokens = details.ttl_12h_tokens or 0
    if ttl_30m_tokens + ttl_12h_tokens != cache_write_tokens:
        raise ValidationError("Cache write lifetime tokens do not match cache_write_tokens.")
    ttl_30m = _lifetime_line(ttl_30m_tokens, cache_write_rate_per_1k_tokens, input_multiplier)
    twelve_hour_rate = (
        None
        if input_rate_per_1k_tokens is None
        else input_rate_per_1k_tokens * OPENAI_CACHE_WRITE_12H_INPUT_MULTIPLIER
    )
    ttl_12h = _lifetime_line(ttl_12h_tokens, twelve_hour_rate, input_multiplier)
    reasons: list[str] = []
    if ttl_30m is None:
        reasons.append("cache_write_price_missing")
    if ttl_12h is None:
        reasons.append("cache_write_12h_price_missing")
    if ttl_30m is None or ttl_12h is None:
        return None, reasons
    tokens = ttl_30m.tokens + ttl_12h.tokens
    amount = ttl_30m.amount + ttl_12h.amount
    blended = (
        ttl_30m.rate_per_1k_tokens
        if tokens == 0
        else amount * 1000 / tokens
    )
    return (
        PricedCacheWriteLifetimes(
            tokens=tokens,
            rate_per_1k_tokens=blended,
            amount=amount,
            ttl_30m=ttl_30m,
            ttl_12h=ttl_12h,
        ),
        [],
    )
