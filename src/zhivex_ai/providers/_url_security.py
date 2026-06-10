from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from ..errors import ValidationError


def validate_provider_url(
    url: str,
    *,
    provider: str,
    purpose: str,
    allowed_suffixes: tuple[str, ...] = (),
) -> str:
    parsed = urlparse(str(url))
    if parsed.scheme != "https":
        raise ValidationError(f'Provider "{provider}" returned an unsafe {purpose} URL.')
    if parsed.username or parsed.password:
        raise ValidationError(f'Provider "{provider}" returned an unsafe {purpose} URL.')
    host = (parsed.hostname or "").strip(".").lower()
    if not host:
        raise ValidationError(f'Provider "{provider}" returned an unsafe {purpose} URL.')
    if _is_blocked_host(host):
        raise ValidationError(f'Provider "{provider}" returned an unsafe {purpose} URL.')
    if allowed_suffixes and not any(host == suffix or host.endswith(f".{suffix}") for suffix in allowed_suffixes):
        raise ValidationError(f'Provider "{provider}" returned an unexpected {purpose} URL host.')
    return str(url)


def _is_blocked_host(host: str) -> bool:
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )
