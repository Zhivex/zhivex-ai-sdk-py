from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from ..errors import ValidationError


def validate_provider_url(
    url: str,
    *,
    provider: str,
    purpose: str,
    allowed_suffixes: tuple[str, ...] = (),
) -> str:
    raw_url = str(url)
    if any(character in raw_url for character in ("\r", "\n", "\t", "\x00")):
        raise ValidationError(f'Provider "{provider}" returned an unsafe {purpose} URL.')
    parsed = urlparse(raw_url)
    if parsed.scheme != "https":
        raise ValidationError(f'Provider "{provider}" returned an unsafe {purpose} URL.')
    if parsed.username or parsed.password:
        raise ValidationError(f'Provider "{provider}" returned an unsafe {purpose} URL.')
    try:
        host = (parsed.hostname or "").strip(".").encode("idna").decode("ascii").lower()
        port = parsed.port
    except (UnicodeError, ValueError):
        raise ValidationError(f'Provider "{provider}" returned an unsafe {purpose} URL.') from None
    if not host:
        raise ValidationError(f'Provider "{provider}" returned an unsafe {purpose} URL.')
    if port is not None and not 1 <= port <= 65535:
        raise ValidationError(f'Provider "{provider}" returned an unsafe {purpose} URL.')
    if _is_blocked_host(host):
        raise ValidationError(f'Provider "{provider}" returned an unsafe {purpose} URL.')
    normalized_suffixes = tuple(suffix.strip(".").encode("idna").decode("ascii").lower() for suffix in allowed_suffixes)
    if normalized_suffixes:
        if not any(host == suffix or host.endswith(f".{suffix}") for suffix in normalized_suffixes):
            raise ValidationError(f'Provider "{provider}" returned an unexpected {purpose} URL host.')
    else:
        _validate_resolved_addresses(host, provider=provider, purpose=purpose)
    return raw_url


def _is_blocked_host(host: str) -> bool:
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        return True
    address = _parse_ip_address(host)
    if address is None:
        return False
    return _is_blocked_address(address)


def _parse_ip_address(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    candidate = host.strip("[]")
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        pass

    # inet_aton recognizes legacy one-part, octal, and hexadecimal IPv4 forms
    # such as 2130706433 and 0x7f000001. Browsers and HTTP stacks may resolve
    # those to loopback even though ipaddress intentionally rejects them.
    try:
        return ipaddress.ip_address(socket.inet_aton(candidate))
    except OSError:
        return None


def _is_blocked_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _validate_resolved_addresses(host: str, *, provider: str, purpose: str) -> None:
    try:
        results = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise ValidationError(f'Provider "{provider}" returned an unresolvable {purpose} URL host.') from None

    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for result in results:
        raw_address = str(result[4][0]).split("%", 1)[0]
        try:
            addresses.add(ipaddress.ip_address(raw_address))
        except ValueError:
            raise ValidationError(f'Provider "{provider}" returned an unsafe {purpose} URL.') from None
    if not addresses or any(_is_blocked_address(address) for address in addresses):
        raise ValidationError(f'Provider "{provider}" returned an unsafe {purpose} URL.')
