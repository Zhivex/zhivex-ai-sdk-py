"""Fail-closed egress allowlists.

An empty allow list denies every call. A declaration that conflicts with the
configured destination is denied before any provider request.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlparse

from .errors import ValidationError, ZhivexAIError

EGRESS_WILDCARD = "*"
_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EgressAllowRule:
    provider: str
    model: str
    region: str
    endpoint: str


@dataclass(frozen=True, slots=True)
class EgressDeclaration:
    region: str | None = None
    endpoint: str | None = None


@dataclass(frozen=True, slots=True)
class EgressCallTarget:
    provider: str
    model: str
    region: str | None = None
    endpoint: str | None = None


@dataclass(frozen=True, slots=True)
class EgressAuditRecord:
    schema_version: int
    type: Literal["egress_call_audit"]
    id: str
    decision: Literal["allowed", "denied"]
    provider: str
    model_id: str
    reason: str
    created_at: int
    region: str | None = None
    endpoint: str | None = None


class EgressDeniedError(ZhivexAIError):
    code = "EGRESS_DENIED"

    def __init__(self, audit: EgressAuditRecord) -> None:
        super().__init__(f"Egress denied for {audit.provider}/{audit.model_id}: {audit.reason}")
        self.provider = audit.provider
        self.model_id = audit.model_id
        self.region = audit.region
        self.endpoint = audit.endpoint
        self.audit = audit


@dataclass(frozen=True, slots=True)
class _Endpoint:
    kind: Literal["any", "host", "url"]
    canonical: str
    host: str | None = None
    protocol: str | None = None
    port: str | None = None
    path: str | None = None


def _read_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _try_normalize(value: str) -> tuple[_Endpoint | None, str | None]:
    if value == EGRESS_WILDCARD:
        return _Endpoint(kind="any", canonical=EGRESS_WILDCARD), None
    parsed = urlparse(value)
    if parsed.scheme:
        if parsed.username or parsed.password:
            return None, "endpoint must not include credentials"
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None, "endpoint must use http or https"
        path = parsed.path.rstrip("/")
        port = parsed.port
        canonical = f"{parsed.scheme}://{parsed.hostname.lower()}"
        if port is not None:
            canonical = f"{canonical}:{port}"
        canonical = f"{canonical}{path}"
        return _Endpoint(
            kind="url",
            canonical=canonical,
            host=parsed.hostname.lower(),
            protocol=f"{parsed.scheme}:",
            port="" if port is None else str(port),
            path=path,
        ), None
    if not value or value.startswith(".") or value.endswith(".") or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for character in value):
        return None, "endpoint must be a hostname, an absolute http(s) URL, or *"
    host = value.lower()
    return _Endpoint(kind="host", canonical=host, host=host), None


def _normalize_rule_endpoint(value: str) -> _Endpoint:
    endpoint, reason = _try_normalize(value)
    if endpoint is None:
        raise ValidationError(f"Egress allow rule {reason}.")
    return endpoint


def _endpoint_matches(rule: _Endpoint, call: _Endpoint | None) -> bool:
    if rule.kind == "any":
        return True
    if call is None or call.kind == "any":
        return False
    if rule.host != call.host:
        return False
    if rule.kind == "host":
        return True
    if call.kind != "url" or rule.protocol != call.protocol or (rule.port or "") != (call.port or ""):
        return False
    if not rule.path:
        return True
    return call.path == rule.path or bool(call.path and call.path.startswith(f"{rule.path}/"))


def _dimension_matches(rule_value: str, call_value: str | None, *, case_insensitive: bool = False) -> bool:
    if rule_value == EGRESS_WILDCARD:
        return True
    if not call_value:
        return False
    if case_insensitive:
        return rule_value.lower() == call_value.lower()
    return rule_value == call_value


def _require_field(rule: EgressAllowRule | Mapping[str, object], field: str) -> str:
    value = rule[field] if isinstance(rule, Mapping) else getattr(rule, field)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f'Egress allow rule {field} must be a non-empty string or "*".')
    return value.strip()


@dataclass(frozen=True, slots=True)
class _CompiledRule:
    provider: str
    model: str
    region: str
    endpoint: _Endpoint


class EgressPolicy:
    def __init__(self, allow: list[EgressAllowRule], compiled: list[_CompiledRule], on_audit: Any) -> None:
        self.allow = tuple(allow)
        self.records: list[EgressAuditRecord] = []
        self._compiled = compiled
        self._on_audit = on_audit

    async def assert_call(self, target: EgressCallTarget | Mapping[str, object]) -> EgressAuditRecord:
        provider = _read_string(target["provider"] if isinstance(target, Mapping) else target.provider) or ""
        model = _read_string(target["model"] if isinstance(target, Mapping) else target.model) or ""
        region = _read_string(target.get("region") if isinstance(target, Mapping) else target.region)
        raw_endpoint = target.get("endpoint") if isinstance(target, Mapping) else target.endpoint
        endpoint_result: tuple[_Endpoint | None, str | None] | None
        if raw_endpoint is None:
            endpoint_result = None
        else:
            endpoint_result = _try_normalize(str(raw_endpoint).strip())
        endpoint = endpoint_result[0] if endpoint_result and endpoint_result[0] is not None else None
        decision: Literal["allowed", "denied"] = "denied"
        reason = "no allow rule matched"
        if not provider or not model:
            reason = "provider and model are required"
        elif endpoint_result is not None and endpoint_result[0] is None:
            reason = endpoint_result[1] or "endpoint is not a valid URL"
        elif not self._compiled:
            reason = "allowlist is empty"
        else:
            match = next(
                (
                    rule for rule in self._compiled
                    if _dimension_matches(rule.provider, provider)
                    and _dimension_matches(rule.model, model)
                    and _dimension_matches(rule.region, region, case_insensitive=True)
                    and _endpoint_matches(rule.endpoint, endpoint)
                ),
                None,
            )
            if match is not None:
                decision = "allowed"
                reason = "allow rule matched"
        record = EgressAuditRecord(
            schema_version=_SCHEMA_VERSION,
            type="egress_call_audit",
            id=str(uuid.uuid4()),
            decision=decision,
            provider=provider or "unknown",
            model_id=model or "unknown",
            region=region,
            endpoint=None if endpoint is None or endpoint.kind == "any" else endpoint.canonical,
            reason=reason,
            created_at=int(datetime.now(timezone.utc).timestamp() * 1000),
        )
        await self._write(record)
        if decision == "denied":
            raise EgressDeniedError(record)
        return record

    async def _write(self, record: EgressAuditRecord) -> None:
        self.records.append(record)
        if self._on_audit is None:
            return
        try:
            result = self._on_audit(record)
            if hasattr(result, "__await__"):
                await result
        except Exception as error:
            if record.decision == "denied":
                raise EgressDeniedError(record) from error
            raise


def create_egress_policy(
    *,
    allow: list[EgressAllowRule | Mapping[str, object]],
    on_audit: Any = None,
) -> EgressPolicy:
    if not isinstance(allow, list):
        raise ValidationError("Egress policy allow must be an array.")
    compiled: list[_CompiledRule] = []
    public: list[EgressAllowRule] = []
    for rule in allow:
        provider = _require_field(rule, "provider")
        model = _require_field(rule, "model")
        region = _require_field(rule, "region")
        endpoint = _normalize_rule_endpoint(_require_field(rule, "endpoint"))
        compiled.append(_CompiledRule(provider=provider, model=model, region=region, endpoint=endpoint))
        public.append(EgressAllowRule(provider=provider, model=model, region=region, endpoint=endpoint.canonical))
    return EgressPolicy(public, compiled, on_audit)


def _read_attr(source: object, keys: tuple[str, ...]) -> str | None:
    if source is None:
        return None
    if isinstance(source, Mapping):
        for key in keys:
            value = _read_string(source.get(key))
            if value:
                return value
        return None
    for key in keys:
        value = _read_string(getattr(source, key, None))
        if value:
            return value
    return None


def _canonical(value: str | None) -> str | None:
    if not value:
        return None
    endpoint, _reason = _try_normalize(value)
    return endpoint.canonical if endpoint is not None else value


def resolve_egress_target(
    model: object,
    declaration: EgressDeclaration | Mapping[str, object] | None = None,
    provider_options: Mapping[str, object] | None = None,
    *,
    authoritative_destination: bool = False,
) -> tuple[EgressCallTarget, str | None]:
    declared_region = _read_attr(declaration, ("region",))
    option_region = _read_attr(provider_options, ("region", "location"))
    model_region = _read_attr(model, ("region", "egress_region"))
    declared_endpoint = _read_attr(declaration, ("endpoint",))
    option_endpoint = _read_attr(provider_options, ("endpoint", "base_url", "baseURL"))
    model_endpoint = _read_attr(model, ("endpoint", "base_url", "baseURL", "egress_endpoint"))
    provider = str(getattr(model, "provider", "") or "")
    model_id = str(getattr(model, "model_id", "") or "")
    if authoritative_destination:
        region_conflict = any(
            value is not None and (value or "").lower() != (model_region or "").lower()
            for value in (declared_region, option_region)
        )
        endpoint_conflict = any(
            value is not None and _canonical(value) != _canonical(model_endpoint)
            for value in (declared_endpoint, option_endpoint)
        )
        target = EgressCallTarget(provider=provider, model=model_id, region=model_region, endpoint=model_endpoint)
        if region_conflict:
            return target, "declared region does not match the configured destination"
        if endpoint_conflict:
            return target, "declared endpoint does not match the configured destination"
        return target, None
    regions = [value for value in (declared_region, option_region, model_region) if value]
    endpoints = [value for value in (declared_endpoint, option_endpoint, model_endpoint) if value]
    if len({value.lower() for value in regions}) > 1:
        return EgressCallTarget(provider=provider, model=model_id, region=regions[0] if regions else None), "declared region does not match the model region"
    canonical_endpoints = {_canonical(value) for value in endpoints}
    if len(canonical_endpoints) > 1:
        return (
            EgressCallTarget(provider=provider, model=model_id, region=regions[0] if regions else None, endpoint=endpoints[0]),
            "declared endpoint does not match the model endpoint",
        )
    return EgressCallTarget(
        provider=provider,
        model=model_id,
        region=regions[0] if regions else None,
        endpoint=endpoints[0] if endpoints else None,
    ), None


async def enforce_egress_policy(
    policy: EgressPolicy,
    model: object,
    declaration: EgressDeclaration | Mapping[str, object] | None = None,
    provider_options: Mapping[str, object] | None = None,
    *,
    authoritative_destination: bool = False,
) -> EgressAuditRecord:
    target, conflict = resolve_egress_target(
        model,
        declaration,
        provider_options,
        authoritative_destination=authoritative_destination,
    )
    if conflict:
        normalized, _reason = _try_normalize(target.endpoint) if target.endpoint else (None, None)
        record = EgressAuditRecord(
            schema_version=_SCHEMA_VERSION,
            type="egress_call_audit",
            id=str(uuid.uuid4()),
            decision="denied",
            provider=str(getattr(model, "provider", "unknown")),
            model_id=str(getattr(model, "model_id", "unknown")),
            region=target.region,
            endpoint=None if normalized is None or normalized.kind == "any" else normalized.canonical,
            reason=conflict,
            created_at=int(datetime.now(timezone.utc).timestamp() * 1000),
        )
        await policy._write(record)
        raise EgressDeniedError(record)
    return await policy.assert_call(target)
