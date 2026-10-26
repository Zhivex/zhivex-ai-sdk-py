"""Optional ``provider/model`` resolution.

Direct provider factories remain supported. Nothing is registered by import.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .catalog import ModelCatalog, ModelCatalogEntry
from .errors import ConfigurationError
from .providers.base import ProviderAdapter
from .types import LanguageModel, ModelCapabilities

_MAX_PROVIDER_LENGTH = 128
_MAX_MODEL_ID_LENGTH = 256
_MAX_ALIAS_LENGTH = 128
_PROVIDER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\Z")

ModelResolutionErrorCode = str


class ModelResolutionError(ConfigurationError):
    """Preflight failure raised before a provider call."""

    def __init__(
        self,
        code: ModelResolutionErrorCode,
        message: str,
        details: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True, slots=True)
class ModelReference:
    provider: str
    model_id: str


@dataclass(frozen=True, slots=True)
class ModelResolverAlias:
    alias: str
    target: str | ModelReference


class ModelResolverBackend(Protocol):
    name: str

    def language_model(self, reference: ModelReference) -> LanguageModel: ...


@dataclass(frozen=True, slots=True)
class ModelResolutionMetadata:
    schema_version: int
    identifier: str
    requested: ModelReference
    resolved: ModelReference
    source: Mapping[str, str]
    catalog: Mapping[str, Any]
    catalog_entry: ModelCatalogEntry
    capabilities: ModelCapabilities
    alias: str | None = None


class ModelResolver(Protocol):
    def resolve(self, identifier: str) -> tuple[LanguageModel, ModelResolutionMetadata]: ...

    def model(self, identifier: str) -> LanguageModel: ...


def _control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _valid_provider(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _MAX_PROVIDER_LENGTH
        and value == value.strip()
        and _PROVIDER_RE.fullmatch(value) is not None
    )


def _valid_model_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _MAX_MODEL_ID_LENGTH
        and value == value.strip()
        and "://" not in value
        and not _control_characters(value)
    )


def _invalid_identifier(message: str) -> ModelResolutionError:
    return ModelResolutionError("invalid_identifier", message)


def _normalize_reference(value: str | ModelReference | Mapping[str, object]) -> ModelReference:
    if isinstance(value, str):
        if value == "" or value != value.strip() or _control_characters(value):
            raise _invalid_identifier(
                "Model identifiers must be non-empty and must not contain surrounding whitespace or control characters."
            )
        separator = value.find("/")
        if separator <= 0 or separator == len(value) - 1:
            raise _invalid_identifier('Model identifiers must use the explicit "provider/model" format.')
        return _normalize_reference({"provider": value[:separator], "model_id": value[separator + 1 :]})
    provider = value.provider if isinstance(value, ModelReference) else value.get("provider")
    model_id = value.model_id if isinstance(value, ModelReference) else value.get("model_id")
    if not _valid_provider(provider) or not _valid_model_id(model_id):
        raise _invalid_identifier(
            "Model references require a provider of at most "
            f"{_MAX_PROVIDER_LENGTH} safe characters and a model_id of at most {_MAX_MODEL_ID_LENGTH} characters."
        )
    return ModelReference(provider=str(provider), model_id=str(model_id))


def _normalize_alias(alias: object) -> str:
    if (
        not isinstance(alias, str)
        or alias == ""
        or len(alias) > _MAX_ALIAS_LENGTH
        or alias != alias.strip()
        or "/" in alias
        or _control_characters(alias)
    ):
        raise ModelResolutionError(
            "invalid_configuration",
            f"Model aliases must be non-empty, contain no slash, and be at most {_MAX_ALIAS_LENGTH} characters.",
        )
    return alias


def _assert_language_model(model: object) -> LanguageModel:
    if (
        not _valid_provider(getattr(model, "provider", None))
        or not _valid_model_id(getattr(model, "model_id", None))
        or not isinstance(getattr(model, "capabilities", None), ModelCapabilities)
        or not callable(getattr(model, "generate", None))
    ):
        raise ModelResolutionError(
            "invalid_resolved_model",
            "The configured adapter or backend returned an invalid language model before any model call was made.",
        )
    return model  # type: ignore[return-value]


def _catalog_trace(catalog: ModelCatalog) -> dict[str, Any]:
    metadata = getattr(catalog, "metadata", None)
    payload: dict[str, Any] = {
        "schema_version": int(getattr(metadata, "schema_version", 1)),
        "contract_version": str(getattr(metadata, "contract_version", "python")),
        "snapshot_version": str(getattr(metadata, "snapshot_version", "unreleased")),
    }
    pricing_version = getattr(metadata, "pricing_version", None)
    currency = getattr(metadata, "currency", None)
    if pricing_version is not None:
        payload["pricing_version"] = pricing_version
    if currency is not None:
        payload["currency"] = currency
    return payload


def _clone_capabilities(capabilities: ModelCapabilities) -> ModelCapabilities:
    agent = capabilities.agent_capabilities
    return replace(capabilities, agent_capabilities=None if agent is None else replace(agent))


class _Resolver:
    def __init__(
        self,
        *,
        catalog: ModelCatalog,
        adapters: Mapping[str, ProviderAdapter] | None,
        backend: ModelResolverBackend | None,
        aliases: Mapping[str, ModelReference],
        on_resolve: Callable[[ModelResolutionMetadata], None] | None,
    ) -> None:
        self._catalog = catalog
        self._adapters = dict(adapters) if adapters is not None else None
        self._backend = backend
        self._aliases = dict(aliases)
        self._on_resolve = on_resolve
        self._providers = {entry.provider for entry in catalog.list()}
        self._catalog_metadata = _catalog_trace(catalog)

    def _find(self, reference: ModelReference) -> ModelCatalogEntry:
        if reference.provider not in self._providers:
            raise ModelResolutionError(
                "unknown_provider",
                "The requested model provider is not present in the configured catalog.",
            )
        if self._adapters is not None and reference.provider not in self._adapters:
            raise ModelResolutionError(
                "unknown_provider",
                "The requested model provider is not configured in this resolver.",
            )
        entry = self._catalog.find(reference.provider, reference.model_id)
        if entry is None:
            raise ModelResolutionError(
                "unknown_model",
                "The requested model is not present for its provider in the configured catalog.",
            )
        return entry

    def resolve(self, identifier: str) -> tuple[LanguageModel, ModelResolutionMetadata]:
        if not isinstance(identifier, str):
            raise _invalid_identifier("Model identifiers must be strings.")
        alias_target = self._aliases.get(identifier)
        if "/" not in identifier and alias_target is None:
            _normalize_alias(identifier)
            raise ModelResolutionError(
                "unknown_alias",
                "The requested model alias is not configured in this resolver.",
            )
        requested = alias_target if alias_target is not None else _normalize_reference(identifier)
        entry = self._find(requested)
        canonical = ModelReference(provider=entry.provider, model_id=entry.model_id)
        if self._adapters is None:
            assert self._backend is not None
            source = {"kind": "backend", "name": self._backend.name}
            model = _assert_language_model(self._backend.language_model(canonical))
        else:
            source = {"kind": "adapter", "name": canonical.provider}
            model = _assert_language_model(self._adapters[canonical.provider].language_model(canonical.model_id))
        metadata = ModelResolutionMetadata(
            schema_version=1,
            identifier=identifier,
            requested=requested,
            resolved=canonical,
            source=source,
            catalog=dict(self._catalog_metadata),
            catalog_entry=entry,
            capabilities=_clone_capabilities(model.capabilities),
            alias=identifier if alias_target is not None else None,
        )
        if self._on_resolve is not None:
            self._on_resolve(metadata)
        return model, metadata

    def model(self, identifier: str) -> LanguageModel:
        return self.resolve(identifier)[0]


def create_model_resolver(
    *,
    catalog: ModelCatalog,
    adapters: Mapping[str, ProviderAdapter] | None = None,
    backend: ModelResolverBackend | None = None,
    aliases: Sequence[ModelResolverAlias | Mapping[str, Any]] | None = None,
    on_resolve: Callable[[ModelResolutionMetadata], None] | None = None,
) -> ModelResolver:
    if not isinstance(catalog, ModelCatalog) or not callable(getattr(catalog, "find", None)) or not callable(getattr(catalog, "list", None)):
        raise ModelResolutionError("invalid_configuration", "create_model_resolver requires an explicit ModelCatalog instance.")
    if (adapters is None) == (backend is None):
        raise ModelResolutionError(
            "invalid_configuration",
            "create_model_resolver requires exactly one explicit source: adapters or backend.",
        )
    if on_resolve is not None and not callable(on_resolve):
        raise ModelResolutionError("invalid_configuration", "create_model_resolver on_resolve must be a function.")
    adapter_snapshot: dict[str, ProviderAdapter] | None = None
    resolved_backend: ModelResolverBackend | None = None
    if adapters is not None:
        if not isinstance(adapters, Mapping):
            raise ModelResolutionError("invalid_configuration", "create_model_resolver adapters must be a mapping.")
        adapter_snapshot = {}
        for provider, adapter in adapters.items():
            if not _valid_provider(provider) or not callable(getattr(adapter, "language_model", None)):
                raise ModelResolutionError(
                    "invalid_configuration",
                    "Every model resolver adapter requires a safe provider key and a languageModel factory.",
                )
            adapter_snapshot[provider] = adapter
    else:
        assert backend is not None
        if not _valid_provider(getattr(backend, "name", None)) or not callable(getattr(backend, "language_model", None)):
            raise ModelResolutionError(
                "invalid_configuration",
                "A model resolver backend requires a safe name and a languageModel factory.",
            )
        resolved_backend = backend
    alias_map: dict[str, ModelReference] = {}
    finder = _Resolver(catalog=catalog, adapters=adapter_snapshot, backend=resolved_backend, aliases={}, on_resolve=None)
    if aliases is not None:
        if isinstance(aliases, (str, bytes)) or not isinstance(aliases, Sequence):
            raise ModelResolutionError("invalid_configuration", "create_model_resolver aliases must be a sequence.")
        for definition in aliases:
            if isinstance(definition, ModelResolverAlias):
                raw_alias: object = definition.alias
                raw_target: object = definition.target
            elif isinstance(definition, Mapping):
                raw_alias = definition.get("alias")
                raw_target = definition.get("target")
            else:
                raise ModelResolutionError("invalid_configuration", "Every model resolver alias must be an object.")
            alias = _normalize_alias(raw_alias)
            if alias in alias_map:
                raise ModelResolutionError("alias_collision", "A model alias is configured more than once.")
            if not isinstance(raw_target, (str, ModelReference, Mapping)):
                raise ModelResolutionError(
                    "invalid_configuration",
                    "Every model resolver alias target must be an explicit provider/model string or reference.",
                )
            target = _normalize_reference(raw_target)
            finder._find(target)
            alias_map[alias] = target
    return _Resolver(
        catalog=catalog,
        adapters=adapter_snapshot,
        backend=resolved_backend,
        aliases=alias_map,
        on_resolve=on_resolve,
    )


_REGISTERED: dict[str, ProviderAdapter] = {}


def register_model_provider(adapter: ProviderAdapter) -> None:
    if not callable(getattr(adapter, "language_model", None)) or not isinstance(getattr(adapter, "name", None), str) or not adapter.name:
        raise ModelResolutionError(
            "invalid_configuration",
            "register_model_provider requires a provider adapter with a name and language_model factory.",
        )
    _REGISTERED[adapter.name] = adapter


def clear_registered_model_providers() -> None:
    _REGISTERED.clear()


def resolve_language_model(
    model: LanguageModel | str,
    model_resolver: ModelResolver | None = None,
) -> LanguageModel:
    # Existing callers pass duck-typed models, including test doubles whose
    # attributes are not plain strings. Only provider/model strings are resolved.
    if not isinstance(model, str):
        return model
    if model_resolver is not None:
        return model_resolver.model(model)
    if model == "" or model != model.strip():
        raise ModelResolutionError(
            "invalid_identifier",
            "Model identifiers must be non-empty and must not contain surrounding whitespace.",
            {"identifier": model},
        )
    separator = model.find("/")
    if separator <= 0 or separator == len(model) - 1:
        raise ModelResolutionError(
            "invalid_identifier",
            'Model identifiers must use the explicit "provider/model" format.',
            {"identifier": model},
        )
    provider = model[:separator]
    model_id = model[separator + 1 :]
    adapter = _REGISTERED.get(provider)
    if adapter is None:
        raise ModelResolutionError(
            "unknown_provider",
            f'No language model provider is registered for "{provider}". Register one with register_model_provider() or pass model_resolver.',
            {"identifier": model, "provider": provider, "model_id": model_id},
        )
    resolved = adapter.language_model(model_id)
    if not isinstance(getattr(resolved, "provider", None), str) or not isinstance(getattr(resolved, "model_id", None), str) or not callable(getattr(resolved, "generate", None)):
        raise ModelResolutionError(
            "invalid_resolved_model",
            "The registered provider returned an invalid language model before any model call was made.",
            {"identifier": model, "provider": provider, "model_id": model_id},
        )
    return resolved
