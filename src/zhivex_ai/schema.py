from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from typing import Any, Protocol

from pydantic import BaseModel, TypeAdapter


def _looks_like_json_schema(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    json_schema_keys = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "anyOf",
        "oneOf",
        "allOf",
        "$defs",
        "$ref",
        "definitions",
    }
    return any(key in schema for key in json_schema_keys)


class _RawJsonSchemaAdapter:
    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = deepcopy(schema)

    def validate_python(self, value: Any) -> Any:
        return value

    def json_schema(self) -> dict[str, Any]:
        return deepcopy(self._schema)


class _SchemaAdapterProtocol(Protocol):
    def validate_python(self, value: Any) -> Any: ...

    def json_schema(self) -> dict[str, Any]: ...


class SchemaAdapter:
    def __init__(self, schema: Any) -> None:
        self.schema = schema
        self.adapter = self._to_adapter(schema)
        self._json_schema_cache: dict[str, Any] | None = None

    def validate_python(self, value: Any) -> Any:
        return self.adapter.validate_python(value)

    def json_schema(self) -> dict[str, Any]:
        if self._json_schema_cache is None:
            self._json_schema_cache = self.adapter.json_schema()
        return deepcopy(self._json_schema_cache)

    @staticmethod
    def _to_adapter(schema: Any) -> _SchemaAdapterProtocol:
        if isinstance(schema, TypeAdapter):
            return schema
        if _looks_like_json_schema(schema):
            return _RawJsonSchemaAdapter(schema)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return TypeAdapter(schema)
        return TypeAdapter(schema)


@lru_cache(maxsize=256)
def _create_hashable_schema_adapter(schema: Any) -> SchemaAdapter:
    return SchemaAdapter(schema)


def create_schema_adapter(schema: Any) -> SchemaAdapter:
    try:
        hash(schema)
    except TypeError:
        return SchemaAdapter(schema)
    return _create_hashable_schema_adapter(schema)
