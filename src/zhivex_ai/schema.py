from __future__ import annotations

from typing import Any

from pydantic import BaseModel, TypeAdapter


class SchemaAdapter:
    def __init__(self, schema: Any) -> None:
        self.schema = schema
        self.adapter = self._to_adapter(schema)

    def validate_python(self, value: Any) -> Any:
        return self.adapter.validate_python(value)

    def json_schema(self) -> dict[str, Any]:
        return self.adapter.json_schema()

    @staticmethod
    def _to_adapter(schema: Any) -> TypeAdapter[Any]:
        if isinstance(schema, TypeAdapter):
            return schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return TypeAdapter(schema)
        return TypeAdapter(schema)


def create_schema_adapter(schema: Any) -> SchemaAdapter:
    return SchemaAdapter(schema)
