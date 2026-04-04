from __future__ import annotations

from typing import Any


def drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: cleaned for key, item in value.items() if (cleaned := drop_none(item)) is not None}
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := drop_none(item)) is not None]
    return value
