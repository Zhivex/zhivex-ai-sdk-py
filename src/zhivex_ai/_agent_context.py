"""Shared message context formatting, without runtime dependencies."""

from __future__ import annotations

from collections.abc import Iterable

from .types import ModelMessage, TextPart


def _text_from_message(message: ModelMessage) -> str:
    return "".join(part.text for part in message.parts if isinstance(part, TextPart))


def _message_text(messages: Iterable[ModelMessage]) -> str:
    chunks: list[str] = []
    for message in messages:
        text = _text_from_message(message).strip()
        if text:
            chunks.append(f"{message.role}: {text}")
    return "\n".join(chunks)
