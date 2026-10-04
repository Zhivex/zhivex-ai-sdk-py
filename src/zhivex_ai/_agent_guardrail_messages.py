"""Caller provenance for the mutable message list exposed to input guardrails."""

from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher
from operator import index as integer_index
from typing import Any, Self, SupportsIndex, cast, overload

from .errors import ValidationError
from .types import ModelMessage


class _GuardrailMessages(list[ModelMessage]):
    def __init__(self, messages: list[ModelMessage], caller_count: int) -> None:
        super().__init__(messages)
        self._caller = [
            index >= len(messages) - caller_count for index in range(len(messages))
        ]
        self._known: dict[int, tuple[ModelMessage, bool]] = {}
        self._remember(messages, self._caller)

    def _remember(
        self, messages: Iterable[ModelMessage], origins: Iterable[bool]
    ) -> None:
        for message, caller in zip(messages, origins, strict=True):
            previous = self._known.get(id(message))
            self._known[id(message)] = (
                message,
                caller and (previous[1] if previous else True),
            )

    def _origin(self, message: ModelMessage, default: bool) -> bool:
        known = self._known.get(id(message))
        return known[1] if known is not None else default

    def _insertion_origin(self, index: int) -> bool:
        return not any(not caller for caller in self._caller[index:])

    def caller_messages(self) -> list[ModelMessage]:
        return [
            message
            for message, caller in zip(self, self._caller, strict=True)
            if caller
        ]

    def replace_all(self, messages: Iterable[ModelMessage]) -> None:
        values = list(messages)
        origins = self._reconcile(list(self), self._caller, values)
        super().__setitem__(slice(None), values)
        self._caller[:] = origins
        self._remember(values, origins)

    def _reconcile(
        self,
        previous: list[ModelMessage],
        origins: list[bool],
        values: list[ModelMessage],
    ) -> list[bool]:
        result: list[bool] = []
        matcher = SequenceMatcher(
            a=[id(message) for message in previous],
            b=[id(message) for message in values],
            autojunk=False,
        )
        for tag, first, last, start, stop in matcher.get_opcodes():
            labels = origins[first:last]
            replacements = values[start:stop]
            if tag == "equal":
                result.extend(labels)
                continue
            # Cloned but unchanged context remains identifiable by its value.
            prefix = 0
            while (
                prefix < min(len(labels), len(replacements))
                and previous[first + prefix] == replacements[prefix]
            ):
                result.append(labels[prefix])
                prefix += 1
            labels = labels[prefix:]
            replacements = replacements[prefix:]
            if not replacements:
                continue
            if labels and any(labels) and not all(labels):
                raise ValidationError(
                    "Input guardrail replaced both context and caller messages without preserving their provenance. "
                    "Edit caller messages in place or retain the context messages when replacing the list."
                )
            default = (
                labels[0] if labels else not any(not label for label in origins[first:])
            )
            result.extend(self._origin(message, default) for message in replacements)
        return result

    @overload
    def __setitem__(self, index: SupportsIndex, value: ModelMessage) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[ModelMessage]) -> None: ...

    def __setitem__(
        self, index: SupportsIndex | slice, value: ModelMessage | Iterable[ModelMessage]
    ) -> None:
        if not isinstance(index, slice):
            index = integer_index(index)
            if not isinstance(value, ModelMessage):
                raise TypeError("Message assignment requires a ModelMessage.")
            origin = self._origin(value, self._caller[index])
            super().__setitem__(index, value)
            self._caller[index] = origin
            self._remember([value], [origin])
            return
        if isinstance(value, ModelMessage):
            raise TypeError("Message slice assignment requires an iterable.")
        values = list(value)
        labels = self._caller[index]
        if len(labels) == len(values):
            origins = [
                self._origin(message, label)
                for message, label in zip(values, labels, strict=True)
            ]
        elif not labels or all(label == labels[0] for label in labels):
            start, _, _ = index.indices(len(self))
            origin = labels[0] if labels else self._insertion_origin(start)
            origins = [self._origin(message, origin) for message in values]
        else:
            origins = self._reconcile(self[index], labels, values)
        super().__setitem__(index, values)
        self._caller[index] = origins
        self._remember(values, origins)

    def __delitem__(self, index: SupportsIndex | slice) -> None:
        super().__delitem__(index)
        del self._caller[index]

    def append(self, message: ModelMessage) -> None:
        origin = self._origin(message, True)
        super().append(message)
        self._caller.append(origin)
        self._remember([message], [origin])

    def extend(self, messages: Iterable[ModelMessage]) -> None:
        for message in list(messages):
            self.append(message)

    def insert(self, index: SupportsIndex, message: ModelMessage) -> None:
        index = integer_index(index)
        index = max(0, min(index if index >= 0 else len(self) + index, len(self)))
        origin = self._origin(message, self._insertion_origin(index))
        super().insert(index, message)
        self._caller.insert(index, origin)
        self._remember([message], [origin])

    def pop(self, index: SupportsIndex = -1) -> ModelMessage:
        message = super().pop(index)
        self._caller.pop(index)
        return message

    def remove(self, message: ModelMessage) -> None:
        del self[self.index(message)]

    def clear(self) -> None:
        super().clear()
        self._caller.clear()

    def reverse(self) -> None:
        super().reverse()
        self._caller.reverse()

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        pairs = sorted(
            zip(self, self._caller, strict=True),
            key=lambda pair: key(pair[0]) if key is not None else cast(Any, pair[0]),
            reverse=reverse,
        )
        super().__setitem__(slice(None), [message for message, _ in pairs])
        self._caller[:] = [origin for _, origin in pairs]

    # Mypy compares += with list's heterogeneous __add__ overloads. The
    # instrumented list still accepts only ModelMessage values here.
    def __iadd__(self, messages: Iterable[ModelMessage]) -> Self:  # type: ignore[override, misc]
        self.extend(messages)
        return self

    def __imul__(self, count: SupportsIndex) -> Self:
        count = integer_index(count)
        super().__imul__(count)
        self._caller *= count
        return self
