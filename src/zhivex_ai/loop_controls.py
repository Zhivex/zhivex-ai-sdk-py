"""Optional generation loop controls.

Existing numeric defaults stay in place. ``"unlimited"`` and ``stop_when`` are
opt-in. A ``stop_when`` condition without ``max_steps`` uses a 20-step ceiling.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from .errors import ValidationError
from .types import GenerateTextStep, ToolSet

MaxSteps = int | Literal["unlimited"]
DEFAULT_STOP_WHEN_MAX_STEPS = 20


@dataclass(frozen=True, slots=True)
class StopContext:
    steps: tuple[GenerateTextStep, ...]
    step_count: int


StopCondition = Callable[[StopContext], bool | Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class PrepareStepContext:
    """``step_number`` is the zero-based index of this call. ``step`` is one-based."""

    step_number: int
    step: int
    steps: tuple[GenerateTextStep, ...]
    model: object
    messages: list
    tools: ToolSet | None


def validate_max_steps(value: MaxSteps | None) -> MaxSteps:
    if value == "unlimited":
        return value
    resolved = 1 if value is None else value
    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved < 1:
        raise ValidationError('"max_steps" must be a positive integer or "unlimited".')
    return resolved


def resolve_step_ceiling(
    max_steps: MaxSteps | None,
    stop_when: StopCondition | Sequence[StopCondition] | None,
) -> MaxSteps:
    if max_steps is not None:
        return validate_max_steps(max_steps)
    if stop_when is not None:
        return DEFAULT_STOP_WHEN_MAX_STEPS
    return validate_max_steps(None)


def step_count_is(count: int) -> StopCondition:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValidationError("step_count_is requires a positive integer.")

    def condition(context: StopContext) -> bool:
        return context.step_count >= count

    return condition


def has_tool_call(name: str | None = None) -> StopCondition:
    def condition(context: StopContext) -> bool:
        for step in context.steps:
            for message in step.response.messages or []:
                for part in message.parts:
                    if getattr(part, "type", None) != "tool-call":
                        continue
                    tool_call = getattr(part, "tool_call", None)
                    if name is None or getattr(tool_call, "name", None) == name:
                        return True
        return False

    return condition


async def stop_condition_met(
    stop_when: StopCondition | Sequence[StopCondition] | None,
    steps: Sequence[GenerateTextStep],
) -> bool:
    if stop_when is None:
        return False
    conditions = stop_when if isinstance(stop_when, Sequence) and not isinstance(stop_when, (str, bytes)) else (stop_when,)
    context = StopContext(steps=tuple(steps), step_count=len(steps))
    for condition in conditions:
        result = condition(context)
        if hasattr(result, "__await__"):
            result = await result  # type: ignore[misc]
        if result:
            return True
    return False


def select_active_tools(tools: ToolSet | None, active_tools: Sequence[str] | None) -> ToolSet | None:
    if not tools or active_tools is None:
        return tools
    selected: ToolSet = {}
    for name in active_tools:
        if not isinstance(name, str) or name not in tools or tools[name] is None:
            raise ValidationError(f'active_tools entry "{name}" is not a registered tool.')
        selected[name] = tools[name]
    return selected or None
