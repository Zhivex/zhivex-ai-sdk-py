"""Task-local cumulative budget accounting for agent-owned generation loops."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._agent_contracts import GuardrailTripwireTriggered
from .types import GenerateResult, TokenUsage

if TYPE_CHECKING:
    from .safety import BudgetGuard


@dataclass
class _BudgetLedger:
    guard: BudgetGuard
    steps: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def check(self, *, dispatch: bool = False) -> None:
        outcome = self.guard._evaluate_counts(
            steps=self.steps,
            tool_calls=self.tool_calls,
            tool_errors=self.tool_errors,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            dispatch=dispatch,
        )
        if outcome.tripwire_triggered:
            raise GuardrailTripwireTriggered(
                stage="output",
                guardrail_name="BudgetGuard",
                reason=outcome.reason,
            )

    def add_usage(self, usage: TokenUsage | None) -> None:
        if usage is not None:
            self.input_tokens += usage.input_tokens or 0
            self.output_tokens += usage.output_tokens or 0
            self.total_tokens += (
                usage.total_tokens
                if usage.total_tokens is not None
                else (usage.input_tokens or 0) + (usage.output_tokens or 0)
            )


@dataclass
class _BudgetScope:
    ledgers: list[_BudgetLedger] = field(default_factory=list)

    def add_guardrails(self, guardrails: list[Any]) -> None:
        # Deferred import avoids the public safety/agent import cycle.
        from .safety import BudgetGuard

        for guardrail in guardrails:
            owner = getattr(guardrail, "__self__", None)
            if isinstance(owner, BudgetGuard) and all(
                ledger.guard is not owner for ledger in self.ledgers
            ):
                self.ledgers.append(_BudgetLedger(owner))


_current: ContextVar[_BudgetScope | None] = ContextVar(
    "agent_budget_scope", default=None
)


def enter_budget_scope(guardrails: list[Any]) -> tuple[_BudgetScope, Token]:
    inherited = _current.get()
    scope = _BudgetScope(
        [ledger for ledger in inherited.ledgers if ledger.guard.include_child_runs]
        if inherited is not None
        else []
    )
    scope.add_guardrails(guardrails)
    return scope, _current.set(scope)


def exit_budget_scope(token: Token) -> None:
    _current.reset(token)


def before_model() -> None:
    scope = _current.get()
    if scope is not None:
        for ledger in scope.ledgers:
            ledger.check(dispatch=True)
        for ledger in scope.ledgers:
            ledger.steps += 1
            ledger.check()


def after_model(response: GenerateResult) -> None:
    scope = _current.get()
    if scope is not None:
        messages = response.messages or ([response.message] if response.message else [])
        calls = sum(
            part.type == "tool-call" for message in messages for part in message.parts
        )
        for ledger in scope.ledgers:
            ledger.add_usage(response.usage)
            ledger.tool_calls += calls
            ledger.check()


def before_tool() -> None:
    scope = _current.get()
    if scope is not None:
        for ledger in scope.ledgers:
            ledger.check(dispatch=True)


def after_tool(*, is_error: bool) -> None:
    scope = _current.get()
    if scope is not None:
        for ledger in scope.ledgers:
            ledger.tool_errors += int(is_error)
            ledger.check()
