from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .agent import (
    Agent,
    ApprovalDecision,
    ApprovalPolicy,
    GuardrailResult,
    InputGuardrail,
    InputGuardrailRequest,
    OutputGuardrail,
    OutputGuardrailRequest,
    RunLimits,
    ToolApprovalRequest,
)
from .agent_state import AgentRunState
from .types import JsonValue, ModelMessage, TextPart, ToolExecutionOptions

SafetyPolicyPreset = Literal["permissive", "review_sensitive", "locked_down"]
ApprovalPolicyPreset = SafetyPolicyPreset


@dataclass(slots=True)
class RedactionRule:
    pattern: str | re.Pattern[str]
    replacement: str = "[REDACTED]"
    name: str | None = None


@dataclass(slots=True)
class RedactionPolicy:
    rules: list[RedactionRule]

    def redact_text(self, text: str) -> str:
        redacted = text
        for rule in self.rules:
            redacted = re.sub(rule.pattern, rule.replacement, redacted)
        return redacted

    def redact_json(self, value: JsonValue | None) -> JsonValue | None:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, list):
            return [self.redact_json(item) for item in value]  # type: ignore[list-item]
        if isinstance(value, dict):
            return {key: self.redact_json(item) for key, item in value.items()}  # type: ignore[dict-item]
        return value

    def redact_messages(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        from .messages import create_text_message

        redacted: list[ModelMessage] = []
        for message in messages:
            parts = [part for part in message.parts if isinstance(part, TextPart)]
            if len(parts) == len(message.parts):
                redacted.append(create_text_message(message.role, self.redact_text("".join(part.text for part in parts))))
            else:
                redacted.append(message)
        return redacted

    async def input_guardrail(self, request: InputGuardrailRequest) -> GuardrailResult:
        request.messages[:] = self.redact_messages(request.messages)
        if request.prompt is not None:
            request.prompt = self.redact_text(request.prompt)
        return GuardrailResult()

    async def output_guardrail(self, request: OutputGuardrailRequest) -> GuardrailResult:
        request.text = self.redact_text(request.text)
        request.messages[:] = self.redact_messages(request.messages)
        return GuardrailResult()


@dataclass(slots=True)
class BudgetGuard:
    max_steps: int | None = None
    max_tool_calls: int | None = None
    max_tool_errors: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    include_child_runs: bool = True

    def evaluate_state(self, state: AgentRunState) -> GuardrailResult:
        child_runs = state.child_runs if self.include_child_runs else []
        steps = state.current_step + sum(child.steps for child in child_runs)
        tool_calls = sum(len(step.tool_calls) for step in state.steps) + sum(child.tool_calls for child in child_runs)
        tool_errors = sum(1 for result in state.tool_results if result.is_error) + sum(child.tool_errors for child in child_runs)
        usage = state.usage
        input_tokens = (usage.input_tokens if usage else 0) or 0
        output_tokens = (usage.output_tokens if usage else 0) or 0
        total_tokens = (usage.total_tokens if usage else None) or input_tokens + output_tokens
        for child in child_runs:
            if child.usage is None:
                continue
            input_tokens += child.usage.input_tokens or 0
            output_tokens += child.usage.output_tokens or 0
            total_tokens += child.usage.total_tokens or (child.usage.input_tokens or 0) + (child.usage.output_tokens or 0)
        checks = (
            (self.max_steps, steps, "steps"),
            (self.max_tool_calls, tool_calls, "tool calls"),
            (self.max_tool_errors, tool_errors, "tool errors"),
            (self.max_input_tokens, input_tokens, "input tokens"),
            (self.max_output_tokens, output_tokens, "output tokens"),
            (self.max_total_tokens, total_tokens, "total tokens"),
        )
        for limit, actual, label in checks:
            if limit is not None and actual > limit:
                return GuardrailResult(True, f"Agent run exceeded budget for {label}: {actual} > {limit}.")
        return GuardrailResult()

    async def input_guardrail(self, request: InputGuardrailRequest) -> GuardrailResult:
        return GuardrailResult()

    async def output_guardrail(self, request: OutputGuardrailRequest) -> GuardrailResult:
        return GuardrailResult()


@dataclass(slots=True)
class ApprovalPolicyOptions:
    preset: ApprovalPolicyPreset = "review_sensitive"
    sensitive_tool_names: list[str] = field(default_factory=list)
    allow_tool_names: list[str] = field(default_factory=list)
    deny_tool_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SafetyPolicy:
    preset: SafetyPolicyPreset
    approval_policy: ApprovalPolicy | None = None
    input_guardrails: list[InputGuardrail] = field(default_factory=list)
    output_guardrails: list[OutputGuardrail] = field(default_factory=list)
    tool_execution: ToolExecutionOptions | None = None
    redaction: RedactionPolicy | None = None
    budget: BudgetGuard | None = None
    run_limits: RunLimits | None = None


_SENSITIVE_TOOL_NAMES = {
    "delete",
    "delete_file",
    "write",
    "write_file",
    "exec",
    "execute",
    "shell",
    "bash",
    "terminal",
    "apply_patch",
    "deploy",
    "http",
    "request",
    "post",
}
_SENSITIVE_PERMISSIONS = {"write", "filesystem", "shell", "code-execution", "network", "external-side-effect"}
_SENSITIVE_HOSTED_CLASSES = {"computer-use", "code-execution", "remote-mcp", "toolset"}


def create_approval_policy(
    *,
    preset: ApprovalPolicyPreset = "review_sensitive",
    sensitive_tool_names: list[str] | None = None,
    allow_tool_names: list[str] | None = None,
    deny_tool_names: list[str] | None = None,
) -> ApprovalPolicy:
    extra_sensitive = {name.lower() for name in sensitive_tool_names or []}
    allow = {name.lower() for name in allow_tool_names or []}
    deny = {name.lower() for name in deny_tool_names or []}

    async def policy(request: ToolApprovalRequest) -> ApprovalDecision:
        name = request.tool_name.lower()
        if name in deny:
            return ApprovalDecision(False, f'Tool "{request.tool_name}" is denied by policy.')
        if name in allow or preset == "permissive":
            return ApprovalDecision(True)
        hosted_class = str(request.tool_metadata.get("hosted_tool_class", ""))
        high_risk = (
            name in _SENSITIVE_TOOL_NAMES
            or name in extra_sensitive
            or any(permission in _SENSITIVE_PERMISSIONS for permission in request.tool_permissions)
            or hosted_class in _SENSITIVE_HOSTED_CLASSES
            or bool(request.tool_metadata.get("requires_approval"))
        )
        if preset == "locked_down" or high_risk:
            return ApprovalDecision(False, f'Tool "{request.tool_name}" requires explicit approval under {preset}.')
        return ApprovalDecision(True)

    return policy


def create_redaction_policy(
    *,
    rules: list[RedactionRule] | None = None,
    include_emails: bool = False,
    replacement: str = "[REDACTED]",
) -> RedactionPolicy:
    default_rules = [
        RedactionRule(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", replacement, "bearer-token"),
        RedactionRule(r"\bBasic\s+[A-Za-z0-9+/=-]+", replacement, "basic-auth"),
        RedactionRule(r"\b(?:api[_-]?key|apikey|secret|token)\s*[:=]\s*[\"']?[A-Za-z0-9._~+/=-]{8,}[\"']?", replacement, "api-key"),
    ]
    if include_emails:
        default_rules.append(RedactionRule(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", replacement, "email"))
    return RedactionPolicy([*default_rules, *(rules or [])])


def create_budget_guard(
    *,
    max_steps: int | None = None,
    max_tool_calls: int | None = None,
    max_tool_errors: int | None = None,
    max_input_tokens: int | None = None,
    max_output_tokens: int | None = None,
    max_total_tokens: int | None = None,
    include_child_runs: bool = True,
) -> BudgetGuard:
    return BudgetGuard(
        max_steps=max_steps,
        max_tool_calls=max_tool_calls,
        max_tool_errors=max_tool_errors,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_total_tokens=max_total_tokens,
        include_child_runs=include_child_runs,
    )


def create_safety_policy(
    *,
    preset: SafetyPolicyPreset = "review_sensitive",
    approval: ApprovalPolicy | ApprovalPolicyOptions | bool | None = None,
    redaction: RedactionPolicy | bool | None = None,
    budget: BudgetGuard | bool | None = None,
    tool_execution: ToolExecutionOptions | None = None,
    input_guardrails: list[InputGuardrail] | None = None,
    output_guardrails: list[OutputGuardrail] | None = None,
) -> SafetyPolicy:
    if approval is False:
        approval_policy = None
    elif isinstance(approval, ApprovalPolicyOptions):
        approval_policy = create_approval_policy(
            preset=approval.preset,
            sensitive_tool_names=approval.sensitive_tool_names,
            allow_tool_names=approval.allow_tool_names,
            deny_tool_names=approval.deny_tool_names,
        )
    elif callable(approval):
        approval_policy = approval
    else:
        approval_policy = create_approval_policy(preset=preset)

    redaction_policy = None if redaction is False else redaction if isinstance(redaction, RedactionPolicy) else create_redaction_policy()
    budget_guard = None if budget is False else budget if isinstance(budget, BudgetGuard) else create_budget_guard()
    policy_input_guardrails = list(input_guardrails or [])
    policy_output_guardrails = list(output_guardrails or [])
    if redaction_policy is not None:
        policy_input_guardrails.append(redaction_policy.input_guardrail)
        policy_output_guardrails.append(redaction_policy.output_guardrail)
    if budget_guard is not None:
        policy_input_guardrails.append(budget_guard.input_guardrail)
        policy_output_guardrails.append(budget_guard.output_guardrail)
    return SafetyPolicy(
        preset=preset,
        approval_policy=approval_policy,
        input_guardrails=policy_input_guardrails,
        output_guardrails=policy_output_guardrails,
        tool_execution=tool_execution,
        redaction=redaction_policy,
        budget=budget_guard,
    )


def apply_safety_policy_to_agent(agent: Agent, policy: SafetyPolicy) -> Agent:
    approval_policy = policy.approval_policy or agent.approval_policy
    input_guardrails = [*agent.input_guardrails, *policy.input_guardrails]
    output_guardrails = [*agent.output_guardrails, *policy.output_guardrails]
    metadata: dict[str, Any] = {**agent.metadata, "safety_policy": policy.preset}
    return replace(
        agent,
        approval_policy=approval_policy,
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
        metadata=metadata,
    )
