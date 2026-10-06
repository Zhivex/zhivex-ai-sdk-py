"""Agent approvals helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

from ._agent_contracts import AgentCancellationToken as AgentCancellationToken
from ._agent_contracts import AgentContext as AgentContext
from ._agent_contracts import AgentHooks as AgentHooks
from ._agent_contracts import ApprovalDecision as ApprovalDecision
from ._agent_contracts import ApprovalPolicy as ApprovalPolicy
from ._agent_contracts import ToolApprovalRequest as ToolApprovalRequest
from ._agent_execution import (
    _await_with_agent_cancellation,
    _call_agent_hooks,
    _guardrail_name,
    _maybe_await,
    _new_id,
    _now_ms,
    _resolve_tool_registry,
)
from ._agent_tools import ToolRegistry as ToolRegistry
from ._agent_tools import _tool_definition_fingerprint as _tool_definition_fingerprint
from .agent_state import AgentRunState, PendingApproval
from .errors import AgentRunCancelled, ToolExecutionOutcomeUnknown, ValidationError
from .messages import is_callable_tool_definition, serialize_json_value
from .schema import create_schema_adapter
from .types import (
    ToolExecutionContext,
    ToolExecutionError,
    ToolExecutionOptions,
    ToolExecutionResult,
    ToolGuardrailResult,
    ToolGuardrailTripwireTriggered,
    ToolOutputGuardrailRequest,
    ToolSet,
    ToolSource,
)

if TYPE_CHECKING:
    from .agent import Agent


async def allow_all_approval_policy(request: ToolApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(approved=True)


async def deny_all_approval_policy(request: ToolApprovalRequest) -> ApprovalDecision:
    return ApprovalDecision(approved=False, reason="Tool execution denied by policy.")


def permission_allowlist_approval_policy(*allowed_permissions: str) -> ApprovalPolicy:
    allowed = set(allowed_permissions)

    async def policy(request: ToolApprovalRequest) -> ApprovalDecision:
        if not request.tool_permissions:
            return ApprovalDecision(approved=True)
        missing = [
            permission
            for permission in request.tool_permissions
            if permission not in allowed
        ]
        if missing:
            return ApprovalDecision(
                approved=False, reason=f"Missing permissions: {', '.join(missing)}"
            )
        return ApprovalDecision(approved=True)

    return policy


def _normalize_approval_decision(
    value: ApprovalDecision | bool | None,
) -> ApprovalDecision:
    if isinstance(value, ApprovalDecision):
        return value
    if value is False or value is None:
        return ApprovalDecision(approved=False)
    if value is True:
        return ApprovalDecision(approved=True)
    raise ValidationError("Approval policies must return ApprovalDecision or bool.")


def _pending_approval_from_request(
    request: ToolApprovalRequest,
    decision: ApprovalDecision,
    *,
    tool_call_id: str,
    tool_fingerprint: str,
) -> PendingApproval:
    from pydantic import BaseModel

    # Schema validation yields typed inputs; durable approvals store JSON and
    # validate that JSON against the tool schema again when resuming.
    arguments = request.tool_input
    if isinstance(arguments, BaseModel):
        arguments = arguments.model_dump(mode="json")
    return PendingApproval(
        id=decision.approval_id or _new_id("approval"),
        name=request.tool_name,
        arguments=serialize_json_value(arguments),
        provider=str(request.tool_metadata.get("provider") or "") or None,
        reason=decision.reason,
        tool_call_id=tool_call_id or None,
        permissions=list(request.tool_permissions),
        source=request.tool_source,
        metadata={
            str(key): serialize_json_value(value)
            for key, value in request.tool_metadata.items()
        },
        created_at_ms=_now_ms(),
        handoff_path=list(request.handoff_path),
        tool_fingerprint=tool_fingerprint,
    )


async def _execute_resolved_approval_tool(
    *,
    agent: Agent,
    state: AgentRunState,
    pending: PendingApproval,
    approved: bool,
    reason: str | None,
    tools: ToolSet | ToolRegistry | None,
    tool_execution: ToolExecutionOptions | None = None,
    deps: Any = None,
    cancellation_token: AgentCancellationToken | None = None,
    hooks: Iterable[AgentHooks] | None = None,
) -> ToolExecutionResult:
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled(state.run_id)
    agent_context = AgentContext(
        run_id=state.run_id,
        session_id=state.session_id or "",
        agent_name=state.agent_name,
        metadata=dict(state.metadata),
        handoff_path=list(pending.handoff_path),
        deps=deps,
        cancellation_token=cancellation_token,
    )
    effective_hooks = list(hooks or [])
    request = ToolApprovalRequest(
        run_id=state.run_id,
        session_id=state.session_id or "",
        agent_name=state.agent_name,
        tool_name=pending.name,
        tool_input=pending.arguments,
        tool_permissions=list(pending.permissions),
        tool_source=pending.source,
        tool_metadata=dict(pending.metadata),
        context=agent_context,
        handoff_path=list(pending.handoff_path),
    )
    await _call_agent_hooks(
        effective_hooks,
        "on_approval",
        agent_context,
        agent,
        request,
        ApprovalDecision(approved=approved, reason=reason),
    )
    native_metadata = next((dict(call.provider_metadata) for step in state.steps for call in step.tool_calls
                            if call.id == pending.tool_call_id), {})
    is_computer = bool(pending.metadata.get("zhivex_native_computer"))
    if is_computer and native_metadata.get("item_type") != "computer_call":
        raise ValidationError("Pending computer approval lost its native call correlation.")
    if not approved:
        if is_computer:
            raise ValidationError("Computer execution denied by approval decision.")
        return ToolExecutionResult(
            tool_call_id=pending.tool_call_id or pending.id,
            tool_name=pending.name,
            error=ToolExecutionError(
                message=reason
                or pending.reason
                or "Tool execution denied by approval decision."
            ),
            is_error=True,
            provider_metadata={"approval_id": pending.id, "approval_status": "denied"},
        )
    registry = _resolve_tool_registry(agent, tools)
    definition = registry.get(pending.name)
    if definition is None or not is_callable_tool_definition(definition):
        raise ValidationError(
            f'Pending approval references unknown local tool "{pending.name}".'
        )
    if not pending.tool_fingerprint:
        raise ValidationError(
            f'Pending approval "{pending.id}" predates tool fingerprinting and cannot be executed safely. '
            "Start a new agent run to request approval again."
        )
    if _tool_definition_fingerprint(definition) != pending.tool_fingerprint:
        raise ValidationError(
            f'Pending approval "{pending.id}" no longer matches the registered tool definition. '
            "Start a new agent run to approve the current tool version."
        )
    context = ToolExecutionContext(
        tool_name=pending.name,
        tool_call_id=pending.tool_call_id or pending.id,
        idempotency_key=(
            f"{agent.metadata.get('zhivex_workflow_step_idempotency_key') or state.run_id}:"
            f"{pending.tool_call_id or pending.name}"
        ),
        run_id=state.run_id,
        session_id=state.session_id or "",
        agent_name=state.agent_name,
        permissions=list(pending.permissions),
        source=cast(ToolSource, pending.source),
        metadata={**pending.metadata, "provider_metadata": native_metadata},
        handoff_path=list(pending.handoff_path),
        deps=deps,
        cancellation_token=cancellation_token,
    )
    await _call_agent_hooks(
        effective_hooks,
        "on_tool_start",
        agent_context,
        agent,
        definition,
        pending.arguments,
        context,
    )
    try:
        context.raise_if_cancelled()
        timeout_ms = tool_execution.timeout_ms if tool_execution is not None else None
        if timeout_ms is not None and timeout_ms <= 0:
            raise ValidationError(
                'The "tool_execution.timeout_ms" field must be greater than zero.'
            )
        context.deadline_ms = _now_ms() + timeout_ms if timeout_ms is not None else None
        parsed_input = create_schema_adapter(definition.schema).validate_python(
            pending.arguments
        )
        execution = _await_with_agent_cancellation(
            registry.execute(definition, parsed_input, context),
            cancellation_token=cancellation_token,
            run_id=state.run_id,
        )
        try:
            output = (
                await asyncio.wait_for(execution, timeout_ms / 1000)
                if timeout_ms is not None
                else await execution
            )
        except TimeoutError as error:
            raise ToolExecutionOutcomeUnknown(
                f'Tool "{pending.name}" exceeded its {timeout_ms} ms timeout; its external outcome is unknown. '
                f'Reconcile the side effect with idempotency key "{context.idempotency_key}" before retrying.',
                tool_name=pending.name,
                tool_call_id=pending.tool_call_id or pending.id,
                timeout_ms=cast(int, timeout_ms),
                idempotency_key=context.idempotency_key
                or f"{state.run_id}:{pending.name}",
            ) from error
        context.raise_if_cancelled()
        guarded_output = output
        for guardrail in definition.output_guardrails:
            guardrail_name = _guardrail_name(guardrail)
            guardrail_request = ToolOutputGuardrailRequest(
                tool_name=pending.name,
                input=pending.arguments,
                output=guarded_output,
                context=context,
            )
            try:
                raw_outcome = await _maybe_await(guardrail(guardrail_request))
                if isinstance(raw_outcome, ToolGuardrailResult):
                    outcome = raw_outcome
                elif isinstance(raw_outcome, bool):
                    outcome = ToolGuardrailResult(tripwire_triggered=raw_outcome)
                elif raw_outcome is None:
                    outcome = ToolGuardrailResult()
                else:
                    raise TypeError(
                        "Tool guardrails must return ToolGuardrailResult, bool, or None."
                    )
            except Exception as error:
                raise ToolGuardrailTripwireTriggered(
                    stage="output",
                    tool_name=pending.name,
                    guardrail_name=guardrail_name,
                    reason="Guardrail evaluation failed.",
                ) from error
            if outcome.tripwire_triggered:
                raise ToolGuardrailTripwireTriggered(
                    stage="output",
                    tool_name=pending.name,
                    guardrail_name=guardrail_name,
                    reason=outcome.reason,
                    metadata=outcome.metadata,
                )
            if outcome.replace:
                guarded_output = outcome.replacement
        output = guarded_output
    except (AgentRunCancelled, ToolExecutionOutcomeUnknown):
        raise
    except Exception as error:
        if is_computer:
            raise ValidationError(f"Computer approval continuation stopped: {error}") from error
        await _call_agent_hooks(
            effective_hooks,
            "on_tool_error",
            agent_context,
            agent,
            definition,
            pending.arguments,
            context,
            error,
            reverse=True,
        )
        return ToolExecutionResult(
            tool_call_id=pending.tool_call_id or pending.id,
            tool_name=pending.name,
            error=ToolExecutionError(message=str(error) or "Tool execution failed."),
            is_error=True,
            provider_metadata={
                "approval_id": pending.id,
                "approval_status": "approved",
            },
        )
    await _call_agent_hooks(
        effective_hooks,
        "on_tool_end",
        agent_context,
        agent,
        definition,
        pending.arguments,
        context,
        output,
        reverse=True,
    )
    return ToolExecutionResult(
        tool_call_id=pending.tool_call_id or pending.id,
        tool_name=pending.name,
        output=serialize_json_value(output),
        is_error=False,
        provider_metadata={**native_metadata, "approval_id": pending.id, "approval_status": "approved"},
    )


def _validate_resolved_approval_tool(
    *,
    agent: Agent,
    pending: PendingApproval,
    tools: ToolSet | ToolRegistry | None,
) -> None:
    registry = _resolve_tool_registry(agent, tools)
    definition = registry.get(pending.name)
    if definition is None or not is_callable_tool_definition(definition):
        raise ValidationError(
            f'Pending approval references unknown local tool "{pending.name}".'
        )
    if not pending.tool_fingerprint:
        raise ValidationError(
            f'Pending approval "{pending.id}" predates tool fingerprinting and cannot be executed safely. '
            "Start a new agent run to request approval again."
        )
    if _tool_definition_fingerprint(definition) != pending.tool_fingerprint:
        raise ValidationError(
            f'Pending approval "{pending.id}" no longer matches the registered tool definition. '
            "Start a new agent run to approve the current tool version."
        )
