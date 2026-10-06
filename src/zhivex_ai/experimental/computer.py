"""Experimental application-owned OpenAI Responses GA computer execution.

No desktop driver is included. Approval is for one immutable batch and its
provider safety checks. An exception after executor entry has an unknown external
outcome; stop and reconcile it before starting another run.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import time
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agent import Agent
    from .._agent_contracts import AgentRunResult

from ..errors import ToolExecutionOutcomeUnknown, ValidationError
from ..types import ToolDefinition, ToolExecutionContext


@dataclass(frozen=True, slots=True)
class ComputerApproval:
    approved: bool
    acknowledged_safety_check_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComputerScreenshot:
    """A bounded image from the app-owned session; the host validates its pixels."""

    image_url: str


def _validate_action(action: Any) -> None:
    if not isinstance(action, dict):
        raise ValidationError("Computer action must be an object.")
    kind = action.get("type")
    fields = {
        "click": {"type", "x", "y", "button"}, "double_click": {"type", "x", "y"},
        "move": {"type", "x", "y"}, "drag": {"type", "path"},
        "scroll": {"type", "x", "y", "scroll_x", "scroll_y"},
        "keypress": {"type", "keys"}, "type": {"type", "text"},
        "screenshot": {"type"}, "wait": {"type"},
    }
    if not isinstance(kind, str) or kind not in fields or set(action) - fields[kind]:
        raise ValidationError("Unsupported OpenAI GA computer action or fields.")
    def point(value: Any) -> bool:
        return isinstance(value, dict) and all(type(value.get(key)) is int and value[key] >= 0 for key in ("x", "y"))
    if kind in {"click", "double_click", "move", "scroll"} and not point(action):
        raise ValidationError("Computer coordinates must be nonnegative integers in host screenshot pixels.")
    if kind == "click" and action.get("button", "left") not in {"left", "right", "wheel", "back", "forward"}:
        raise ValidationError("Unsupported computer mouse button.")
    if kind == "scroll" and any(type(action.get(key)) is not int for key in ("scroll_x", "scroll_y")):
        raise ValidationError("Computer scroll deltas must be integers.")
    if kind == "drag":
        path = action.get("path")
        if not isinstance(path, list) or not 1 <= len(path) <= 1000 or any(not point(p) or set(p) != {"x", "y"} for p in path):
            raise ValidationError("Computer drag must contain a bounded path of points.")
    if kind == "keypress":
        keys = action.get("keys")
        if not isinstance(keys, list) or not 1 <= len(keys) <= 16 or any(not isinstance(key, str) or not 1 <= len(key) <= 128 for key in keys):
            raise ValidationError("Computer keypress must contain bounded nonempty keys.")
    if kind == "type" and (not isinstance(action.get("text"), str) or len(action["text"]) > 16000):
        raise ValidationError("Computer typing requires text of at most 16000 characters.")


async def _bounded_callback(awaitable: Awaitable[Any], seconds: float) -> Any:
    # wait_for waits for cancellation acknowledgement and can hang forever if a
    # driver suppresses cancellation. Stop waiting, but observe late exceptions.
    task = asyncio.ensure_future(awaitable)
    def observe(done: asyncio.Future[Any]) -> None:
        if not done.cancelled():
            done.exception()
    try:
        done, _ = await asyncio.wait({task}, timeout=seconds)
        if task not in done:
            raise TimeoutError("Computer callback deadline exceeded.")
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            task.add_done_callback(observe)


def _validate_call(value: Any, call_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("call_id") != call_id or not call_id:
        raise ValidationError("Computer call correlation is missing or changed.")
    actions = value.get("actions")
    if not isinstance(actions, list) or not 1 <= len(actions) <= 100:
        raise ValidationError("Computer actions must contain between 1 and 100 actions.")
    for action in actions:
        _validate_action(action)
    checks = value.get("pending_safety_checks", [])
    if not isinstance(checks, list) or any(
        not isinstance(check, dict) or not isinstance(check.get("id"), str) or not check["id"]
        for check in checks
    ):
        raise ValidationError("Computer safety checks must have nonempty identifiers.")
    if len({check["id"] for check in checks}) != len(checks):
        raise ValidationError("Computer safety check identifiers must be unique.")
    return deepcopy(value)


def _validate_screenshot(value: Any) -> str:
    if not isinstance(value, ComputerScreenshot):
        raise ValidationError("Computer executor must return ComputerScreenshot after the complete batch.")
    url = value.image_url
    if not isinstance(url, str) or len(url) > 12 * 1024 * 1024:
        raise ValidationError("Computer screenshot exceeds the image limit.")
    header, separator, data = url.partition(",")
    if not separator or header not in {"data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"}:
        raise ValidationError("Computer screenshot must be a PNG, JPEG or WebP base64 data URL.")
    try:
        decoded = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValidationError("Computer screenshot contains invalid base64.") from error
    if not decoded or len(decoded) > 8 * 1024 * 1024:
        raise ValidationError("Computer screenshot must contain at most 8 MiB of image data.")
    return url


def openai_computer_tool(
    *,
    authorize: Callable[[dict[str, Any], ToolExecutionContext[Any]], Awaitable[ComputerApproval]],
    execute: Callable[[dict[str, Any], ToolExecutionContext[Any]], Awaitable[ComputerScreenshot]],
    callback_timeout_ms: int = 30_000,
) -> ToolDefinition:
    """Create a native GA ``computer`` tool for generate_text/Agent.

    Both callbacks must be async and cancellation-cooperative. The authorizer
    receives call_id, ordered actions and complete pending_safety_checks. It must
    explicitly acknowledge exactly those IDs. Separate snapshots prevent an
    authorizer from rewriting the executed batch. Execute sequentially, recheck
    the authorized session/observation before effects, and return a fresh image.
    Generic Agent approval is additional and does not acknowledge provider checks.
    """
    for callback in (authorize, execute):
        if not (inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(getattr(callback, "__call__", None))):
            raise ValidationError("Computer callbacks must be async and cancellation-cooperative.")
    if isinstance(callback_timeout_ms, bool) or not isinstance(callback_timeout_ms, int) or callback_timeout_ms <= 0:
        raise ValidationError("callback_timeout_ms must be a positive integer.")

    async def invoke(value: Any, context: ToolExecutionContext[Any]) -> dict[str, Any]:
        batch = _validate_call(value, context.tool_call_id)
        native = context.metadata.get("provider_metadata", {})
        expected = {"call_id": native.get("call_id"), "actions": native.get("actions"), "pending_safety_checks": native.get("pending_safety_checks", [])}
        if native.get("item_type") != "computer_call" or batch != expected:
            raise ValidationError("Computer batch differs from the provider call; request fresh authorization.")
        context.raise_if_cancelled()
        deadline = int(time.time() * 1000) + callback_timeout_ms
        if context.deadline_ms is not None:
            deadline = min(deadline, context.deadline_ms)
        context.deadline_ms = deadline
        def remaining() -> float:
            return max(0, (deadline - int(time.time() * 1000)) / 1000)
        approval = await _bounded_callback(authorize(deepcopy(batch), replace(context, metadata=deepcopy(context.metadata))), remaining())
        required = [check["id"] for check in batch.get("pending_safety_checks", [])]
        if (
            not isinstance(approval, ComputerApproval)
            or approval.approved is not True
            or sorted(approval.acknowledged_safety_check_ids) != sorted(required)
        ):
            raise ValidationError("Computer batch denied or provider safety checks not explicitly acknowledged.")
        context.raise_if_cancelled()
        if remaining() <= 0:
            raise TimeoutError("Computer callback deadline expired before execution.")
        try:
            screenshot = await _bounded_callback(execute(deepcopy(batch), context), remaining())
            image_url = _validate_screenshot(screenshot)
            context.raise_if_cancelled()
        except BaseException as error:
            if not isinstance(error, (Exception, asyncio.CancelledError)):
                raise
            raise ToolExecutionOutcomeUnknown(
                "Computer execution was interrupted or returned no valid screenshot; external outcome is unknown. Reconcile before retrying.",
                tool_name="computer_use", tool_call_id=context.tool_call_id,
                timeout_ms=callback_timeout_ms,
                idempotency_key=context.idempotency_key or context.tool_call_id,
            ) from error
        return {
            "output": {"type": "computer_screenshot", "image_url": image_url},
            "acknowledged_safety_checks": deepcopy(batch.get("pending_safety_checks", [])),
        }

    return ToolDefinition(
        name="computer_use", description="Application-owned OpenAI GA computer executor.",
        schema=dict, execute=invoke,
        metadata={"zhivex_native_computer": "openai-ga"},
    )


@dataclass(frozen=True, slots=True)
class ComputerRunResult:
    """Model termination and independently verified task success are separate."""

    run: AgentRunResult[Any]
    verified: bool


async def run_computer_use(
    *,
    agent: Agent[Any, Any],
    is_complete: Callable[[AgentRunResult[Any]], Awaitable[bool]],
    verification_timeout_ms: int = 30_000,
    **run_options: Any,
) -> ComputerRunResult:
    """Run an Agent and check the app's read-only postcondition, including text-only finals.

    This is an experimental verification wrapper for the native tool integration,
    not a portable computer protocol. Suspended/failed/cancelled runs cannot be
    verified. Exceptions/timeouts from verification propagate; retain application
    receipts and reconcile existing effects before retrying the task.
    """
    from ..agent import run_agent

    if not (inspect.iscoroutinefunction(is_complete) or inspect.iscoroutinefunction(getattr(is_complete, "__call__", None))):
        raise ValidationError("is_complete must be an async read-only application callback.")
    if isinstance(verification_timeout_ms, bool) or not isinstance(verification_timeout_ms, int) or verification_timeout_ms <= 0:
        raise ValidationError("verification_timeout_ms must be a positive integer.")
    result = await run_agent(agent=agent, **run_options)
    if result.state is not None and result.state.status != "completed":
        return ComputerRunResult(run=result, verified=False)
    verified = await _bounded_callback(is_complete(result), verification_timeout_ms / 1000)
    return ComputerRunResult(run=result, verified=verified is True)
