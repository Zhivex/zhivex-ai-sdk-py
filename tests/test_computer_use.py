"""Offline contracts, also executed against the built wheel without source imports."""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import pytest

from zhivex_ai import (
    Agent, ApprovalDecision, ToolExecutionOutcomeUnknown, ValidationError,
    create_in_memory_agent_run_store, create_openai, deny_all_approval_policy,
    generate_text, openai_computer_use_tool, resume_agent_run, run_agent, stream_text,
)
from zhivex_ai.experimental import ComputerApproval, ComputerScreenshot, openai_computer_tool

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a4uoAAAAASUVORK5CYII="
CALL = {"type": "computer_call", "id": "item-1", "call_id": "call-1", "actions": [
    {"type": "click", "x": 1, "y": 2, "button": "left"}, {"type": "screenshot"},
], "pending_safety_checks": [{"id": "safety-1", "code": "irreversible", "message": "Confirm this effect."}]}


class Response:
    status_code = 200
    headers = {}

    def __init__(self, payload, stream=False):
        self.payload = payload
        self.streaming = stream

    async def json(self):
        return self.payload

    async def text(self):
        return json.dumps(self.payload)

    async def iter_lines(self):
        for item in self.payload["output"]:
            yield "event: response.output_item.done"
            yield "data: " + json.dumps({"type": "response.output_item.done", "item": item})
            yield ""
        yield "event: response.completed"
        yield "data: " + json.dumps({"type": "response.completed", "response": self.payload})
        yield ""


def fixture(call=None, repeat=False):
    requests = []
    async def fetch(url, **kwargs):
        requests.append(deepcopy(kwargs["json_body"]))
        output = [deepcopy(CALL if call is None else call)] if len(requests) == 1 or repeat else [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}
        ]
        return Response({"id": "resp-" + str(len(requests)), "status": "completed", "output": output}, kwargs.get("stream"))
    return create_openai(api_key="test", fetch=fetch).native.language_model("gpt-5.4"), requests


async def allow(batch, context):
    return ComputerApproval(True, tuple(check["id"] for check in batch["pending_safety_checks"]))


async def screenshot(batch, context):
    return ComputerScreenshot(PNG)


def native_tools(authorize=allow, execute=screenshot, timeout=1000):
    return {"computer_use": openai_computer_tool(authorize=authorize, execute=execute, callback_timeout_ms=timeout)}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["generate", "agent", "stream"])
async def test_native_batch_round_trip_and_safety(mode):
    model, requests = fixture()
    effects = []
    async def execute(batch, context):
        effects.extend(batch["actions"])
        assert context.tool_call_id == "call-1"
        assert context.deadline_ms is not None
        return ComputerScreenshot(PNG)
    tools = native_tools(execute=execute)
    if mode == "agent":
        result = await run_agent(agent=Agent(name="computer", model=model, tools=tools), prompt="inspect", max_steps=3)
    elif mode == "stream":
        result = await stream_text(model=model, prompt="inspect", tools=tools, max_steps=3).collect()
    else:
        result = await generate_text(model=model, prompt="inspect", tools=tools, max_steps=3)
    assert effects == CALL["actions"]
    assert requests[0]["tools"] == [{"type": "computer"}]
    replay = requests[1]["input"]
    assert next(item for item in replay if item["type"] == "computer_call") == CALL
    output = next(item for item in replay if item["type"] == "computer_call_output")
    assert output == {"type": "computer_call_output", "call_id": "call-1", "output": {"type": "computer_screenshot", "image_url": PNG, "detail": "original"}, "acknowledged_safety_checks": CALL["pending_safety_checks"]}
    assert len(result.tool_results) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [None, {"computer": openai_computer_use_tool(tool_type="computer")}])
async def test_missing_executor_cannot_complete(tools):
    model, requests = fixture()
    with pytest.raises(ValidationError, match="not registered"):
        await run_agent(agent=Agent(name="missing", model=model, tools=tools), prompt="go")
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_agent_deny_policy_runs_before_authorizer_and_executor():
    model, requests = fixture()
    async def forbidden(*args):
        pytest.fail("Must not reach callbacks")
    agent = Agent(name="deny", model=model, tools=native_tools(forbidden, forbidden), approval_policy=deny_all_approval_policy)
    with pytest.raises(ValidationError, match="denied"):
        await run_agent(agent=agent, prompt="go", max_steps=3)
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("approval", [ComputerApproval(False), ComputerApproval(True), ComputerApproval(True, ("wrong",)), ComputerApproval(True, ("safety-1", "safety-1"))])
async def test_safety_acknowledgement_is_exact(approval):
    model, requests = fixture()
    async def authorize(batch, context):
        return approval
    async def forbidden(*args):
        pytest.fail("Must not execute")
    with pytest.raises(ValidationError, match="acknowledged"):
        await generate_text(model=model, prompt="go", tools=native_tools(authorize, forbidden), max_steps=3)
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_authorizer_mutation_does_not_change_effect():
    model, _ = fixture()
    async def authorize(batch, context):
        batch["actions"][0]["x"] = 9999
        batch["pending_safety_checks"][0]["message"] = "rewritten"
        return ComputerApproval(True, ("safety-1",))
    async def execute(batch, context):
        assert batch["actions"][0]["x"] == 1
        assert batch["pending_safety_checks"] == CALL["pending_safety_checks"]
        return ComputerScreenshot(PNG)
    await generate_text(model=model, prompt="go", tools=native_tools(authorize, execute), max_steps=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("actions", [None, [], [{}], [{"type": "submit"}]])
async def test_invalid_or_empty_batch_stops(actions):
    call = {**CALL, "actions": actions}
    model, requests = fixture(call)
    async def forbidden(*args):
        pytest.fail("Must not authorize malformed action")
    with pytest.raises(ValidationError):
        await generate_text(model=model, prompt="go", tools=native_tools(forbidden, forbidden), max_steps=3)
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["approval", "executor"])
async def test_callback_timeout_stops_without_retry(phase):
    model, requests = fixture()
    stopped = asyncio.Event()
    async def wait(*args):
        try:
            await asyncio.Future()
        finally:
            stopped.set()
    tools = native_tools(wait if phase == "approval" else allow, wait if phase == "executor" else screenshot, 20)
    error = ValidationError if phase == "approval" else ToolExecutionOutcomeUnknown
    with pytest.raises(error):
        await generate_text(model=model, prompt="go", tools=tools, max_steps=3)
    assert stopped.is_set()
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["approval", "executor"])
async def test_callback_cancellation_stops_without_retry(phase):
    model, requests = fixture()
    started = asyncio.Event()
    async def wait(*args):
        started.set()
        await asyncio.Future()
    tools = native_tools(wait if phase == "approval" else allow, wait if phase == "executor" else screenshot)
    task = asyncio.create_task(generate_text(model=model, prompt="go", tools=tools, max_steps=3))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError if phase == "approval" else ToolExecutionOutcomeUnknown):
        await task
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["failure", "none", "invalid"])
async def test_effect_without_screenshot_is_unknown(outcome):
    model, requests = fixture()
    effects = []
    async def execute(batch, context):
        effects.append("possible click")
        if outcome == "failure":
            raise RuntimeError("second action failed")
        return None if outcome == "none" else ComputerScreenshot("not-an-image")
    with pytest.raises(ToolExecutionOutcomeUnknown) as caught:
        await generate_text(model=model, prompt="go", tools=native_tools(execute=execute), max_steps=3)
    assert caught.value.tool_call_id == "call-1"
    assert effects == ["possible click"]
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_duplicate_call_is_not_reexecuted():
    model, requests = fixture(repeat=True)
    effects = []
    async def execute(batch, context):
        effects.append(1)
        return ComputerScreenshot(PNG)
    with pytest.raises(ValidationError, match="Repeated computer"):
        await generate_text(model=model, prompt="go", tools=native_tools(execute=execute), max_steps=3)
    assert effects == [1]
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_durable_approval_resume_keeps_native_correlation(approved):
    model, requests = fixture()
    async def suspend(request):
        assert request.tool_input["pending_safety_checks"] == CALL["pending_safety_checks"]
        return ApprovalDecision.require_human(approval_id="review")
    store = create_in_memory_agent_run_store()
    agent = Agent(name="resume", model=model, tools=native_tools(), approval_policy=suspend, run_store=store)
    pending = await run_agent(agent=agent, prompt="go", max_steps=3)
    assert pending.state.status == "suspended"
    assert len(requests) == 1
    if not approved:
        with pytest.raises(ValidationError, match="denied"):
            await resume_agent_run(agent=agent, run_id=pending.run_id, approved=False, max_steps=3)
        assert len(requests) == 1
        return
    result = await resume_agent_run(agent=agent, run_id=pending.run_id, approved=True, max_steps=3)
    assert result.state.status == "completed"
    output = next(item for item in requests[1]["input"] if item["type"] == "computer_call_output")
    assert output["call_id"] == "call-1"
    assert output["output"]["detail"] == "original"
    assert output["acknowledged_safety_checks"] == CALL["pending_safety_checks"]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["approval", "executor"])
async def test_noncooperative_callback_is_bounded(phase):
    model, requests = fixture()
    release = asyncio.Event()
    ended = asyncio.Event()
    async def stubborn(*args):
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release.wait()
        finally:
            ended.set()
        return ComputerScreenshot(PNG)
    try:
        error = ValidationError if phase == "approval" else ToolExecutionOutcomeUnknown
        with pytest.raises(error):
            async with asyncio.timeout(0.3):
                await generate_text(model=model, prompt="go", tools=native_tools(stubborn if phase == "approval" else allow, stubborn if phase == "executor" else screenshot, 10), max_steps=3)
        assert len(requests) == 1
        assert not ended.is_set()
    finally:
        release.set()
        await asyncio.wait_for(ended.wait(), 1)


@pytest.mark.asyncio
async def test_agent_cancellation_preserves_unknown_external_effect():
    from zhivex_ai import AgentCancellationToken
    model, requests = fixture()
    token = AgentCancellationToken()
    started = asyncio.Event()
    async def execute(batch, context):
        started.set()
        await asyncio.Future()
    agent = Agent(name="cancel", model=model, tools=native_tools(execute=execute))
    task = asyncio.create_task(run_agent(agent=agent, prompt="go", cancellation_token=token))
    await started.wait()
    token.cancel("stop")
    with pytest.raises(ToolExecutionOutcomeUnknown) as caught:
        await task
    assert caught.value.tool_call_id == "call-1"
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("verified", [True, False])
async def test_text_only_final_always_checks_postcondition(verified):
    from zhivex_ai.experimental import run_computer_use
    model, _ = fixture({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]})
    checks = []
    async def is_complete(result):
        checks.append(result.text)
        return verified
    result = await run_computer_use(agent=Agent(name="verify", model=model), prompt="go", is_complete=is_complete)
    assert checks == ["done"]
    assert result.verified is verified


@pytest.mark.asyncio
async def test_verification_timeout_does_not_report_success():
    from zhivex_ai.experimental import run_computer_use
    model, requests = fixture({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]})
    async def is_complete(result):
        await asyncio.Future()
    with pytest.raises(TimeoutError):
        await run_computer_use(agent=Agent(name="verify", model=model), prompt="go", is_complete=is_complete, verification_timeout_ms=10)
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [
    {"type": "click"}, {"type": "click", "x": True, "y": 1},
    {"type": "click", "x": -1, "y": 0}, {"type": "wait", "url": "unapproved"},
    {"type": "drag", "path": []}, {"type": "keypress", "keys": []},
    {"type": "scroll", "x": 0, "y": 0}, {"type": "type", "text": None},
])
async def test_malformed_action_never_reaches_authorizer(action):
    model, _ = fixture({**CALL, "actions": [action]})
    async def forbidden(*args):
        pytest.fail("Malformed action reached application")
    with pytest.raises(ValidationError):
        await generate_text(model=model, prompt="go", tools=native_tools(forbidden, forbidden))


def test_sync_callbacks_are_rejected_before_model_call():
    with pytest.raises(ValidationError, match="async"):
        openai_computer_tool(authorize=lambda *args: ComputerApproval(True), execute=screenshot)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "incomplete", "cancelled", None])
async def test_noncompleted_response_cannot_execute(status):
    async def fetch(url, **kwargs):
        return Response({"status": status, "output": [CALL]})
    model = create_openai(api_key="test", fetch=fetch).native.language_model("gpt-5.4")
    async def forbidden(*args):
        pytest.fail("Incomplete response executed")
    with pytest.raises(ValidationError, match="successfully completed"):
        await generate_text(model=model, prompt="go", tools=native_tools(forbidden, forbidden))


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["eof", "done", "failed", "incomplete"])
@pytest.mark.parametrize("mode", ["stream", "agent"])
async def test_interrupted_stream_cannot_execute(ending, mode):
    from zhivex_ai import stream_agent
    class Interrupted(Response):
        async def iter_lines(self):
            yield "data: " + json.dumps({"type": "response.output_item.done", "item": CALL})
            yield ""
            if ending == "done":
                yield "data: [DONE]"
                yield ""
            elif ending != "eof":
                yield "data: " + json.dumps({"type": "response." + ending, "response": {"status": ending, "output": [CALL]}})
                yield ""
    async def fetch(url, **kwargs):
        return Interrupted({})
    model = create_openai(api_key="test", fetch=fetch).native.language_model("gpt-5.4")
    async def forbidden(*args):
        pytest.fail("Interrupted stream reached application")
    tools = native_tools(forbidden, forbidden)
    with pytest.raises(Exception, match="terminal event|successfully completed"):
        if mode == "stream":
            await stream_text(model=model, prompt="go", tools=tools).collect()
        else:
            await stream_agent(agent=Agent(name="broken", model=model, tools=tools), prompt="go").collect()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["generate", "stream", "agent"])
@pytest.mark.parametrize("status", ["in_progress", "incomplete", "failed", "cancelled"])
async def test_unfinished_computer_item_never_executes(mode, status):
    model, requests = fixture({**CALL, "status": status})
    async def forbidden(*args):
        pytest.fail("Unfinished item reached application")
    tools = native_tools(forbidden, forbidden)
    with pytest.raises(ValidationError, match="computer.*status|Computer.*status"):
        if mode == "generate":
            await generate_text(model=model, prompt="go", tools=tools)
        elif mode == "stream":
            await stream_text(model=model, prompt="go", tools=tools).collect()
        else:
            await run_agent(agent=Agent(name="unfinished", model=model, tools=tools), prompt="go")
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("failure", ["guardrail", "hook", "hook_cancel", "serialization"])
async def test_post_effect_failure_remains_unknown(resume, failure):
    from zhivex_ai import AgentHooks, ToolGuardrailResult
    model, requests = fixture({**CALL, "status": "completed"})
    effects = []
    async def execute(batch, context):
        effects.append(context.tool_call_id)
        return ComputerScreenshot(PNG)
    async def guardrail(request):
        if failure == "serialization":
            return ToolGuardrailResult(replace=True, replacement=object())
        raise RuntimeError("output guardrail failed after effect")
    class Hooks(AgentHooks):
        async def on_tool_end(self, *args):
            if failure == "hook_cancel":
                raise asyncio.CancelledError("cancelled after effect")
            raise RuntimeError("hook failed after effect")
    async def suspend(request):
        return ApprovalDecision.require_human(approval_id="review")
    tools = native_tools(execute=execute)
    if failure in {"guardrail", "serialization"}:
        tools["computer_use"].output_guardrails = [guardrail]
    agent = Agent(name="after-effect", model=model, tools=tools,
                  run_store=create_in_memory_agent_run_store(),
                  hooks=[] if failure in {"guardrail", "serialization"} else [Hooks()],
                  approval_policy=suspend if resume else None)
    if resume:
        pending = await run_agent(agent=agent, prompt="go")
        assert effects == []
        operation = resume_agent_run(agent=agent, run_id=pending.run_id)
    else:
        operation = run_agent(agent=agent, prompt="go")
    with pytest.raises(ToolExecutionOutcomeUnknown) as caught:
        await operation
    assert caught.value.tool_call_id == "call-1"
    assert caught.value.idempotency_key.endswith(":call-1")
    assert caught.value.outcome_unknown
    assert effects == ["call-1"]
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
async def test_outer_timeout_during_approval_is_not_an_unknown_effect(resume):
    from zhivex_ai import ToolExecutionOptions
    model, requests = fixture()
    async def wait(*args):
        await asyncio.Future()
    async def forbidden(*args):
        pytest.fail("Timed-out approval executed")
    async def suspend(request):
        return ApprovalDecision.require_human(approval_id="review")
    agent = Agent(name="approval-timeout", model=model, tools=native_tools(wait, forbidden),
                  run_store=create_in_memory_agent_run_store(), approval_policy=suspend if resume else None)
    if resume:
        pending = await run_agent(agent=agent, prompt="go")
        operation = resume_agent_run(agent=agent, run_id=pending.run_id, tool_execution=ToolExecutionOptions(timeout_ms=10))
    else:
        operation = run_agent(agent=agent, prompt="go", tool_execution=ToolExecutionOptions(timeout_ms=10))
    with pytest.raises(ValidationError):
        await operation
    assert len(requests) == 1
