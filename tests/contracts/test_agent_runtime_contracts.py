from __future__ import annotations

import asyncio
import sys
import tempfile
from collections.abc import AsyncIterable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    Agent,
    AgentCancellationToken,
    AgentEventDeliveryError,
    AgentErrorEvent,
    AgentFinishEvent,
    AgentRunCancelled,
    AgentRunStartEvent,
    AgentRuntime,
    ApprovalDecision,
    AgentRunState,
    AgentRunStore,
    GuardrailResult,
    PendingApproval,
    ToolApprovalRequest,
    ToolExecutionContext,
    ToolExecutionOptions,
    ToolExecutionOutcomeUnknown,
    ToolGuardrailResult,
    cancel_agent_run,
    cancel_agent_run_tree,
    create_agent_run_snapshot,
    create_agent_session,
    create_in_memory_agent_run_store,
    create_mock_language_model,
    create_sqlite_agent_memory_store,
    create_sqlite_agent_run_store,
    create_sqlite_checkpoint_store,
    create_text_message,
    deny_all_approval_policy,
    fail_agent_run_resume_claim,
    get_pending_agent_approvals,
    handoff_to,
    permission_allowlist_approval_policy,
    replay_agent_run,
    resume_agent,
    resume_agent_run,
    run_agent,
    stream_agent,
    to_ui_message_stream,
    tool,
)
from zhivex_ai.types import (  # noqa: E402
    GenerateResult,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    ToolCall,
    ToolCallPart,
    TokenUsage,
)
from zhivex_ai.errors import ValidationError  # noqa: E402
from zhivex_ai.agent_state import deserialize_agent_run_state  # noqa: E402


BASE_CAPABILITIES = ModelCapabilities(
    streaming=True,
    tools=True,
    structured_output=True,
    json_mode=True,
    tool_choice=True,
    parallel_tool_calls=False,
    vision=False,
    files=False,
    audio_input=False,
    audio_output=False,
    embeddings=False,
    reasoning=False,
    web_search=False,
)


class EchoModel:
    provider = "contract"
    model_id = "echo"
    capabilities = BASE_CAPABILITIES

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        user_message = next((message for message in reversed(input.messages) if message.role == "user"), None)
        text = "".join(part.text for part in (user_message.parts if user_message else []) if part.type == "text")
        return GenerateResult(messages=[create_text_message("assistant", f"echo:{text}")], text=f"echo:{text}")

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        async def generator() -> AsyncIterable[object]:
            yield StreamTextDeltaEvent(text_delta="stream")
            yield StreamTextDeltaEvent(text_delta=" ok")
            yield StreamFinishEvent(finish_reason="stop", usage=TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3))

        return generator()


class ToolModel:
    provider = "contract"
    model_id = "tool"
    capabilities = BASE_CAPABILITIES

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        if not any(message.role == "tool" for message in input.messages):
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[ToolCallPart(tool_call=ToolCall(id="call_1", name="lookup", input={"item": "apollo"}))],
                    )
                ],
                finish_reason="tool-calls",
            )
        return GenerateResult(messages=[create_text_message("assistant", "tool done")], text="tool done", finish_reason="stop")

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        async def generator() -> AsyncIterable[object]:
            yield StreamTextDeltaEvent(text_delta="tool done")
            yield StreamFinishEvent(finish_reason="stop")

        return generator()


class MultiToolModel:
    provider = "contract"
    model_id = "multi-tool"
    capabilities = replace(BASE_CAPABILITIES, parallel_tool_calls=True)

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        result_count = sum(message.role == "tool" for message in input.messages)
        if result_count < 2:
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[
                            ToolCallPart(tool_call=ToolCall(id="call_safe", name="safe", input={})),
                            ToolCallPart(tool_call=ToolCall(id="call_write", name="write", input={})),
                        ],
                    )
                ],
                finish_reason="tool-calls",
            )
        return GenerateResult(messages=[create_text_message("assistant", "both done")], text="both done")

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        async def generator() -> AsyncIterable[object]:
            yield StreamFinishEvent(finish_reason="stop")

        return generator()


class StreamingToolModel(ToolModel):
    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        async def generator() -> AsyncIterable[object]:
            if not any(message.role == "tool" for message in input.messages):
                yield StreamToolCallEvent(tool_call=ToolCall(id="call_1", name="lookup", input={"item": "apollo"}))
                yield StreamFinishEvent(finish_reason="tool-calls")
                return
            yield StreamTextDeltaEvent(text_delta="tool done")
            yield StreamFinishEvent(finish_reason="stop")

        return generator()


class HandoffModel:
    provider = "contract"
    model_id = "handoff"
    capabilities = BASE_CAPABILITIES

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        if not any(message.role == "tool" for message in input.messages):
            return GenerateResult(
                messages=[
                    ModelMessage(
                        role="assistant",
                        parts=[ToolCallPart(tool_call=ToolCall(id="call_1", name="delegate", input={"task": "research"}))],
                    )
                ],
                finish_reason="tool-calls",
            )
        return GenerateResult(messages=[create_text_message("assistant", "handoff done")], text="handoff done")

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        async def generator() -> AsyncIterable[object]:
            yield StreamFinishEvent(finish_reason="stop")

        return generator()


async def _approval_policy(request: ToolApprovalRequest) -> bool:
    return request.tool_name == "lookup"


def _run_store_factories() -> list[tuple[str, Callable[[], tuple[AgentRunStore, tempfile.TemporaryDirectory[str] | None]]]]:
    def memory_factory() -> tuple[AgentRunStore, None]:
        return create_in_memory_agent_run_store(), None

    def sqlite_factory() -> tuple[AgentRunStore, tempfile.TemporaryDirectory[str]]:
        directory = tempfile.TemporaryDirectory()
        return create_sqlite_agent_run_store(str(Path(directory.name) / "runs.sqlite3")), directory

    return [("in-memory", memory_factory), ("sqlite", sqlite_factory)]


@pytest.mark.asyncio
async def test_run_stream_and_resume_contract() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "agent.sqlite3")
        memory = create_sqlite_agent_memory_store(path)
        checkpoints = create_sqlite_checkpoint_store(path)
        session = create_agent_session()
        agent = Agent(name="assistant", model=EchoModel(), memory=memory, checkpoint_store=checkpoints)

        first = await run_agent(agent=agent, session=session, prompt="hello")
        stream = stream_agent(agent=agent, session=first.session, prompt="hello")
        stream_deltas = [event.text_delta async for event in stream.event_stream() if event.type == "text-delta"]
        streamed = await stream.collect()
        resumed = await resume_agent(agent=agent, session_id=session.id, prompt="again")

        assert first.text == "echo:hello"
        assert "".join(stream_deltas) == "stream ok"
        assert streamed.text == "stream ok"
        assert resumed.text == "echo:again"
        assert resumed.resumed_from_checkpoint is not None


@pytest.mark.asyncio
async def test_tool_approval_allow_and_deny_contract() -> None:
    allowed = Agent(
        name="assistant",
        model=ToolModel(),
        approval_policy=_approval_policy,
        tools={
            "lookup": tool(
                name="lookup",
                schema=dict[str, str],
                execute=lambda input: {"item": input["item"], "status": "ok"},
                requires_approval=True,
                permissions=["project:read"],
            )
        },
    )
    result = await run_agent(agent=allowed, prompt="lookup")
    assert result.text == "tool done"
    assert [event.type for event in result.trace.events].count("tool-approval") == 1  # type: ignore[union-attr]

    denied = Agent(
        name="assistant",
        model=ToolModel(),
        approval_policy=deny_all_approval_policy,
        tools={
            "lookup": tool(
                name="lookup",
                schema=dict[str, str],
                execute=lambda input: {"item": input["item"]},
                requires_approval=True,
            )
        },
    )
    denied_result = await run_agent(agent=denied, prompt="lookup")
    assert denied_result.tool_results[0].is_error
    assert "denied" in (denied_result.tool_results[0].error.message if denied_result.tool_results[0].error else "")


@pytest.mark.asyncio
async def test_required_local_tool_approval_fails_closed_without_policy() -> None:
    executions = 0

    async def execute(input: dict[str, str]) -> dict[str, str]:
        nonlocal executions
        executions += 1
        return input

    agent = Agent(
        name="assistant",
        model=ToolModel(),
        tools={
            "lookup": tool(
                name="lookup",
                schema=dict[str, str],
                execute=execute,
                requires_approval=True,
            )
        },
    )

    result = await run_agent(agent=agent, prompt="lookup")

    assert executions == 0
    assert result.tool_results[0].is_error
    assert "approval_policy" in (result.tool_results[0].error.message if result.tool_results[0].error else "")


@pytest.mark.asyncio
async def test_tool_approval_can_suspend_and_resume_from_run_store() -> None:
    store = create_in_memory_agent_run_store()

    async def suspend_policy(request: ToolApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.require_human("manager approval required", approval_id="approval_lookup")

    agent = Agent(
        name="assistant",
        model=ToolModel(),
        run_store=store,
        approval_policy=suspend_policy,
        tools={
            "lookup": tool(
                name="lookup",
                schema=dict[str, str],
                execute=lambda input: {"item": input["item"], "status": "approved"},
                requires_approval=True,
                permissions=["project:read"],
            )
        },
    )
    suspended = await run_agent(agent=agent, prompt="lookup", idempotency_key="approval-job")
    assert suspended.state is not None
    assert suspended.state.status == "suspended"
    assert suspended.state.pending_approvals[0].id == "approval_lookup"
    assert suspended.finish_reason == "tool-calls"

    pending = await get_pending_agent_approvals(store, suspended.run_id)
    assert len(pending) == 1
    assert pending[0].name == "lookup"
    assert pending[0].arguments == {"item": "apollo"}
    assert pending[0].permissions == ["project:read"]

    resumed = await resume_agent_run(agent=agent, run_id=suspended.run_id, approval_id="approval_lookup")
    assert resumed.text == "tool done"
    assert resumed.state is not None
    assert resumed.state.parent_run_id == suspended.run_id

    final_state = await store.load(suspended.run_id)
    assert final_state is not None
    assert final_state.status == "completed"
    assert final_state.output_text == "tool done"
    assert final_state.pending_approvals == []
    assert final_state.child_runs[0].run_id == resumed.run_id
    assert final_state.metadata["resolved_approval"]["approved"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "factory"), _run_store_factories(), ids=[name for name, _factory in _run_store_factories()])
async def test_concurrent_approval_resume_executes_tool_once(
    name: str,
    factory: Callable[[], tuple[AgentRunStore, Any]],
) -> None:
    del name
    store, cleanup = factory()
    execution_started = asyncio.Event()
    release_execution = asyncio.Event()
    executions = 0

    async def suspend_policy(request: ToolApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.require_human("manager approval required", approval_id="approval_lookup")

    async def execute(input: dict[str, str]) -> dict[str, str]:
        nonlocal executions
        executions += 1
        execution_started.set()
        await release_execution.wait()
        return {"item": input["item"], "status": "approved"}

    try:
        agent = Agent(
            name="assistant",
            model=ToolModel(),
            run_store=store,
            approval_policy=suspend_policy,
            tools={
                "lookup": tool(
                    name="lookup",
                    schema=dict[str, str],
                    execute=execute,
                    requires_approval=True,
                )
            },
        )
        suspended = await run_agent(agent=agent, prompt="lookup")
        first = asyncio.create_task(
            resume_agent_run(agent=agent, run_id=suspended.run_id, approval_id="approval_lookup")
        )
        await asyncio.wait_for(execution_started.wait(), timeout=1)
        second = asyncio.create_task(
            resume_agent_run(agent=agent, run_id=suspended.run_id, approval_id="approval_lookup")
        )
        await asyncio.sleep(0)
        release_execution.set()
        results = await asyncio.gather(first, second, return_exceptions=True)

        assert executions == 1
        assert sum(not isinstance(item, BaseException) for item in results) == 1
        errors = [item for item in results if isinstance(item, BaseException)]
        assert len(errors) == 1
        assert isinstance(errors[0], ValidationError)
        assert "not suspended" in str(errors[0]) or "already being resumed" in str(errors[0])
    finally:
        release_execution.set()
        if cleanup is not None:
            cleanup.cleanup()


@pytest.mark.asyncio
async def test_suspension_preserves_completed_calls_and_defers_later_side_effects() -> None:
    store = create_in_memory_agent_run_store()
    executions: list[str] = []

    async def suspend_write(request: ToolApprovalRequest) -> ApprovalDecision:
        if request.tool_name == "write":
            return ApprovalDecision.require_human("review write", approval_id="approval_write")
        return ApprovalDecision(True)

    agent = Agent(
        name="assistant",
        model=MultiToolModel(),
        run_store=store,
        approval_policy=suspend_write,
        tools={
            "safe": tool(
                name="safe",
                schema=dict,
                execute=lambda _input: executions.append("safe") or {"ok": True},
                requires_approval=False,
            ),
            "write": tool(
                name="write",
                schema=dict,
                execute=lambda _input: executions.append("write") or {"ok": True},
                requires_approval=True,
            ),
        },
    )

    suspended = await run_agent(agent=agent, prompt="run both")

    assert executions == ["safe"]
    assert [result.tool_name for result in suspended.tool_results] == ["safe"]
    resumed = await resume_agent_run(agent=agent, run_id=suspended.run_id, approval_id="approval_write")
    assert executions == ["safe", "write"]
    assert resumed.text == "both done"


@pytest.mark.asyncio
async def test_approval_resume_rejects_changed_closure_state() -> None:
    store = create_in_memory_agent_run_store()
    config = {"destination": "original"}

    async def suspend_policy(_request: ToolApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.require_human("review", approval_id="approval_lookup")

    def execute(_input: dict[str, str]) -> dict[str, str]:
        return {"destination": config["destination"]}

    agent = Agent(
        name="assistant",
        model=ToolModel(),
        run_store=store,
        approval_policy=suspend_policy,
        tools={"lookup": tool(name="lookup", schema=dict[str, str], execute=execute, requires_approval=True)},
    )
    suspended = await run_agent(agent=agent, prompt="lookup")
    config["destination"] = "changed"

    with pytest.raises(ValidationError, match="no longer matches"):
        await resume_agent_run(agent=agent, run_id=suspended.run_id, approval_id="approval_lookup")


@pytest.mark.asyncio
async def test_approval_resume_rejects_changed_tool_guardrail() -> None:
    store = create_in_memory_agent_run_store()
    policy = {"version": "original"}

    async def suspend_policy(_request: ToolApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.require_human("review", approval_id="approval_lookup")

    async def input_policy(_request: object) -> ToolGuardrailResult:
        _ = policy["version"]
        return ToolGuardrailResult()

    agent = Agent(
        name="assistant",
        model=ToolModel(),
        run_store=store,
        approval_policy=suspend_policy,
        tools={
            "lookup": tool(
                name="lookup",
                schema=dict[str, str],
                execute=lambda _input: {"ok": True},
                requires_approval=True,
                input_guardrails=[input_policy],
            )
        },
    )
    suspended = await run_agent(agent=agent, prompt="lookup")
    policy["version"] = "changed"

    with pytest.raises(ValidationError, match="no longer matches"):
        await resume_agent_run(agent=agent, run_id=suspended.run_id, approval_id="approval_lookup")


@pytest.mark.asyncio
async def test_approval_resume_applies_output_guardrail_and_propagates_cancellation_context() -> None:
    store = create_in_memory_agent_run_store()
    token = AgentCancellationToken()
    seen_tokens: list[object] = []

    async def suspend_policy(_request: ToolApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.require_human("review", approval_id="approval_lookup")

    def execute(_input: dict[str, str], context: ToolExecutionContext) -> dict[str, str]:
        seen_tokens.append(context.cancellation_token)
        return {"secret": "raw"}

    async def redact_output(_request: object) -> ToolGuardrailResult:
        return ToolGuardrailResult(replacement={"secret": "[REDACTED]"}, replace=True)

    agent = Agent(
        name="assistant",
        model=ToolModel(),
        run_store=store,
        approval_policy=suspend_policy,
        tools={
            "lookup": tool(
                name="lookup",
                schema=dict[str, str],
                execute=execute,
                requires_approval=True,
                output_guardrails=[redact_output],
            )
        },
    )
    suspended = await run_agent(agent=agent, prompt="lookup")
    await resume_agent_run(
        agent=agent,
        run_id=suspended.run_id,
        approval_id="approval_lookup",
        cancellation_token=token,
    )

    stored = await store.load(suspended.run_id)
    assert stored is not None
    assert seen_tokens == [token]
    assert stored.tool_results[0].output == {"secret": "[REDACTED]"}


@pytest.mark.asyncio
async def test_approval_resume_preserves_unknown_outcome_on_tool_timeout() -> None:
    store = create_in_memory_agent_run_store()

    async def suspend_policy(_request: ToolApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.require_human("review", approval_id="approval_lookup")

    async def slow_execute(_input: dict[str, str]) -> dict[str, bool]:
        await asyncio.sleep(1)
        return {"ok": True}

    agent = Agent(
        name="assistant",
        model=ToolModel(),
        run_store=store,
        approval_policy=suspend_policy,
        tools={
            "lookup": tool(
                name="lookup",
                schema=dict[str, str],
                execute=slow_execute,
                requires_approval=True,
            )
        },
    )
    suspended = await run_agent(agent=agent, prompt="lookup")

    with pytest.raises(ToolExecutionOutcomeUnknown, match="outcome is unknown"):
        await resume_agent_run(
            agent=agent,
            run_id=suspended.run_id,
            approval_id="approval_lookup",
            tool_execution=ToolExecutionOptions(timeout_ms=10),
        )

    stored = await store.load(suspended.run_id)
    assert stored is not None
    assert stored.status == "failed"


@pytest.mark.asyncio
async def test_idempotency_key_is_claimed_atomically() -> None:
    store = create_in_memory_agent_run_store()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class BlockingModel(EchoModel):
        async def generate(self, input: ModelGenerateInput) -> GenerateResult:
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return await super().generate(input)

    agent = Agent(name="assistant", model=BlockingModel(), run_store=store)
    first_task = asyncio.create_task(run_agent(agent=agent, prompt="first", idempotency_key="same"))
    await started.wait()
    second = await run_agent(agent=agent, prompt="second", idempotency_key="same")
    release.set()
    first = await first_task

    assert calls == 1
    assert first.run_id == second.run_id
    assert first.session.id == second.session.id


@pytest.mark.asyncio
async def test_tool_approval_can_suspend_and_resume_with_denial() -> None:
    store = create_in_memory_agent_run_store()

    async def suspend_policy(request: ToolApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.require_human("manager approval required", approval_id="approval_lookup")

    agent = Agent(
        name="assistant",
        model=ToolModel(),
        run_store=store,
        approval_policy=suspend_policy,
        tools={
            "lookup": tool(
                name="lookup",
                schema=dict[str, str],
                execute=lambda input: {"item": input["item"], "status": "approved"},
                requires_approval=True,
            )
        },
    )
    suspended = await run_agent(agent=agent, prompt="lookup", idempotency_key="approval-deny-job")

    resumed = await resume_agent_run(
        agent=agent,
        run_id=suspended.run_id,
        approval_id="approval_lookup",
        approved=False,
        reason="manager rejected request",
    )

    assert resumed.text == "tool done"
    assert resumed.state is not None
    assert resumed.state.parent_run_id == suspended.run_id
    final_state = await store.load(suspended.run_id)
    assert final_state is not None
    assert final_state.status == "completed"
    assert final_state.pending_approvals == []
    assert final_state.metadata["resolved_approval"]["approved"] is False
    assert final_state.metadata["resolved_approval"]["reason"] == "manager rejected request"
    assert final_state.tool_results[0].is_error is True
    assert final_state.tool_results[0].provider_metadata["approval_status"] == "denied"


@pytest.mark.asyncio
async def test_suspended_tool_approval_generates_stable_event_and_state_id() -> None:
    store = create_in_memory_agent_run_store()

    async def suspend_policy(request: ToolApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.require_human("manager approval required")

    agent = Agent(
        name="assistant",
        model=ToolModel(),
        run_store=store,
        approval_policy=suspend_policy,
        tools={
            "lookup": tool(
                name="lookup",
                schema=dict[str, str],
                execute=lambda input: {"item": input["item"], "status": "approved"},
                requires_approval=True,
            )
        },
    )

    result = await run_agent(agent=agent, prompt="lookup")

    assert result.trace is not None
    approval_events = [event for event in result.trace.events if event.type == "tool-approval"]
    assert len(approval_events) == 1
    assert result.state is not None
    approval_id = result.state.pending_approvals[0].id
    assert approval_id.startswith("approval_")
    assert approval_events[0].approval_request_id == approval_id


@pytest.mark.asyncio
async def test_suspended_tool_approval_streams_to_ui_message_chunk() -> None:
    store = create_in_memory_agent_run_store()

    async def suspend_policy(request: ToolApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.require_human("manager approval required", approval_id="approval_lookup")

    agent = Agent(
        name="assistant",
        model=StreamingToolModel(),
        run_store=store,
        approval_policy=suspend_policy,
        tools={
            "lookup": tool(
                name="lookup",
                schema=dict[str, str],
                execute=lambda input: {"item": input["item"], "status": "approved"},
                requires_approval=True,
            )
        },
    )

    stream = stream_agent(agent=agent, prompt="lookup")
    chunks = [chunk async for chunk in to_ui_message_stream(stream, message_id="assistant-approval")]
    result = await stream.collect()

    approval_chunks = [chunk for chunk in chunks if chunk.type == "tool-approval"]
    assert result.state is not None
    assert result.state.status == "suspended"
    assert len(approval_chunks) == 1
    assert approval_chunks[0].approval_request_id == "approval_lookup"
    assert approval_chunks[0].metadata["suspended"] is True


@pytest.mark.asyncio
async def test_handoff_contract_records_orchestration_path() -> None:
    researcher = Agent(name="researcher", model=EchoModel())
    triage = Agent(
        name="triage",
        model=HandoffModel(),
        subagents={"researcher": researcher},
        tools={
            "delegate": tool(
                name="delegate",
                schema=dict[str, str],
                execute=lambda input: handoff_to("researcher", input=input["task"]),
            )
        },
    )

    result = await run_agent(agent=triage, prompt="delegate")

    assert result.orchestration_path == ["triage", "researcher"]
    assert result.trace is not None
    assert result.trace.handoff_count == 1
    assert result.text == "echo:research"


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "factory"), _run_store_factories(), ids=[name for name, _factory in _run_store_factories()])
async def test_run_store_idempotency_cancel_and_replay_contract(name: str, factory: Callable[[], tuple[AgentRunStore, Any]]) -> None:
    del name
    store, cleanup = factory()
    try:
        parent = AgentRunState(run_id="parent", agent_name="parent", provider="mock", model_id="model")
        child = AgentRunState(run_id="child", agent_name="child", provider="mock", model_id="model", parent_run_id="parent")
        await store.save(parent)
        await store.save(child)

        cancelled = await cancel_agent_run_tree(store, "parent", reason="stop")
        assert [state.run_id for state in cancelled.cancelled] == ["parent", "child"]
        assert (await store.load("child")).status == "cancelled"  # type: ignore[union-attr]

        agent = Agent(name="assistant", model=create_mock_language_model(), run_store=store)
        first = await run_agent(agent=agent, prompt="hello", idempotency_key="idem")
        second = await run_agent(agent=agent, prompt="ignored", idempotency_key="idem")
        state = await store.load(first.run_id)

        assert first.run_id == second.run_id
        assert state is not None
        assert create_agent_run_snapshot(state).run_id == first.run_id
        assert replay_agent_run(state).timeline[0].type == "run-start"
    finally:
        if cleanup is not None:
            cleanup.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "factory"), _run_store_factories(), ids=[name for name, _factory in _run_store_factories()])
async def test_cancelled_run_cannot_be_overwritten_by_stale_worker(
    name: str,
    factory: Callable[[], tuple[AgentRunStore, Any]],
) -> None:
    del name
    store, cleanup = factory()
    try:
        state = AgentRunState(run_id="run", agent_name="assistant", provider="mock", model_id="model")
        await store.save(state)
        stale_worker_state = await store.load("run")
        assert stale_worker_state is not None

        cancelled = await cancel_agent_run(store, "run", reason="operator-stop", now_ms=123)
        assert cancelled is not None
        assert cancelled.status == "cancelled"
        assert cancelled.revision == 1

        stale_worker_state.status = "completed"
        stale_worker_state.output_text = "late result"
        with pytest.raises(ValidationError, match="revision conflict"):
            await store.save(stale_worker_state)

        persisted = await store.load("run")
        assert persisted is not None
        assert persisted.status == "cancelled"
        assert persisted.cancellation_reason == "operator-stop"
        assert persisted.output_text == ""
        assert persisted.revision == 1
    finally:
        if cleanup is not None:
            cleanup.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "factory"), _run_store_factories(), ids=[name for name, _factory in _run_store_factories()])
async def test_agent_worker_completion_cannot_resurrect_cancelled_run(
    name: str,
    factory: Callable[[], tuple[AgentRunStore, Any]],
) -> None:
    del name
    store, cleanup = factory()
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingModel(EchoModel):
        async def generate(self, input: ModelGenerateInput) -> GenerateResult:
            started.set()
            await release.wait()
            return await super().generate(input)

    try:
        agent = Agent(name="assistant", model=BlockingModel(), run_store=store)
        events: list[Any] = []

        async def capture(event: Any) -> None:
            events.append(event)

        worker = asyncio.create_task(
            AgentRuntime().run(
                agent=agent,
                prompt="hello",
                idempotency_key="cancel-race",
                emit=capture,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        running = await store.find_by_idempotency_key("cancel-race")
        assert running is not None

        cancelled = await cancel_agent_run(store, running.run_id, reason="operator-stop", now_ms=123)
        assert cancelled is not None
        release.set()
        with pytest.raises(AgentRunCancelled, match="operator-stop") as cancelled_error:
            await worker
        assert cancelled_error.value.run_id == running.run_id
        assert cancelled_error.value.reason == "operator-stop"

        persisted = await store.load(running.run_id)
        assert persisted is not None
        assert persisted.status == "cancelled"
        assert persisted.revision == 1
        assert not any(isinstance(event, AgentFinishEvent) for event in events)
        assert isinstance(events[-1], AgentErrorEvent)
        assert isinstance(events[-1].error, AgentRunCancelled)
    finally:
        release.set()
        if cleanup is not None:
            cleanup.cleanup()


@pytest.mark.asyncio
async def test_finish_event_delivery_failure_does_not_rewrite_completed_state() -> None:
    store = create_in_memory_agent_run_store()

    async def fail_finish(event: Any) -> None:
        if isinstance(event, AgentFinishEvent):
            raise RuntimeError("event sink unavailable")

    agent = Agent(name="assistant", model=EchoModel(), run_store=store)
    with pytest.raises(AgentEventDeliveryError) as delivery_error:
        await AgentRuntime().run(
            agent=agent,
            prompt="hello",
            idempotency_key="finish-event-failure",
            emit=fail_finish,
        )

    assert delivery_error.value.event_type == "finish"
    assert delivery_error.value.durable_state_committed is True
    state = await store.find_by_idempotency_key("finish-event-failure")
    assert state is not None
    assert state.status == "completed"
    assert state.output_text == "echo:hello"


@pytest.mark.asyncio
async def test_run_start_event_delivery_failure_persists_failed_state() -> None:
    store = create_in_memory_agent_run_store()

    async def fail_start(event: Any) -> None:
        if isinstance(event, AgentRunStartEvent):
            raise RuntimeError("event sink unavailable")

    agent = Agent(name="assistant", model=EchoModel(), run_store=store)
    with pytest.raises(AgentEventDeliveryError) as delivery_error:
        await AgentRuntime().run(
            agent=agent,
            prompt="hello",
            idempotency_key="start-event-failure",
            emit=fail_start,
        )

    assert delivery_error.value.event_type == "run-start"
    assert delivery_error.value.durable_state_committed is False
    state = await store.find_by_idempotency_key("start-event-failure")
    assert state is not None
    assert state.status == "failed"
    assert state.finish_reason == "error"


@pytest.mark.asyncio
async def test_resume_reconciles_parent_when_resumed_child_is_cancelled() -> None:
    store = create_in_memory_agent_run_store()
    resume_started = asyncio.Event()
    release_resume = asyncio.Event()

    class BlockingResumeModel(ToolModel):
        async def generate(self, input: ModelGenerateInput) -> GenerateResult:
            if any(message.role == "tool" for message in input.messages):
                resume_started.set()
                await release_resume.wait()
            return await super().generate(input)

    async def suspend_policy(_request: ToolApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.require_human("review", approval_id="approval_lookup")

    agent = Agent(
        name="assistant",
        model=BlockingResumeModel(),
        run_store=store,
        approval_policy=suspend_policy,
        tools={
            "lookup": tool(
                name="lookup",
                schema=dict[str, str],
                execute=lambda input: {"item": input["item"]},
                requires_approval=True,
            )
        },
    )
    suspended = await run_agent(agent=agent, prompt="lookup")
    resume_task = asyncio.create_task(
        resume_agent_run(agent=agent, run_id=suspended.run_id, approval_id="approval_lookup")
    )
    await asyncio.wait_for(resume_started.wait(), timeout=1)
    children = await store.find_by_parent_run_id(suspended.run_id)
    assert len(children) == 1
    cancelled_child = await cancel_agent_run(store, children[0].run_id, reason="stop-child")
    assert cancelled_child is not None
    release_resume.set()

    with pytest.raises(AgentRunCancelled) as cancelled_error:
        await resume_task
    assert cancelled_error.value.run_id == children[0].run_id

    parent = await store.load(suspended.run_id)
    assert parent is not None
    assert parent.status == "failed"
    assert parent.metadata["resume_claim_failure"]["claim_token"]  # type: ignore[index]


@pytest.mark.asyncio
async def test_resume_preflights_reconciliation_capability_before_tool_execution() -> None:
    backing = create_in_memory_agent_run_store()
    executions = 0

    class MissingReconciliationStore:
        async def load(self, run_id: str) -> AgentRunState | None:
            return await backing.load(run_id)

        async def find_by_idempotency_key(self, key: str) -> AgentRunState | None:
            return await backing.find_by_idempotency_key(key)

        async def find_by_parent_run_id(self, parent_run_id: str) -> list[AgentRunState]:
            return await backing.find_by_parent_run_id(parent_run_id)

        async def save(self, state: AgentRunState) -> AgentRunState:
            return await backing.save(state)

        async def claim_pending_approval(self, *args: Any, **kwargs: Any) -> AgentRunState | None:
            return await backing.claim_pending_approval(*args, **kwargs)

    async def suspend_policy(_request: ToolApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.require_human("review", approval_id="approval_lookup")

    def execute(input: dict[str, str]) -> dict[str, str]:
        nonlocal executions
        executions += 1
        return input

    store = MissingReconciliationStore()
    agent = Agent(
        name="assistant",
        model=ToolModel(),
        run_store=store,  # type: ignore[arg-type]
        approval_policy=suspend_policy,
        tools={"lookup": tool(name="lookup", schema=dict[str, str], execute=execute, requires_approval=True)},
    )
    suspended = await run_agent(agent=agent, prompt="lookup")

    with pytest.raises(ValidationError, match="resume-claim reconciliation"):
        await resume_agent_run(agent=agent, run_id=suspended.run_id, approval_id="approval_lookup")

    assert executions == 0
    parent = await backing.load(suspended.run_id)
    assert parent is not None
    assert parent.status == "suspended"


@pytest.mark.asyncio
async def test_resume_rejects_an_invalid_claim_before_tool_execution() -> None:
    backing = create_in_memory_agent_run_store()
    executions = 0

    class InvalidClaimStore:
        async def load(self, run_id: str) -> AgentRunState | None:
            return await backing.load(run_id)

        async def find_by_idempotency_key(self, key: str) -> AgentRunState | None:
            return await backing.find_by_idempotency_key(key)

        async def find_by_parent_run_id(self, parent_run_id: str) -> list[AgentRunState]:
            return await backing.find_by_parent_run_id(parent_run_id)

        async def save(self, state: AgentRunState) -> AgentRunState:
            return await backing.save(state)

        async def claim_pending_approval(self, run_id: str, *args: Any, **kwargs: Any) -> AgentRunState | None:
            return await backing.load(run_id)

        async def fail_resume_claim(self, *args: Any, **kwargs: Any) -> AgentRunState | None:
            return await backing.fail_resume_claim(*args, **kwargs)

    async def suspend_policy(_request: ToolApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.require_human("review", approval_id="approval_lookup")

    def execute(input: dict[str, str]) -> dict[str, str]:
        nonlocal executions
        executions += 1
        return input

    store = InvalidClaimStore()
    agent = Agent(
        name="assistant",
        model=ToolModel(),
        run_store=store,  # type: ignore[arg-type]
        approval_policy=suspend_policy,
        tools={"lookup": tool(name="lookup", schema=dict[str, str], execute=execute, requires_approval=True)},
    )
    suspended = await run_agent(agent=agent, prompt="lookup")

    with pytest.raises(ValidationError, match="invalid claim"):
        await resume_agent_run(agent=agent, run_id=suspended.run_id, approval_id="approval_lookup")

    assert executions == 0
    parent = await backing.load(suspended.run_id)
    assert parent is not None
    assert parent.status == "suspended"


def test_agent_run_state_schema_rejects_future_versions() -> None:
    payload = {
        "schema_version": 999,
        "revision": 0,
        "run_id": "future",
        "agent_name": "assistant",
        "provider": "mock",
        "model_id": "model",
    }

    with pytest.raises(ValidationError, match="unsupported future schema_version"):
        deserialize_agent_run_state(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "factory"), _run_store_factories(), ids=[name for name, _factory in _run_store_factories()])
async def test_stale_resume_claim_can_only_be_failed_by_its_claim_token(
    name: str,
    factory: Callable[[], tuple[AgentRunStore, Any]],
) -> None:
    del name
    store, cleanup = factory()
    try:
        state = AgentRunState(
            run_id="approval-run",
            agent_name="assistant",
            provider="mock",
            model_id="model",
            status="suspended",
            pending_approvals=[PendingApproval(id="approval", name="write")],
        )
        await store.save(state)
        claimed = await store.claim_pending_approval(
            "approval-run",
            "approval",
            claim_token="worker-claim",
            claimed_at_ms=100,
        )
        assert claimed is not None
        assert claimed.status == "running"
        assert claimed.revision == 1

        wrong_claim = await fail_agent_run_resume_claim(
            store,
            "approval-run",
            claim_token="other-worker",
            reason="lease expired",
            now_ms=200,
        )
        assert wrong_claim is None

        reconciled = await fail_agent_run_resume_claim(
            store,
            "approval-run",
            claim_token="worker-claim",
            reason="resume worker lease expired; outcome requires operator review",
            now_ms=200,
        )
        assert reconciled is not None
        assert reconciled.status == "failed"
        assert reconciled.revision == 2
        assert reconciled.pending_approvals[0].id == "approval"
        assert reconciled.metadata["resume_claim_failure"]["claim_token"] == "worker-claim"  # type: ignore[index]

        repeated = await fail_agent_run_resume_claim(
            store,
            "approval-run",
            claim_token="worker-claim",
            reason="retry",
            now_ms=300,
        )
        assert repeated is None
    finally:
        if cleanup is not None:
            cleanup.cleanup()


@pytest.mark.asyncio
async def test_cancel_and_resume_reconciliation_assign_timestamps_by_default() -> None:
    store = create_in_memory_agent_run_store()
    await store.save(
        AgentRunState(
            run_id="untimed-cancel",
            agent_name="assistant",
            provider="mock",
            model_id="model",
        )
    )
    cancelled = await cancel_agent_run(store, "untimed-cancel")
    assert cancelled is not None
    assert isinstance(cancelled.updated_at_ms, int) and cancelled.updated_at_ms > 0
    assert cancelled.finished_at_ms == cancelled.updated_at_ms

    await store.save(
        AgentRunState(
            run_id="untimed-resume",
            agent_name="assistant",
            provider="mock",
            model_id="model",
            status="suspended",
            pending_approvals=[PendingApproval(id="approval", name="write")],
        )
    )
    claimed = await store.claim_pending_approval(
        "untimed-resume",
        "approval",
        claim_token="lease-owner",
        claimed_at_ms=100,
    )
    assert claimed is not None
    failed = await fail_agent_run_resume_claim(
        store,
        "untimed-resume",
        claim_token="lease-owner",
        reason="lease expired",
    )
    assert failed is not None
    assert isinstance(failed.updated_at_ms, int) and failed.updated_at_ms > 0
    assert failed.finished_at_ms == failed.updated_at_ms


@pytest.mark.asyncio
async def test_stream_agent_reuses_idempotency_key_from_run_store() -> None:
    store = create_in_memory_agent_run_store()
    agent = Agent(name="assistant", model=EchoModel(), run_store=store)

    first_stream = stream_agent(agent=agent, prompt="hello", idempotency_key="stream-idem")
    first = await first_stream.collect()
    second_stream = stream_agent(agent=agent, prompt="ignored", idempotency_key="stream-idem")
    second = await second_stream.collect()

    assert first.run_id == second.run_id
    assert second.state is not None
    assert second.state.idempotency_key == "stream-idem"


@pytest.mark.asyncio
async def test_agent_cancellation_token_interrupts_model_and_persists_cancelled_state() -> None:
    started = asyncio.Event()
    provider_cancelled = asyncio.Event()

    class BlockingModel(EchoModel):
        async def generate(self, input: ModelGenerateInput) -> GenerateResult:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                provider_cancelled.set()
                raise
            return await super().generate(input)

    store = create_in_memory_agent_run_store()
    token = AgentCancellationToken()
    agent = Agent(name="assistant", model=BlockingModel(), run_store=store)
    running = asyncio.create_task(
        run_agent(
            agent=agent,
            prompt="wait",
            idempotency_key="cancel-token",
            cancellation_token=token,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    running_state = await store.find_by_idempotency_key("cancel-token")
    assert running_state is not None
    await cancel_agent_run_tree(
        store,
        running_state.run_id,
        reason="operator stop",
        cancellation_token=token,
    )

    with pytest.raises(AgentRunCancelled, match="operator stop"):
        await asyncio.wait_for(running, timeout=1)
    assert provider_cancelled.is_set()
    state = await store.find_by_idempotency_key("cancel-token")
    assert state is not None
    assert state.status == "cancelled"
    assert state.cancellation_reason == "operator stop"


@pytest.mark.asyncio
async def test_failed_runs_are_persisted_with_error_metadata() -> None:
    store = create_in_memory_agent_run_store()
    agent = Agent(
        name="assistant",
        model=HandoffModel(),
        run_store=store,
        tools={"delegate": tool(name="delegate", schema=dict[str, str], execute=lambda input: handoff_to("missing"))},
    )

    with pytest.raises(RuntimeError, match="Unknown handoff target"):
        await run_agent(agent=agent, prompt="delegate", idempotency_key="failed")

    state = await store.find_by_idempotency_key("failed")
    assert state is not None
    assert state.status == "failed"
    assert "Unknown handoff target" in (state.error or "")


@pytest.mark.asyncio
async def test_guardrail_tripwire_is_a_runtime_failure_contract() -> None:
    store = create_in_memory_agent_run_store()

    async def block_input(_request: Any) -> GuardrailResult:
        return GuardrailResult(tripwire_triggered=True, reason="blocked")

    agent = Agent(name="assistant", model=EchoModel(), run_store=store, input_guardrails=[block_input])

    with pytest.raises(Exception, match="blocked"):
        await run_agent(agent=agent, prompt="secret", idempotency_key="guardrail")

    state = await store.find_by_idempotency_key("guardrail")
    assert state is not None
    assert state.status == "failed"
    assert "blocked" in (state.error or "")


@pytest.mark.asyncio
async def test_tool_registry_context_and_permission_policy_contract() -> None:
    registry_tool = tool(
        name="lookup",
        schema=dict[str, str],
        execute=lambda input, context: {"tool": context.tool_name, "run": context.run_id, "item": input["item"]},
        permissions=["project:read"],
        requires_approval=True,
    )
    agent = Agent(
        name="assistant",
        model=ToolModel(),
        tools={"lookup": registry_tool},
        approval_policy=permission_allowlist_approval_policy("project:read"),
    )

    result = await run_agent(agent=agent, prompt="lookup")

    assert result.text == "tool done"
    assert result.tool_results[0].output["tool"] == "lookup"
    assert result.tool_results[0].output["run"] == result.run_id
