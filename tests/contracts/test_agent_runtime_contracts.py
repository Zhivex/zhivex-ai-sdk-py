from __future__ import annotations

import sys
import tempfile
from collections.abc import AsyncIterable, Callable
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from zhivex_ai import (  # noqa: E402
    Agent,
    ApprovalDecision,
    AgentRunState,
    AgentRunStore,
    GuardrailResult,
    ToolApprovalRequest,
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
    get_pending_agent_approvals,
    handoff_to,
    permission_allowlist_approval_policy,
    replay_agent_run,
    resume_agent,
    resume_agent_run,
    run_agent,
    stream_agent,
    tool,
)
from zhivex_ai.types import (  # noqa: E402
    GenerateResult,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    ToolCall,
    ToolCallPart,
    TokenUsage,
)


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
