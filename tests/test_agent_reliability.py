from __future__ import annotations

import asyncio

import pytest

from zhivex_ai import (
    Agent,
    AgentEventDeliveryError,
    AgentRuntime,
    GuardrailTripwireTriggered,
    RedactionRule,
    SummaryConfig,
    apply_safety_policy_to_agent,
    create_agent_session,
    create_budget_guard,
    create_in_memory_agent_memory_store,
    create_in_memory_agent_run_store,
    create_in_memory_checkpoint_store,
    create_redaction_policy,
    create_safety_policy,
    create_text_message,
    handoff_to,
    replay_agent_run,
    run_agent,
    stream_agent,
    tool,
)
from zhivex_ai.types import (
    GenerateResult,
    ModelCapabilities,
    ModelGenerateInput,
    ModelMessage,
    StreamFinishEvent,
    StreamTextDeltaEvent,
    StreamToolCallEvent,
    TokenUsage,
    ToolCall,
    ToolCallPart,
)

CAPABILITIES = ModelCapabilities(
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


class ScriptedModel:
    provider = "test"
    model_id = "reliability"
    capabilities = CAPABILITIES

    def __init__(self, responses: list[GenerateResult]):
        self.responses = responses
        self.inputs: list[ModelGenerateInput] = []

    async def generate(self, request):
        self.inputs.append(request)
        return self.responses[min(len(self.inputs) - 1, len(self.responses) - 1)]

    async def stream(self, request):
        response = await self.generate(request)

        async def events():
            if response.text:
                yield StreamTextDeltaEvent(text_delta=response.text)
            for message in response.messages or []:
                for part in message.parts:
                    if isinstance(part, ToolCallPart):
                        yield StreamToolCallEvent(tool_call=part.tool_call)
            yield StreamFinishEvent(
                finish_reason=response.finish_reason, usage=response.usage
            )

        return events()


def answer(text="done", tokens=2):
    return GenerateResult(
        text=text,
        messages=[create_text_message("assistant", text)],
        finish_reason="stop",
        usage=TokenUsage(input_tokens=tokens, output_tokens=1),
    )


def calls(*names, tokens=2):
    return GenerateResult(
        messages=[
            ModelMessage(
                role="assistant",
                parts=[
                    ToolCallPart(
                        tool_call=ToolCall(
                            id=f"call-{i}", name=name, input={"prompt": "hello"}
                        )
                    )
                    for i, name in enumerate(names)
                ],
            )
        ],
        finish_reason="tool-calls",
        usage=TokenUsage(input_tokens=tokens, output_tokens=1),
    )


def protected(agent, **limits):
    return apply_safety_policy_to_agent(
        agent,
        create_safety_policy(
            redaction=False,
            approval=False,
            budget=create_budget_guard(**limits),
        ),
    )


@pytest.mark.parametrize("input_kind", ["prompt", "messages"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_redacted_input_is_canonical_for_all_stores_and_replay(
    input_kind, streaming
):
    canary = "fictional-canary"
    model = ScriptedModel([answer(canary)])
    memory = create_in_memory_agent_memory_store(
        summary_config=SummaryConfig(max_messages=1, preserve_recent_messages=1)
    )
    checkpoints = create_in_memory_checkpoint_store()
    runs = create_in_memory_agent_run_store()
    instructions = f"Internal instructions {canary}"
    session = create_agent_session(
        messages=[
            create_text_message("system", f"Caller system {canary}"),
            create_text_message("user", f"Prior history {canary}"),
        ],
        summary=f"Prior summary {canary}",
    )
    agent = apply_safety_policy_to_agent(
        Agent(
            name="test",
            model=model,
            instructions=instructions,
            memory=memory,
            run_store=runs,
            checkpoint_store=checkpoints,
        ),
        create_safety_policy(
            approval=False,
            budget=False,
            redaction=create_redaction_policy(
                rules=[RedactionRule(canary)],
            ),
        ),
    )
    kwargs = (
        {"prompt": f"New input {canary}"}
        if input_kind == "prompt"
        else {
            "messages": [
                create_text_message("system", f"New caller system {canary}"),
                create_text_message("user", canary),
            ]
        }
    )
    result = await (
        stream_agent(agent=agent, session=session, **kwargs).collect()
        if streaming
        else run_agent(agent=agent, session=session, **kwargs)
    )
    stored = await runs.load(result.run_id)
    assert stored is not None
    for surface in [
        model.inputs,
        result,
        await memory.load(session.id),
        await checkpoints.list(run_id=result.run_id),
        stored,
        replay_agent_run(stored),
    ]:
        assert canary not in repr(surface)
    assert all(
        "Internal instructions" not in repr(message) for message in session.messages
    )
    assert "Caller system" in repr(session.messages)
    if input_kind == "messages":
        assert "New caller system" in repr(session.messages)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "limit", ["max_input_tokens", "max_output_tokens", "max_total_tokens"]
)
async def test_token_budget_stops_before_tools_and_next_model(streaming, limit):
    executed = []
    model = ScriptedModel([calls("work", tokens=4), answer()])
    agent = protected(
        Agent(
            name="test",
            model=model,
            tools={
                "work": tool(
                    name="work",
                    schema={"type": "object"},
                    execute=lambda _: executed.append("work"),
                )
            },
        ),
        **{limit: 0 if limit == "max_output_tokens" else 3},
    )
    with pytest.raises(GuardrailTripwireTriggered):
        await (
            stream_agent(agent=agent, prompt="test").collect()
            if streaming
            else run_agent(agent=agent, prompt="test")
        )
    assert len(model.inputs) <= 1
    assert executed == []


async def test_tool_error_budget_stops_sequential_batch_before_next_side_effect():
    executed = []
    model = ScriptedModel([calls("fail", "work"), answer()])

    async def fail(_):
        raise ValueError("fictional failure")

    agent = protected(
        Agent(
            name="test",
            model=model,
            tools={
                "fail": tool(name="fail", schema={"type": "object"}, execute=fail),
                "work": tool(
                    name="work",
                    schema={"type": "object"},
                    execute=lambda _: executed.append("work"),
                ),
            },
        ),
        max_tool_errors=0,
    )
    with pytest.raises(GuardrailTripwireTriggered, match="tool errors"):
        await run_agent(agent=agent, prompt="test")
    assert executed == []
    assert len(model.inputs) == 1


@pytest.mark.parametrize("include_children", [True, False])
async def test_child_usage_counts_once_and_stops_parent_dispatch(include_children):
    child_model = ScriptedModel([answer(tokens=4)])
    parent_model = ScriptedModel([calls("child"), answer()])
    agent = protected(
        Agent(
            name="parent",
            model=parent_model,
            subagents={
                "child": Agent(name="child", model=child_model),
            },
        ),
        max_total_tokens=6,
        include_child_runs=include_children,
    )
    if include_children:
        with pytest.raises(GuardrailTripwireTriggered, match="total tokens"):
            await run_agent(agent=agent, prompt="test")
        assert len(parent_model.inputs) == 1
    else:
        result = await run_agent(agent=agent, prompt="test")
        assert len(parent_model.inputs) == 2
        assert result.state.child_runs[0].usage.input_tokens == 4
    assert len(child_model.inputs) == 1


async def test_handoff_budget_accumulates_across_agents():
    target_model = ScriptedModel([answer(tokens=4)])
    parent_model = ScriptedModel([calls("transfer")])
    agent = protected(
        Agent(
            name="parent",
            model=parent_model,
            subagents={
                "target": Agent(name="target", model=target_model),
            },
            tools={
                "transfer": tool(
                    name="transfer",
                    schema={"type": "object"},
                    execute=lambda _: handoff_to("target"),
                )
            },
        ),
        max_total_tokens=6,
    )
    with pytest.raises(GuardrailTripwireTriggered, match="total tokens"):
        await run_agent(agent=agent, prompt="test", max_steps=1)
    assert len(parent_model.inputs) == 1
    assert len(target_model.inputs) == 1


class CloseableIterator:
    def __init__(self, *, count=1):
        self.closed = False
        self.count = count
        self.started = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.started.set()
        if self.count:
            self.count -= 1
            return StreamTextDeltaEvent(text_delta="x")
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


class CloseableModel:
    provider = "test"
    model_id = "closeable"
    capabilities = CAPABILITIES

    def __init__(self):
        self.iterator = CloseableIterator()

    async def stream(self, request):
        return self.iterator


@pytest.mark.parametrize("exit_mode", ["callback", "cancel", "timeout"])
async def test_agent_closes_provider_iterator_on_every_early_exit(exit_mode):
    model = CloseableModel()
    agent = Agent(name="test", model=model)
    if exit_mode == "callback":

        async def emit(event):
            if event.type == "text-delta":
                raise ValueError("fictional sink failure")

        with pytest.raises(AgentEventDeliveryError):
            await AgentRuntime().run(
                agent=agent, prompt="test", live_stream=True, emit=emit
            )
    elif exit_mode == "timeout":
        with pytest.raises(TimeoutError):
            await stream_agent(
                agent=agent, prompt="test", total_timeout_ms=20
            ).collect()
    else:
        result = stream_agent(agent=agent, prompt="test")
        await asyncio.wait_for(model.iterator.started.wait(), 1)
        await result.aclose()
    assert model.iterator.closed


async def test_agent_inner_replay_is_bounded_without_changing_public_replay(
    monkeypatch,
):
    import zhivex_ai.agent as agent_module
    from zhivex_ai import stream_text
    from zhivex_ai.types import StreamFinishEvent

    inner = []
    original = agent_module.stream_text

    def capture(**kwargs):
        result = original(**kwargs)
        inner.append(result)
        return result

    monkeypatch.setattr(agent_module, "stream_text", capture)

    class ManyEventsModel(ScriptedModel):
        async def stream(self, request):
            async def events():
                for _ in range(5000):
                    yield StreamTextDeltaEvent(text_delta="x")
                yield StreamFinishEvent(finish_reason="stop")

            return events()

    model = ManyEventsModel([])
    results = await asyncio.gather(
        *(
            stream_agent(
                agent=Agent(name="test", model=model, trace_event_limit=8),
                prompt="test",
                stream_buffer_size=8,
            ).collect()
            for _ in range(3)
        )
    )
    assert all(result.text == "x" * 5000 for result in results)
    assert all(len(result._broadcast.history) <= 1 for result in inner)
    public = stream_text(model=model, prompt="test")
    await public.collect()
    assert len(public._broadcast.history) == 5001


@pytest.mark.parametrize("streaming", [False, True])
async def test_multi_step_redaction_preserves_generated_outputs_and_request_snapshots(
    streaming,
):
    canary = "fictional-loop-canary"
    first = calls("work")
    first.messages[0].parts.insert(0, create_text_message("assistant", canary).parts[0])
    first.text = canary
    model = ScriptedModel([first, answer(canary)])
    memory = create_in_memory_agent_memory_store()
    checkpoints = create_in_memory_checkpoint_store()
    agent = apply_safety_policy_to_agent(
        Agent(
            name="test",
            model=model,
            memory=memory,
            checkpoint_store=checkpoints,
            tools={
                "work": tool(
                    name="work", schema={"type": "object"}, execute=lambda _: "ok"
                )
            },
        ),
        create_safety_policy(
            approval=False,
            budget=False,
            redaction=create_redaction_policy(rules=[RedactionRule(canary)]),
        ),
    )
    session = create_agent_session(
        messages=[create_text_message("assistant", "Prior answer")]
    )
    result = await (
        stream_agent(agent=agent, session=session, prompt=canary).collect()
        if streaming
        else run_agent(agent=agent, session=session, prompt=canary)
    )
    assert result.text == "[REDACTED]"
    assert all(step.response.text == "[REDACTED]" for step in result.steps)
    assert canary not in repr(result)
    assert canary not in repr(await checkpoints.list(run_id=result.run_id))
    assert canary not in repr(model.inputs[-1])
    assert "Prior answer" in repr(session.messages)


async def test_caller_system_message_equal_to_instructions_is_retained():
    session = create_agent_session(
        messages=[create_text_message("system", "Shared wording")]
    )
    result = await run_agent(
        agent=Agent(
            name="test", model=ScriptedModel([answer()]), instructions="Shared wording"
        ),
        session=session,
        prompt="test",
    )
    assert result.session.messages[0].role == "system"
    assert (
        sum("Shared wording" in repr(message) for message in result.session.messages)
        == 1
    )


async def test_equal_token_budget_allows_final_answer_but_blocks_continuation():
    model = ScriptedModel([answer()])
    await run_agent(
        agent=protected(Agent(name="test", model=model), max_total_tokens=3),
        prompt="test",
    )
    tool_model = ScriptedModel([calls("work"), answer()])
    executed = []
    agent = protected(
        Agent(
            name="test",
            model=tool_model,
            tools={
                "work": tool(
                    name="work",
                    schema={"type": "object"},
                    execute=lambda _: executed.append("work"),
                )
            },
        ),
        max_total_tokens=3,
    )
    with pytest.raises(GuardrailTripwireTriggered):
        await run_agent(agent=agent, prompt="test")
    assert executed == []
    assert len(tool_model.inputs) == 1


@pytest.mark.parametrize("limit", ["max_steps", "max_tool_calls"])
async def test_child_work_obeys_shared_step_and_tool_ceilings(limit):
    executed = []
    child_model = ScriptedModel([calls("work"), answer()])
    parent_model = ScriptedModel([calls("child"), answer()])
    child = Agent(
        name="child",
        model=child_model,
        tools={
            "work": tool(
                name="work",
                schema={"type": "object"},
                execute=lambda _: executed.append("work"),
            )
        },
    )
    ceiling = 1
    agent = protected(
        Agent(name="parent", model=parent_model, subagents={"child": child}),
        **{limit: ceiling},
    )
    with pytest.raises(GuardrailTripwireTriggered):
        await run_agent(agent=agent, prompt="test")
    assert executed == []
    assert len(parent_model.inputs) == 1
    assert len(child_model.inputs) == (0 if limit == "max_steps" else 1)


async def test_parallel_tool_tripwire_cancels_and_joins_started_siblings():
    from zhivex_ai import ToolExecutionOptions

    started, stopped = asyncio.Event(), asyncio.Event()

    async def wait(_):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def fail(_):
        await started.wait()
        raise ValueError("fictional failure")

    agent = protected(
        Agent(
            name="test",
            model=ScriptedModel([calls("wait", "fail")]),
            tools={
                "wait": tool(name="wait", schema={"type": "object"}, execute=wait),
                "fail": tool(name="fail", schema={"type": "object"}, execute=fail),
            },
        ),
        max_tool_errors=0,
    )
    with pytest.raises(GuardrailTripwireTriggered, match="tool errors"):
        await asyncio.wait_for(
            run_agent(
                agent=agent,
                prompt="test",
                tool_execution=ToolExecutionOptions(parallel=True, max_concurrency=2),
            ),
            1,
        )
    assert stopped.is_set()


async def test_concurrent_runs_using_same_policy_keep_independent_ledgers():
    guard = create_budget_guard(max_total_tokens=3)
    agent = apply_safety_policy_to_agent(
        Agent(name="test", model=ScriptedModel([answer()])),
        create_safety_policy(redaction=False, approval=False, budget=guard),
    )
    results = await asyncio.gather(
        *(run_agent(agent=agent, prompt="test") for _ in range(4))
    )
    assert all(result.text == "done" for result in results)


@pytest.mark.parametrize("exit_mode", ["cancel", "timeout"])
async def test_event_delivery_wait_closes_provider_on_cancellation_and_timeout(
    exit_mode,
):
    model = CloseableModel()
    entered = asyncio.Event()

    async def emit(event):
        if event.type == "text-delta":
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(
        AgentRuntime().run(
            agent=Agent(name="test", model=model),
            prompt="test",
            live_stream=True,
            emit=emit,
            total_timeout_ms=100 if exit_mode == "timeout" else None,
        )
    )
    await asyncio.wait_for(entered.wait(), 1)
    if exit_mode == "cancel":
        task.cancel()
    with pytest.raises(
        asyncio.CancelledError if exit_mode == "cancel" else TimeoutError
    ):
        await task
    assert model.iterator.closed


async def test_sqlite_replay_idempotency_and_memory_resume_keep_redacted_input(tmp_path):
    from zhivex_ai import create_sqlite_agent_run_store, resume_agent

    canary = "fictional-durable-canary"
    model = ScriptedModel([answer(canary)])
    memory = create_in_memory_agent_memory_store()
    runs = create_sqlite_agent_run_store(str(tmp_path / "runs.db"))
    agent = apply_safety_policy_to_agent(
        Agent(name="test", model=model, memory=memory, run_store=runs),
        create_safety_policy(approval=False, budget=False, redaction=create_redaction_policy(rules=[RedactionRule(canary)])),
    )
    first = await run_agent(agent=agent, prompt=canary, idempotency_key="fictional-request")
    replayed = await run_agent(agent=agent, prompt=canary, idempotency_key="fictional-request")
    assert len(model.inputs) == 1
    assert replayed.run_id == first.run_id
    resumed = await resume_agent(agent=agent, session_id=first.session.id, prompt=canary)
    assert len(model.inputs) == 2
    assert canary not in repr(model.inputs)
    assert canary not in repr(replayed)
    assert canary not in repr(resumed)
    assert canary not in repr(await runs.load(resumed.run_id))
    assert all(canary.encode() not in path.read_bytes() for path in tmp_path.glob("runs.db*"))


async def test_failed_run_persists_redacted_prior_history():
    canary = "fictional-failure-canary"
    runs = create_in_memory_agent_run_store()
    session = create_agent_session(messages=[create_text_message("user", canary)])
    agent = apply_safety_policy_to_agent(
        Agent(name="test", model=ScriptedModel([answer()]), run_store=runs),
        create_safety_policy(approval=False, budget=create_budget_guard(max_total_tokens=1), redaction=create_redaction_policy(rules=[RedactionRule(canary)])),
    )
    identities = []

    async def emit(event):
        if event.type == "run-start":
            identities.append(event.run_id)

    with pytest.raises(GuardrailTripwireTriggered):
        await AgentRuntime().run(agent=agent, session=session, prompt=canary, emit=emit)
    state = await runs.load(identities[0])
    assert state.status == "failed"
    assert canary not in repr(state)
    assert canary not in repr(session)
