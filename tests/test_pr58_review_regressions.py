from __future__ import annotations

from dataclasses import replace

import pytest

from tests.test_agent_reliability import ScriptedModel, answer
from zhivex_ai import (
    Agent,
    RedactionRule,
    apply_safety_policy_to_agent,
    create_agent_session,
    create_in_memory_agent_memory_store,
    create_in_memory_agent_run_store,
    create_in_memory_checkpoint_store,
    create_redaction_policy,
    create_safety_policy,
    create_text_message,
    provider_data_part,
    run_agent,
    stream_agent,
)
from zhivex_ai.types import (
    OpenAIMcpApprovalRequest,
    StreamFinishEvent,
    StreamProviderDataEvent,
    StreamTextDeltaEvent,
)


class ApprovalContinuationModel(ScriptedModel):
    provider = "openai"

    async def stream(self, request):
        response = await self.generate(request)

        async def events():
            for message in response.messages or []:
                for part in message.parts:
                    if part.type == "text":
                        yield StreamTextDeltaEvent(text_delta=part.text)
                    elif part.type == "provider-data":
                        yield StreamProviderDataEvent(
                            provider=part.provider, data=part.data
                        )
            yield StreamFinishEvent(
                finish_reason=response.finish_reason, usage=response.usage
            )

        return events()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("approved", [False, True])
async def test_all_approval_continuations_are_guarded_without_rewriting_old_history(
    streaming, approved
):
    responses = []
    for index in range(3):
        response = answer(f"fictional-approval-canary-{index}")
        if index < 2:
            response.messages[0].parts.append(
                provider_data_part(
                    "openai",
                    OpenAIMcpApprovalRequest(
                        id=f"approval-{index}",
                        name="lookup",
                        arguments='{"query":"fictional"}',
                        server_label="fixture",
                    ),
                )
            )
        responses.append(response)
    model = ApprovalContinuationModel(responses)
    memory = create_in_memory_agent_memory_store()
    checkpoints = create_in_memory_checkpoint_store()
    runs = create_in_memory_agent_run_store()
    observed = []

    async def observe_output(request):
        observed.extend(request.messages)

    agent = apply_safety_policy_to_agent(
        Agent(
            name="test",
            model=model,
            memory=memory,
            checkpoint_store=checkpoints,
            run_store=runs,
            approval_policy=lambda _: approved,
            output_guardrails=[observe_output],
        ),
        create_safety_policy(
            approval=False,
            budget=False,
            redaction=create_redaction_policy(
                rules=[RedactionRule(r"fictional-approval-canary-\d")],
            ),
        ),
    )
    session = create_agent_session(
        messages=[create_text_message("assistant", "Previous answer unaffected")]
    )
    result = await (
        stream_agent(agent=agent, session=session, prompt="test").collect()
        if streaming
        else run_agent(agent=agent, session=session, prompt="test")
    )
    assert len(model.inputs) == 3
    assert len(observed) == 3
    assert "Previous answer" not in repr(observed)
    for surface in [
        result,
        await memory.load(session.id),
        await checkpoints.list(run_id=result.run_id),
        await runs.load(result.run_id),
    ]:
        assert "fictional-approval-canary" not in repr(surface)
    assert session.messages[0] == create_text_message(
        "assistant", "Previous answer unaffected"
    )
    assert result.text == "[REDACTED]"
    assert [step.response.text for step in result.steps] == ["[REDACTED]"] * 3
    approvals = [
        part.data
        for message in session.messages
        for part in message.parts
        if part.type == "provider-data"
        and getattr(part.data, "type", None) == "mcp_approval_response"
    ]
    assert [approval.approval_request_id for approval in approvals] == [
        "approval-0",
        "approval-1",
    ]
    assert all(approval.approve is approved for approval in approvals)


@pytest.mark.parametrize(
    "mutation",
    [
        "remove_all",
        "remove_one",
        "append",
        "replace",
        "reassign",
        "reassign_cloned_context",
    ],
)
@pytest.mark.parametrize("input_kind", ["prompt", "messages"])
async def test_guarded_caller_provenance_survives_cardinality_changes(
    mutation, input_kind
):
    history = [
        create_text_message("system", "Caller policy"),
        create_text_message("assistant", "Prior answer"),
    ]
    original = [create_text_message("user", "First input")]
    if input_kind == "messages":
        original.append(create_text_message("user", "Second input"))
    count = len(original)
    canonical = create_text_message("user", "Canonical input")
    appended = create_text_message("user", "Appended input")
    expected = []

    async def transform(request):
        if mutation == "remove_all":
            del request.messages[-count:]
        elif mutation == "remove_one":
            request.messages.pop()
            expected.extend(original[:-1])
        elif mutation == "append":
            request.messages.append(appended)
            expected.extend([*original, appended])
        elif mutation == "replace":
            request.messages[-count:] = [canonical]
            expected.append(canonical)
        else:
            prefix = request.messages[:-count]
            if mutation == "reassign_cloned_context":
                prefix = [
                    replace(message, parts=list(message.parts)) for message in prefix
                ]
            request.messages = [
                *prefix,
                canonical,
                appended,
                create_text_message("system", "Canonical caller system"),
            ]
            expected.extend(request.messages[-3:])

    model = ScriptedModel([answer()])
    session = create_agent_session(messages=history)
    memory = create_in_memory_agent_memory_store()
    runs = create_in_memory_agent_run_store()
    agent = apply_safety_policy_to_agent(
        Agent(
            name="test",
            model=model,
            instructions="Internal deployment instruction",
            input_guardrails=[transform],
            memory=memory,
            run_store=runs,
        ),
        create_safety_policy(approval=False, budget=False),
    )
    kwargs = (
        {"prompt": "First input"} if input_kind == "prompt" else {"messages": original}
    )
    result = await run_agent(agent=agent, session=session, **kwargs)
    assert result.session.messages == [
        *history,
        *expected,
        create_text_message("assistant", "done"),
    ]
    assert model.inputs[0].messages == [
        create_text_message("system", "Internal deployment instruction"),
        *history,
        *expected,
    ]
    assert "Internal deployment instruction" not in repr(result.session.messages)
    assert (await memory.load(session.id)).messages == result.session.messages
    assert "Internal deployment instruction" not in repr(
        (await runs.load(result.run_id)).metadata["session_messages"]
    )


@pytest.mark.parametrize(
    "mutation", ["reverse", "sort", "insert", "extend", "multiply", "clear_then_append"]
)
async def test_caller_ownership_survives_reordering_and_multiple_guardrails(mutation):
    original = [
        create_text_message("user", "First caller"),
        create_text_message("system", "Second caller"),
    ]
    history = [create_text_message("assistant", "Prior answer")]
    added = create_text_message("user", "fictional-added-canary")
    expected = []

    async def mutate(request):
        if mutation == "reverse":
            request.messages.reverse()
            expected.extend(reversed(original))
        elif mutation == "sort":
            request.messages.sort(
                key=lambda message: message.parts[0].text, reverse=True
            )
            expected.extend(
                sorted(
                    original, key=lambda message: message.parts[0].text, reverse=True
                )
            )
        elif mutation == "insert":
            request.messages.insert(-2, added)
            expected.extend([create_text_message("user", "[REDACTED]"), *original])
        elif mutation == "extend":
            request.messages += [added]
            expected.extend([*original, create_text_message("user", "[REDACTED]")])
        elif mutation == "multiply":
            request.messages *= 2
            expected.extend([*original, *original])
        else:
            request.messages.clear()
            request.messages.append(added)
            expected.append(create_text_message("user", "[REDACTED]"))

    async def second_guard(request):
        request.messages = list(request.messages)

    agent = apply_safety_policy_to_agent(
        Agent(
            name="test",
            model=ScriptedModel([answer()]),
            instructions="Internal instructions",
            input_guardrails=[mutate, second_guard],
        ),
        create_safety_policy(
            approval=False,
            budget=False,
            redaction=create_redaction_policy(
                rules=[RedactionRule("fictional-added-canary")]
            ),
        ),
    )
    result = await run_agent(
        agent=agent, session=create_agent_session(messages=history), messages=original
    )
    assert result.session.messages == [
        *history,
        *expected,
        create_text_message("assistant", "done"),
    ]
    assert "Internal instructions" not in repr(result.session.messages)
    assert "fictional-added-canary" not in repr(result)


async def test_ambiguous_wholesale_rewrite_fails_before_model_or_input_persistence():
    from zhivex_ai import ValidationError

    model = ScriptedModel([answer()])
    runs = create_in_memory_agent_run_store()
    session = create_agent_session(
        messages=[create_text_message("assistant", "History")]
    )

    async def replace_everything(request):
        request.messages = [
            create_text_message("system", "Changed internal context"),
            create_text_message("user", "Changed input"),
        ]

    agent = Agent(
        name="test",
        model=model,
        instructions="Original internal context",
        input_guardrails=[replace_everything],
        run_store=runs,
    )
    with pytest.raises(ValidationError, match="provenance"):
        await run_agent(agent=agent, session=session, prompt="Original input")
    assert model.inputs == []
    assert session.messages == [create_text_message("assistant", "History")]


async def test_guardrail_can_add_canonical_caller_input_to_a_history_only_run():
    model = ScriptedModel([answer()])
    history = [create_text_message("assistant", "History")]
    added = create_text_message("system", "New caller policy")

    async def add_input(request):
        request.messages.append(added)

    result = await run_agent(
        agent=Agent(
            name="test",
            model=model,
            instructions="Internal instructions",
            input_guardrails=[add_input],
        ),
        session=create_agent_session(messages=history),
    )
    assert result.session.messages == [
        *history,
        added,
        create_text_message("assistant", "done"),
    ]
    assert model.inputs[0].messages[-1] == added
