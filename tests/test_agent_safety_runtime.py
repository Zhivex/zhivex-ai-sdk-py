from __future__ import annotations

from collections.abc import AsyncIterable

import pytest

from zhivex_ai import (
    Agent,
    GuardrailTripwireTriggered,
    ToolExecutionOptions,
    apply_safety_policy_to_agent,
    create_budget_guard,
    create_in_memory_agent_memory_store,
    create_in_memory_checkpoint_store,
    create_safety_policy,
    create_text_message,
    run_agent,
)
from zhivex_ai.types import (
    GenerateResult,
    ModelCapabilities,
    ModelGenerateInput,
    StreamFinishEvent,
    TokenUsage,
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


class SensitiveModel:
    provider = "test"
    model_id = "sensitive"
    capabilities = CAPABILITIES

    def __init__(self) -> None:
        self.last_input: ModelGenerateInput | None = None

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        self.last_input = input
        return GenerateResult(
            messages=[create_text_message("assistant", "api_key=anothersecret123")],
            text="api_key=anothersecret123",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=2, output_tokens=2, total_tokens=4),
        )

    async def stream(self, input: ModelGenerateInput) -> AsyncIterable[object]:
        async def generator() -> AsyncIterable[object]:
            yield StreamFinishEvent(
                finish_reason="stop",
                usage=TokenUsage(input_tokens=2, output_tokens=2, total_tokens=4),
            )

        return generator()


@pytest.mark.asyncio
async def test_safety_redaction_changes_model_input_result_memory_and_checkpoint() -> None:
    model = SensitiveModel()
    memory = create_in_memory_agent_memory_store()
    checkpoints = create_in_memory_checkpoint_store()
    policy = create_safety_policy(approval=False, budget=False)
    agent = apply_safety_policy_to_agent(
        Agent(name="assistant", model=model, memory=memory, checkpoint_store=checkpoints),
        policy,
    )

    result = await run_agent(agent=agent, prompt="token=supersecret123")

    assert model.last_input is not None
    assert "supersecret123" not in repr(model.last_input.messages)
    assert result.text == "[REDACTED]"
    assert "anothersecret123" not in repr(result.session.messages)
    saved = await checkpoints.list(run_id=result.run_id)
    assert saved
    assert "anothersecret123" not in repr(saved)


@pytest.mark.asyncio
async def test_budget_guard_enforces_usage_and_policy_runtime_options() -> None:
    model = SensitiveModel()
    policy = create_safety_policy(
        approval=False,
        redaction=False,
        budget=create_budget_guard(max_steps=2, max_tool_calls=3, max_total_tokens=1),
        tool_execution=ToolExecutionOptions(parallel=False, max_concurrency=1),
    )
    agent = apply_safety_policy_to_agent(Agent(name="assistant", model=model), policy)

    assert agent.run_limits.max_steps == 2
    assert agent.run_limits.max_tool_calls == 3
    assert agent.tool_execution == policy.tool_execution
    with pytest.raises(GuardrailTripwireTriggered, match="total tokens"):
        await run_agent(agent=agent, prompt="hello")
