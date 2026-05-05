from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass

from zhivex_ai import (
    Agent,
    AgentRunState,
    ModelCapabilities,
    ProviderHTTPError,
    TokenUsage,
    apply_safety_policy_to_agent,
    create_budget_guard,
    create_circuit_breaker_middleware,
    create_redaction_policy,
    create_safety_policy,
    create_telemetry_middleware,
    create_text_message,
    wrap_language_model,
)
from zhivex_ai.types import GenerateResult, ModelGenerateInput


class OfflineOperationsModel:
    provider = "offline"
    model_id = "operations-demo"
    capabilities = ModelCapabilities(
        streaming=False,
        tools=False,
        structured_output=False,
        json_mode=False,
        tool_choice=False,
        parallel_tool_calls=False,
        vision=False,
        files=False,
        audio_input=False,
        audio_output=False,
        embeddings=False,
        reasoning=False,
        web_search=False,
    )

    def __init__(self) -> None:
        self.fail_next = True

    async def generate(self, input: ModelGenerateInput) -> GenerateResult:
        if self.fail_next:
            self.fail_next = False
            raise ProviderHTTPError("upstream unavailable", 503, retryable=True, retry_after_ms=250)
        return GenerateResult(
            message=create_text_message("assistant", "request correlated and guarded"),
            text="request correlated and guarded",
            usage=TokenUsage(input_tokens=8, output_tokens=4, total_tokens=12),
        )

    async def stream(self, input: ModelGenerateInput):
        raise RuntimeError("streaming is not used by this offline example")


@dataclass(slots=True)
class OperationsHardeningSummary:
    request_id: str
    session_id: str
    run_id: str
    redacted_prompt: str
    budget_blocked: bool
    retryable_error: bool
    retry_after_ms: int | None
    telemetry_events: list[str]
    circuit_transitions: list[str]
    response_text: str
    agent_safety_policy: str


async def run_operations_hardening_demo() -> OperationsHardeningSummary:
    request_id = "req_offline_001"
    session_id = "sess_offline_001"
    run_id = "run_offline_001"
    telemetry_events: list[str] = []
    circuit_transitions: list[str] = []
    retryable_error = False
    retry_after_ms: int | None = None

    redaction = create_redaction_policy(include_emails=True)
    budget = create_budget_guard(max_total_tokens=10)
    safety = create_safety_policy(redaction=redaction, budget=budget)
    agent = apply_safety_policy_to_agent(
        Agent(
            name="operations-assistant",
            model=OfflineOperationsModel(),
            metadata={"request_id": request_id, "session_id": session_id, "run_id": run_id},
        ),
        safety,
    )

    redacted_prompt = redaction.redact_text("token=abcdefghij contact user@example.com")
    budget_result = budget.evaluate_state(
        AgentRunState(
            run_id=run_id,
            agent_name=agent.name,
            provider="offline",
            model_id="operations-demo",
            usage=TokenUsage(input_tokens=6, output_tokens=7, total_tokens=13),
        )
    )

    model = wrap_language_model(
        OfflineOperationsModel(),
        [
            create_telemetry_middleware(on_event=lambda event: telemetry_events.append(str(event["type"]))),
            create_circuit_breaker_middleware(
                failure_threshold=1,
                cooldown_ms=0,
                on_state_change=lambda event: circuit_transitions.append(str(event["status"])),
            ),
        ],
    )

    try:
        await model.generate(ModelGenerateInput(messages=[create_text_message("user", "hello")]))
    except ProviderHTTPError as error:
        retryable_error = error.retryable
        retry_after_ms = error.retry_after_ms

    result = await model.generate(ModelGenerateInput(messages=[create_text_message("user", "hello")]))

    return OperationsHardeningSummary(
        request_id=request_id,
        session_id=session_id,
        run_id=run_id,
        redacted_prompt=redacted_prompt,
        budget_blocked=budget_result.tripwire_triggered,
        retryable_error=retryable_error,
        retry_after_ms=retry_after_ms,
        telemetry_events=telemetry_events,
        circuit_transitions=circuit_transitions,
        response_text=result.text or "",
        agent_safety_policy=str(agent.metadata["safety_policy"]),
    )


async def main() -> None:
    summary = await run_operations_hardening_demo()
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
