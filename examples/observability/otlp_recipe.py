"""Opt-in OTLP recipe: synthetic generation, gateway fallback, agent run/resume."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from zhivex_ai import (
    Agent,
    ApprovalDecision,
    GatewayConfig,
    GatewayMessage,
    GatewayModelTarget,
    ModelMessage,
    OTelAgentObserver,
    ToolCall,
    create_gateway,
    create_in_memory_agent_run_store,
    generate_text,
    resume_agent_run,
    run_agent,
    tool,
    tool_call_part,
)
from zhivex_ai.evals import GenerateResult, create_mock_language_model


async def exercise(tracer: Any, meter: Any) -> str:
    counter = meter.create_counter("zhivex.operations")

    # Never use run/session/request/model IDs as metric labels.
    def count(operation: str, ok: bool) -> None:
        counter.add(1, {"operation": operation, "outcome": "ok" if ok else "error"})

    with tracer.start_as_current_span(
        "application.request", attributes={"request.id": "synthetic-request"}
    ) as root:
        with tracer.start_as_current_span("zhivex.generation"):
            await generate_text(
                model=create_mock_language_model(), prompt="synthetic", max_retries=0
            )
            count("generation", True)

        def attempt(event: dict[str, Any]) -> None:
            with tracer.start_as_current_span("zhivex.gateway.attempt") as span:
                span.set_attribute("gateway.attempt.id", event["attemptId"])
                span.set_attribute("gateway.ok", event["ok"])
                span.set_attribute("gateway.retry", event["retry"])
                # No exception message, payload or arbitrary model metadata.
                count("gateway", event["ok"])

        class Adapter:
            def language_model(self, model_id: str) -> Any:
                return create_mock_language_model()

        gateway = create_gateway(
            GatewayConfig(
                adapters={"openai": Adapter()}, max_retries=0, on_attempt=attempt
            )
        )
        await gateway.generate(
            messages=[GatewayMessage(role="user", content="synthetic")],
            primary=GatewayModelTarget(provider="anthropic", model_id="missing"),
            fallbacks=[GatewayModelTarget(provider="openai", model_id="fixture")],
        )

        async def review(_request: Any) -> ApprovalDecision:
            return ApprovalDecision.require_human(approval_id="synthetic-approval")

        agent = Agent(
            name="synthetic-agent",
            run_store=create_in_memory_agent_run_store(),
            approval_policy=review,
            tools={
                "lookup": tool(
                    name="lookup",
                    schema=dict[str, str],
                    execute=lambda data: data,
                    requires_approval=True,
                )
            },
            model=create_mock_language_model(
                responses=[
                    GenerateResult(
                        messages=[
                            ModelMessage(
                                role="assistant",
                                parts=[
                                    tool_call_part(
                                        ToolCall(
                                            id="synthetic-call",
                                            name="lookup",
                                            input={"item": "synthetic"},
                                        )
                                    )
                                ],
                            )
                        ],
                        finish_reason="tool-calls",
                    ),
                    GenerateResult(text="done", finish_reason="stop"),
                ]
            ),
        )
        observer = OTelAgentObserver(tracer)
        pending = await run_agent(agent=agent, prompt="synthetic", observer=observer)
        await resume_agent_run(
            agent=agent,
            run_id=pending.run_id,
            approval_id="synthetic-approval",
            observer=observer,
        )
        count("agent", True)
        return format(root.get_span_context().trace_id, "032x")


def main() -> None:
    # Export is only configured when this script is explicitly executed.
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(
        resource=Resource.create({"service.name": "zhivex-recipe"}),
        sampler=ParentBased(TraceIdRatioBased(1.0)),
    )
    provider.add_span_processor(
        SimpleSpanProcessor(
            OTLPSpanExporter(
                endpoint=os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"], timeout=5
            )
        )
    )
    metrics = MeterProvider()
    try:
        trace_id = asyncio.run(
            exercise(provider.get_tracer("recipe"), metrics.get_meter("recipe"))
        )
        if not provider.force_flush():
            raise RuntimeError("trace flush failed")
        print(json.dumps({"trace_id": trace_id}))
    finally:
        provider.shutdown()
        metrics.shutdown()


if __name__ == "__main__":
    main()
