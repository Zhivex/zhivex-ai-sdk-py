from __future__ import annotations

import asyncio
from uuid import uuid4

from zhivex_ai import (
    GatewayConfig,
    GatewayMessage,
    GatewayModelTarget,
    create_anthropic,
    create_gateway,
    create_openai,
    create_telemetry_middleware,
    generate_text,
    wrap_language_model,
)


async def main() -> None:
    request_id = f"req_{uuid4().hex[:8]}"
    openai = create_openai()

    model = wrap_language_model(
        openai("gpt-5.4-mini"),
        [
            create_telemetry_middleware(
                on_event=lambda event: print(
                    {
                        "requestId": request_id,
                        "type": event["type"],
                        "provider": event["model"].provider,
                        "modelId": event["model"].model_id,
                        "latencyMs": event.get("latencyMs"),
                    }
                )
            )
        ],
    )

    result = await generate_text(
        model=model,
        prompt="Explain why request IDs are useful in one sentence.",
        timeout_ms=30_000,
    )
    print({"requestId": request_id, "text": result.text})

    gateway = create_gateway(
        GatewayConfig(
            adapters={
                "openai": openai,
                "anthropic": create_anthropic(),
            },
            on_attempt=lambda payload: print({"requestId": request_id, "gateway": payload}),
        )
    )
    gateway_result = await gateway.generate(
        messages=[GatewayMessage(role="user", content="Say hello in one short sentence.")],
        primary=GatewayModelTarget(provider="openai", model_id="gpt-5.4-mini"),
        fallbacks=[GatewayModelTarget(provider="anthropic", model_id="claude-sonnet-4-20250514")],
    )
    print(
        {
            "requestId": request_id,
            "providerUsed": gateway_result.provider_used,
            "modelUsed": gateway_result.model_used,
            "latencyMs": gateway_result.latency_ms,
            "text": gateway_result.text,
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
