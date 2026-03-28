import asyncio

from zhivex_ai import GatewayConfig, GatewayMessage, GatewayModelTarget, create_anthropic, create_gateway, create_openai


async def main() -> None:
    gateway = create_gateway(
        GatewayConfig(
            adapters={
                "openai": create_openai(),
                "anthropic": create_anthropic(),
            }
        )
    )
    result = await gateway.generate(
        messages=[GatewayMessage(role="user", content="Say hello in one short sentence.")],
        primary=GatewayModelTarget(provider="openai", model_id="gpt-4o-mini"),
        fallbacks=[GatewayModelTarget(provider="anthropic", model_id="claude-3-5-sonnet")],
    )
    print(result.text)
    print(result.provider_used, result.model_used)


if __name__ == "__main__":
    asyncio.run(main())
