# Gateway Routing

The gateway is the SDK's fallback-routing layer. It lets application code define a primary model target and ordered fallbacks while keeping routing policy server-side.

## Recommended Use

Use `create_gateway(...)` when an API or worker needs:

- explicit primary/fallback targets
- retry and fallback observability
- a stable application response contract across providers
- provider/model metadata in the result

Keep routing decisions in your API or service layer. Do not let public clients choose arbitrary provider fallbacks unless your product explicitly supports that policy.

## Minimal Shape

```python
from zhivex_ai import GatewayConfig, GatewayMessage, GatewayModelTarget, create_gateway

gateway = create_gateway(GatewayConfig(adapters={"openai": openai, "anthropic": anthropic}))
result = await gateway.generate(
    messages=[GatewayMessage(role="user", content="Summarize this.")],
    primary=GatewayModelTarget(provider="openai", model_id="gpt-5.4-mini"),
    fallbacks=[GatewayModelTarget(provider="anthropic", model_id="claude-sonnet-4-20250514")],
)
```

The result includes `text`, `provider_used`, `model_used`, attempt metadata, and latency fields.

## Operations

Use `GatewayConfig(on_attempt=...)` to log every attempt. Include your request ID in the closure so gateway logs can be correlated with API logs.

For production APIs:

- define one timeout budget per endpoint
- return normalized errors to clients
- log selected provider and model
- keep fallback order explicit and reviewed
- prefer tier-1 providers for long-term public contracts

See [../PRODUCTION_APIS.md](../PRODUCTION_APIS.md) and `examples/integrations/fastapi_gateway_api.py` for API wiring.
