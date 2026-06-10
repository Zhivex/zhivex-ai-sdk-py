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

The result includes `text`, `provider_used`, `model_used`, normalized finish reasons, attempt metadata, and latency fields.

For production routes where every configured provider should be present, enable fail-fast adapter validation:

```python
gateway = create_gateway(
    GatewayConfig(
        adapters={"openai": openai, "anthropic": anthropic},
        fail_on_missing_adapter=True,
    )
)
```

With this setting, a missing adapter raises `GatewayError` instead of silently moving to a fallback target.

Gateway routing treats a generated refusal as a fallback signal by default. When a provider returns `finish_reason="refusal"` or `provider_finish_reason="refusal"`, the attempt is recorded and the gateway moves to the next configured fallback. Set `GatewayConfig(fallback_on_refusal=False)` to return refusals from the selected target without trying fallback models.

## Operations

Use `GatewayConfig(on_attempt=...)` to log every attempt. Include your request ID in the closure so gateway logs can be correlated with API logs.

`on_attempt` receives payloads for executed attempts and skipped targets. That includes missing adapters, capability skips, vision skips, cost-budget skips, retries, and successful attempts.

For production APIs:

- define one timeout budget per endpoint
- return normalized errors to clients
- log selected provider and model
- log skipped gateway targets, not just the final provider
- keep fallback order explicit and reviewed
- prefer tier-1 providers for long-term public contracts

See [../PRODUCTION_APIS.md](../PRODUCTION_APIS.md), `examples/integrations/fastapi_gateway_api.py`, and `examples/production/fastapi_agent_api.py` for API wiring.
