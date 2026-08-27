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
    primary=GatewayModelTarget(provider="openai", model_id="gpt-5.6-terra"),
    fallbacks=[GatewayModelTarget(provider="anthropic", model_id="claude-sonnet-5")],
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

Gateway routing returns generated refusals by default. When a provider returns `finish_reason="refusal"` or `provider_finish_reason="refusal"`, the attempt is recorded and the refusal is returned from the selected target. Set `GatewayConfig(fallback_on_refusal=True)` to explicitly retry refusals on fallback models.

## Cost Budgets

`max_cost_per_1k_tokens` is a strict routing-rate ceiling. When it is set, the gateway resolves a non-negative finite rate for every target in this order:

1. `GatewayConfig.model_costs_per_1k_tokens[provider][model_id]`
2. a canonical-model override reached through the configured catalog's aliases
3. `ModelCatalogEntry.cost_per_1k_tokens` from `GatewayConfig.model_catalog`
4. the deprecated provider-wide `GatewayConfig.provider_costs_per_1k_tokens` compatibility fallback

The model-specific rate always wins over the provider-wide default. If no verifiable rate exists, or the selected value is invalid, the target is skipped with `reason="cost_unknown"`; a known rate above the ceiling is skipped with `reason="cost_exceeds_budget"`. Equality is allowed. When no budget is passed, unknown-cost targets remain eligible for backward compatibility.

```python
gateway = create_gateway(
    GatewayConfig(
        adapters={"openai": openai, "anthropic": anthropic},
        # Illustrative application-owned rates; validate them for your account and effective date.
        model_costs_per_1k_tokens={
            "openai": {"gpt-5.6-luna": 0.25},
            "anthropic": {"claude-sonnet-5": 3.0},
        },
    )
)

result = await gateway.generate(
    messages=[GatewayMessage(role="user", content="Summarize this.")],
    primary=GatewayModelTarget(provider="openai", model_id="gpt-5.6-luna"),
    fallbacks=[GatewayModelTarget(provider="anthropic", model_id="claude-sonnet-5")],
    max_cost_per_1k_tokens=1.0,
)
```

The SDK treats each value as one application-owned monetary rate per 1,000 total tokens and only compares routing rates; it does not calculate the final invoice for a request. Use one currency across the route. When a provider publishes different input, output, cached-token, or tiered rates, supply a conservative blended rate that matches your policy. Keep pricing source, effective date, and review cadence application-owned. The SDK does not refresh prices from the Internet.

`provider_costs_per_1k_tokens` remains accepted so existing configurations keep working, but it is deprecated because one provider can expose models with materially different prices. Migrate each production route to `model_costs_per_1k_tokens` or a reviewed `ModelCatalog` before a future breaking release.

## Operations

Use `GatewayConfig(on_attempt=...)` to log every attempt. Include your request ID in the closure so gateway logs can be correlated with API logs.

`on_attempt` receives payloads for executed attempts and skipped targets. That includes missing adapters, capability skips, vision skips, cost-budget skips, refusals selected for fallback, retries, and successful attempts. The machine-readable `reason` is one of `missing_adapter`, `vision_unsupported`, `capability_mismatch`, `cost_unknown`, `cost_exceeds_budget`, or `provider_refusal` for policy skips, and `None` for ordinary executions and transport/provider failures. Error text remains a sanitized human-readable companion; do not parse it as policy state.

For production APIs:

- define one timeout budget per endpoint
- return normalized errors to clients
- log selected provider and model
- log skipped gateway targets, not just the final provider
- keep fallback order explicit and reviewed
- prefer tier-1 providers for long-term public contracts

See [../PRODUCTION_APIS.md](../PRODUCTION_APIS.md), `examples/integrations/fastapi_gateway_api.py`, and `examples/production/fastapi_agent_api.py` for API wiring.
