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

Gateway routing returns generated refusals by default. When a provider returns `finish_reason="refusal"` or `provider_finish_reason="refusal"`, the attempt is recorded with `ok=False`, `reason="provider_refusal"`, and `error_type="refusal"`, while the refusal response is returned from the selected target. Set `GatewayConfig(fallback_on_refusal=True)` to explicitly retry refusals on fallback models.

## Catalog-Driven Routing

`GatewayConfig.model_catalog` accepts a typed `ModelCatalog`. The caller's primary target always remains first. For fallback targets found in the catalog, the gateway scores `ModelCatalogEntry.recommended_for` for the requested `routing_mode` and `task_intent`; it does not inspect substrings such as `pro`, `flash`, or `lite`. Targets absent from the configured catalog keep the legacy name heuristic so existing configurations do not change silently.

```python
from zhivex_ai import GatewayConfig, ModelCatalogEntry, create_model_catalog
from zhivex_ai.catalog import ModelPricing

catalog = create_model_catalog(
    [
        ModelCatalogEntry(
            provider="openai",
            model_id="application-fast-model",
            aliases=["fast"],
            recommended_for=["chat", "speed"],
            pricing=ModelPricing(
                currency="USD",
                source_url="https://billing.example.com/models",
                input_per_1m_tokens=5,
                output_per_1m_tokens=20,
                effective_from="2026-08-01",
            ),
            availability="stable",
        ),
        ModelCatalogEntry(
            provider="anthropic",
            model_id="application-reasoning-model",
            recommended_for=["chat", "reasoning", "tools"],
            availability="stable",
        ),
    ]
)

config = GatewayConfig(adapters=adapters, model_catalog=catalog)
```

`recommended_for` influences ranking only. It never implies a runtime capability. For strict `required_capabilities`, set `ModelCatalogEntry.capabilities` to a reviewed `ModelCapabilities` value. A cataloged target is skipped with `reason="capability_mismatch"` when required metadata is absent or false, before any provider invocation. The adapter's runtime capability metadata is then checked as a second guard. For compatibility, uncataloged targets continue to use adapter capabilities.

Catalog aliases must be API-level aliases for the same provider route. A snapshot, previous version, preview, or separately billable model must have its own entry and lifecycle state. Entries marked `retired` are skipped with `reason="model_unavailable"`; entries whose `api_surface` is not `language` are skipped with `reason="unsupported_api_surface"`. Preview, limited, and deprecated fallbacks remain eligible but receive a ranking penalty and stay visible in route evidence.

Every result exposes the evidence used in `result.route_decision.target_evidence`, aligned with `ordered_targets`. Each entry records:

- requested and canonical model IDs
- `scoring_source` (`model_catalog` or `legacy_heuristic`) and numeric score
- catalog recommendations, known capabilities, and availability
- normalized cost and its source (`model_override`, `model_catalog`, `provider_default`, or `unknown`)
- catalog price currency, source URL, and effective window when typed pricing supplied the rate

`route_decision.required_capabilities` records the true-valued requirements evaluated for that request. These fields explain routing policy; they do not claim live provider certification or calculate the final invoice.

Migration is additive: existing configurations may omit `model_catalog` and retain primary-first plus legacy fallback scoring. Adopt catalog routing by adding reviewed entries, then confirm `target_evidence` in tests before removing application-side heuristics. See [ADR 0001](./adr/0001-catalog-driven-gateway-routing.md) for the compatibility decision.

## Cost Budgets

`max_cost_per_1k_tokens` is a strict routing-rate ceiling. When it is set, the gateway resolves a non-negative finite rate for every target in this order:

1. `GatewayConfig.model_costs_per_1k_tokens[provider][model_id]`
2. a canonical-model override reached through the configured catalog's aliases
3. typed `ModelCatalogEntry.pricing`, converted conservatively from the highest known input/output rate per million tokens
4. legacy `ModelCatalogEntry.cost_per_1k_tokens` for application catalogs created before typed pricing
5. the deprecated provider-wide `GatewayConfig.provider_costs_per_1k_tokens` compatibility fallback

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

Application overrides remain one monetary rate per 1,000 total tokens. Typed `ModelPricing` preserves the provider's input/output rates per million tokens, currency, source, and effective window; the gateway compares the highest input/output rate divided by 1,000. An expired or not-yet-effective catalog price becomes unknown and therefore fails closed when a budget is set. This conservative routing rate is not an invoice estimator. Use one currency across the route, override regional or tiered prices in the application, and keep a review cadence: the SDK does not refresh prices from the Internet.

`provider_costs_per_1k_tokens` remains accepted so existing configurations keep working, but it is deprecated because one provider can expose models with materially different prices. Migrate each production route to `model_costs_per_1k_tokens` or a reviewed `ModelCatalog` before a future breaking release.

## Operations

Use `GatewayConfig(on_attempt=...)` to log every terminal attempt. Include your request ID in the closure so gateway logs can be correlated with API logs.

`on_attempt` emits exactly one payload after each executed retry or skipped target. It does not emit a pre-call/start payload. Every event includes `phase="finished"`, `terminal=True`, `attemptId="{targetRank}:{retry}"`, provider, model, `targetRank`, `retry`, `ok`, `latencyMs`, `retryable`, `reason`, `errorType`, and sanitized `errorMessage`. Combine the locally deterministic `attemptId` with the application request ID; it is not globally unique by itself.

Executed success events use `ok=True`; timeout, refusal, transport/provider error, and failed-policy events use `ok=False`. Executed attempts report provider duration with a minimum one-millisecond representation, while targets skipped before invocation use zero. Observer execution time is excluded. Synchronous and asynchronous observers are supported, and observer exceptions are ignored so telemetry cannot falsify a provider result or trigger a duplicate retry. If observer delivery itself needs monitoring, wrap the callback in application-owned handling.

The machine-readable `reason` is one of `missing_adapter`, `model_unavailable`, `unsupported_api_surface`, `vision_unsupported`, `capability_mismatch`, `cost_unknown`, `cost_exceeds_budget`, or `provider_refusal` for policy skips/refusals, and `None` for ordinary executions and transport/provider failures. `errorType` distinguishes `policy_skip`, `refusal`, `timeout`, `provider_http_error`, `transport_error`, `gateway_error`, and `provider_error`; it is `None` on success. Error text remains a sanitized human-readable companion and generic exception messages are not copied into telemetry. Do not parse error text as policy state.

Migration: older builds emitted a premature success-shaped callback with `latencyMs=0` before provider invocation and no terminal callback on success. That event is removed. Consumers should count the terminal payloads described above and use an application-level event before calling the gateway only when they separately need route-start telemetry.

For production APIs:

- define one timeout budget per endpoint
- return normalized errors to clients
- log selected provider and model
- log skipped gateway targets, not just the final provider
- keep fallback order explicit and reviewed
- prefer tier-1 providers for long-term public contracts

See [../PRODUCTION_APIS.md](../PRODUCTION_APIS.md), `examples/integrations/fastapi_gateway_api.py`, and `examples/production/fastapi_agent_api.py` for API wiring.
