# ADR 0001: Catalog-Driven Gateway Routing

- Status: Accepted
- Date: 2026-08-29
- Scope: `GatewayConfig.model_catalog`, fallback scoring, capability policy, and route evidence

## Context

The Stable gateway preserves an explicit primary target and ranks fallbacks. Before this decision, fallback scoring inferred quality and speed from model-name substrings such as `pro`, `flash`, and `lite`. `GatewayConfig.model_catalog` influenced budget resolution but not the routing decision, so application-maintained capability and availability metadata could not act as the configured source of truth.

The model-catalog helpers remain Beta. The gateway configuration and response are Stable, so the change must be additive and must not silently reorder primary targets or remove the legacy behavior for callers that have not adopted a catalog.

## Decision

1. `GatewayConfig.model_catalog` is typed as `ModelCatalog | None`.
2. The primary target remains first in every routing mode.
3. Cataloged fallbacks are scored only from typed `recommended_for` metadata, reviewed cost, provider latency bias, routing mode, and task intent. Their model IDs are not parsed for routing hints.
4. Uncataloged fallbacks retain the legacy name heuristic as an explicit compatibility path.
5. `ModelCatalogEntry.capabilities` accepts typed `ModelCapabilities`. Recommendations influence ranking only and never imply capabilities; dedicated legacy fields remain compatibility metadata.
6. A cataloged target fails closed when a requested capability is missing or false. Adapter capabilities are checked afterward as a second runtime guard. Uncataloged targets retain adapter-based capability checks.
7. A configured cost ceiling continues to fail closed on missing or invalid price evidence.
8. Canonical IDs and aliases are unique per provider. Only an API-level alias may resolve to another ID; versions, snapshots, previews, and separately billed models remain distinct entries.
9. Retired and non-language catalog targets fail closed before provider invocation. Preview, limited, and deprecated fallbacks stay eligible with a ranking penalty.
10. `ModelPricing` preserves per-million input/output rates, currency, provenance, and effective dates; expired prices resolve as unknown. The legacy scalar remains compatible for application catalogs.
11. `GatewayRouteDecision.target_evidence` records the per-target metadata, score, lifecycle, and pricing provenance so applications can test and audit routing without parsing prose.

## Consequences

- Configuring a catalog can demonstrably reorder fallbacks; this is the intended opt-in behavior.
- Primary-first behavior and configurations without a catalog remain compatible.
- Catalog maintainers must treat missing capability metadata as a policy denial when applications request that capability.
- Catalog construction fails on canonical/alias collisions, entries are immutable, and lookups return defensive copies.
- Catalog evidence is routing metadata, not provider availability discovery, live certification, billing, or authorization.
- The legacy heuristic can be removed only through the Stable deprecation workflow after users have a documented migration window.

## Verification

- Misleading model names prove that cataloged routing ignores `pro`, `flash`, and `lite`.
- The same target list proves that adding `model_catalog` changes fallback selection.
- Missing required capability metadata proves fail-closed behavior without provider invocation.
- Retired/non-language targets and expired prices prove fail-closed behavior without provider invocation.
- Route-decision assertions cover canonical aliases, recommendations, capabilities, availability, score, cost, and evidence source.
