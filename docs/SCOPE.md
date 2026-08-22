# Product Scope

Zhivex AI SDK for Python is a provider-portable runtime for building reliable agents in Python. Its primary job is to keep application code stable across model providers while supplying the execution contracts that production agents repeatedly need: tools, streaming, handoffs, durable runs, approvals, replay, routing, and bounded observability.

This document separates that core product from extension areas that are useful but are not part of the SDK's central promise.

## Core Product

The core adoption path is:

1. Choose a provider through the portable model contract.
2. Define an `Agent` and application-owned tools.
3. Run or stream the agent.
4. Add durable state and approval resume when the application needs recovery.
5. Add gateway fallback and observability at the application boundary.

The core includes:

- portable text generation, streaming, structured output, grounding, and embeddings
- normalized messages, results, errors, and provider capability validation
- the agent runtime, tools, handoffs, typed dependencies, hooks, and middleware contracts
- durable agent run state, idempotency, pending approvals, resume, cancellation, snapshots, and replay
- thin provider adapters with explicit portable and provider-native namespaces
- gateway fallback and transport helpers used by backend applications

## Extension Areas

The repository also contains extension areas for teams that need them. They do not expand the core product promise and should be imported from their focused namespaces:

- `zhivex_ai.evals`: evaluation fixtures, experiments, metrics, gates, and artifacts
- `zhivex_ai.workflows`: declarative workflows, durable graphs, checkpoints, leases, and external-engine adapter contracts
- `zhivex_ai.integrations.protocols`: A2A, AG-UI, and Responses-compatible hosting adapters
- `zhivex_ai.experimental`: realtime/live-agent and non-portable provider experiments

The general CLI, local playground, packaged-skill registry, and provider-native resource clients are developer tools or provider integrations. They remain available, but they are not required to adopt the portable agent runtime.

Existing top-level imports remain available for compatibility. New Beta or Experimental APIs should be added to a focused namespace rather than expanding the `zhivex_ai` package root. Stability follows the individual symbol; moving a recommended import path does not promote an API.

## Application Responsibilities

The SDK orchestrates model and agent execution. Applications continue to own:

- authentication, authorization, tenants, and approval identity
- business rules and regulated decision policy
- durable domain storage and external side-effect idempotency
- public API compatibility, rate limits, DLP, and audit retention
- provider selection policy and release-specific live certification
- workflow-engine deployment when using Temporal, Prefect, DBOS, Restate, or another external runtime

## Provider Evidence

Provider support and live certification are separate claims:

- **Contract-supported** means the adapter participates in documented metadata, deterministic contract tests, provider tests, and smoke configuration.
- **Release-certified** means a recorded live run passed for an exact provider, model, operation set, built artifact, and source revision.
- **Experimental/native-only** means the provider or surface does not carry the portable compatibility promise.

A contract-supported provider is not automatically release-certified. Documentation and release notes must identify the exact evidence behind any live claim.

## Non-Goals For The Core

The core SDK is not intended to become:

- a general-purpose durable workflow engine
- a hosted agent platform or public control plane
- an approval UI or identity system
- a universal protocol server
- a remote-code package marketplace
- a replacement for provider-native SDKs across every resource lifecycle

Those capabilities may exist as integrations or incubating modules, but they must not make the core agent path harder to understand, install, certify, or maintain.
