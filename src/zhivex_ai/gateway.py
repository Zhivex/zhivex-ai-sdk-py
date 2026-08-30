from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .catalog import ModelAvailability, ModelCatalog, ModelCatalogEntry, RecommendedUse
from .errors import ProviderHTTPError, ZhivexAIError, redact_provider_error_body
from .generate_object import generate_object, stream_object
from .generate_text import generate_text, stream_text
from .providers.base import resolve_provider_adapter
from .types import FinishReason, ModelMessage, TokenUsage

GatewayProviderId = Literal[
    "openai",
    "anthropic",
    "gemini",
    "vertex",
    "qwen",
    "kimi",
    "deepseek",
    "bedrock",
    "ollama",
    "azure-openai",
    "openrouter",
    "vllm",
    "meta",
]
GatewayRoutingMode = Literal["speed", "balanced", "quality"]
GatewayTaskIntent = Literal["chat", "reasoning", "tool-heavy"]
_GatewayAttemptReason = Literal[
    "missing_adapter",
    "model_unavailable",
    "unsupported_api_surface",
    "vision_unsupported",
    "capability_mismatch",
    "cost_unknown",
    "cost_exceeds_budget",
    "provider_refusal",
]
_GatewayCostSource = Literal[
    "model_override", "model_catalog", "provider_default", "unknown"
]
_GatewayRoutingEvidenceSource = Literal["model_catalog", "legacy_heuristic"]
_GatewayAttemptErrorType = Literal[
    "policy_skip",
    "refusal",
    "timeout",
    "provider_http_error",
    "transport_error",
    "gateway_error",
    "provider_error",
]


@dataclass(slots=True)
class GatewayImageAttachment:
    data_url: str
    mime_type: str


@dataclass(slots=True)
class GatewayMessage:
    role: Literal["user", "assistant"]
    content: str
    images: list[GatewayImageAttachment] = field(default_factory=list)


@dataclass(slots=True)
class GatewayModelTarget:
    provider: GatewayProviderId
    model_id: str


@dataclass(slots=True)
class GatewayAttempt:
    provider: GatewayProviderId
    model_id: str
    ok: bool
    latency_ms: int
    error_message: str | None = None
    retryable: bool = False
    reason: _GatewayAttemptReason | None = None
    error_type: _GatewayAttemptErrorType | None = None


@dataclass(slots=True)
class GatewayRouteTargetEvidence:
    provider: GatewayProviderId
    model_id: str
    canonical_model_id: str | None
    scoring_source: _GatewayRoutingEvidenceSource
    score: float
    recommended_for: tuple[RecommendedUse, ...] = ()
    capabilities: dict[str, bool] = field(default_factory=dict)
    availability: ModelAvailability | None = None
    cost_per_1k_tokens: float | None = None
    cost_source: _GatewayCostSource = "unknown"
    pricing_currency: str | None = None
    pricing_source_url: str | None = None
    pricing_effective_from: str | None = None
    pricing_effective_until: str | None = None


@dataclass(slots=True)
class GatewayRouteDecision:
    mode: GatewayRoutingMode
    intent: GatewayTaskIntent
    ordered_targets: list[GatewayModelTarget]
    reason: str
    target_evidence: list[GatewayRouteTargetEvidence] = field(default_factory=list)
    required_capabilities: tuple[str, ...] = ()


@dataclass(slots=True)
class GatewayResponse:
    text: str
    provider_used: GatewayProviderId
    model_used: str
    latency_ms: int
    attempts: list[GatewayAttempt]
    usage: TokenUsage
    usage_estimated: bool
    route_decision: GatewayRouteDecision
    finish_reason: FinishReason | None = None
    provider_finish_reason: str | None = None


@dataclass(slots=True)
class GatewayObjectResponse(GatewayResponse):
    object: Any = None
    object_mode: Literal["native", "prompted"] = "prompted"


@dataclass(slots=True)
class GatewayConfig:
    adapters: dict[GatewayProviderId, Any]
    model_catalog: ModelCatalog | None = None
    provider_costs_per_1k_tokens: dict[GatewayProviderId, float] = field(
        default_factory=dict
    )
    latency_bias_ms: dict[GatewayProviderId, int] = field(default_factory=dict)
    max_retries: int = 2
    attempt_timeout_ms: int = 20_000
    attempt_timeouts_ms: dict[GatewayProviderId, int] = field(default_factory=dict)
    retry_backoff_ms: int = 200
    fail_on_missing_adapter: bool = False
    fallback_on_refusal: bool = False
    on_attempt: Any = None
    model_costs_per_1k_tokens: dict[GatewayProviderId, dict[str, float]] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class _GatewayCostResolution:
    cost_per_1k_tokens: float | None
    source: _GatewayCostSource
    currency: str | None = None
    pricing_source_url: str | None = None
    effective_from: str | None = None
    effective_until: str | None = None


class GatewayError(ZhivexAIError):
    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def supports_vision_input(provider: GatewayProviderId, model_id: str) -> bool:
    model = model_id.lower()
    if provider == "gemini":
        return "embedding" not in model
    if provider == "bedrock":
        return "nova" in model or "claude-3" in model or "claude-4" in model
    if provider == "deepseek":
        return False
    return True


def _messages_require_vision(messages: list[GatewayMessage]) -> bool:
    return any(message.images for message in messages)


def _find_catalog_entry(
    config: GatewayConfig, target: GatewayModelTarget
) -> ModelCatalogEntry | None:
    if config.model_catalog is None:
        return None
    try:
        return config.model_catalog.find(target.provider, target.model_id)
    except Exception:
        return None


_MODEL_CAPABILITY_FIELDS = (
    "streaming",
    "tools",
    "structured_output",
    "json_mode",
    "tool_choice",
    "parallel_tool_calls",
    "vision",
    "files",
    "audio_input",
    "audio_output",
    "embeddings",
    "reasoning",
    "web_search",
    "realtime",
    "realtime_audio_input",
    "realtime_audio_output",
    "realtime_tools",
    "realtime_browser_tokens",
)
_CAPABILITY_ALIASES = {"structuredOutput": "structured_output", "jsonMode": "json_mode"}


def _catalog_capabilities(entry: ModelCatalogEntry) -> dict[str, bool]:
    capabilities: dict[str, bool] = {}
    if entry.capabilities is not None:
        capabilities.update(
            {
                name: bool(getattr(entry.capabilities, name))
                for name in _MODEL_CAPABILITY_FIELDS
            }
        )
    if entry.structured_output is not None:
        capabilities["structured_output"] = entry.structured_output
    if entry.parallel_tool_calls is not None:
        capabilities["parallel_tool_calls"] = entry.parallel_tool_calls
    return capabilities


def _target_supports_vision(
    adapter: Any, target: GatewayModelTarget, config: GatewayConfig
) -> bool:
    capabilities = (
        resolve_provider_adapter(adapter).language_model(target.model_id).capabilities
    )
    catalog_entry = _find_catalog_entry(config, target)
    if catalog_entry is not None:
        return (
            _catalog_capabilities(catalog_entry).get("vision") is True
            and capabilities.vision
        )
    return capabilities.vision and supports_vision_input(
        target.provider, target.model_id
    )


def gateway_messages_to_model_messages(
    messages: list[GatewayMessage], system_prompt: str | None = None
) -> list[ModelMessage]:
    mapped: list[ModelMessage] = []
    if system_prompt:
        from .messages import system

        mapped.append(system(system_prompt))
    from .types import ContentPart, ImagePart, TextPart

    for message in messages:
        parts: list[ContentPart] = [TextPart(text=message.content)]
        parts.extend(
            ImagePart(image=image.data_url, media_type=image.mime_type)
            for image in message.images
        )
        mapped.append(ModelMessage(role=message.role, parts=parts))
    return mapped


def create_route_decision(
    mode: GatewayRoutingMode,
    intent: GatewayTaskIntent,
    ordered_targets: list[GatewayModelTarget],
    *,
    config: GatewayConfig | None = None,
    required_capabilities: dict[str, bool] | None = None,
) -> GatewayRouteDecision:
    target_evidence = (
        [
            _target_routing_evidence(mode, intent, target, config)
            for target in ordered_targets
        ]
        if config is not None
        else []
    )
    cataloged = sum(item.scoring_source == "model_catalog" for item in target_evidence)
    legacy = len(target_evidence) - cataloged
    evidence_summary = (
        f" Catalog metadata used for {cataloged} target(s); legacy heuristics used for {legacy} uncataloged target(s)."
        if target_evidence
        else ""
    )
    return GatewayRouteDecision(
        mode=mode,
        intent=intent,
        ordered_targets=ordered_targets,
        reason=f"Primary target preserved first; fallbacks ordered by {mode} mode with {intent} intent.{evidence_summary}",
        target_evidence=target_evidence,
        required_capabilities=tuple(
            sorted(
                key
                for key, required in (required_capabilities or {}).items()
                if required is True
            )
        ),
    )


def _normalize_cost(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    cost = float(value)
    if not math.isfinite(cost) or cost < 0:
        return None
    return cost


def _resolve_target_cost(
    config: GatewayConfig, target: GatewayModelTarget
) -> _GatewayCostResolution:
    model_costs = config.model_costs_per_1k_tokens.get(target.provider, {})
    if target.model_id in model_costs:
        return _GatewayCostResolution(
            _normalize_cost(model_costs[target.model_id]), "model_override"
        )

    catalog_entry = _find_catalog_entry(config, target)

    if catalog_entry is not None:
        canonical_model_id = getattr(catalog_entry, "model_id", None)
        if isinstance(canonical_model_id, str) and canonical_model_id in model_costs:
            return _GatewayCostResolution(
                _normalize_cost(model_costs[canonical_model_id]), "model_override"
            )
        catalog_cost = getattr(catalog_entry, "cost_per_1k_tokens", None)
        if catalog_cost is not None:
            return _GatewayCostResolution(
                _normalize_cost(catalog_cost), "model_catalog"
            )
        pricing = getattr(catalog_entry, "pricing", None)
        if pricing is not None:
            routing_cost = _normalize_cost(pricing.conservative_cost_per_1k_tokens())
            if routing_cost is not None:
                return _GatewayCostResolution(
                    routing_cost,
                    "model_catalog",
                    currency=pricing.currency,
                    pricing_source_url=pricing.source_url,
                    effective_from=pricing.effective_from,
                    effective_until=pricing.effective_until,
                )

    if target.provider in config.provider_costs_per_1k_tokens:
        return _GatewayCostResolution(
            _normalize_cost(config.provider_costs_per_1k_tokens[target.provider]),
            "provider_default",
        )
    return _GatewayCostResolution(None, "unknown")


def _score_target(
    mode: GatewayRoutingMode,
    intent: GatewayTaskIntent,
    target: GatewayModelTarget,
    config: GatewayConfig,
) -> float:
    catalog_entry = _find_catalog_entry(config, target)
    local_boost = -2 if target.provider == "ollama" else 0
    if catalog_entry is not None:
        recommendations = set(catalog_entry.recommended_for)
        quality_boost = (
            2
            if "reasoning" in recommendations
            else (1 if "chat" in recommendations else 0)
        )
        speed_boost = 2 if "speed" in recommendations else 0
        reasoning_boost = 2 if "reasoning" in recommendations else 0
        tools_boost = 2 if "tools" in recommendations else 0
    else:
        model = target.model_id.lower()
        quality_boost = 2 if ("pro" in model or "claude" in model) else 0
        speed_boost = 2 if ("flash" in model or "lite" in model) else 0
        reasoning_boost = 2 if ("pro" in model or "claude" in model) else 0
        tools_boost = 0
    availability_penalty = (
        {
            "stable": 0.0,
            "preview": 0.5,
            "limited": 1.0,
            "deprecated": 3.0,
            "retired": 100.0,
        }.get(catalog_entry.availability, 0.0)
        if catalog_entry is not None
        else 0.0
    )
    cost_penalty = _resolve_target_cost(config, target).cost_per_1k_tokens or 0
    latency_penalty = config.latency_bias_ms.get(target.provider, 0) / 100
    intent_boost = (
        reasoning_boost
        if intent == "reasoning"
        else tools_boost
        if intent == "tool-heavy"
        else 0
    )
    if mode == "speed":
        return speed_boost + local_boost - latency_penalty - availability_penalty
    if mode == "quality":
        return quality_boost + intent_boost - cost_penalty - availability_penalty
    return (
        speed_boost
        + quality_boost
        + local_boost
        + intent_boost
        - cost_penalty
        - latency_penalty
        - availability_penalty
    )


def _target_routing_evidence(
    mode: GatewayRoutingMode,
    intent: GatewayTaskIntent,
    target: GatewayModelTarget,
    config: GatewayConfig,
) -> GatewayRouteTargetEvidence:
    catalog_entry = _find_catalog_entry(config, target)
    cost = _resolve_target_cost(config, target)
    return GatewayRouteTargetEvidence(
        provider=target.provider,
        model_id=target.model_id,
        canonical_model_id=catalog_entry.model_id
        if catalog_entry is not None
        else None,
        scoring_source="model_catalog"
        if catalog_entry is not None
        else "legacy_heuristic",
        score=_score_target(mode, intent, target, config),
        recommended_for=tuple(catalog_entry.recommended_for)
        if catalog_entry is not None
        else (),
        capabilities=_catalog_capabilities(catalog_entry)
        if catalog_entry is not None
        else {},
        availability=catalog_entry.availability if catalog_entry is not None else None,
        cost_per_1k_tokens=cost.cost_per_1k_tokens,
        cost_source=cost.source,
        pricing_currency=cost.currency,
        pricing_source_url=cost.pricing_source_url,
        pricing_effective_from=cost.effective_from,
        pricing_effective_until=cost.effective_until,
    )


def _order_targets(
    mode: GatewayRoutingMode,
    intent: GatewayTaskIntent,
    primary: GatewayModelTarget,
    fallbacks: list[GatewayModelTarget],
    config: GatewayConfig,
) -> list[GatewayModelTarget]:
    ordered: list[GatewayModelTarget] = [primary]
    seen: set[tuple[str, str]] = {(primary.provider, primary.model_id)}
    ranked_fallbacks: list[GatewayModelTarget] = []
    for target in fallbacks:
        key = (target.provider, target.model_id)
        if key in seen:
            continue
        seen.add(key)
        ranked_fallbacks.append(target)
    ranked_fallbacks.sort(
        key=lambda target: _score_target(mode, intent, target, config), reverse=True
    )
    ordered.extend(ranked_fallbacks)
    return ordered


def _supports_required_capabilities(
    adapter: Any,
    target: GatewayModelTarget,
    required: dict[str, bool] | None,
    config: GatewayConfig,
) -> bool:
    if not required:
        return True
    catalog_entry = _find_catalog_entry(config, target)
    catalog_capabilities = (
        _catalog_capabilities(catalog_entry) if catalog_entry is not None else None
    )
    required_names = [
        _CAPABILITY_ALIASES.get(key, key)
        for key, value in required.items()
        if value is True
    ]
    if catalog_capabilities is not None and any(
        catalog_capabilities.get(capability) is not True
        for capability in required_names
    ):
        return False

    adapter_capabilities = (
        resolve_provider_adapter(adapter).language_model(target.model_id).capabilities
    )
    for capability in required_names:
        if getattr(adapter_capabilities, capability, None) is not True:
            return False
    return True


def _cost_budget_reason(
    config: GatewayConfig,
    max_cost: float | None,
    target: GatewayModelTarget,
) -> Literal["cost_unknown", "cost_exceeds_budget"] | None:
    if max_cost is None:
        return None
    effective_cost = _resolve_target_cost(config, target).cost_per_1k_tokens
    if effective_cost is None:
        return "cost_unknown"
    if not effective_cost <= max_cost:
        return "cost_exceeds_budget"
    return None


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text.strip()) + 3) // 4)


def _normalize_usage(
    usage: TokenUsage | None, input_text: str, output_text: str
) -> tuple[TokenUsage, bool]:
    input_tokens = (
        usage.input_tokens
        if usage and usage.input_tokens is not None
        else _estimate_tokens(input_text)
    )
    output_tokens = (
        usage.output_tokens
        if usage and usage.output_tokens is not None
        else _estimate_tokens(output_text)
    )
    total_tokens = (
        usage.total_tokens
        if usage and usage.total_tokens is not None
        else input_tokens + output_tokens
    )
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    ), (
        usage is None
        or usage.input_tokens is None
        or usage.output_tokens is None
        or usage.total_tokens is None
    )


def _is_refusal_result(result: Any) -> bool:
    finish_reason = str(getattr(result, "finish_reason", "") or "").lower()
    provider_finish_reason = str(
        getattr(result, "provider_finish_reason", "") or ""
    ).lower()
    return finish_reason == "refusal" or provider_finish_reason == "refusal"


def _normalize_error(error: Exception) -> GatewayError:
    if isinstance(error, GatewayError):
        return error
    if isinstance(error, asyncio.TimeoutError):
        return GatewayError("Gateway attempt timed out.", True)
    if isinstance(error, ProviderHTTPError):
        return GatewayError(str(error), error.retryable)
    if isinstance(error, OSError):
        return GatewayError(str(error), True)
    message = str(error).lower()
    if any(
        token in message
        for token in ("timed out", "429", "rate", "connect", "econnrefused", "503")
    ):
        return GatewayError(str(error), True)
    return GatewayError(str(error), False)


def _final_gateway_error(attempts: list[GatewayAttempt]) -> GatewayError:
    if not attempts:
        return GatewayError("All gateway attempts failed.", False)
    retryable = False
    for attempt in reversed(attempts):
        if not attempt.ok and attempt.error_message:
            retryable = attempt.retryable
            break
    summary = "; ".join(
        f"{attempt.provider}/{attempt.model_id}: {attempt.error_message or 'unknown error'}"
        for attempt in attempts
        if not attempt.ok
    )
    return GatewayError(summary or "All gateway attempts failed.", retryable)


async def _maybe_await(value: Any) -> Any:
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


def _attempt_payload(
    attempt: GatewayAttempt, retry: int, target_rank: int
) -> dict[str, Any]:
    return {
        "attemptId": f"{target_rank}:{retry}",
        "phase": "finished",
        "terminal": True,
        "provider": attempt.provider,
        "modelId": attempt.model_id,
        "ok": attempt.ok,
        "latencyMs": attempt.latency_ms,
        "errorMessage": attempt.error_message,
        "errorType": attempt.error_type,
        "retryable": attempt.retryable,
        "reason": attempt.reason,
        "retry": retry,
        "targetRank": target_rank,
    }


async def _emit_attempt(
    observer: Any, attempt: GatewayAttempt, retry: int, target_rank: int
) -> None:
    if observer is None:
        return
    try:
        await _maybe_await(observer(_attempt_payload(attempt, retry, target_rank)))
    except Exception:
        # Gateway observers are non-authoritative telemetry sinks. An observer
        # failure must never turn a provider success into a failed attempt or
        # cause a retry that could duplicate external work.
        return


def _attempt_latency_ms(started_at_ns: int) -> int:
    elapsed_ns = max(0, time.monotonic_ns() - started_at_ns)
    return max(1, math.ceil(elapsed_ns / 1_000_000))


def _attempt_error_type(error: Exception) -> _GatewayAttemptErrorType:
    if isinstance(error, asyncio.TimeoutError):
        return "timeout"
    if isinstance(error, ProviderHTTPError):
        return "provider_http_error"
    if isinstance(error, OSError):
        return "transport_error"
    if isinstance(error, GatewayError):
        return "gateway_error"
    return "provider_error"


def _safe_attempt_error_message(
    error: Exception,
    normalized: GatewayError,
    target: GatewayModelTarget,
    timeout_ms: int,
) -> str:
    if isinstance(error, asyncio.TimeoutError):
        return f'Gateway attempt timed out for "{target.provider}/{target.model_id}" after {timeout_ms} ms.'
    if isinstance(error, ProviderHTTPError):
        return (
            redact_provider_error_body(str(error))
            or f"Provider request failed with status {error.status}."
        )
    if isinstance(error, OSError):
        return f'Gateway transport error for "{target.provider}/{target.model_id}".'
    if isinstance(error, GatewayError):
        return redact_provider_error_body(str(normalized)) or "Gateway attempt failed."
    return f'Gateway provider error ({type(error).__name__}) for "{target.provider}/{target.model_id}".'


def create_gateway(config: GatewayConfig):
    class Gateway:
        async def _run_generate(
            self,
            *,
            messages: list[GatewayMessage],
            primary: GatewayModelTarget,
            fallbacks: list[GatewayModelTarget] | None,
            system_prompt: str | None,
            temperature: float | None,
            max_tokens: int | None,
            required_capabilities: dict[str, bool] | None,
            max_cost_per_1k_tokens: float | None,
            routing_mode: GatewayRoutingMode,
            task_intent: GatewayTaskIntent,
            run: Any,
        ) -> tuple[
            Any,
            GatewayProviderId,
            str,
            list[GatewayAttempt],
            GatewayRouteDecision,
            float,
        ]:
            attempts: list[GatewayAttempt] = []
            started_at = time.monotonic()
            ordered_targets = _order_targets(
                routing_mode, task_intent, primary, fallbacks or [], config
            )
            route_decision = create_route_decision(
                routing_mode,
                task_intent,
                ordered_targets,
                config=config,
                required_capabilities=required_capabilities,
            )
            refusal_result: (
                tuple[
                    Any,
                    GatewayProviderId,
                    str,
                    list[GatewayAttempt],
                    GatewayRouteDecision,
                    float,
                ]
                | None
            ) = None

            async def record_skipped_attempt(
                target: GatewayModelTarget,
                target_index: int,
                error_message: str,
                reason: _GatewayAttemptReason,
            ) -> None:
                attempts.append(
                    GatewayAttempt(
                        provider=target.provider,
                        model_id=target.model_id,
                        ok=False,
                        latency_ms=0,
                        error_message=error_message,
                        retryable=False,
                        reason=reason,
                        error_type="policy_skip",
                    )
                )
                await _emit_attempt(config.on_attempt, attempts[-1], 0, target_index)

            for target_index, target in enumerate(ordered_targets):
                catalog_entry = _find_catalog_entry(config, target)
                if (
                    catalog_entry is not None
                    and catalog_entry.availability == "retired"
                ):
                    replacement = (
                        f' Use "{catalog_entry.replacement_model_id}" instead.'
                        if catalog_entry.replacement_model_id
                        else ""
                    )
                    await record_skipped_attempt(
                        target,
                        target_index,
                        f"Skipped because the catalog marks this model as retired.{replacement}",
                        "model_unavailable",
                    )
                    continue
                if (
                    catalog_entry is not None
                    and catalog_entry.api_surface != "language"
                ):
                    await record_skipped_attempt(
                        target,
                        target_index,
                        f'Skipped because catalog surface "{catalog_entry.api_surface}" is not language generation.',
                        "unsupported_api_surface",
                    )
                    continue
                adapter = config.adapters.get(target.provider)
                if adapter is None:
                    await record_skipped_attempt(
                        target,
                        target_index,
                        f'No adapter registered for provider "{target.provider}".',
                        "missing_adapter",
                    )
                    if config.fail_on_missing_adapter:
                        raise _final_gateway_error(attempts)
                    continue
                cost_budget_reason = _cost_budget_reason(
                    config, max_cost_per_1k_tokens, target
                )
                if cost_budget_reason is not None:
                    await record_skipped_attempt(
                        target,
                        target_index,
                        (
                            "Skipped because model cost is unknown under the configured budget."
                            if cost_budget_reason == "cost_unknown"
                            else "Skipped because model cost exceeds the configured budget."
                        ),
                        cost_budget_reason,
                    )
                    continue
                if _messages_require_vision(messages) and not _target_supports_vision(
                    adapter, target, config
                ):
                    await record_skipped_attempt(
                        target,
                        target_index,
                        "Skipped because the request contains images and the target does not support vision input.",
                        "vision_unsupported",
                    )
                    continue
                if not _supports_required_capabilities(
                    adapter, target, required_capabilities, config
                ):
                    await record_skipped_attempt(
                        target,
                        target_index,
                        "Skipped because model capabilities do not satisfy the request.",
                        "capability_mismatch",
                    )
                    continue
                for retry in range(max(0, config.max_retries) + 1):
                    attempt_started_ns = time.monotonic_ns()
                    try:

                        async def invoke() -> Any:
                            return await _maybe_await(
                                run(
                                    adapter,
                                    target,
                                    messages,
                                    system_prompt,
                                    temperature,
                                    max_tokens,
                                )
                            )

                        result = await asyncio.wait_for(
                            invoke(),
                            timeout=(
                                config.attempt_timeouts_ms.get(
                                    target.provider, config.attempt_timeout_ms
                                )
                                / 1000
                            ),
                        )
                        latency_ms = _attempt_latency_ms(attempt_started_ns)
                        if _is_refusal_result(result):
                            attempts.append(
                                GatewayAttempt(
                                    provider=target.provider,
                                    model_id=target.model_id,
                                    ok=False,
                                    latency_ms=latency_ms,
                                    error_message="Provider returned refusal stop reason.",
                                    retryable=False,
                                    reason="provider_refusal",
                                    error_type="refusal",
                                )
                            )
                            await _emit_attempt(
                                config.on_attempt, attempts[-1], retry, target_index
                            )
                            if config.fallback_on_refusal:
                                refusal_result = (
                                    result,
                                    target.provider,
                                    target.model_id,
                                    attempts,
                                    route_decision,
                                    started_at,
                                )
                                break
                            return (
                                result,
                                target.provider,
                                target.model_id,
                                attempts,
                                route_decision,
                                started_at,
                            )
                        attempts.append(
                            GatewayAttempt(
                                provider=target.provider,
                                model_id=target.model_id,
                                ok=True,
                                latency_ms=latency_ms,
                            )
                        )
                        await _emit_attempt(
                            config.on_attempt, attempts[-1], retry, target_index
                        )
                        return (
                            result,
                            target.provider,
                            target.model_id,
                            attempts,
                            route_decision,
                            started_at,
                        )
                    except Exception as error:
                        normalized = _normalize_error(error)
                        latency_ms = _attempt_latency_ms(attempt_started_ns)
                        timeout_ms = config.attempt_timeouts_ms.get(
                            target.provider, config.attempt_timeout_ms
                        )
                        error_message = _safe_attempt_error_message(
                            error, normalized, target, timeout_ms
                        )
                        attempts.append(
                            GatewayAttempt(
                                provider=target.provider,
                                model_id=target.model_id,
                                ok=False,
                                latency_ms=latency_ms,
                                error_message=str(normalized),
                                retryable=normalized.retryable,
                                error_type=_attempt_error_type(error),
                            )
                        )
                        attempts[-1].error_message = error_message
                        await _emit_attempt(
                            config.on_attempt, attempts[-1], retry, target_index
                        )
                        if retry < max(0, config.max_retries) and normalized.retryable:
                            await asyncio.sleep(
                                config.retry_backoff_ms * (retry + 1) / 1000
                            )
                            continue
                        break
            if refusal_result is not None:
                return refusal_result
            raise _final_gateway_error(attempts)

        def _build_response(
            self,
            *,
            result: Any,
            provider_used: GatewayProviderId,
            model_used: str,
            attempts: list[GatewayAttempt],
            route_decision: GatewayRouteDecision,
            started_at: float,
            messages: list[GatewayMessage],
            system_prompt: str | None,
        ) -> GatewayResponse:
            input_text = f"{system_prompt or ''}\n" + "\n".join(
                message.content for message in messages
            )
            usage, estimated = _normalize_usage(
                getattr(result, "usage", None), input_text.strip(), result.text
            )
            return GatewayResponse(
                text=result.text,
                provider_used=provider_used,
                model_used=model_used,
                finish_reason=getattr(result, "finish_reason", None),
                provider_finish_reason=getattr(result, "provider_finish_reason", None),
                latency_ms=int((time.monotonic() - started_at) * 1000),
                attempts=attempts,
                usage=usage,
                usage_estimated=estimated,
                route_decision=route_decision,
            )

        async def generate(
            self,
            *,
            messages: list[GatewayMessage],
            primary: GatewayModelTarget,
            fallbacks: list[GatewayModelTarget] | None = None,
            system_prompt: str | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
            required_capabilities: dict[str, bool] | None = None,
            max_cost_per_1k_tokens: float | None = None,
            routing_mode: GatewayRoutingMode = "balanced",
            task_intent: GatewayTaskIntent = "chat",
        ) -> GatewayResponse:
            (
                result,
                provider_used,
                model_used,
                attempts,
                route_decision,
                started_at,
            ) = await self._run_generate(
                messages=messages,
                primary=primary,
                fallbacks=fallbacks,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                required_capabilities=required_capabilities,
                max_cost_per_1k_tokens=max_cost_per_1k_tokens,
                routing_mode=routing_mode,
                task_intent=task_intent,
                run=lambda adapter, target, messages, system_prompt, temperature, max_tokens: (
                    generate_text(
                        model=resolve_provider_adapter(adapter).language_model(
                            target.model_id
                        ),
                        messages=gateway_messages_to_model_messages(
                            messages, system_prompt
                        ),
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                ),
            )
            return self._build_response(
                result=result,
                provider_used=provider_used,
                model_used=model_used,
                attempts=attempts,
                route_decision=route_decision,
                started_at=started_at,
                messages=messages,
                system_prompt=system_prompt,
            )

        def stream_text(
            self,
            *,
            messages: list[GatewayMessage],
            primary: GatewayModelTarget,
            fallbacks: list[GatewayModelTarget] | None = None,
            system_prompt: str | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
            required_capabilities: dict[str, bool] | None = None,
            max_cost_per_1k_tokens: float | None = None,
            routing_mode: GatewayRoutingMode = "balanced",
            task_intent: GatewayTaskIntent = "chat",
        ):
            async def start():
                return await self._run_generate(
                    messages=messages,
                    primary=primary,
                    fallbacks=fallbacks,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    required_capabilities=required_capabilities,
                    max_cost_per_1k_tokens=max_cost_per_1k_tokens,
                    routing_mode=routing_mode,
                    task_intent=task_intent,
                    run=lambda adapter, target, messages, system_prompt, temperature, max_tokens: (
                        stream_text(
                            model=resolve_provider_adapter(adapter).language_model(
                                target.model_id
                            ),
                            messages=gateway_messages_to_model_messages(
                                messages, system_prompt
                            ),
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    ),
                )

            selected = asyncio.create_task(start())

            class GatewayStreamTextResult:
                async def event_stream(self):
                    result, *_ = await selected
                    async for event in result.event_stream():
                        yield event

                async def text_stream(self):
                    result, *_ = await selected
                    async for chunk in result.text_stream():
                        yield chunk

                async def collect(self):
                    (
                        result,
                        provider_used,
                        model_used,
                        attempts,
                        route_decision,
                        started_at,
                    ) = await selected
                    final = await result.collect()
                    return self_outer._build_response(
                        result=final,
                        provider_used=provider_used,
                        model_used=model_used,
                        attempts=attempts,
                        route_decision=route_decision,
                        started_at=started_at,
                        messages=messages,
                        system_prompt=system_prompt,
                    )

            self_outer = self
            return GatewayStreamTextResult()

        async def generate_object(
            self,
            *,
            schema: Any,
            messages: list[GatewayMessage],
            primary: GatewayModelTarget,
            fallbacks: list[GatewayModelTarget] | None = None,
            system_prompt: str | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
            mode: str = "auto",
            schema_name: str | None = None,
            schema_description: str | None = None,
            required_capabilities: dict[str, bool] | None = None,
            max_cost_per_1k_tokens: float | None = None,
            routing_mode: GatewayRoutingMode = "balanced",
            task_intent: GatewayTaskIntent = "chat",
        ) -> GatewayObjectResponse:
            (
                result,
                provider_used,
                model_used,
                attempts,
                route_decision,
                started_at,
            ) = await self._run_generate(
                messages=messages,
                primary=primary,
                fallbacks=fallbacks,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                required_capabilities=required_capabilities,
                max_cost_per_1k_tokens=max_cost_per_1k_tokens,
                routing_mode=routing_mode,
                task_intent=task_intent,
                run=lambda adapter, target, messages, system_prompt, temperature, max_tokens: (
                    generate_object(
                        model=resolve_provider_adapter(adapter).language_model(
                            target.model_id
                        ),
                        messages=gateway_messages_to_model_messages(
                            messages, system_prompt
                        ),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        schema=schema,
                        mode=mode,
                        schema_name=schema_name,
                        schema_description=schema_description,
                    )
                ),
            )
            text_response = self._build_response(
                result=result,
                provider_used=provider_used,
                model_used=model_used,
                attempts=attempts,
                route_decision=route_decision,
                started_at=started_at,
                messages=messages,
                system_prompt=system_prompt,
            )
            return GatewayObjectResponse(
                text=text_response.text,
                provider_used=text_response.provider_used,
                model_used=text_response.model_used,
                finish_reason=text_response.finish_reason,
                provider_finish_reason=text_response.provider_finish_reason,
                latency_ms=text_response.latency_ms,
                attempts=text_response.attempts,
                usage=text_response.usage,
                usage_estimated=text_response.usage_estimated,
                route_decision=text_response.route_decision,
                object=result.object,
                object_mode=result.object_mode,
            )

        def stream_object(
            self,
            *,
            schema: Any,
            messages: list[GatewayMessage],
            primary: GatewayModelTarget,
            fallbacks: list[GatewayModelTarget] | None = None,
            system_prompt: str | None = None,
            temperature: float | None = None,
            max_tokens: int | None = None,
            mode: str = "auto",
            schema_name: str | None = None,
            schema_description: str | None = None,
            required_capabilities: dict[str, bool] | None = None,
            max_cost_per_1k_tokens: float | None = None,
            routing_mode: GatewayRoutingMode = "balanced",
            task_intent: GatewayTaskIntent = "chat",
        ):
            async def start():
                return await self._run_generate(
                    messages=messages,
                    primary=primary,
                    fallbacks=fallbacks,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    required_capabilities=required_capabilities,
                    max_cost_per_1k_tokens=max_cost_per_1k_tokens,
                    routing_mode=routing_mode,
                    task_intent=task_intent,
                    run=lambda adapter, target, messages, system_prompt, temperature, max_tokens: (
                        stream_object(
                            model=resolve_provider_adapter(adapter).language_model(
                                target.model_id
                            ),
                            messages=gateway_messages_to_model_messages(
                                messages, system_prompt
                            ),
                            temperature=temperature,
                            max_tokens=max_tokens,
                            schema=schema,
                            mode=mode,
                            schema_name=schema_name,
                            schema_description=schema_description,
                        )
                    ),
                )

            selected = asyncio.create_task(start())

            class GatewayStreamObjectResult:
                async def event_stream(self):
                    result, *_ = await selected
                    async for event in result.event_stream():
                        yield event

                async def text_stream(self):
                    result, *_ = await selected
                    async for chunk in result.text_stream():
                        yield chunk

                async def partial_object_stream(self):
                    result, *_ = await selected
                    async for item in result.partial_object_stream():
                        yield item

                async def collect(self):
                    (
                        result,
                        provider_used,
                        model_used,
                        attempts,
                        route_decision,
                        started_at,
                    ) = await selected
                    final = await result.collect()
                    text_response = self_outer._build_response(
                        result=final,
                        provider_used=provider_used,
                        model_used=model_used,
                        attempts=attempts,
                        route_decision=route_decision,
                        started_at=started_at,
                        messages=messages,
                        system_prompt=system_prompt,
                    )
                    return GatewayObjectResponse(
                        text=text_response.text,
                        provider_used=text_response.provider_used,
                        model_used=text_response.model_used,
                        latency_ms=text_response.latency_ms,
                        attempts=text_response.attempts,
                        usage=text_response.usage,
                        usage_estimated=text_response.usage_estimated,
                        route_decision=text_response.route_decision,
                        object=final.object,
                        object_mode=final.object_mode,
                    )

            self_outer = self
            return GatewayStreamObjectResult()

    return Gateway()
