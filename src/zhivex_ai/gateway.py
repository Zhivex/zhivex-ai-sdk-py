from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .providers.base import resolve_provider_adapter
from .errors import ProviderHTTPError, ZhivexAIError
from .generate_object import generate_object, stream_object
from .generate_text import generate_text, stream_text
from .types import ModelMessage, TokenUsage

GatewayProviderId = Literal[
    "openai", "anthropic", "gemini", "vertex", "qwen", "kimi", "bedrock", "ollama", "azure-openai", "openrouter"
]
GatewayRoutingMode = Literal["speed", "balanced", "quality"]
GatewayTaskIntent = Literal["chat", "reasoning", "tool-heavy"]


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


@dataclass(slots=True)
class GatewayRouteDecision:
    mode: GatewayRoutingMode
    intent: GatewayTaskIntent
    ordered_targets: list[GatewayModelTarget]
    reason: str


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


@dataclass(slots=True)
class GatewayObjectResponse(GatewayResponse):
    object: Any = None
    object_mode: Literal["native", "prompted"] = "prompted"


@dataclass(slots=True)
class GatewayConfig:
    adapters: dict[GatewayProviderId, Any]
    model_catalog: Any = None
    provider_costs_per_1k_tokens: dict[GatewayProviderId, float] = field(default_factory=dict)
    latency_bias_ms: dict[GatewayProviderId, int] = field(default_factory=dict)
    max_retries: int = 2
    attempt_timeout_ms: int = 20_000
    attempt_timeouts_ms: dict[GatewayProviderId, int] = field(default_factory=dict)
    retry_backoff_ms: int = 200
    on_attempt: Any = None


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
    return True


def _messages_require_vision(messages: list[GatewayMessage]) -> bool:
    return any(message.images for message in messages)


def _target_supports_vision(adapter: Any, target: GatewayModelTarget) -> bool:
    capabilities = resolve_provider_adapter(adapter).language_model(target.model_id).capabilities
    return capabilities.vision and supports_vision_input(target.provider, target.model_id)


def gateway_messages_to_model_messages(messages: list[GatewayMessage], system_prompt: str | None = None) -> list[ModelMessage]:
    mapped: list[ModelMessage] = []
    if system_prompt:
        from .messages import system

        mapped.append(system(system_prompt))
    from .types import ImagePart, TextPart

    for message in messages:
        parts = [TextPart(text=message.content)]
        parts.extend(ImagePart(image=image.data_url, media_type=image.mime_type) for image in message.images)
        mapped.append(ModelMessage(role=message.role, parts=parts))
    return mapped


def create_route_decision(mode: GatewayRoutingMode, intent: GatewayTaskIntent, ordered_targets: list[GatewayModelTarget]) -> GatewayRouteDecision:
    return GatewayRouteDecision(
        mode=mode,
        intent=intent,
        ordered_targets=ordered_targets,
        reason=f"Primary target preserved first; fallbacks ordered by {mode} mode with {intent} intent.",
    )


def _score_target(mode: GatewayRoutingMode, intent: GatewayTaskIntent, target: GatewayModelTarget, config: GatewayConfig) -> float:
    model = target.model_id.lower()
    local_boost = -2 if target.provider == "ollama" else 0
    quality_boost = 2 if ("pro" in model or "claude" in model) else 0
    speed_boost = 2 if ("flash" in model or "lite" in model) else 0
    reasoning_boost = 2 if ("pro" in model or "claude" in model) else 0
    cost_penalty = config.provider_costs_per_1k_tokens.get(target.provider, 0)
    latency_penalty = config.latency_bias_ms.get(target.provider, 0) / 100
    if mode == "speed":
        return speed_boost + local_boost - latency_penalty
    if mode == "quality":
        return quality_boost + (reasoning_boost if intent == "reasoning" else 0) - cost_penalty
    return speed_boost + quality_boost + local_boost + (1 if intent == "reasoning" else 0) - cost_penalty - latency_penalty


def _order_targets(mode: GatewayRoutingMode, intent: GatewayTaskIntent, primary: GatewayModelTarget, fallbacks: list[GatewayModelTarget], config: GatewayConfig) -> list[GatewayModelTarget]:
    ordered: list[GatewayModelTarget] = [primary]
    seen: set[tuple[str, str]] = {(primary.provider, primary.model_id)}
    ranked_fallbacks: list[GatewayModelTarget] = []
    for target in fallbacks:
        key = (target.provider, target.model_id)
        if key in seen:
            continue
        seen.add(key)
        ranked_fallbacks.append(target)
    ranked_fallbacks.sort(key=lambda target: _score_target(mode, intent, target, config), reverse=True)
    ordered.extend(ranked_fallbacks)
    return ordered


def _supports_required_capabilities(adapter: Any, target: GatewayModelTarget, required: dict[str, bool] | None) -> bool:
    if not required:
        return True
    capabilities = resolve_provider_adapter(adapter).language_model(target.model_id).capabilities
    mapping = {"structuredOutput": "structured_output", "jsonMode": "json_mode"}
    for key, value in required.items():
        if value is not True:
            continue
        if getattr(capabilities, mapping.get(key, key)) is not True:
            return False
    return True


def _within_cost_budget(config: GatewayConfig, max_cost: float | None, target: GatewayModelTarget) -> bool:
    if max_cost is None:
        return True
    effective_cost = config.provider_costs_per_1k_tokens.get(target.provider)
    return effective_cost is None or effective_cost <= max_cost


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text.strip()) + 3) // 4)


def _normalize_usage(usage: TokenUsage | None, input_text: str, output_text: str) -> tuple[TokenUsage, bool]:
    input_tokens = usage.input_tokens if usage and usage.input_tokens is not None else _estimate_tokens(input_text)
    output_tokens = usage.output_tokens if usage and usage.output_tokens is not None else _estimate_tokens(output_text)
    total_tokens = usage.total_tokens if usage and usage.total_tokens is not None else input_tokens + output_tokens
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens), (
        usage is None or usage.input_tokens is None or usage.output_tokens is None or usage.total_tokens is None
    )


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
    if any(token in message for token in ("timed out", "429", "rate", "connect", "econnrefused", "503")):
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
    if asyncio.iscoroutine(value):
        return await value
    return value


def _attempt_payload(attempt: GatewayAttempt, retry: int, target_rank: int) -> dict[str, Any]:
    return {
        "provider": attempt.provider,
        "modelId": attempt.model_id,
        "ok": attempt.ok,
        "latencyMs": attempt.latency_ms,
        "errorMessage": attempt.error_message,
        "retryable": attempt.retryable,
        "retry": retry,
        "targetRank": target_rank,
    }


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
        ) -> tuple[Any, GatewayProviderId, str, list[GatewayAttempt], GatewayRouteDecision, float]:
            attempts: list[GatewayAttempt] = []
            started_at = time.time()
            ordered_targets = _order_targets(routing_mode, task_intent, primary, fallbacks or [], config)
            route_decision = create_route_decision(routing_mode, task_intent, ordered_targets)
            for target_index, target in enumerate(ordered_targets):
                adapter = config.adapters.get(target.provider)
                if adapter is None:
                    attempts.append(GatewayAttempt(provider=target.provider, model_id=target.model_id, ok=False, latency_ms=0, error_message=f'No adapter registered for provider "{target.provider}".'))
                    continue
                if _messages_require_vision(messages) and not _target_supports_vision(adapter, target):
                    attempts.append(
                        GatewayAttempt(
                            provider=target.provider,
                            model_id=target.model_id,
                            ok=False,
                            latency_ms=0,
                            error_message="Skipped because the request contains images and the target does not support vision input.",
                        )
                    )
                    continue
                if not _supports_required_capabilities(adapter, target, required_capabilities):
                    attempts.append(GatewayAttempt(provider=target.provider, model_id=target.model_id, ok=False, latency_ms=0, error_message="Skipped because model capabilities do not satisfy the request."))
                    continue
                if not _within_cost_budget(config, max_cost_per_1k_tokens, target):
                    attempts.append(GatewayAttempt(provider=target.provider, model_id=target.model_id, ok=False, latency_ms=0, error_message="Skipped because provider cost exceeds the configured budget."))
                    continue
                for retry in range(max(0, config.max_retries) + 1):
                    attempt_started = time.time()
                    try:
                        if config.on_attempt:
                            await _maybe_await(
                                config.on_attempt(
                                    _attempt_payload(
                                        GatewayAttempt(provider=target.provider, model_id=target.model_id, ok=True, latency_ms=0),
                                        retry,
                                        target_index,
                                    )
                                )
                            )
                        async def invoke() -> Any:
                            return await _maybe_await(run(adapter, target, messages, system_prompt, temperature, max_tokens))

                        result = await asyncio.wait_for(
                            invoke(),
                            timeout=(config.attempt_timeouts_ms.get(target.provider, config.attempt_timeout_ms) / 1000),
                        )
                        latency_ms = int((time.time() - attempt_started) * 1000)
                        attempts.append(GatewayAttempt(provider=target.provider, model_id=target.model_id, ok=True, latency_ms=latency_ms))
                        return result, target.provider, target.model_id, attempts, route_decision, started_at
                    except Exception as error:
                        normalized = _normalize_error(error)
                        latency_ms = int((time.time() - attempt_started) * 1000)
                        error_message = str(normalized) or (
                            f'Gateway attempt failed for "{target.provider}/{target.model_id}".'
                        )
                        if isinstance(error, asyncio.TimeoutError):
                            timeout_ms = config.attempt_timeouts_ms.get(target.provider, config.attempt_timeout_ms)
                            error_message = (
                                f'Gateway attempt timed out for "{target.provider}/{target.model_id}" after {timeout_ms} ms.'
                            )
                        attempts.append(
                            GatewayAttempt(
                                provider=target.provider,
                                model_id=target.model_id,
                                ok=False,
                                latency_ms=latency_ms,
                                error_message=str(normalized),
                                retryable=normalized.retryable,
                            )
                        )
                        attempts[-1].error_message = error_message
                        if config.on_attempt:
                            await _maybe_await(config.on_attempt(_attempt_payload(attempts[-1], retry, target_index)))
                        if retry < max(0, config.max_retries) and normalized.retryable:
                            await asyncio.sleep(config.retry_backoff_ms * (retry + 1) / 1000)
                            continue
                        break
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
            input_text = f'{system_prompt or ""}\n' + "\n".join(message.content for message in messages)
            usage, estimated = _normalize_usage(getattr(result, "usage", None), input_text.strip(), result.text)
            return GatewayResponse(
                text=result.text,
                provider_used=provider_used,
                model_used=model_used,
                latency_ms=int((time.time() - started_at) * 1000),
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
            result, provider_used, model_used, attempts, route_decision, started_at = await self._run_generate(
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
                run=lambda adapter, target, messages, system_prompt, temperature, max_tokens: generate_text(
                    model=resolve_provider_adapter(adapter).language_model(target.model_id),
                    messages=gateway_messages_to_model_messages(messages, system_prompt),
                    temperature=temperature,
                    max_tokens=max_tokens,
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
                    run=lambda adapter, target, messages, system_prompt, temperature, max_tokens: stream_text(
                        model=resolve_provider_adapter(adapter).language_model(target.model_id),
                        messages=gateway_messages_to_model_messages(messages, system_prompt),
                        temperature=temperature,
                        max_tokens=max_tokens,
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
                    result, provider_used, model_used, attempts, route_decision, started_at = await selected
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
            result, provider_used, model_used, attempts, route_decision, started_at = await self._run_generate(
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
                run=lambda adapter, target, messages, system_prompt, temperature, max_tokens: generate_object(
                    model=resolve_provider_adapter(adapter).language_model(target.model_id),
                    messages=gateway_messages_to_model_messages(messages, system_prompt),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    schema=schema,
                    mode=mode,
                    schema_name=schema_name,
                    schema_description=schema_description,
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
                    run=lambda adapter, target, messages, system_prompt, temperature, max_tokens: stream_object(
                        model=resolve_provider_adapter(adapter).language_model(target.model_id),
                        messages=gateway_messages_to_model_messages(messages, system_prompt),
                        temperature=temperature,
                        max_tokens=max_tokens,
                        schema=schema,
                        mode=mode,
                        schema_name=schema_name,
                        schema_description=schema_description,
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
                    result, provider_used, model_used, attempts, route_decision, started_at = await selected
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
