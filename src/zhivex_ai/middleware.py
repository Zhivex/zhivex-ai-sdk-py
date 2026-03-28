from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .types import GenerateResult, LanguageModel, ModelGenerateInput


def _serialize_input(input: Any) -> str:
    return json.dumps(input, default=lambda value: getattr(value, "__dict__", str(value)), sort_keys=True)


class GenerateCache(Protocol):
    async def get(self, key: str) -> GenerateResult | None: ...

    async def set(self, key: str, value: GenerateResult) -> None: ...


@dataclass(slots=True)
class MiddlewareContext:
    model: LanguageModel
    input: ModelGenerateInput


MiddlewareNext = Callable[[], Awaitable[GenerateResult]]
GenerateMiddleware = Callable[[MiddlewareContext, MiddlewareNext], Awaitable[GenerateResult]]


def wrap_language_model(model: LanguageModel, middlewares: list[GenerateMiddleware]) -> LanguageModel:
    if not middlewares:
        return model

    class WrappedLanguageModel:
        provider = model.provider
        model_id = model.model_id
        capabilities = model.capabilities

        async def generate(self, input: ModelGenerateInput) -> GenerateResult:
            index = -1

            async def run(position: int) -> GenerateResult:
                nonlocal index
                if position <= index:
                    raise RuntimeError("Language model middleware called next() multiple times.")
                index = position
                if position >= len(middlewares):
                    return await model.generate(input)
                return await middlewares[position](MiddlewareContext(model=model, input=input), lambda: run(position + 1))

            return await run(0)

        async def stream(self, input: ModelGenerateInput):
            return await model.stream(input)

    return WrappedLanguageModel()


def create_telemetry_middleware(*, on_event: Callable[[dict[str, Any]], Awaitable[None] | None]) -> GenerateMiddleware:
    async def middleware(context: MiddlewareContext, next: MiddlewareNext) -> GenerateResult:
        started_at = int(time.time() * 1000)
        maybe = on_event({"type": "generate-start", "model": context.model, "input": context.input, "startedAt": started_at})
        if maybe is not None:
            await maybe
        try:
            output = await next()
            finished_at = int(time.time() * 1000)
            maybe = on_event(
                {
                    "type": "generate-finish",
                    "model": context.model,
                    "input": context.input,
                    "output": output,
                    "startedAt": started_at,
                    "finishedAt": finished_at,
                    "latencyMs": finished_at - started_at,
                }
            )
            if maybe is not None:
                await maybe
            return output
        except Exception as error:
            finished_at = int(time.time() * 1000)
            maybe = on_event(
                {
                    "type": "generate-error",
                    "model": context.model,
                    "input": context.input,
                    "error": error,
                    "startedAt": started_at,
                    "finishedAt": finished_at,
                    "latencyMs": finished_at - started_at,
                }
            )
            if maybe is not None:
                await maybe
            raise

    return middleware


def create_cached_generate_middleware(
    *,
    cache: GenerateCache,
    get_key: Callable[[ModelGenerateInput, LanguageModel], str] | None = None,
) -> GenerateMiddleware:
    async def middleware(context: MiddlewareContext, next: MiddlewareNext) -> GenerateResult:
        key = get_key(context.input, context.model) if get_key else _serialize_input(
            {"provider": context.model.provider, "model_id": context.model.model_id, "input": context.input}
        )
        cached = await cache.get(key)
        if cached is not None:
            return cached
        output = await next()
        await cache.set(key, output)
        return output

    return middleware


class InMemoryGenerateCache:
    def __init__(self) -> None:
        self._store: dict[str, GenerateResult] = {}

    async def get(self, key: str) -> GenerateResult | None:
        return self._store.get(key)

    async def set(self, key: str, value: GenerateResult) -> None:
        self._store[key] = value


def create_in_memory_generate_cache() -> InMemoryGenerateCache:
    return InMemoryGenerateCache()


class FileGenerateCache:
    def __init__(self, dir: str) -> None:
        self._dir = Path(dir)

    def _path(self, key: str) -> Path:
        filename = hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json"
        return self._dir / filename

    async def get(self, key: str) -> GenerateResult | None:
        path = self._path(key)
        if not path.exists():
            return None
        return GenerateResult(**json.loads(path.read_text("utf-8")))

    async def set(self, key: str, value: GenerateResult) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        serializable = json.loads(_serialize_input(value))
        self._path(key).write_text(json.dumps(serializable), "utf-8")


def create_file_generate_cache(*, dir: str) -> FileGenerateCache:
    return FileGenerateCache(dir)


@dataclass(slots=True)
class CircuitBreakerState:
    failures: int = 0
    opened_at: int | None = None


def create_circuit_breaker_middleware(
    *,
    failure_threshold: int = 3,
    cooldown_ms: int = 30_000,
    is_failure: Callable[[Exception], bool] | None = None,
    on_state_change: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
) -> GenerateMiddleware:
    state = CircuitBreakerState()
    threshold = max(1, failure_threshold)
    cooldown = max(0, cooldown_ms)

    async def emit(payload: dict[str, Any]) -> None:
        if on_state_change is None:
            return
        maybe = on_state_change(payload)
        if maybe is not None:
            await maybe

    async def middleware(context: MiddlewareContext, next: MiddlewareNext) -> GenerateResult:
        now = int(time.time() * 1000)
        if state.opened_at and now - state.opened_at < cooldown:
            raise RuntimeError(f'Circuit breaker open for model "{context.model.provider}/{context.model.model_id}".')
        if state.opened_at and now - state.opened_at >= cooldown:
            await emit({"failures": state.failures, "openedAt": state.opened_at, "model": context.model, "status": "half-open"})
        try:
            result = await next()
            state.failures = 0
            state.opened_at = None
            await emit({"failures": state.failures, "openedAt": state.opened_at, "model": context.model, "status": "closed"})
            return result
        except Exception as error:
            if is_failure is not None and not is_failure(error):
                raise
            state.failures += 1
            if state.failures >= threshold:
                state.opened_at = int(time.time() * 1000)
                await emit({"failures": state.failures, "openedAt": state.opened_at, "model": context.model, "status": "open"})
            raise

    return middleware
