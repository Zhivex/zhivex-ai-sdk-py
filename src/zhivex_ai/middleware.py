from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import stat
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from .errors import ValidationError
from .types import GenerateResult, LanguageModel, ModelGenerateInput


_FILE_CACHE_TEMP_PREFIX = ".zhivex-generate-cache-"
_FILE_CACHE_TEMP_SUFFIX = ".tmp"
_FILE_CACHE_STALE_TEMP_SECONDS = 24 * 60 * 60
_FILE_CACHE_CLEANUP_SCAN_LIMIT = 256
_FILE_CACHE_CLEANUP_DELETE_LIMIT = 32


def _serialize_input(input: Any) -> str:
    def default(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        return getattr(value, "__dict__", str(value))

    return json.dumps(input, default=default, sort_keys=True)


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
    in_flight: dict[str, asyncio.Task[GenerateResult]] = {}
    in_flight_lock = asyncio.Lock()

    def consume_task_exception(task: asyncio.Task[GenerateResult]) -> None:
        if not task.cancelled():
            task.exception()

    async def resolve(key: str, next: MiddlewareNext) -> GenerateResult:
        try:
            cached = await cache.get(key)
            if cached is not None:
                return cached
            output = await next()
            await cache.set(key, output)
            return output
        finally:
            task = asyncio.current_task()
            async with in_flight_lock:
                if in_flight.get(key) is task:
                    del in_flight[key]

    async def middleware(context: MiddlewareContext, next: MiddlewareNext) -> GenerateResult:
        key = get_key(context.input, context.model) if get_key else _serialize_input(
            {"provider": context.model.provider, "model_id": context.model.model_id, "input": context.input}
        )
        async with in_flight_lock:
            task = in_flight.get(key)
            if task is None:
                task = asyncio.create_task(resolve(key, next))
                task.add_done_callback(consume_task_exception)
                in_flight[key] = task
        return await asyncio.shield(task)

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

    @staticmethod
    def _digest(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / f"{self._digest(key)}.json"

    def _temp_prefix(self, key: str) -> str:
        return f"{_FILE_CACHE_TEMP_PREFIX}{self._digest(key)}-"

    @staticmethod
    def _corrupt_entry(path: Path) -> ValidationError:
        return ValidationError(
            f'File generate cache entry "{path.name}" is corrupt, incompatible, or not a regular file.'
        )

    async def get(self, key: str) -> GenerateResult | None:
        return await asyncio.to_thread(self._get_sync, key)

    def _get_sync(self, key: str) -> GenerateResult | None:
        path = self._path(key)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        if no_follow:
            flags |= no_follow
        else:
            try:
                if stat.S_ISLNK(os.lstat(path).st_mode):
                    raise self._corrupt_entry(path)
            except FileNotFoundError:
                return None

        descriptor: int | None = None
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            return None
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise self._corrupt_entry(path) from error
            raise

        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise self._corrupt_entry(path)
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = None
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise self._corrupt_entry(path)
            return GenerateResult(**payload)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
            raise self._corrupt_entry(path) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)

    async def set(self, key: str, value: GenerateResult) -> None:
        await asyncio.to_thread(self._set_sync, key, value)

    def _set_sync(self, key: str, value: GenerateResult) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        serializable = json.loads(_serialize_input(value))
        payload = json.dumps(serializable, separators=(",", ":"), sort_keys=True)
        self._cleanup_stale_temp_files(key)

        descriptor: int | None = None
        temp_path: str | None = None
        try:
            descriptor, temp_path = tempfile.mkstemp(
                dir=self._dir,
                prefix=self._temp_prefix(key),
                suffix=_FILE_CACHE_TEMP_SUFFIX,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                descriptor = None
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temp_path, self._path(key))
            temp_path = None
            self._fsync_directory()
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    def _cleanup_stale_temp_files(self, key: str) -> None:
        prefix = self._temp_prefix(key)
        cutoff = time.time() - _FILE_CACHE_STALE_TEMP_SECONDS
        deleted = 0
        try:
            with os.scandir(self._dir) as entries:
                for scanned, entry in enumerate(entries):
                    if scanned >= _FILE_CACHE_CLEANUP_SCAN_LIMIT or deleted >= _FILE_CACHE_CLEANUP_DELETE_LIMIT:
                        break
                    if not entry.name.startswith(prefix) or not entry.name.endswith(_FILE_CACHE_TEMP_SUFFIX):
                        continue
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mtime > cutoff:
                            continue
                        os.unlink(entry.path)
                        deleted += 1
                    except (FileNotFoundError, PermissionError):
                        continue
        except (FileNotFoundError, PermissionError):
            return

    def _fsync_directory(self) -> None:
        if os.name == "nt":
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(self._dir, flags)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EINVAL, errno.ENOTSUP}:
                return
            raise
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in {errno.EBADF, errno.EINVAL, errno.ENOTSUP}:
                raise
        finally:
            os.close(descriptor)


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
