"""Lazy, owned or borrowed asyncpg pools for agent persistence."""

from __future__ import annotations

import asyncio
from typing import Any, Self

from .errors import ValidationError


class _ConnectionLease:
    """Keep the existing private connection hook while releasing pool leases."""

    def __init__(self, pool: Any, connection: Any) -> None:
        self._pool = pool
        self._connection = connection
        self._released = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    async def close(self) -> None:
        if not self._released:
            self._released = True
            await self._pool.release(self._connection)


class PostgresStoreBackend:
    """Internal backend; a supplied pool remains owned by the caller."""

    def _configure_pool(
        self,
        dsn: str | None,
        *,
        pool: Any | None,
        pool_min_size: int,
        pool_max_size: int,
    ) -> None:
        if pool is None and (not isinstance(dsn, str) or not dsn.strip()):
            raise ValidationError(
                'Postgres storage requires a non-empty "dsn" or an asyncpg pool.'
            )
        if (
            isinstance(pool_min_size, bool)
            or isinstance(pool_max_size, bool)
            or not isinstance(pool_min_size, int)
            or not isinstance(pool_max_size, int)
            or not 0 <= pool_min_size <= pool_max_size
            or pool_max_size < 1
        ):
            raise ValidationError(
                "Postgres pool sizes require integers with 0 <= pool_min_size <= pool_max_size and pool_max_size >= 1."
            )
        self._dsn = dsn
        self._pool = pool
        self._owns_pool = pool is None
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool_lock = asyncio.Lock()
        self._schema_lock = asyncio.Lock()
        self._schema_ready = False
        self._closed = False

    async def _get_pool(self) -> Any:
        async with self._pool_lock:
            if self._closed:
                raise RuntimeError("Postgres store is closed.")
            if self._pool is None:
                try:
                    import asyncpg  # type: ignore[import-not-found,import-untyped]
                except ImportError as error:
                    raise RuntimeError(
                        'Postgres support requires the optional dependency "asyncpg".'
                    ) from error
                self._pool = await asyncpg.create_pool(
                    dsn=self._dsn,
                    min_size=self._pool_min_size,
                    max_size=self._pool_max_size,
                )
            return self._pool

    async def _ensure_schema(self, connection: Any) -> None:
        raise NotImplementedError

    async def _connect(self) -> Any:
        pool = await self._get_pool()
        connection = _ConnectionLease(pool, await pool.acquire())
        try:
            if not self._schema_ready:
                async with self._schema_lock:
                    if not self._schema_ready:
                        await self._ensure_schema(connection)
                        self._schema_ready = True
        except BaseException:
            await connection.close()
            raise
        return connection

    async def initialize(self) -> None:
        """Create/migrate the schema once, before serving requests if desired."""
        connection = await self._connect()
        await connection.close()

    async def close(self) -> None:
        """Close the store and its owned pool; borrowed pools remain available."""
        async with self._pool_lock:
            if self._closed:
                return
            self._closed = True
            if self._owns_pool and self._pool is not None:
                await self._pool.close()

    async def __aenter__(self) -> Self:
        try:
            await self.initialize()
        except BaseException:
            await self.close()
            raise
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()
