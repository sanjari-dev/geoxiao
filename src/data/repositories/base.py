"""Shared PostgreSQL repository utilities.

All persistence in Geoxiao goes to PostgreSQL.  ClickHouse remains strictly
read-only and is intentionally not referenced from this package.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import asyncpg
import structlog

from src.config.settings import settings

log = structlog.get_logger(__name__)


def postgres_asyncpg_dsn() -> str:
    """Return a DSN accepted by asyncpg.

    The project exposes ``PG_DSN`` as a SQLAlchemy-style async URL
    (``postgresql+asyncpg://...``), while asyncpg itself expects
    ``postgresql://...``.  Prefer PG_DSN so a user only has to update one env
    var, and fall back to PG_DSN_SYNC for older local configs.
    """

    dsn = (getattr(settings, "PG_DSN", "") or "").strip()
    if not dsn:
        dsn = (getattr(settings, "PG_DSN_SYNC", "") or "").strip()

    if dsn.startswith("postgresql+asyncpg://"):
        return "postgresql://" + dsn[len("postgresql+asyncpg://") :]
    if dsn.startswith("postgres+asyncpg://"):
        return "postgres://" + dsn[len("postgres+asyncpg://") :]
    return dsn


def postgres_sync_dsn() -> str:
    """Return a plain ``postgresql://`` DSN for sync clients.

    SQLAlchemy/ConnectorX and asyncpg can all consume this normalized form for
    the connection strings used in this project.
    """

    return postgres_asyncpg_dsn()


def as_uuid(value: Any) -> uuid.UUID:
    """Normalize UUID-ish values for asyncpg UUID parameters."""

    if isinstance(value, uuid.UUID):
        return value
    if value is None:
        raise ValueError("UUID value cannot be None")
    return uuid.UUID(str(value))


def as_json(value: Any) -> str:
    """Encode JSONB values consistently for ``$n::jsonb`` parameters."""

    if value is None:
        value = {}
    return json.dumps(value, default=str, sort_keys=True)


def from_json(value: Any) -> Any:
    """Decode asyncpg JSON/JSONB values when they come back as strings."""

    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def as_decimal(value: Any) -> Decimal | None:
    """Convert floats/strings to Decimal for PostgreSQL NUMERIC columns."""

    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def require_keys(record: dict[str, Any], required: set[str], *, context: str) -> None:
    missing = sorted(key for key in required if record.get(key) is None)
    if missing:
        raise ValueError(f"{context} missing required keys: {', '.join(missing)}")


class AsyncPostgresRepository:
    """Small lazy asyncpg pool wrapper used by repository classes."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        self._dsn = dsn or postgres_asyncpg_dsn()
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool | None = None

    @property
    def dsn(self) -> str:
        return self._dsn

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        retryable_types = (
            ConnectionError,
            ConnectionResetError,
            TimeoutError,
            OSError,
            asyncio.TimeoutError,
            asyncpg.InterfaceError,
            asyncpg.PostgresConnectionError,
        )
        current: BaseException | None = exc
        while current is not None:
            if isinstance(current, retryable_types):
                return True
            current = current.__cause__ or current.__context__
        return False

    async def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=self._min_size,
                max_size=self._max_size,
            )
            log.info("PostgreSQL repository pool opened", repo=type(self).__name__)
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            log.info("PostgreSQL repository pool closed", repo=type(self).__name__)

    async def _run_with_retry(
        self,
        operation,
        *,
        context: str,
        attempts: int = 3,
    ):
        for attempt in range(1, attempts + 1):
            try:
                pool = await self.pool()
                async with pool.acquire() as conn:
                    return await operation(conn)
            except Exception as exc:
                if not self._is_retryable_exception(exc) or attempt >= attempts:
                    raise
                log.warning(
                    "PostgreSQL operation failed; resetting pool and retrying",
                    repo=type(self).__name__,
                    context=context,
                    attempt=attempt,
                    error=repr(exc),
                )
                await self.close()
                await asyncio.sleep(min(2 ** (attempt - 1), 5))

    async def __aenter__(self):
        await self.pool()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


__all__ = [
    "AsyncPostgresRepository",
    "as_date",
    "as_decimal",
    "as_int",
    "as_json",
    "as_uuid",
    "from_json",
    "postgres_asyncpg_dsn",
    "postgres_sync_dsn",
    "require_keys",
]


def as_date(value: Any) -> date:
    """Normalize DATE parameters from date/datetime/ISO strings."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        raise ValueError("date value cannot be None")
    return date.fromisoformat(str(value)[:10])
