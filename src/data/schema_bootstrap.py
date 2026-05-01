"""PostgreSQL schema bootstrap for Geoxiao runtime tables.

This module is intentionally PostgreSQL-only.  ClickHouse remains strictly
read-only and is never modified by this bootstrap.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import structlog

from src.data.repositories.base import postgres_sync_dsn

log = structlog.get_logger(__name__)

REQUIRED_TABLES = frozenset(
    {
        "strategy_dna",
        "trial_logs",
        "trade_logs",
        "monthly_metrics",
    }
)

MIGRATION_PATH = Path("src/data/migrations/001_initial_schema.sql")


class SchemaBootstrapError(RuntimeError):
    """Raised when the PostgreSQL schema cannot be safely bootstrapped."""


def get_existing_geoxiao_tables() -> set[str]:
    """Return existing required Geoxiao tables from PostgreSQL public schema."""

    query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = ANY(%s)
    """
    with psycopg.connect(postgres_sync_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (list(REQUIRED_TABLES),))
            return {row[0] for row in cur.fetchall()}


def ensure_postgres_schema(*, auto_apply: bool = True) -> bool:
    """Ensure PostgreSQL runtime tables exist.

    Args:
        auto_apply: If true, apply the initial migration only when none of the
            required Geoxiao tables exist. Partial schemas are not auto-mutated
            because that can hide manual drift or failed migrations.

    Returns:
        True when the migration was applied, False when schema already existed.

    Raises:
        SchemaBootstrapError: If schema is missing and cannot be safely applied.
    """

    existing = get_existing_geoxiao_tables()
    missing = REQUIRED_TABLES - existing

    if not missing:
        log.info("PostgreSQL Geoxiao schema verified", tables=sorted(REQUIRED_TABLES))
        return False

    if existing:
        raise SchemaBootstrapError(
            "Partial Geoxiao PostgreSQL schema detected; refusing automatic bootstrap. "
            f"Existing={sorted(existing)}, missing={sorted(missing)}. "
            "Run/repair migrations manually before starting evolution."
        )

    if not auto_apply:
        raise SchemaBootstrapError(
            "Geoxiao PostgreSQL schema is missing and auto_apply=False. "
            f"Missing={sorted(missing)}"
        )

    if not MIGRATION_PATH.exists():
        raise SchemaBootstrapError(f"Initial schema migration not found: {MIGRATION_PATH}")

    sql = MIGRATION_PATH.read_text(encoding="utf-8")
    with psycopg.connect(postgres_sync_dsn()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)

    existing_after = get_existing_geoxiao_tables()
    missing_after = REQUIRED_TABLES - existing_after
    if missing_after:
        raise SchemaBootstrapError(
            "Initial schema migration completed but required tables are still missing: "
            f"{sorted(missing_after)}"
        )

    log.info("PostgreSQL Geoxiao schema bootstrapped", tables=sorted(REQUIRED_TABLES))
    return True


__all__ = [
    "REQUIRED_TABLES",
    "SchemaBootstrapError",
    "ensure_postgres_schema",
    "get_existing_geoxiao_tables",
]
