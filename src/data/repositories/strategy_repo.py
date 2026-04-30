"""Repository for ``strategy_dna`` records."""

from __future__ import annotations

from typing import Any

import structlog

from src.data.repositories.base import (
    AsyncPostgresRepository,
    as_int,
    as_json,
    as_uuid,
    from_json,
)
from src.strategy.base_strategy import StrategyDNA

log = structlog.get_logger(__name__)


class StrategyRepository(AsyncPostgresRepository):
    """Persist and query generated strategy DNA in PostgreSQL."""

    VALID_STATUSES = {"pending", "backtesting", "passed", "eliminated", "archived"}

    _UPSERT_SQL = """
        INSERT INTO strategy_dna (
            id, generation, individual_id, tree_repr, tree_depth, tree_nodes,
            params_json, symbol, timeframe, status
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9,$10)
        ON CONFLICT (generation, individual_id)
        DO UPDATE SET
            tree_repr = EXCLUDED.tree_repr,
            tree_depth = EXCLUDED.tree_depth,
            tree_nodes = EXCLUDED.tree_nodes,
            params_json = EXCLUDED.params_json,
            symbol = EXCLUDED.symbol,
            timeframe = EXCLUDED.timeframe,
            status = EXCLUDED.status,
            updated_at = NOW()
        RETURNING id::text
    """

    async def save(self, dna: StrategyDNA) -> str:
        """Insert or update a strategy DNA record.

        Returns the persisted UUID string.  If an existing row is found by the
        schema's natural key ``(generation, individual_id)``, ``dna.id`` is
        updated to the existing row id so downstream trial/trade FKs stay valid.
        """

        if dna.status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid strategy status: {dna.status}")

        pool = await self.pool()
        async with pool.acquire() as conn:
            strategy_id = await conn.fetchval(
                self._UPSERT_SQL,
                as_uuid(dna.id),
                int(dna.generation),
                dna.individual_id,
                dna.tree_repr,
                as_int(dna.tree_depth),
                as_int(dna.tree_nodes),
                as_json(dna.params),
                dna.symbol,
                dna.timeframe,
                dna.status,
            )

        dna.id = str(strategy_id)
        log.info(
            "Strategy DNA saved",
            strategy_id=dna.id,
            generation=dna.generation,
            individual_id=dna.individual_id,
            status=dna.status,
        )
        return dna.id

    async def get(self, strategy_id: str) -> StrategyDNA | None:
        """Fetch a strategy by id."""

        pool = await self.pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id::text, generation, individual_id, tree_repr,
                       tree_depth, tree_nodes, params_json, symbol, timeframe,
                       status
                FROM strategy_dna
                WHERE id = $1
                """,
                as_uuid(strategy_id),
            )
        return self._row_to_dna(row) if row else None

    async def list_by_generation(self, generation: int) -> list[StrategyDNA]:
        """Return all strategies for a generation ordered by creation time."""

        pool = await self.pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id::text, generation, individual_id, tree_repr,
                       tree_depth, tree_nodes, params_json, symbol, timeframe,
                       status
                FROM strategy_dna
                WHERE generation = $1
                ORDER BY created_at ASC
                """,
                int(generation),
            )
        return [self._row_to_dna(row) for row in rows]

    async def update_status(self, strategy_id: str, status: str) -> None:
        """Update only the strategy lifecycle status."""

        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid strategy status: {status}")

        pool = await self.pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE strategy_dna
                SET status = $2, updated_at = NOW()
                WHERE id = $1
                """,
                as_uuid(strategy_id),
                status,
            )

        log.info("Strategy status updated", strategy_id=strategy_id, status=status, result=result)

    @staticmethod
    def _row_to_dna(row: Any) -> StrategyDNA:
        params = from_json(row["params_json"]) or {}
        return StrategyDNA(
            id=str(row["id"]),
            generation=int(row["generation"]),
            individual_id=row["individual_id"],
            tree_repr=row["tree_repr"],
            tree_depth=int(row["tree_depth"]),
            tree_nodes=int(row["tree_nodes"]),
            params=params,
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            status=row["status"],
        )


__all__ = ["StrategyRepository"]
