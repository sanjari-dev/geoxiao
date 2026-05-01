"""Repository for ``trade_logs`` records."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable

import structlog

from src.data.repositories.base import (
    AsyncPostgresRepository,
    as_date,
    as_decimal,
    as_uuid,
    require_keys,
)

log = structlog.get_logger(__name__)


class TradeRepository(AsyncPostgresRepository):
    """Persist individual trade logs produced by backtests."""

    VALID_SIDES = {"BUY", "SELL"}
    VALID_ORDER_TYPES = {"LIMIT", "STOP_LIMIT"}
    DEFAULT_CHUNK_SIZE = 1_000

    _INSERT_SQL = """
        INSERT INTO trade_logs (
            trial_id, strategy_id, symbol, side, order_type,
            entry_price, exit_price, sl_price, tp_price,
            entry_time, exit_time, raw_pips,
            spread_pips, slippage_pips, commission_pips,
            exit_reason, backtest_month
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
    """

    _REQUIRED = {
        "trial_id",
        "strategy_id",
        "symbol",
        "side",
        "order_type",
        "entry_price",
        "sl_price",
        "entry_time",
        "backtest_month",
    }

    async def insert(self, trade: dict[str, Any]) -> None:
        """Insert one trade row."""

        await self.batch_insert([trade])

    async def batch_insert(
        self,
        trades: Iterable[dict[str, Any]],
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> int:
        """Insert trade logs in chunks inside PostgreSQL transactions.

        ``net_pips`` is a generated column and is deliberately never inserted.
        Returns the number of inserted rows.
        """

        trade_list = list(trades)
        if not trade_list:
            return 0

        rows = [self._to_tuple(t) for t in trade_list]
        pool = await self.pool()

        inserted = 0
        async with pool.acquire() as conn:
            async with conn.transaction():
                for start in range(0, len(rows), chunk_size):
                    chunk = rows[start : start + chunk_size]
                    await conn.executemany(self._INSERT_SQL, chunk)
                    inserted += len(chunk)

        log.info("Trade logs inserted", count=inserted)
        return inserted

    async def list_by_trial(self, trial_id: str, *, limit: int = 1_000) -> list[dict[str, Any]]:
        """Fetch recent trade logs for a trial for diagnostics/tests."""

        pool = await self.pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id::text, trial_id::text, strategy_id::text, symbol, side,
                       order_type, entry_price, exit_price, sl_price, tp_price,
                       entry_time, exit_time, raw_pips, spread_pips,
                       slippage_pips, commission_pips, net_pips, exit_reason,
                       backtest_month, created_at
                FROM trade_logs
                WHERE trial_id = $1
                ORDER BY entry_time ASC
                LIMIT $2
                """,
                as_uuid(trial_id),
                int(limit),
            )
        return [dict(row) for row in rows]

    @classmethod
    def _to_tuple(cls, trade: dict[str, Any]) -> tuple:
        require_keys(trade, cls._REQUIRED, context="trade_logs")

        side = str(trade["side"]).upper()
        order_type = str(trade["order_type"]).upper()
        if side not in cls.VALID_SIDES:
            raise ValueError(f"Invalid trade side: {trade['side']}")
        if order_type not in cls.VALID_ORDER_TYPES:
            raise ValueError(f"Invalid order_type: {trade['order_type']}")

        entry_time = trade["entry_time"]
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))

        exit_time = trade.get("exit_time")
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))

        return (
            as_uuid(trade["trial_id"]),
            as_uuid(trade["strategy_id"]),
            trade["symbol"],
            side,
            order_type,
            as_decimal(trade["entry_price"]),
            as_decimal(trade.get("exit_price")),
            as_decimal(trade["sl_price"]),
            as_decimal(trade.get("tp_price")),
            entry_time,
            exit_time,
            as_decimal(trade.get("raw_pips")),
            as_decimal(trade.get("spread_pips", 0)),
            as_decimal(trade.get("slippage_pips", 0)),
            as_decimal(trade.get("commission_pips", 0)),
            trade.get("exit_reason"),
            as_date(trade["backtest_month"]),
        )


__all__ = ["TradeRepository"]
