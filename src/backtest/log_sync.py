# src/backtest/log_sync.py
# Referensi: Blueprint §4.2

from __future__ import annotations
import asyncio
from collections import deque
from datetime import date
import asyncpg
from src.data.repositories.base import postgres_asyncpg_dsn
import structlog

log = structlog.get_logger(__name__)


class AsyncTradeLogSync:
    """
    Buffer trade logs dan flush ke PostgreSQL secara batch async.

    Pattern penggunaan:
        sync = AsyncTradeLogSync()
        await sync.start()          # Panggil di awal session
        sync.enqueue(trade_dict)    # Di-call dari backtest (sync)
        await sync.stop()           # Final flush + close pool
    """

    BATCH_SIZE: int = 500
    FLUSH_INTERVAL_SEC: float = 5.0

    # SQL INSERT — net_pips tidak di-insert (GENERATED COLUMN)
    _INSERT_SQL = '''
        INSERT INTO trade_logs (
            trial_id, strategy_id, symbol, side, order_type,
            entry_price, sl_price, tp_price, entry_time,
            exit_price, exit_time, raw_pips,
            spread_pips, slippage_pips, commission_pips,
            exit_reason, backtest_month
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
    '''

    def __init__(self) -> None:
        self._buffer: deque[dict] = deque()
        self._pool: asyncpg.Pool | None = None
        self._flush_task: asyncio.Task | None = None
        self._total_flushed: int = 0

    async def start(self) -> None:
        """Buat connection pool dan mulai periodic flush task."""
        self._pool = await asyncpg.create_pool(
            dsn=postgres_asyncpg_dsn(),
            min_size=2,
            max_size=10,
        )
        self._flush_task = asyncio.create_task(self._periodic_flush())
        log.info('AsyncTradeLogSync started')

    async def stop(self) -> None:
        """Hentikan periodic flush, lakukan final flush, tutup pool."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush_now()   # WAJIB — flush sisa buffer
        if self._pool:
            await self._pool.close()
        log.info('AsyncTradeLogSync stopped', total_flushed=self._total_flushed)

    def enqueue(self, trade: dict) -> None:
        """
        Non-blocking append ke buffer.
        Dipanggil dari backtest event handler (synchronous context).

        trade dict HARUS berisi key:
        trial_id, strategy_id, symbol, side, order_type,
        entry_price, sl_price, tp_price, entry_time,
        exit_price, exit_time, raw_pips, spread_pips,
        slippage_pips, commission_pips, exit_reason, backtest_month
        """
        self._buffer.append(trade)

    async def _periodic_flush(self) -> None:
        """Flush buffer setiap FLUSH_INTERVAL_SEC detik."""
        while True:
            await asyncio.sleep(self.FLUSH_INTERVAL_SEC)
            await self._flush_now()

    async def _flush_now(self) -> None:
        """Ambil batch dari buffer dan INSERT ke PostgreSQL."""
        if not self._buffer:
            return

        batch: list[dict] = []
        while self._buffer and len(batch) < self.BATCH_SIZE:
            batch.append(self._buffer.popleft())

        if not batch:
            return

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    self._INSERT_SQL,
                    [self._to_tuple(t) for t in batch],
                )
            self._total_flushed += len(batch)
            log.info('Trade logs flushed', count=len(batch),
                     total=self._total_flushed)
        except Exception as e:
            log.error('Flush failed — returning to buffer', error=str(e))
            # Kembalikan batch ke buffer untuk retry
            self._buffer.extendleft(reversed(batch))

    @staticmethod
    def _to_tuple(t: dict) -> tuple:
        """Konversi trade dict ke ordered tuple untuk asyncpg executemany."""
        return (
            t['trial_id'],
            t['strategy_id'],
            t['symbol'],
            t['side'],
            t['order_type'],
            t['entry_price'],
            t['sl_price'],
            t['tp_price'],
            t['entry_time'],
            t.get('exit_price'),
            t.get('exit_time'),
            t.get('raw_pips'),
            t.get('spread_pips', 0),
            t.get('slippage_pips', 0),
            t.get('commission_pips', 0),
            t.get('exit_reason'),
            t['backtest_month'],  # date object
        )
