# src/backtest/nautilus_adapter.py
# Referensi: Blueprint §4.1

from __future__ import annotations
import polars as pl
import clickhouse_connect
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from src.config.settings import settings

from src.data.retry import db_retry

import structlog

log = structlog.get_logger(__name__)


class ClickHouseNautilusAdapter:
    """
    Bridge antara ClickHouse market data store dan NautilusTrader.

    CATATAN PENTING:
    - Satu instance per backtest session (tidak thread-safe untuk multi-symbol).
    - fetch_tick_data() melakukan satu query besar per backtest period.
    - Data di-cache sebagai polars DataFrame sebelum konversi ke QuoteTick.
    """

    # Skema yang diharapkan dari ClickHouse
    EXPECTED_COLUMNS = {'timestamp', 'bid', 'ask', 'bid_size', 'ask_size'}

    def __init__(self) -> None:
        self._client = clickhouse_connect.get_client(
            host=settings.CH_HOST,
            port=settings.CH_PORT,
            database=settings.CH_DATABASE,
            username=settings.CH_USER,
            password=settings.CH_PASSWORD,
            connect_timeout=10,
            send_receive_timeout=300,
        )
        log.info('ClickHouse connection established',
                 host=settings.CH_HOST, database=settings.CH_DATABASE)

    @db_retry
    def fetch_tick_data(
        self,
        symbol: str,
        start: str,   # Format: 'YYYY-MM-DD HH:MM:SS'
        end: str,
    ) -> pl.DataFrame:
        """
        Query tick data dari ClickHouse secara READ-ONLY.
        Disesuaikan dengan skema tabel `ticks` milik sistem ingestion eksternal.
        """
        # Menggunakan ALIAS agar nama kolom sesuai dengan ekspektasi Geoxiao
        query = f"""
            SELECT 
                time AS timestamp, 
                bid, 
                ask, 
                bid_volume AS bid_size, 
                ask_volume AS ask_size
            FROM ticks
            WHERE instrument = '{symbol}'
              AND time BETWEEN '{start}' AND '{end}'
            ORDER BY time ASC
        """
        log.info('Fetching tick data (READ-ONLY)', symbol=symbol, start=start, end=end)

        pandas_df = self._client.query_df(query)
        df = pl.from_pandas(pandas_df)  # Konversi ke polars SEGERA

        # Validasi skema internal Geoxiao
        missing = self.EXPECTED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f'Missing columns dari ClickHouse setelah mapping: {missing}')

        log.info('Tick data fetched', symbol=symbol, rows=len(df))
        return df

    def to_nautilus_quote_ticks(
        self,
        df: pl.DataFrame,
        instrument_id: InstrumentId,
        price_precision: int = 5,
        size_precision: int = 0,
    ) -> list[QuoteTick]:
        """
        Konversi polars DataFrame ke list[QuoteTick] NautilusTrader.

        PENTING: Konversi timestamp ke nanoseconds (int).
        NautilusTrader menggunakan UNIX nanoseconds untuk semua timestamps.
        """
        ticks: list[QuoteTick] = []

        for row in df.iter_rows(named=True):
            ts_ns = int(row['timestamp'].timestamp() * 1_000_000_000)
            tick = QuoteTick(
                instrument_id=instrument_id,
                bid_price=Price(row['bid'], price_precision),
                ask_price=Price(row['ask'], price_precision),
                bid_size=Quantity(row['bid_size'], size_precision),
                ask_size=Quantity(row['ask_size'], size_precision),
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
            ticks.append(tick)

        log.info('QuoteTick conversion complete',
                 instrument=str(instrument_id), count=len(ticks))
        return ticks

    def fetch_and_convert(
        self,
        symbol: str,
        start: str,
        end: str,
        instrument_id: InstrumentId,
    ) -> tuple[pl.DataFrame, list[QuoteTick]]:
        """
        Convenience method: fetch + convert dalam satu panggilan.
        Return tuple (raw_df, ticks) untuk debugging dan backtest.
        """
        df = self.fetch_tick_data(symbol, start, end)
        ticks = self.to_nautilus_quote_ticks(df, instrument_id)
        return df, ticks
