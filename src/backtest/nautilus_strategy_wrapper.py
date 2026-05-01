# src/backtest/nautilus_strategy_wrapper.py
# Bridge antara BaseStrategy (Geoxiao) dan NautilusTrader Strategy actor

from __future__ import annotations
import polars as pl
from collections import deque
from datetime import timedelta

from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Price, Quantity

from src.strategy.base_strategy import BaseStrategy


class NautilusAdapterConfig(StrategyConfig, frozen=True):
    """Config untuk NautilusStrategyAdapter — harus frozen=True."""
    instrument_id_str: str = 'EURUSD_USD.SIM'
    max_tick_buffer: int = 200


class NautilusStrategyAdapter(Strategy):
    """
    Adapter yang membungkus BaseStrategy menjadi NautilusTrader Strategy.

    Lifecycle NautilusTrader:
    1. on_start() — subscribe ke data
    2. on_quote_tick() — dipanggil setiap tick
    3. on_stop() — cleanup
    """

    TICK_BUFFER_SIZE = 200

    def __init__(
        self,
        strategy: BaseStrategy,
        instrument_id: InstrumentId,
        tick_buffer: pl.DataFrame,
        config: NautilusAdapterConfig | None = None,
    ) -> None:
        super().__init__(config or NautilusAdapterConfig(
            instrument_id_str=str(instrument_id)))
        self._strategy = strategy
        self._instrument_id = instrument_id
        self._tick_buf_df = tick_buffer  # Full tick data sebagai polars DF
        self._live_ticks: deque = deque(maxlen=self.TICK_BUFFER_SIZE)
        self._position_open = False

    def on_start(self) -> None:
        """Subscribe ke QuoteTick data."""
        self.subscribe_quote_ticks(self._instrument_id)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        """
        Handler untuk setiap QuoteTick yang masuk dari engine.
        Buffer ticks, hitung features, generate signal, submit order.
        """
        self._live_ticks.append({
            'timestamp': tick.ts_event,
            'bid': float(tick.bid_price),
            'ask': float(tick.ask_price),
            'bid_size': float(tick.bid_size),
            'ask_size': float(tick.ask_size),
        })

        if len(self._live_ticks) < self.TICK_BUFFER_SIZE:
            return  # Tunggu buffer penuh

        if self._position_open:
            return  # Satu posisi sekaligus

        # Konversi buffer ke polars DataFrame
        buf_df = pl.DataFrame(list(self._live_ticks))
        buf_df = buf_df.with_columns(
            pl.from_epoch('timestamp', time_unit='ns').alias('timestamp')
        )

        try:
            features = self._strategy.compute_features(buf_df)
            signal = self._strategy.generate_signal(features)
        except Exception:
            return

        if signal is None:
            return

        self._submit_order(tick, signal)

    def _submit_order(
        self,
        tick: QuoteTick,
        signal: dict,
    ) -> None:
        """Submit LIMIT order ke NautilusTrader engine."""
        instrument = self.cache.instrument(self._instrument_id)
        if instrument is None:
            return

        side = OrderSide.BUY if signal['side'] == 'BUY' else OrderSide.SELL
        pip = instrument.price_increment  # e.g., 0.0001 untuk FX

        # LIMIT order: entry sedikit di atas/bawah current price
        offset = float(pip) * 2
        if side == OrderSide.BUY:
            price = float(tick.ask_price) + offset
        else:
            price = float(tick.bid_price) - offset

        order = self.order_factory.limit(
            instrument_id=self._instrument_id,
            order_side=side,
            quantity=instrument.make_qty(100_000),  # 1 lot standard
            price=instrument.make_price(price),
            time_in_force=TimeInForce.GTD,
            expire_time=self.clock.utc_now() + timedelta(hours=1),  # GTD 1 jam
        )
        self.submit_order(order)
        self._position_open = True

    def on_stop(self) -> None:
        """Flatten semua posisi terbuka saat backtest selesai."""
        self.cancel_all_orders(self._instrument_id)
        self.close_all_positions(self._instrument_id)
