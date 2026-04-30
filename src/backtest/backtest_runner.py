# src/backtest/backtest_runner.py — Bagian 1: Remote Worker

from __future__ import annotations
import asyncio
import uuid
from datetime import date

import ray
import polars as pl

from src.strategy.base_strategy import StrategyDNA
import structlog

log = structlog.get_logger(__name__)


@ray.remote
def _backtest_one_remote(
    dna_dict: dict,
    backtest_start: str,
    backtest_end: str,
) -> tuple[dict, list[dict]] | None:
    """
    Ray remote function untuk menjalankan satu backtest terisolasi.

    PENTING — Deferred Imports:
    Semua import dilakukan di dalam fungsi untuk menghindari
    serialization issue Ray dengan module-level objects.

    Args:
        dna_dict: StrategyDNA.to_db_dict() — JSON-serializable.
        backtest_start, backtest_end: ISO date strings 'YYYY-MM-DD'.

    Returns:
        tuple(dna_dict, list[trade_dict]) atau None jika gagal.
    """
    # ── Deferred imports (wajib di dalam @ray.remote) ────────────────
    from src.strategy.base_strategy import StrategyDNA
    from src.evolution.gp_generator import DEAPGenerator
    from src.backtest.nautilus_adapter import ClickHouseNautilusAdapter
    from src.config.settings import settings

    # NautilusTrader imports
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model.currencies import USD
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
    from nautilus_trader.model.objects import Money
    from nautilus_trader.test_kit.providers import TestInstrumentProvider

    import structlog
    log = structlog.get_logger('ray_worker')

    try:
        # ── 1. Rekonstruksi StrategyDNA dari dict ────────────────────
        dna = StrategyDNA(**{
            k: v for k, v in dna_dict.items()
            if k in StrategyDNA.__dataclass_fields__
        })
        dna.params = dna_dict.get('params_json', {})

        log.info('Worker: backtest start', dna_id=dna.id[:8],
                 tree_nodes=dna.tree_nodes)

        # ── 2. Generate strategy class dari pohon GP ──────────────────
        generator = DEAPGenerator(symbol=dna.symbol, timeframe=dna.timeframe)
        StrategyClass = generator.to_strategy_class(dna)
        strategy_instance = StrategyClass(dna)
        strategy_instance.validate_params()  # Raise jika params invalid

        # ── 3. Fetch market data dari ClickHouse ──────────────────────
        adapter = ClickHouseNautilusAdapter()
        instrument_id = InstrumentId(
            Symbol(dna.symbol.replace('/', '_')),
            Venue('SIM'),
        )
        raw_df, quote_ticks = adapter.fetch_and_convert(
            symbol=dna.symbol,
            start=backtest_start,
            end=backtest_end,
            instrument_id=instrument_id,
        )

        if len(quote_ticks) < 1000:
            log.warning('Worker: insufficient tick data', count=len(quote_ticks))
            return None

        # ── 4. Setup NautilusTrader BacktestEngine ────────────────────
        engine = BacktestEngine(
            config=BacktestEngineConfig(
                logging=LoggingConfig(log_level='ERROR'),  # Suppress NT logs
            )
        )

        # Instrumen dan akun
        instrument = TestInstrumentProvider.default_fx_ccy(dna.symbol)
        engine.add_instrument(instrument)
        engine.add_data(quote_ticks)

        engine.add_account(
            account_type=AccountType.MARGIN,
            base_currency=USD,
            starting_balances=[Money(100_000, USD)],
            oms_type=OmsType.HEDGING,
        )

        # ── 5. Register strategy sebagai NautilusTrader strategy actor ─
        # NautilusTrader menggunakan internal actor system.
        # GeneratedStrategy HARUS mengimplementasikan on_quote_tick() dan on_start().
        # Ini dilakukan via NautilusStrategyAdapter wrapper di bawah.
        from src.backtest.nautilus_strategy_wrapper import NautilusStrategyAdapter
        adapter_strategy = NautilusStrategyAdapter(
            strategy=strategy_instance,
            instrument_id=instrument_id,
            tick_buffer=raw_df,
        )
        engine.add_strategy(adapter_strategy)

        # ── 6. Jalankan backtest ──────────────────────────────────────
        engine.run()

        # ── 7. Ekstrak trade logs dari engine ─────────────────────────
        trade_logs = _extract_trade_logs(engine, dna)

        log.info('Worker: backtest complete',
                 dna_id=dna.id[:8], trades=len(trade_logs))

        return dna.to_db_dict(), trade_logs

    except Exception as e:
        import traceback
        log.error('Worker: backtest failed',
                  dna_id=dna_dict.get('id', '?')[:8],
                  error=str(e),
                  tb=traceback.format_exc()[-500:])
        return None


def _extract_trade_logs(
    engine,
    dna: 'StrategyDNA',
) -> list[dict]:
    """
    Ekstrak trade logs dari NautilusTrader engine setelah run() selesai.
    Converts fills/positions menjadi format trade_logs schema.
    """
    from nautilus_trader.model.enums import OrderSide
    from datetime import timezone
    import datetime

    trades = []
    orders = engine.trader.generate_order_fills_report()

    for _, row in orders.iterrows():
        try:
            entry_ts = row.get('last_modified_timestamp_utc', None)
            if entry_ts is None:
                continue

            entry_time = entry_ts if hasattr(entry_ts, 'tzinfo') else \
                         datetime.datetime.fromtimestamp(entry_ts/1e9, tz=timezone.utc)
            backtest_month = date(entry_time.year, entry_time.month, 1)

            side_val = row.get('side', '')
            side = 'BUY' if 'BUY' in str(side_val).upper() else 'SELL'

            entry_price = float(row.get('avg_px', 0))
            sl_pips = dna.params.get('sl_pips', 20.0)
            tp_pips = dna.params.get('tp_pips', 40.0)
            pip_val = 0.0001

            if side == 'BUY':
                sl_price = entry_price - (sl_pips * pip_val)
                tp_price = entry_price + (tp_pips * pip_val)
            else:
                sl_price = entry_price + (sl_pips * pip_val)
                tp_price = entry_price - (tp_pips * pip_val)

            trades.append({
                'trial_id': None,     # Di-set oleh RayBacktestRunner setelah INSERT trial
                'strategy_id': dna.id,
                'symbol': dna.symbol,
                'side': side,
                'order_type': 'LIMIT',
                'entry_price': entry_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'entry_time': entry_time,
                'exit_price': None,
                'exit_time': None,
                'raw_pips': None,
                'spread_pips': 1.5,     # Default untuk FX major
                'slippage_pips': 0.5,
                'commission_pips': 0.0,
                'exit_reason': 'SIMULATED',
                'backtest_month': backtest_month,
            })
        except Exception:
            continue

    return trades

# src/backtest/backtest_runner.py — Bagian 2: RayBacktestRunner class
# Tambahkan di bawah _backtest_one_remote() dan _extract_trade_logs()

class RayBacktestRunner:
    """
    Orkestrasi backtest seluruh populasi via Ray parallel execution.

    Pipeline per individu:
    1. OptunaTuner.tune() → update dna.params (numerical optimization)
    2. StrategyRepository.save(dna) → persist ke PostgreSQL
    3. TrialRepository.save(trial) → buat trial record
    4. _backtest_one_remote.remote(dna) → Ray task
    5. AsyncTradeLogSync: flush trade_logs ke PostgreSQL
    6. MetricsCalculator.compute_monthly_metrics() → polars DataFrame
    7. Return (dna, monthly_metrics) ke run_evolution.py
    """

    # Periode backtest default: 6 bulan
    DEFAULT_START = '2023-07-01'
    DEFAULT_END   = '2023-12-31'

    def __init__(
        self,
        backtest_start: str | None = None,
        backtest_end: str | None = None,
        optuna_n_trials: int = 30,
    ) -> None:
        from src.evolution.optuna_tuner import OptunaTuner
        from src.analytics.metrics_calculator import MetricsCalculator
        from src.data.repositories.strategy_repo import StrategyRepository
        from src.data.repositories.trial_repo import TrialRepository
        from src.data.repositories.trade_repo import TradeRepository
        from src.backtest.nautilus_adapter import ClickHouseNautilusAdapter
        from src.config.settings import settings

        self.backtest_start = backtest_start or self.DEFAULT_START
        self.backtest_end = backtest_end or self.DEFAULT_END
        self.optuna_n_trials = optuna_n_trials

        # Komponen — diinisialisasi sekali, digunakan untuk semua individu
        self.metrics_calc = MetricsCalculator()
        self.strategy_repo = StrategyRepository()
        self.trial_repo = TrialRepository()
        self.trade_repo = TradeRepository()
        self.adapter = ClickHouseNautilusAdapter()

        # Pre-load sample tick data untuk Optuna proxy evaluation
        # Gunakan 30 hari pertama dari periode backtest
        self._sample_tick_data: pl.DataFrame | None = None

        log.info('RayBacktestRunner initialized',
                 start=self.backtest_start, end=self.backtest_end)

    def _get_sample_data(self, symbol: str) -> pl.DataFrame:
        """Load sample tick data untuk Optuna proxy — cached per symbol."""
        if self._sample_tick_data is None:
            # Sample: 7 hari pertama
            sample_end = self.backtest_start[:8]  # Adjust jika perlu
            try:
                self._sample_tick_data = self.adapter.fetch_tick_data(
                    symbol=symbol,
                    start=self.backtest_start,
                    end=self.backtest_start[:7] + '-07',  # +7 hari
                )
            except Exception as e:
                log.warning('Sample data fetch failed', error=str(e))
                self._sample_tick_data = pl.DataFrame()
        return self._sample_tick_data

    async def evaluate_population(
        self,
        population: list[StrategyDNA],
        generation: int,
    ) -> list[tuple[StrategyDNA, pl.DataFrame]]:
        """
        Evaluasi seluruh populasi secara paralel.

        Returns:
            list[(StrategyDNA, monthly_metrics_df)]
            Individu yang gagal backtest TIDAK dimasukkan ke list.
        """
        from src.evolution.gp_generator import DEAPGenerator
        from src.evolution.optuna_tuner import OptunaTuner

        log.info('Evaluating population', generation=generation, size=len(population))

        # ── Step 1: Optuna tuning (sequential, sebelum parallel backtest) ──
        # Optuna study menggunakan shared PostgreSQL storage,
        # sehingga trials dari individu berbeda tidak bentrok.
        tuned_population = []
        for dna in population:
            try:
                tuner = OptunaTuner(
                    study_name=f'gen{generation}',
                    n_trials=self.optuna_n_trials,
                )
                generator = DEAPGenerator(symbol=dna.symbol)
                sample_data = self._get_sample_data(dna.symbol)

                if not sample_data.is_empty():
                    dna = tuner.tune(
                        dna=dna,
                        tick_data=sample_data,
                        strategy_class_factory=generator.to_strategy_class,
                    )
                tuned_population.append(dna)
            except Exception as e:
                log.warning('Optuna tuning failed — using default params',
                            dna_id=dna.id[:8], error=str(e))
                tuned_population.append(dna)

        # ── Step 2: Persist strategy_dna ke PostgreSQL ───────────────────
        for dna in tuned_population:
            await self.strategy_repo.save(dna)

        # ── Step 3: Dispatch Ray parallel backtest tasks ─────────────────
        ray_futures = [
            _backtest_one_remote.remote(
                dna.to_db_dict(),
                self.backtest_start,
                self.backtest_end,
            )
            for dna in tuned_population
        ]

        log.info('Ray tasks dispatched', count=len(ray_futures))

        # ray.get() — tunggu semua tasks selesai
        # timeout: 10 menit per task
        raw_results = ray.get(ray_futures, timeout=600)

        # ── Step 4: Process results ───────────────────────────────────────
        final_results: list[tuple[StrategyDNA, pl.DataFrame]] = []

        for dna, result in zip(tuned_population, raw_results):
            if result is None:
                log.warning('Backtest returned None', dna_id=dna.id[:8])
                continue

            returned_dna_dict, trade_logs = result

            # ── Step 5: Buat trial record ─────────────────────────────
            trial_id = str(uuid.uuid4())
            trial_record = {
                'id': trial_id,
                'strategy_id': dna.id,
                'optuna_trial_id': -1,  # -1 = direct backtest, bukan Optuna trial
                'study_name': f'gen{generation}_{dna.id[:8]}',
                'params_json': dna.params,
            }
            await self.trial_repo.save(trial_record)

            # ── Step 6: Bulk insert trade_logs ────────────────────────
            if trade_logs:
                for t in trade_logs:
                    t['trial_id'] = trial_id  # Set trial_id setelah INSERT
                await self.trade_repo.batch_insert(trade_logs)

            # ── Step 7: Compute monthly metrics via Polars ────────────
            monthly_metrics = self.metrics_calc.compute_monthly_metrics(trial_id)

            if monthly_metrics.is_empty():
                log.warning('Empty monthly metrics', dna_id=dna.id[:8])
                continue

            final_results.append((dna, monthly_metrics))

        log.info('Population evaluation complete',
                 generation=generation,
                 total=len(population),
                 successful=len(final_results),
                 failed=len(population) - len(final_results))

        return final_results
