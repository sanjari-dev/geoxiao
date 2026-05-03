# src/evolution/optuna_tuner.py
# Optuna TPE Parameter Optimizer

from __future__ import annotations
import optuna
import polars as pl
import numpy as np
from typing import Callable
from dataclasses import dataclass

from src.strategy.base_strategy import StrategyDNA
from src.strategy.signal_utils import clamp_signal_threshold, MIN_SIGNAL_THRESHOLD, MAX_SIGNAL_THRESHOLD
from src.config.settings import settings
import structlog

log = structlog.get_logger(__name__)

# Suppress Optuna verbosity — gunakan structlog
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class SignalDiagnostics:
    threshold: float
    windows_evaluated: int
    entry_candidates: int
    min_signal: float
    max_signal: float
    error: str | None = None

    @property
    def viable(self) -> bool:
        return self.error is not None or self.entry_candidates > 0


class OptunaTuner:
    """
    Bayesian hyperparameter optimizer untuk parameter numerik DNA.

    Alur kerja:
    1. Menerima StrategyDNA + sample tick data
    2. Optuna (TPE) menyarankan parameter numerik
    3. Objective function: jalankan proxy signal evaluation
    4. Update dna.params dengan best_params
    5. Return updated DNA — siap untuk full backtest
    """

    def __init__(
        self,
        study_name: str,
        n_trials: int = 30,
        n_jobs: int = 1,
    ) -> None:
        self.study_name = study_name
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        log.info('OptunaTuner initialized',
                 study=study_name, n_trials=n_trials)

    def _get_or_create_study(self, dna_id: str) -> optuna.Study:
        """
        Buat atau load Optuna study dari PostgreSQL storage.
        Setiap DNA memiliki study uniknya sendiri.
        """
        # Sanitasi: pastikan nama study tidak mengandung karakter illegal
        safe_dna_id = dna_id.replace('-', '_')[:8]
        study_name = f'{self.study_name}_{safe_dna_id}'
        
        try:
            return optuna.create_study(
                study_name=study_name,
                storage=settings.OPTUNA_STORAGE,
                direction='maximize',
                sampler=optuna.samplers.TPESampler(
                    seed=42,
                    n_startup_trials=10,  # Random exploration dulu
                ),
                load_if_exists=True,  # Resume jika study sudah ada
            )
        except Exception as e:
            log.warning('Optuna PostgreSQL storage unavailable — using in-memory storage',
                        storage=settings.OPTUNA_STORAGE, error=str(e))
            return optuna.create_study(
                study_name=study_name,
                direction='maximize',
                sampler=optuna.samplers.TPESampler(
                    seed=42,
                    n_startup_trials=10,
                ),
            )

    def tune(
        self,
        dna: StrategyDNA,
        tick_data: pl.DataFrame,
        strategy_class_factory: Callable,
    ) -> StrategyDNA:
        """
        Jalankan Optuna optimization untuk satu DNA.

        Args:
            dna: StrategyDNA dengan tree_repr yang sudah di-set GP.
            tick_data: polars DataFrame tick data (sample, bukan full period).
            strategy_class_factory: callable(dna) → type[BaseStrategy].
                                    Biasanya DEAPGenerator.to_strategy_class.

        Returns:
            StrategyDNA dengan dna.params ter-update dengan best_params.
        """
        study = self._get_or_create_study(dna.id)

        def objective(trial: optuna.Trial) -> float:
            """Proxy objective: evaluasi kualitas parameter tanpa full backtest."""
            params = self._suggest_params(trial)
            return self._proxy_evaluate(dna, params, tick_data, strategy_class_factory)

        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                n_jobs=self.n_jobs,
                show_progress_bar=False,
                catch=(Exception,),  # Tangkap exception per-trial, jangan abort
            )
        except Exception as e:
            log.error('Optuna study failed', dna_id=dna.id, error=str(e))
            return dna  # Return DNA tanpa update jika study gagal total

        best = study.best_params if study.best_params else {}
        dna.params.update(best)

        log.info('Optuna tuning complete',
                 dna_id=dna.id[:8],
                 best_value=study.best_value,
                 n_trials=len(study.trials),
                 sl_pips=best.get('sl_pips'),
                 tp_pips=best.get('tp_pips'))

        return dna

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """
        Definisikan search space untuk semua 13 parameter numerik.
        Semua range-nya merujuk pada tabel di dokumen ini.
        """
        sl_pips = trial.suggest_float('sl_pips', 10.0, 50.0)

        return {
            'sl_pips':             sl_pips,
            # TP minimum 1.5x SL untuk ensure positive expectancy
            'tp_pips':             trial.suggest_float('tp_pips', sl_pips * 1.5, 150.0),
            'signal_threshold':    trial.suggest_float(
                'signal_threshold',
                MIN_SIGNAL_THRESHOLD,
                MAX_SIGNAL_THRESHOLD,
            ),
            'obi_window':          trial.suggest_int('obi_window', 10, 50),
            'tick_vel_window':     trial.suggest_int('tick_vel_window', 5, 30),
            'spread_dyn_window':   trial.suggest_int('spread_dyn_window', 10, 60),
            'tick_den_window_sec': trial.suggest_float('tick_den_window_sec', 30.0, 300.0),
            'vol_skew_window':     trial.suggest_int('vol_skew_window', 10, 60),
            'mid_mom_short':       trial.suggest_int('mid_mom_short', 3, 15),
            'mid_mom_long':        trial.suggest_int('mid_mom_long', 10, 60),
            'skew_window':         trial.suggest_int('skew_window', 15, 60),
            'kurt_window':         trial.suggest_int('kurt_window', 15, 60),
            'vw_spread_window':    trial.suggest_int('vw_spread_window', 10, 50),
        }

    def _proxy_evaluate(
        self,
        dna: StrategyDNA,
        params: dict,
        tick_data: pl.DataFrame,
        strategy_class_factory: Callable,
    ) -> float:
        """
        Proxy evaluation — jalankan strategi pada sample tick data
        tanpa full vectorized backtest (lightweight, cepat).

        Tujuan: ranking parameter set, bukan menghasilkan P&L akurat.
        Menggunakan subset kecil dari tick data (max 50k baris).

        Returns:
            float: proxy score (net_pips estimasi). Semakin tinggi semakin baik.
        """
        try:
            # Update params sementara ke dna
            dna_copy = type(dna)(**{
                f.name: getattr(dna, f.name)
                for f in dna.__dataclass_fields__.values()  # type: ignore
            })
            dna_copy.params = {**dna.params, **params}

            # Instansiasi strategy class dengan params baru
            StrategyClass = strategy_class_factory(dna_copy)
            strategy = StrategyClass(dna_copy)
            strategy.validate_params()

            # Sample max 50k baris untuk kecepatan
            sample = self._prepare_tick_data(tick_data.head(50_000))
            if len(sample) < 100:
                return -9999.0

            sl_pips = params['sl_pips']
            tp_pips = params['tp_pips']
            params['signal_threshold'] = clamp_signal_threshold(params.get('signal_threshold'))

            # Sliding window evaluation
            min_window = max(params.get('mid_mom_long', 20),
                             params.get('obi_window', 20)) + 10

            trades: list[float] = []
            step = max(1, min_window // 4)  # Stride untuk kecepatan

            for i in range(min_window, len(sample) - 1, step):
                window = sample.slice(max(0, i - min_window * 2), min_window * 2)
                features = strategy.compute_features(window)
                signal = strategy.generate_signal(features)

                if signal is None:
                    continue

                # Simulasi sederhana: jika sinyal ada, ambil net_pips acak tapi
                # diarahkan oleh kualitas sinyal
                # Ini hanya proxy — akurasi bukan tujuan
                side = 1.0 if signal['side'] == 'BUY' else -1.0
                # Gunakan mid-price change di tick berikutnya sebagai outcome
                current_mid = (sample['bid'][i] + sample['ask'][i]) / 2
                next_mid = (sample['bid'][i+1] + sample['ask'][i+1]) / 2
                price_move = (next_mid - current_mid) * 10000 * side

                # Estimasi outcome berdasarkan arah
                if price_move > 0:
                    trades.append(min(price_move, tp_pips))
                else:
                    trades.append(max(price_move, -sl_pips))

            if not trades:
                return -9999.0

            net_pips = float(np.sum(trades))
            trade_count = len(trades)

            # Penalti jika trade count sangat rendah atau sangat tinggi
            # (proxy untuk Hard Constraint freq 10-20/bulan)
            # Asumsikan sample = 1 bulan data
            freq_penalty = 0.0
            if trade_count < 10:
                freq_penalty = (10 - trade_count) * sl_pips
            elif trade_count > 20:
                freq_penalty = (trade_count - 20) * sl_pips * 0.5

            return net_pips - freq_penalty

        except Exception as e:
            log.debug('Proxy evaluation failed', error=str(e))
            return -9999.0  # Worst possible score — Optuna akan menghindari params ini

    def analyze_signal_diagnostics(
        self,
        dna: StrategyDNA,
        tick_data: pl.DataFrame,
        strategy_class_factory: Callable,
    ) -> SignalDiagnostics:
        """Estimate whether a DNA can produce any actionable signal at all."""

        params = dict(dna.params or {})
        threshold = clamp_signal_threshold(params.get('signal_threshold'))

        try:
            dna_copy = type(dna)(**{
                f.name: getattr(dna, f.name)
                for f in dna.__dataclass_fields__.values()  # type: ignore
            })
            dna_copy.params = {**params, 'signal_threshold': threshold}
            StrategyClass = strategy_class_factory(dna_copy)
            strategy = StrategyClass(dna_copy)
            strategy.validate_params()
        except Exception as e:
            log.warning('Signal diagnostics setup failed', dna_id=dna.id[:8], error=str(e))
            return SignalDiagnostics(
                threshold=threshold,
                windows_evaluated=0,
                entry_candidates=0,
                min_signal=0.0,
                max_signal=0.0,
                error=str(e),
            )

        sample = self._prepare_tick_data(tick_data.head(50_000))
        if len(sample) < 100:
            return SignalDiagnostics(
                threshold=threshold,
                windows_evaluated=0,
                entry_candidates=0,
                min_signal=0.0,
                max_signal=0.0,
                error='insufficient_sample',
            )

        min_window = max(params.get('mid_mom_long', 20), params.get('obi_window', 20)) + 10
        step = max(1, min_window // 4)
        min_signal = float('inf')
        max_signal = float('-inf')
        entry_candidates = 0
        windows_evaluated = 0

        try:
            for i in range(min_window, len(sample) - 1, step):
                window = sample.slice(max(0, i - min_window * 2), min_window * 2)
                features = strategy.compute_features(window)
                signal_value = float(strategy.compute_signal_value(features))
                min_signal = min(min_signal, signal_value)
                max_signal = max(max_signal, signal_value)
                windows_evaluated += 1
                if signal_value > threshold or signal_value < -threshold:
                    entry_candidates += 1
        except Exception as e:
            log.warning(
                'Signal diagnostics evaluation failed',
                dna_id=dna.id[:8],
                error=str(e),
            )
            return SignalDiagnostics(
                threshold=threshold,
                windows_evaluated=windows_evaluated,
                entry_candidates=0,
                min_signal=0.0 if min_signal == float('inf') else min_signal,
                max_signal=0.0 if max_signal == float('-inf') else max_signal,
                error=str(e),
            )

        if windows_evaluated == 0:
            min_signal = 0.0
            max_signal = 0.0

        return SignalDiagnostics(
            threshold=threshold,
            windows_evaluated=windows_evaluated,
            entry_candidates=entry_candidates,
            min_signal=min_signal,
            max_signal=max_signal,
        )

    @staticmethod
    def _prepare_tick_data(tick_data: pl.DataFrame) -> pl.DataFrame:
        """Normalize ClickHouse-decimal samples into float arrays for NumPy code."""

        if tick_data.is_empty():
            return tick_data

        cast_exprs = []
        for col in ('bid', 'ask', 'bid_size', 'ask_size'):
            if col in tick_data.columns:
                cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False).alias(col))

        if not cast_exprs:
            return tick_data

        return tick_data.with_columns(cast_exprs)
