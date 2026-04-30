# src/evolution/fitness.py
# Referensi: Blueprint §5.1

from __future__ import annotations
from dataclasses import dataclass
import polars as pl
import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ConstraintConfig:
    """Konfigurasi hard constraints. Immutable setelah inisialisasi."""
    min_trades_per_month: int = 10
    max_trades_per_month: int = 20
    min_risk_pips: float = 10.0
    max_risk_pips: float = 50.0
    max_monthly_drawdown_pips: float = 500.0
    min_profit_factor: float = 2.0
    pf_penalty_weight: float = 0.5


class HardConstraintEvaluator:
    """
    Evaluasi semua hard constraints secara independen.

    PENTING: Semua checks HARUS berjalan — jangan short-circuit.
    Ini memungkinkan logging lengkap alasan eliminasi untuk analisis.
    """

    def __init__(self, config: ConstraintConfig | None = None) -> None:
        self.config = config or ConstraintConfig()

    def evaluate(
        self,
        monthly_metrics: pl.DataFrame,
    ) -> dict:
        """
        Evaluasi semua hard constraints.

        Args:
            monthly_metrics: Output dari MetricsCalculator.compute_monthly_metrics()

        Returns:
            {
                'passed': bool,
                'flags': list[str],   # Semua alasan eliminasi
                'summary': dict       # Data untuk logging
            }
        """
        if monthly_metrics.is_empty():
            return {
                'passed': False,
                'flags': ['NO_TRADES: backtest menghasilkan zero trades'],
                'summary': {},
            }

        flags: list[str] = []
        cfg = self.config

        # ── CHECK 1: Trade Frequency (rata-rata bulanan) ──────────────
        avg_trades = monthly_metrics['trade_count'].mean()
        if avg_trades < cfg.min_trades_per_month:
            flags.append(
                f'FREQ_TOO_LOW: avg={avg_trades:.1f} trade/bulan'
                f' (min={cfg.min_trades_per_month})'
            )
        elif avg_trades > cfg.max_trades_per_month:
            flags.append(
                f'FREQ_TOO_HIGH: avg={avg_trades:.1f} trade/bulan'
                f' (max={cfg.max_trades_per_month})'
            )

        # ── CHECK 2: Monthly Drawdown ─────────────────────────────────
        max_dd = monthly_metrics['max_drawdown_pips'].max()
        if max_dd is None:
            flags.append('DD_CALCULATION_ERROR: max_drawdown_pips bernilai null')
        elif max_dd > cfg.max_monthly_drawdown_pips:
            flags.append(
                f'DD_EXCEEDED: max_monthly_dd={max_dd:.1f} pips'
                f' (limit={cfg.max_monthly_drawdown_pips})'
            )

        # ── CHECK 3: Undefined Profit Factor ─────────────────────────
        # NULL profit_factor = bulan tanpa losing trade = undefined PF
        null_pf_count = monthly_metrics['profit_factor'].null_count()
        if null_pf_count > 0:
            flags.append(
                f'UNDEFINED_PF: {null_pf_count} bulan tidak memiliki'
                f' losing trade (gross_loss=0) → Profit Factor undefined'
            )

        # ── CHECK 4: Risk per Trade (avg_risk_pips) ───────────────────
        if 'avg_risk_pips' in monthly_metrics.columns:
            avg_risk = monthly_metrics['avg_risk_pips'].mean()
            if avg_risk is not None:
                if avg_risk < cfg.min_risk_pips:
                    flags.append(
                        f'RISK_TOO_LOW: avg_risk={avg_risk:.1f} pips'
                        f' (min={cfg.min_risk_pips})'
                    )
                elif avg_risk > cfg.max_risk_pips:
                    flags.append(
                        f'RISK_TOO_HIGH: avg_risk={avg_risk:.1f} pips'
                        f' (max={cfg.max_risk_pips})'
                    )

        passed = len(flags) == 0

        result = {
            'passed': passed,
            'flags': flags,
            'summary': {
                'avg_trades_per_month': avg_trades,
                'max_drawdown_pips': max_dd,
                'null_pf_months': null_pf_count,
            },
        }

        if passed:
            log.info('HardConstraints PASSED',
                     avg_trades=f'{avg_trades:.1f}', max_dd=f'{max_dd:.1f}')
        else:
            log.info('HardConstraints FAILED — ELIMINATED',
                     flags=flags)

        return result

class FitnessScorer:
    """
    Hitung composite fitness score.

    HANYA panggil ini setelah HardConstraintEvaluator.evaluate()
    mengembalikan {'passed': True}.
    """

    def __init__(self, config: ConstraintConfig | None = None) -> None:
        self.config = config or ConstraintConfig()

    def compute(self, monthly_metrics: pl.DataFrame) -> float:
        """
        Hitung fitness score komposit.

        Formula:
            score = (0.40 × avg_pf)
                  + (0.30 × avg_net_pips / 100)
                  + (0.20 × consistency)
                  + (0.10 × win_rate)
                  - pf_penalty

        Returns:
            float >= 0. Semakin tinggi = semakin baik.
        """
        cfg = self.config
        m = monthly_metrics

        # ── Komponen 1: Profit Factor ─────────────────────────────────
        avg_pf = m['profit_factor'].drop_nulls().mean() or 0.0

        # ── Komponen 2: Net Pips (normalized) ────────────────────────
        avg_net = m['net_pips'].mean() or 0.0
        normalized_net = avg_net / 100.0

        # ── Komponen 3: Consistency ───────────────────────────────────
        # Consistency = 1 - (std / mean) — makin rendah volatilitas = lebih konsisten
        std_net = m['net_pips'].std() or 0.0
        if avg_net > 0:
            consistency = max(1.0 - (std_net / avg_net), 0.0)
        else:
            consistency = 0.0

        # ── Komponen 4: Win Rate ──────────────────────────────────────
        win_rate = m['win_rate'].mean() or 0.0

        # ── Penalti: PF di bawah minimum ─────────────────────────────
        pf_penalty = 0.0
        if avg_pf < cfg.min_profit_factor:
            gap = cfg.min_profit_factor - avg_pf
            pf_penalty = cfg.pf_penalty_weight * gap

        # ── Composite Score ───────────────────────────────────────────
        score = (
            0.40 * avg_pf
            + 0.30 * normalized_net
            + 0.20 * consistency
            + 0.10 * win_rate
            - pf_penalty
        )

        final_score = max(score, 0.0)  # Clamp ke non-negative

        log.info('FitnessScorer computed',
                 avg_pf=f'{avg_pf:.3f}',
                 avg_net=f'{avg_net:.1f}',
                 consistency=f'{consistency:.3f}',
                 win_rate=f'{win_rate:.3f}',
                 pf_penalty=f'{pf_penalty:.3f}',
                 final_score=f'{final_score:.4f}')

        return final_score
