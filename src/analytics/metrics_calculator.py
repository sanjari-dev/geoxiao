# src/analytics/metrics_calculator.py
# Referensi: Blueprint §4.3

from __future__ import annotations
import polars as pl
import connectorx as cx
from src.config.settings import settings
import structlog

log = structlog.get_logger(__name__)


class MetricsCalculator:
    """
    Baca trade_logs dari PostgreSQL dan hitung monthly metrics.

    Hasil disimpan ke monthly_metrics table DAN dikembalikan
    sebagai polars DataFrame ke HardConstraintEvaluator.
    """

    def compute_monthly_metrics(
        self,
        trial_id: str,
    ) -> pl.DataFrame:
        """
        Hitung agregasi bulanan untuk satu trial.

        Returns:
            polars DataFrame dengan skema:
            [backtest_month: Date, trade_count: Int64,
             winning_trades: Int64, losing_trades: Int64,
             gross_profit: Float64, gross_loss: Float64,
             net_pips: Float64, profit_factor: Float64 | null,
             max_drawdown_pips: Float64, win_rate: Float64]
        """
        query = f"""
            SELECT
                trial_id,
                strategy_id,
                backtest_month,
                net_pips,
                sl_price,
                entry_price,
                side
            FROM trade_logs
            WHERE trial_id = '{trial_id}'
            ORDER BY entry_time ASC
        """

        log.info('Computing monthly metrics', trial_id=trial_id)

        df = cx.read_sql(
            settings.PG_DSN_SYNC,
            query,
            return_type='polars',
        )

        if df.is_empty():
            log.warning('No trades found for trial', trial_id=trial_id)
            return pl.DataFrame()

        # ── Kalkulasi avg risk per trade (SL distance dalam pips) ──
        df = df.with_columns([
            pl.when(pl.col('side') == 'BUY')
              .then((pl.col('entry_price') - pl.col('sl_price')) * 10000)
              .otherwise((pl.col('sl_price') - pl.col('entry_price')) * 10000)
              .abs().alias('risk_pips')
        ])

        # ── Agregasi per bulan ──
        monthly = (
            df.group_by('backtest_month')
            .agg([
                pl.len().alias('trade_count'),
                pl.col('net_pips').filter(pl.col('net_pips') > 0)
                  .sum().fill_null(0).alias('gross_profit'),
                pl.col('net_pips').filter(pl.col('net_pips') <= 0)
                  .sum().abs().fill_null(0).alias('gross_loss'),
                pl.col('net_pips').sum().alias('net_pips'),
                pl.col('net_pips').filter(pl.col('net_pips') > 0)
                  .len().alias('winning_trades'),
                pl.col('risk_pips').mean().alias('avg_risk_pips'),
            ])
            .with_columns([
                # Profit Factor: None jika tidak ada losing trade
                pl.when(pl.col('gross_loss') > 0)
                  .then(pl.col('gross_profit') / pl.col('gross_loss'))
                  .otherwise(None)
                  .alias('profit_factor'),
                # Win Rate
                (pl.col('winning_trades') / pl.col('trade_count'))
                  .alias('win_rate'),
                # Losing trades
                (pl.col('trade_count') - pl.col('winning_trades'))
                  .alias('losing_trades'),
            ])
            .sort('backtest_month')
        )

        # ── Max Drawdown via running equity curve ──
        # Hitung per bulan: kumulatif equity → peak → drawdown
        equity_df = (
            df.sort('backtest_month')
            .with_columns(
                pl.col('net_pips').cum_sum().alias('cum_equity')
            )
            .with_columns(
                pl.col('cum_equity').cum_max().alias('peak')
            )
            .with_columns(
                (pl.col('peak') - pl.col('cum_equity')).alias('drawdown')
            )
            .group_by('backtest_month')
            .agg(pl.col('drawdown').max().alias('max_drawdown_pips'))
        )

        result = monthly.join(equity_df, on='backtest_month', how='left')

        log.info('Monthly metrics computed',
                 trial_id=trial_id,
                 months=len(result),
                 avg_pf=result['profit_factor'].drop_nulls().mean())

        return result

    def get_summary_stats(self, monthly: pl.DataFrame) -> dict:
        """
        Hitung summary statistics dari monthly_metrics DataFrame.
        Digunakan sebagai input FitnessScorer.
        """
        if monthly.is_empty():
            return {'error': 'no_data'}

        return {
            'avg_trade_count': monthly['trade_count'].mean(),
            'avg_net_pips': monthly['net_pips'].mean(),
            'std_net_pips': monthly['net_pips'].std(),
            'avg_profit_factor': monthly['profit_factor'].drop_nulls().mean(),
            'null_pf_months': monthly['profit_factor'].null_count(),
            'max_drawdown': monthly['max_drawdown_pips'].max(),
            'avg_win_rate': monthly['win_rate'].mean(),
        }
