# src/analytics/metrics_calculator.py
# Referensi: Blueprint §4.3

from __future__ import annotations
import math
import polars as pl
import connectorx as cx
import psycopg
from psycopg.types.json import Jsonb
from src.data.repositories.base import as_uuid, postgres_sync_dsn
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
        safe_trial_id = str(as_uuid(trial_id))
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
            WHERE trial_id = '{safe_trial_id}'
            ORDER BY entry_time ASC
        """

        log.info('Computing monthly metrics', trial_id=trial_id)

        df = cx.read_sql(
            postgres_sync_dsn(),
            query,
            return_type='polars',
        )

        if df.is_empty():
            log.warning('No trades found for trial', trial_id=trial_id)
            return pl.DataFrame()

        strategy_id = str(df['strategy_id'][0])

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

        result = (
            monthly.join(equity_df, on='backtest_month', how='left')
            .with_columns([
                pl.lit(safe_trial_id).alias('trial_id'),
                pl.lit(strategy_id).alias('strategy_id'),
            ])
        )

        log.info('Monthly metrics computed',
                 trial_id=trial_id,
                 months=len(result),
                 avg_pf=result['profit_factor'].drop_nulls().mean())

        self._persist_monthly_metrics(result)

        return result

    def _persist_monthly_metrics(self, monthly: pl.DataFrame) -> None:
        """Upsert computed monthly metrics into PostgreSQL."""
        if monthly.is_empty():
            return

        sql = """
            INSERT INTO monthly_metrics (
                trial_id, strategy_id, backtest_month, trade_count,
                winning_trades, losing_trades, gross_profit, gross_loss,
                net_pips, profit_factor, max_drawdown_pips, win_rate,
                avg_risk_pips, constraint_passed, elimination_flags
            )
            VALUES (
                %(trial_id)s::uuid, %(strategy_id)s::uuid, %(backtest_month)s,
                %(trade_count)s, %(winning_trades)s, %(losing_trades)s,
                %(gross_profit)s, %(gross_loss)s, %(net_pips)s,
                %(profit_factor)s, %(max_drawdown_pips)s, %(win_rate)s,
                %(avg_risk_pips)s, FALSE, NULL
            )
            ON CONFLICT (trial_id, backtest_month)
            DO UPDATE SET
                strategy_id = EXCLUDED.strategy_id,
                trade_count = EXCLUDED.trade_count,
                winning_trades = EXCLUDED.winning_trades,
                losing_trades = EXCLUDED.losing_trades,
                gross_profit = EXCLUDED.gross_profit,
                gross_loss = EXCLUDED.gross_loss,
                net_pips = EXCLUDED.net_pips,
                profit_factor = EXCLUDED.profit_factor,
                max_drawdown_pips = EXCLUDED.max_drawdown_pips,
                win_rate = EXCLUDED.win_rate,
                avg_risk_pips = EXCLUDED.avg_risk_pips,
                calculated_at = NOW()
        """

        rows = [
            {
                'trial_id': str(as_uuid(row['trial_id'])),
                'strategy_id': str(as_uuid(row['strategy_id'])),
                'backtest_month': row['backtest_month'],
                'trade_count': int(row['trade_count']),
                'winning_trades': int(row['winning_trades']),
                'losing_trades': int(row['losing_trades']),
                'gross_profit': self._none_if_nan(row['gross_profit']),
                'gross_loss': self._none_if_nan(row['gross_loss']),
                'net_pips': self._none_if_nan(row['net_pips']),
                'profit_factor': self._none_if_nan(row['profit_factor']),
                'max_drawdown_pips': self._none_if_nan(row['max_drawdown_pips']),
                'win_rate': self._none_if_nan(row['win_rate']),
                'avg_risk_pips': self._none_if_nan(row.get('avg_risk_pips')),
            }
            for row in monthly.iter_rows(named=True)
        ]

        with psycopg.connect(postgres_sync_dsn()) as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)

        log.info('Monthly metrics persisted', trial_id=rows[0]['trial_id'], months=len(rows))

    def update_constraint_result(
        self,
        trial_id: str,
        *,
        passed: bool,
        flags: list[str] | None = None,
    ) -> None:
        """Persist hard-constraint result for all monthly rows of a trial."""

        with psycopg.connect(postgres_sync_dsn()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE monthly_metrics
                    SET constraint_passed = %s,
                        elimination_flags = %s
                    WHERE trial_id = %s::uuid
                    """,
                    (passed, Jsonb(flags or []), str(as_uuid(trial_id))),
                )

        log.info(
            'Monthly constraint result persisted',
            trial_id=trial_id,
            passed=passed,
            flags=flags or [],
        )

    @staticmethod
    def _none_if_nan(value):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

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
