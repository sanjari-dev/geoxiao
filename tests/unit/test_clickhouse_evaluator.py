from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.backtest.clickhouse_evaluator import ClickHouseEvaluator, RunningBacktestMetrics
from src.strategy.base_strategy import StrategyDNA


def _evaluator_without_connection() -> ClickHouseEvaluator:
    evaluator = ClickHouseEvaluator.__new__(ClickHouseEvaluator)
    evaluator.database = "geonera"
    evaluator.ticks_table = "ticks"
    evaluator.block_size = 10_000
    evaluator.max_holding_seconds = 86_400
    evaluator.entry_cooldown_seconds = 60
    evaluator.client = None
    return evaluator


def test_translate_dna_to_sql_maps_gp_primitives():
    evaluator = _evaluator_without_connection()
    dna = StrategyDNA(tree_repr="add(obi, mul(mid_mom, -0.5))")

    sql = evaluator._translate_dna_to_sql(dna)

    assert "obi" in sql
    assert "mid_mom" in sql
    assert "*" in sql
    assert "-0.5" in sql


def test_translate_dna_to_sql_rejects_unknown_identifier():
    evaluator = _evaluator_without_connection()
    dna = StrategyDNA(tree_repr="add(obi, __import__)")

    with pytest.raises(ValueError, match="Unsupported feature"):
        evaluator._translate_dna_to_sql(dna)


def test_running_metrics_incrementally_tracks_drawdown():
    metrics = RunningBacktestMetrics()

    for pips in [10.0, -4.0, -9.0, 6.0]:
        metrics.update(pips)

    assert metrics.total_trades == 4
    assert metrics.gross_profit == 16.0
    assert metrics.gross_loss == 13.0
    assert metrics.winning_trades == 2
    assert metrics.win_rate == 0.5
    assert metrics.max_drawdown_pips == 13.0


def test_process_block_formats_trade_logs_and_updates_metrics():
    evaluator = _evaluator_without_connection()
    dna = StrategyDNA(
        id="11111111-1111-1111-1111-111111111111",
        tree_repr="obi",
        symbol="AUDUSD",
        params={"sl_pips": 10.0, "tp_pips": 20.0},
    )
    metrics = RunningBacktestMetrics()
    entry_time = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
    exit_time = datetime(2024, 1, 15, 10, 5, tzinfo=timezone.utc)

    batch = evaluator._process_block(
        [(entry_time, exit_time, "BUY", 1.1000, 1.1020, 20.0)],
        dna=dna,
        trial_id="22222222-2222-2222-2222-222222222222",
        metrics=metrics,
    )

    assert metrics.total_trades == 1
    assert metrics.gross_profit == 20.0
    assert batch[0]["symbol"] == "AUDUSD"
    assert batch[0]["entry_time"] == entry_time
    assert batch[0]["backtest_month"].isoformat() == "2024-01-01"
    assert batch[0]["sl_price"] == pytest.approx(1.099)
    assert batch[0]["tp_price"] == pytest.approx(1.102)
