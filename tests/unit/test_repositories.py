from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from src.config.settings import settings
from src.data.repositories.base import postgres_asyncpg_dsn
from src.data.repositories.base import AsyncPostgresRepository
from src.data.repositories.strategy_repo import StrategyRepository
from src.data.repositories.trade_repo import TradeRepository


def test_postgres_asyncpg_dsn_normalizes_sqlalchemy_url(monkeypatch):
    monkeypatch.setattr(
        settings,
        "PG_DSN",
        "postgresql+asyncpg://user:secret@db.example:5432/geoxiao",
    )

    assert postgres_asyncpg_dsn() == "postgresql://user:secret@db.example:5432/geoxiao"


def test_strategy_row_to_dna_decodes_jsonb_string():
    strategy_id = str(uuid.uuid4())
    row = {
        "id": strategy_id,
        "generation": 3,
        "individual_id": "ind-1",
        "tree_repr": "safe_add(obi, skewness)",
        "tree_depth": 2,
        "tree_nodes": 3,
        "params_json": '{"sl_pips": 15.5}',
        "symbol": "EURUSD",
        "timeframe": "M15",
        "status": "pending",
    }

    dna = StrategyRepository._row_to_dna(row)

    assert dna.id == strategy_id
    assert dna.params == {"sl_pips": 15.5}
    assert dna.symbol == "EURUSD"


def test_trade_to_tuple_validates_and_normalizes_types():
    trial_id = uuid.uuid4()
    strategy_id = uuid.uuid4()
    row = TradeRepository._to_tuple(
        {
            "trial_id": str(trial_id),
            "strategy_id": str(strategy_id),
            "symbol": "EURUSD",
            "side": "buy",
            "order_type": "limit",
            "entry_price": 1.10123,
            "exit_price": "1.10223",
            "sl_price": "1.09923",
            "tp_price": "1.10523",
            "entry_time": "2024-01-15T10:00:00+00:00",
            "exit_time": datetime(2024, 1, 15, 11, 0, tzinfo=timezone.utc),
            "raw_pips": 10,
            "spread_pips": 1.5,
            "slippage_pips": 0.25,
            "commission_pips": 0,
            "exit_reason": "TP",
            "backtest_month": "2024-01-01",
        }
    )

    assert row[0] == trial_id
    assert row[1] == strategy_id
    assert row[3] == "BUY"
    assert row[4] == "LIMIT"
    assert row[5] == Decimal("1.10123")
    assert row[16] == date(2024, 1, 1)


def test_trade_to_tuple_rejects_invalid_side():
    with pytest.raises(ValueError, match="Invalid trade side"):
        TradeRepository._to_tuple(
            {
                "trial_id": str(uuid.uuid4()),
                "strategy_id": str(uuid.uuid4()),
                "symbol": "EURUSD",
                "side": "HOLD",
                "order_type": "LIMIT",
                "entry_price": 1.1,
                "sl_price": 1.0,
                "entry_time": datetime.now(timezone.utc),
                "backtest_month": date(2024, 1, 1),
            }
        )


def test_repository_retry_classifier_flags_connection_errors():
    assert AsyncPostgresRepository._is_retryable_exception(ConnectionResetError("boom")) is True
    assert AsyncPostgresRepository._is_retryable_exception(ValueError("boom")) is False
