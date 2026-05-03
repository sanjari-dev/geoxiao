# tests/unit/test_optuna_tuner.py
import polars as pl
import numpy as np
import pytest
from decimal import Decimal
from src.evolution.gp_generator import DEAPGenerator
from src.evolution.optuna_tuner import OptunaTuner, SignalDiagnostics

@pytest.fixture
def sample_tick_data():
    n = 5000
    bid = np.cumsum(np.random.normal(0, 0.0001, n)) + 1.08
    ask = bid + 0.0002
    bid_size = np.random.uniform(1_000_000, 5_000_000, n)
    ask_size = np.random.uniform(1_000_000, 5_000_000, n)
    ts = np.arange(n, dtype='int64') * int(1e9)  # 1 tick per detik
    return pl.DataFrame({
        'timestamp': pl.Series(ts).cast(pl.Datetime('ns')),
        'bid': bid, 'ask': ask,
        'bid_size': bid_size, 'ask_size': ask_size,
    })

def test_tune_updates_params(sample_tick_data):
    generator = DEAPGenerator()
    pop = generator.initialize_population(4)
    dna = pop[0]

    tuner = OptunaTuner('test_study_unit', n_trials=5)
    tuned_dna = tuner.tune(
        dna=dna,
        tick_data=sample_tick_data,
        strategy_class_factory=generator.to_strategy_class,
    )

    assert 'sl_pips' in tuned_dna.params
    assert 'tp_pips' in tuned_dna.params
    assert 'signal_threshold' in tuned_dna.params
    assert 10.0 <= tuned_dna.params['sl_pips'] <= 50.0
    assert tuned_dna.params['tp_pips'] >= tuned_dna.params['sl_pips'] * 1.5
    assert 0.05 <= tuned_dna.params['signal_threshold'] <= 0.95


def test_prepare_tick_data_casts_decimal_columns():
    df = pl.DataFrame({
        'timestamp': pl.datetime_range(
            start=pl.datetime(2024, 1, 1),
            end=pl.datetime(2024, 1, 1, 0, 0, 1),
            interval='1s',
            eager=True,
        ),
        'bid': [Decimal('1.1000'), Decimal('1.1001')],
        'ask': [Decimal('1.1002'), Decimal('1.1003')],
        'bid_size': [Decimal('10'), Decimal('11')],
        'ask_size': [Decimal('12'), Decimal('13')],
    })

    prepared = OptunaTuner._prepare_tick_data(df)

    assert prepared.schema['bid'] == pl.Float64
    assert prepared.schema['ask'] == pl.Float64
    assert prepared.schema['bid_size'] == pl.Float64
    assert prepared.schema['ask_size'] == pl.Float64


def test_signal_diagnostics_with_error_is_not_auto_skipped():
    diag = SignalDiagnostics(
        threshold=0.5,
        windows_evaluated=0,
        entry_candidates=0,
        min_signal=0.0,
        max_signal=0.0,
        error='boom',
    )

    assert diag.viable is True
