# tests/unit/test_optuna_tuner.py
import polars as pl
import numpy as np
import pytest
from src.evolution.gp_generator import DEAPGenerator
from src.evolution.optuna_tuner import OptunaTuner

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
    assert 10.0 <= tuned_dna.params['sl_pips'] <= 50.0
    assert tuned_dna.params['tp_pips'] >= tuned_dna.params['sl_pips'] * 1.5
