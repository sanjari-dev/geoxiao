# tests/unit/test_deap_generator.py
import pytest
from src.evolution.gp_generator import DEAPGenerator
from src.strategy.base_strategy import BaseStrategy

@pytest.fixture
def generator():
    return DEAPGenerator()

def test_initialize_population(generator):
    pop = generator.initialize_population(10)
    assert len(pop) == 10
    assert all(dna.tree_repr for dna in pop)
    assert all(dna.generation == 0 for dna in pop)

def test_crossover_produces_offspring(generator):
    pop = generator.initialize_population(4)
    off_a, off_b = generator.crossover(pop[0], pop[1])
    assert off_a.generation == 1
    assert off_b.generation == 1

def test_mutate(generator):
    pop = generator.initialize_population(4)
    mutated = generator.mutate(pop[0])
    assert mutated is not pop[0]

def test_to_strategy_class(generator):
    pop = generator.initialize_population(4)
    dna = pop[0]
    dna.params = {'sl_pips': 20.0, 'tp_pips': 40.0, 'signal_threshold': 0.5}
    StrategyClass = generator.to_strategy_class(dna)
    assert issubclass(StrategyClass, BaseStrategy)
    instance = StrategyClass(dna)
    assert instance.validate_params() is True

def test_no_classical_indicators_in_tree(generator):
    """Pastikan tidak ada RSI/MACD/MA dalam tree representation."""
    banned = ['rsi', 'macd', 'sma', 'ema', 'bollinger', 'stochastic', 'atr']
    pop = generator.initialize_population(20)
    for dna in pop:
        tree_lower = dna.tree_repr.lower()
        for indicator in banned:
            assert indicator not in tree_lower, \
                f'Banned indicator {indicator} ditemukan dalam tree: {dna.tree_repr}'


def test_generated_strategy_supports_gp_alias_functions(generator):
    dna = generator.initialize_population(2)[0]
    dna.tree_repr = "sub(add(obi, tick_vel), mul(sign(kurtosis), 0.5))"
    dna.params = {'sl_pips': 20.0, 'tp_pips': 40.0, 'signal_threshold': 0.5}

    StrategyClass = generator.to_strategy_class(dna)
    strategy = StrategyClass(dna)

    signal = strategy.compute_signal_value(
        {
            'obi': 0.2,
            'tick_vel': 0.4,
            'spread_dyn': 0.1,
            'tick_den': 0.1,
            'vol_skew': 0.0,
            'mid_mom': 0.1,
            'skewness': 0.0,
            'kurtosis': -2.0,
            'vw_spread': 0.1,
        }
    )

    assert isinstance(signal, float)
