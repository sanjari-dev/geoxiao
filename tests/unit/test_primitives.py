# tests/unit/test_primitives.py
import numpy as np
import pytest
from src.evolution.primitives import (
    order_book_imbalance, tick_velocity, spread_dynamics,
    safe_div, safe_log, rolling_skewness
)

def test_obi_range():
    bid = np.ones(30)
    ask = np.ones(30) * 2
    result = order_book_imbalance(bid, ask)
    assert -1.0 <= result <= 1.0

def test_safe_div_zero():
    assert safe_div(5.0, 0.0) == 1.0

def test_safe_log_zero():
    assert safe_log(0.0) == 0.0

def test_tick_velocity_nonneg():
    bid = np.random.uniform(1.08, 1.09, 50)
    ask = bid + 0.0002
    result = tick_velocity(bid, ask)
    assert result >= 0.0

def test_skewness_shape():
    bid = np.random.normal(1.08, 0.001, 100)
    ask = bid + 0.0002
    result = rolling_skewness(bid, ask, window=30)
    assert isinstance(result, float)
