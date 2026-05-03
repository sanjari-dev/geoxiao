from __future__ import annotations

import math
from typing import Any

DEFAULT_SIGNAL_THRESHOLD = 0.5
MIN_SIGNAL_THRESHOLD = 0.05
MAX_SIGNAL_THRESHOLD = 0.95
MAX_SIGNAL_ABS_INPUT = 20.0


def normalize_signal_value(value: Any) -> float:
    """Normalize arbitrary GP output to a stable [-1, 1] domain."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(numeric):
        return 0.0

    bounded = max(min(numeric, MAX_SIGNAL_ABS_INPUT), -MAX_SIGNAL_ABS_INPUT)
    return math.tanh(bounded)


def clamp_signal_threshold(
    value: Any,
    *,
    default: float = DEFAULT_SIGNAL_THRESHOLD,
) -> float:
    """Clamp threshold to the normalized signal domain used by the engine."""

    try:
        threshold = abs(float(value))
    except (TypeError, ValueError):
        threshold = default

    if not math.isfinite(threshold):
        threshold = default

    return min(max(threshold, MIN_SIGNAL_THRESHOLD), MAX_SIGNAL_THRESHOLD)

