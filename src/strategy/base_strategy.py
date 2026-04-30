# src/strategy/base_strategy.py
# Referensi: Blueprint §3.1

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass
class StrategyDNA:
    """
    Data Transfer Object untuk representasi genetik strategi.
    Digunakan sebagai antarmuka antara GP Generator, Optuna Tuner,
    NautilusTrader Adapter, dan PostgreSQL Repository.
    """
    id: str = field(default=None)
    generation: int = 0
    individual_id: str = field(default=None)
    tree_repr: str = ''
    tree_depth: int = 0
    tree_nodes: int = 0
    params: dict = field(default_factory=dict)
    symbol: str = 'EURUSD'
    timeframe: str = 'M15'
    # Diisi setelah evaluasi
    fitness_score: float = 0.0
    status: str = 'pending'     # pending|backtesting|passed|eliminated|archived
    eliminated_reason: str | None = None

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.individual_id is None:
            self.individual_id = str(uuid.uuid4())[:16]

    def to_db_dict(self) -> dict:
        """Serialize ke dict untuk PostgreSQL INSERT."""
        return {
            'id': self.id,
            'generation': self.generation,
            'individual_id': self.individual_id,
            'tree_repr': self.tree_repr,
            'tree_depth': self.tree_depth,
            'tree_nodes': self.tree_nodes,
            'params_json': self.params,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'status': self.status,
        }


class BaseStrategy(ABC):
    """
    Kontrak abstract untuk semua strategi yang di-generate GP.

    Lifecycle:
    1. __init__(dna) → inisialisasi dari StrategyDNA
    2. validate_params() → cek parameter sebelum backtest
    3. compute_features(tick_data) → transform ke feature vector
    4. generate_signal(features) → return entry signal atau None
    """

    def __init__(self, dna: StrategyDNA) -> None:
        self.dna = dna

    @abstractmethod
    def compute_features(self, tick_data: Any) -> Any:
        """
        Transform raw tick data menjadi feature vector.

        Args:
            tick_data: polars DataFrame dengan skema:
                       [timestamp: Datetime, bid: Float64, ask: Float64,
                        bid_size: Float64, ask_size: Float64]

        Returns:
            polars DataFrame dengan kolom fitur tambahan.
            WAJIB mempertahankan kolom timestamp sebagai index.
        """
        ...

    @abstractmethod
    def generate_signal(self, features: Any) -> dict | None:
        """
        Generate entry signal dari feature vector.

        Returns:
            dict: {'side': 'BUY'|'SELL',
                   'sl_pips': float,   # HARUS dalam range 10-50
                   'tp_pips': float,
                   'order_type': 'LIMIT'|'STOP_LIMIT'}
            None: tidak ada signal
        """
        ...

    @abstractmethod
    def validate_params(self) -> bool:
        """
        Validasi parameter numerik dari dna.params.

        Raises:
            ValueError: dengan pesan deskriptif jika parameter invalid.

        Returns:
            True jika semua parameter valid.
        """
        ...

    def get_param(self, key: str, default: Any = None) -> Any:
        """Helper untuk akses dna.params dengan default value."""
        return self.dna.params.get(key, default)
