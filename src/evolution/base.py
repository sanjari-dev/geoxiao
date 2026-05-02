# src/evolution/base.py
# Referensi: Blueprint §3.2

from __future__ import annotations
from abc import ABC, abstractmethod
from src.strategy.base_strategy import StrategyDNA, BaseStrategy


class BaseGenerator(ABC):
    """
    Kontrak untuk GP generator (DEAP layer).
    Mengontrol representasi genetik dan seluruh operasi evolusioner.
    """

    @abstractmethod
    def initialize_population(self, pop_size: int) -> list[StrategyDNA]:
        """
        Generate populasi awal sejumlah pop_size individu.
        Setiap individu HARUS memiliki tree_repr, tree_depth, tree_nodes unik.

        Args:
            pop_size: jumlah individu. Nilai minimum: 10.

        Returns:
            list[StrategyDNA] dengan generation=0.
        """
        ...

    @abstractmethod
    def crossover(
        self,
        parent_a: StrategyDNA,
        parent_b: StrategyDNA,
    ) -> tuple[StrategyDNA, StrategyDNA]:
        """
        Subtree crossover antara dua parent GP tree.
        Implementasi HARUS menggunakan tools.cxOnePoint dari DEAP.

        Returns:
            Tuple (offspring_a, offspring_b) dengan generation parent+1.
        """
        ...

    @abstractmethod
    def mutate(self, individual: StrategyDNA, mu: float = 0.1) -> StrategyDNA:
        """
        Mutasi pohon GP.
        Implementasi HARUS menggunakan gp.mutUniform dari DEAP.

        Args:
            mu: probabilitas mutasi per node. Range: 0.05 - 0.3.

        Returns:
            StrategyDNA baru dengan tree yang termutasi.
        """
        ...

    @abstractmethod
    def next_generation(
        self,
        survivors: list[StrategyDNA],
        target_size: int,
    ) -> list[StrategyDNA]:
        """
        Buat generasi berikutnya dari survivors menggunakan
        tournament selection + crossover + mutation.

        Args:
            survivors: individu yang lolos HardConstraintEvaluator.
            target_size: ukuran populasi yang diinginkan.

        Returns:
            list[StrategyDNA] untuk generasi berikutnya.
        """
        ...

    @abstractmethod
    def to_strategy_class(self, dna: StrategyDNA) -> type[BaseStrategy]:
        """
        Konversi pohon DEAP menjadi runnable Python class.
        Class yang dihasilkan HARUS mewarisi BaseStrategy.
        Simpan ke src/strategy/generated/{dna.individual_id}.py.

        Returns:
            Subclass of BaseStrategy yang bisa di-instantiate.
        """
        ...


class BaseEvaluator(ABC):
    """
    Kontrak untuk evaluasi fitness individu.
    Pipeline: ClickHouse vectorized backtest → log → metrics → constraint → fitness.
    JANGAN gabungkan step-step ini dalam satu fungsi.
    """

    @abstractmethod
    async def evaluate(self, dna: StrategyDNA) -> dict:
        """
        Full evaluation pipeline untuk satu individu.

        Urutan eksekusi WAJIB:
        1. Set status 'backtesting' di PostgreSQL
        2. Jalankan ClickHouse vectorized backtest
        3. Flush trade_logs ke PostgreSQL
        4. Hitung monthly_metrics via MetricsCalculator
        5. Jalankan HardConstraintEvaluator
        6. Jika passed: jalankan FitnessScorer
        7. Update status dan fitness di PostgreSQL

        Returns:
            {
                'fitness_score': float,
                'profit_factor': float,
                'max_drawdown_pips': float,
                'trade_count': int,
                'passed': bool,
                'elimination_reason': str | None
            }
        """
        ...

    @abstractmethod
    def check_hard_constraints(
        self,
        metrics: dict,
    ) -> tuple[bool, str | None]:
        """
        Evaluasi hard constraints secara atomic.

        Returns:
            (passed: bool, reason: str | None)
            reason berisi detail constraint yang dilanggar.
        """
        ...
