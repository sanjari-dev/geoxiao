# src/evolution/gp_generator.py
# Implementasi konkret BaseGenerator menggunakan DEAP

from __future__ import annotations
import random
import copy
import textwrap
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from deap import base, creator, gp, tools, algorithms

from src.evolution.base import BaseGenerator
from src.strategy.base_strategy import StrategyDNA, BaseStrategy
from src.strategy.signal_utils import clamp_signal_threshold
from src.config.settings import settings
from src.evolution.primitives import (
    safe_div, safe_log, safe_sqrt, neg, square, cube, sigmoid, sign_fn,
    max2, min2, order_book_imbalance, tick_velocity, spread_dynamics,
    tick_density, volume_clock_skew, mid_price_momentum,
    rolling_skewness, rolling_kurtosis, volume_weighted_spread,
)
import structlog

log = structlog.get_logger(__name__)

# ── DEAP creator setup (dilakukan sekali di module level) ────────────────
# Hindari re-registrasi jika module di-reload
if not hasattr(creator, 'FitnessMax'):
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
if not hasattr(creator, 'Individual'):
    creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMax)


def _add(a, b):
    return a + b


def _sub(a, b):
    return a - b


def _mul(a, b):
    return a * b


def _const_small() -> float:
    return round(random.uniform(-2.0, 2.0), 4)


def _const_large() -> float:
    return round(random.uniform(-10.0, 10.0), 4)


def _build_primitive_set() -> gp.PrimitiveSet:
    """
    Bangun PrimitiveSet DEAP.

    Input arguments pohon GP (in_0 sampai in_8) adalah nilai fitur
    microstructure yang sudah dihitung sebelumnya:
      in_0: obi           (Order Book Imbalance)
      in_1: tick_vel      (Tick Velocity)
      in_2: spread_dyn    (Spread Dynamics)
      in_3: tick_den      (Tick Density)
      in_4: vol_skew      (Volume Clock Skew)
      in_5: mid_mom       (Mid-Price Momentum)
      in_6: skewness      (Rolling Skewness)
      in_7: kurtosis      (Rolling Kurtosis)
      in_8: vw_spread     (Volume-Weighted Spread)
    """
    pset = gp.PrimitiveSet('MAIN', arity=9)

    # Rename arguments ke nama deskriptif
    pset.renameArguments(
        ARG0='obi', ARG1='tick_vel', ARG2='spread_dyn',
        ARG3='tick_den', ARG4='vol_skew', ARG5='mid_mom',
        ARG6='skewness', ARG7='kurtosis', ARG8='vw_spread'
    )

    # ── Binary operators ─────────────────────────────────────────────
    pset.addPrimitive(_add,             2, name='add')
    pset.addPrimitive(_sub,             2, name='sub')
    pset.addPrimitive(_mul,             2, name='mul')
    pset.addPrimitive(safe_div,         2, name='div')
    pset.addPrimitive(max2,             2, name='max2')
    pset.addPrimitive(min2,             2, name='min2')

    # ── Unary operators ──────────────────────────────────────────────
    pset.addPrimitive(neg,      1, name='neg')
    pset.addPrimitive(square,   1, name='square')
    pset.addPrimitive(cube,     1, name='cube')
    pset.addPrimitive(safe_log, 1, name='log')
    pset.addPrimitive(safe_sqrt,1, name='sqrt')
    pset.addPrimitive(sigmoid,  1, name='sigmoid')
    pset.addPrimitive(sign_fn,  1, name='sign')

    # ── Ephemeral constants (random float di generate time) ──────────
    pset.addEphemeralConstant('const_small', _const_small)
    pset.addEphemeralConstant('const_large', _const_large)

    return pset


class DEAPGenerator(BaseGenerator):
    """
    Implementasi konkret BaseGenerator menggunakan DEAP.

    Pohon GP merepresentasikan 'signal function' — fungsi komposit
    dari fitur microstructure yang menghasilkan scalar signal.
    Signal > threshold → BUY; signal < -threshold → SELL; else → HOLD.
    """

    GENERATED_DIR = Path('src/strategy/generated')
    MIN_TREE_DEPTH = 2
    MAX_TREE_DEPTH = 6
    TOURNAMENT_SIZE = 3

    def __init__(self, symbol: str | None = None, timeframe: str | None = None) -> None:
        self.symbol = symbol or settings.SYMBOL
        self.timeframe = timeframe or settings.TIMEFRAME
        self.pset = _build_primitive_set()
        self.toolbox = self._build_toolbox()
        self.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        log.info('DEAPGenerator initialized', symbol=self.symbol, timeframe=self.timeframe)

    def _build_toolbox(self) -> base.Toolbox:
        """Register semua operator DEAP ke toolbox."""
        tb = base.Toolbox()

        # Tree generation
        tb.register('expr', gp.genHalfAndHalf, pset=self.pset,
                    min_=self.MIN_TREE_DEPTH, max_=self.MAX_TREE_DEPTH)
        tb.register('individual', tools.initIterate,
                    creator.Individual, tb.expr)
        tb.register('population', tools.initRepeat, list, tb.individual)

        # Genetic operators
        tb.register('mate',    gp.cxOnePoint)
        tb.register('mutate',  gp.mutUniform, expr=tb.expr, pset=self.pset)
        tb.register('select',  tools.selTournament,
                    tournsize=self.TOURNAMENT_SIZE)

        # Bloat control — cegah pohon terlalu besar
        tb.decorate('mate',   gp.staticLimit(key=len, max_value=60))
        tb.decorate('mutate', gp.staticLimit(key=len, max_value=60))

        return tb

    # ── BaseGenerator interface ──────────────────────────────────────

    def initialize_population(self, pop_size: int) -> list[StrategyDNA]:
        """Generate populasi awal dengan pohon GP acak."""
        assert pop_size >= 2, 'Minimum population size adalah 2'
        deap_pop = self.toolbox.population(n=pop_size)
        dna_list = [self._deap_to_dna(ind, generation=0) for ind in deap_pop]
        log.info('Population initialized', size=pop_size)
        return dna_list

    def crossover(
        self,
        parent_a: StrategyDNA,
        parent_b: StrategyDNA,
    ) -> tuple[StrategyDNA, StrategyDNA]:
        """One-point subtree crossover antara dua parent."""
        ind_a = self._dna_to_deap(parent_a)
        ind_b = self._dna_to_deap(parent_b)
        off_a, off_b = self.toolbox.mate(ind_a, ind_b)
        gen = parent_a.generation + 1
        return self._deap_to_dna(off_a, gen), self._deap_to_dna(off_b, gen)

    def mutate(
        self,
        individual: StrategyDNA,
        mu: float = 0.1,
    ) -> StrategyDNA:
        """Uniform subtree mutation."""
        ind = self._dna_to_deap(individual)
        mutated, = self.toolbox.mutate(ind)
        return self._deap_to_dna(mutated, individual.generation + 1)

    def next_generation(
        self,
        survivors: list[StrategyDNA],
        target_size: int,
    ) -> list[StrategyDNA]:
        """
        Buat generasi berikutnya:
        1. Tournament selection dari survivors
        2. Crossover pasangan terpilih (cx_prob=0.7)
        3. Mutation pada offspring (mut_prob=0.2)
        4. Elitism: 2 individu terbaik selalu masuk
        """
        if not survivors:
            log.warning('No survivors — reinitializing population')
            return self.initialize_population(target_size)

        cx_prob = 0.7
        mut_prob = 0.2

        # Set fitness untuk DEAP selection
        deap_survivors = []
        for dna in survivors:
            ind = self._dna_to_deap(dna)
            ind.fitness.values = (dna.fitness_score,)
            deap_survivors.append(ind)

        # Elitism: pertahankan 2 terbaik
        elite = tools.selBest(deap_survivors, k=min(2, len(deap_survivors)))

        # Tournament selection untuk sisanya
        offspring = self.toolbox.select(deap_survivors, k=target_size - len(elite))
        offspring = [copy.deepcopy(ind) for ind in offspring]

        # Crossover
        for i in range(0, len(offspring)-1, 2):
            if random.random() < cx_prob:
                offspring[i], offspring[i+1] = self.toolbox.mate(
                    offspring[i], offspring[i+1]
                )
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values

        # Mutation
        for ind in offspring:
            if random.random() < mut_prob:
                ind, = self.toolbox.mutate(ind)
                del ind.fitness.values

        next_gen = [copy.deepcopy(e) for e in elite] + offspring
        next_gen_num = survivors[0].generation + 1 if survivors else 1

        result = [self._deap_to_dna(ind, next_gen_num) for ind in next_gen]
        log.info('Next generation created', generation=next_gen_num, size=len(result))
        return result

    def to_strategy_class(self, dna: StrategyDNA) -> type[BaseStrategy]:
        """
        Konversi pohon GP menjadi Python class yang mewarisi BaseStrategy.
        Kode di-generate, di-write ke file, lalu di-import secara dinamis.
        """
        class_name = f'Strategy_{dna.individual_id.replace("-", "_")}'
        file_path = self.GENERATED_DIR / f'{class_name}.py'

        code = self._generate_strategy_code(dna, class_name)
        file_path.write_text(code, encoding='utf-8')

        # Dynamic import
        spec = importlib.util.spec_from_file_location(class_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return getattr(module, class_name)

    # ── Helper methods ───────────────────────────────────────────────

    def _deap_to_dna(self, individual: gp.PrimitiveTree, generation: int) -> StrategyDNA:
        """Konversi DEAP Individual ke StrategyDNA."""
        tree_str = str(individual)
        return StrategyDNA(
            generation=generation,
            tree_repr=tree_str,
            tree_depth=individual.height,
            tree_nodes=len(individual),
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    def _dna_to_deap(self, dna: StrategyDNA) -> gp.PrimitiveTree:
        """Rekonstruksi DEAP Individual dari tree_repr string."""
        return creator.Individual(
            gp.PrimitiveTree.from_string(dna.tree_repr, self.pset)
        )

    def dna_from_dict(self, d: dict) -> StrategyDNA:
        """Rekonstruksi StrategyDNA dari dict (checkpoint resume)."""
        dna = StrategyDNA(**{k: v for k, v in d.items() if k in StrategyDNA.__dataclass_fields__})
        return dna

    def _generate_strategy_code(self, dna: StrategyDNA, class_name: str) -> str:
        """
        Generate Python source code untuk strategy class.
        Pohon GP di-compile menjadi Python expression via gp.compile().
        """
        deap_ind = self._dna_to_deap(dna)

        # Ekstrak params yang sudah di-tune Optuna
        sl_pips    = dna.params.get('sl_pips', 20.0)
        tp_pips    = dna.params.get('tp_pips', 40.0)
        threshold  = clamp_signal_threshold(dna.params.get('signal_threshold', 0.5))
        obi_w      = dna.params.get('obi_window', 20)
        tv_w       = dna.params.get('tick_vel_window', 10)
        sd_w       = dna.params.get('spread_dyn_window', 20)
        td_wsec    = dna.params.get('tick_den_window_sec', 60.0)
        vs_w       = dna.params.get('vol_skew_window', 30)
        mm_sw      = dna.params.get('mid_mom_short', 5)
        mm_lw      = dna.params.get('mid_mom_long', 20)
        sk_w       = dna.params.get('skew_window', 30)
        ku_w       = dna.params.get('kurt_window', 30)
        vws_w      = dna.params.get('vw_spread_window', 20)

        tree_expr = str(deap_ind)

        code = textwrap.dedent(f'''
            # AUTO-GENERATED by DEAPGenerator — DO NOT EDIT MANUALLY
            # DNA ID: {dna.id}
            # Generation: {dna.generation}
            # Tree: {tree_expr}
            from __future__ import annotations
            import numpy as np
            from src.strategy.base_strategy import BaseStrategy, StrategyDNA
            from src.strategy.signal_utils import normalize_signal_value
            from src.evolution.primitives import (
                safe_div, safe_log, safe_sqrt, neg, square, cube, sigmoid, sign_fn,
                max2, min2, order_book_imbalance, tick_velocity, spread_dynamics,
                tick_density, volume_clock_skew, mid_price_momentum,
                rolling_skewness, rolling_kurtosis, volume_weighted_spread,
            )

            add = lambda a, b: a + b
            sub = lambda a, b: a - b
            mul = lambda a, b: a * b
            div = safe_div
            log = safe_log
            sqrt = safe_sqrt
            sign = sign_fn

            class {class_name}(BaseStrategy):
                """Auto-generated strategy from DEAP GP tree."""

                SL_PIPS   = {sl_pips}
                TP_PIPS   = {tp_pips}
                THRESHOLD = {threshold}

                def compute_features(self, tick_data) -> dict:
                    bid = tick_data['bid'].to_numpy()
                    ask = tick_data['ask'].to_numpy()
                    bid_size = tick_data['bid_size'].to_numpy()
                    ask_size = tick_data['ask_size'].to_numpy()
                    ts = tick_data['timestamp'].cast(int).to_numpy() / 1e9
                    return {{
                        'obi':        order_book_imbalance(bid_size, ask_size, {obi_w}),
                        'tick_vel':   tick_velocity(bid, ask, {tv_w}),
                        'spread_dyn': spread_dynamics(bid, ask, {sd_w}),
                        'tick_den':   tick_density(bid, ask, ts, {td_wsec}),
                        'vol_skew':   volume_clock_skew(bid_size, ask_size, {vs_w}),
                        'mid_mom':    mid_price_momentum(bid, ask, {mm_sw}, {mm_lw}),
                        'skewness':   rolling_skewness(bid, ask, {sk_w}),
                        'kurtosis':   rolling_kurtosis(bid, ask, {ku_w}),
                        'vw_spread':  volume_weighted_spread(bid, ask, bid_size, ask_size, {vws_w}),
                    }}

                def compute_signal_value(self, features: dict) -> float:
                    obi        = features['obi']
                    tick_vel   = features['tick_vel']
                    spread_dyn = features['spread_dyn']
                    tick_den   = features['tick_den']
                    vol_skew   = features['vol_skew']
                    mid_mom    = features['mid_mom']
                    skewness   = features['skewness']
                    kurtosis   = features['kurtosis']
                    vw_spread  = features['vw_spread']
                    raw_signal = {tree_expr}
                    return normalize_signal_value(raw_signal)

                def generate_signal(self, features: dict) -> dict | None:
                    signal = self.compute_signal_value(features)
                    if signal > self.THRESHOLD:
                        return {{'side': 'BUY',  'sl_pips': self.SL_PIPS,
                                'tp_pips': self.TP_PIPS, 'order_type': 'LIMIT'}}
                    elif signal < -self.THRESHOLD:
                        return {{'side': 'SELL', 'sl_pips': self.SL_PIPS,
                                'tp_pips': self.TP_PIPS, 'order_type': 'LIMIT'}}
                    return None

                def validate_params(self) -> bool:
                    sl = self.SL_PIPS
                    tp = self.TP_PIPS
                    if not (10 <= sl <= 50):
                        raise ValueError(f'SL {{sl}} pips di luar range 10-50')
                    if tp <= 0:
                        raise ValueError(f'TP {{tp}} harus positif')
                    return True
        ''')
        return code
