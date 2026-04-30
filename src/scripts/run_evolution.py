# src/scripts/run_evolution.py
# Referensi: Blueprint §6.4

from __future__ import annotations
import asyncio
import signal
import uuid
import ray
import structlog
from src.config.logging import configure_logging
from src.config.settings import settings
from src.evolution.gp_generator import DEAPGenerator
from src.evolution.optuna_tuner import OptunaTuner
from src.backtest.backtest_runner import RayBacktestRunner
from src.evolution.fitness import HardConstraintEvaluator, FitnessScorer
from src.evolution.persistence import CheckpointManager, EvolutionCheckpoint
from src.analytics.metrics_calculator import MetricsCalculator

# Jika logger belum didefinisikan secara penuh di src/config/logging.py, kita setup sederhana
try:
    configure_logging()
except Exception:
    pass

log = structlog.get_logger(__name__)

# ── Graceful shutdown flag ────────────────────────────────────────────────
STOP_FLAG = False

def _handle_signal(sig, frame) -> None:
    global STOP_FLAG
    log.warning('Signal received — akan stop setelah generasi ini selesai',
                signal=sig)
    STOP_FLAG = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


async def run_evolution(
    run_id: str | None = None,
    resume: bool = False,
    n_generations: int | None = None,
    pop_size: int | None = None,
) -> None:
    """
    Main evolution loop.

    Pipeline per generasi:
    1. evaluate_population() via Ray (parallel backtest)
    2. HardConstraintEvaluator → filter survivors
    3. FitnessScorer → score survivors
    4. DEAPGenerator.next_generation() → buat generasi berikutnya
    5. Checkpoint setiap 5 generasi
    """
    global STOP_FLAG
    STOP_FLAG = False  # Reset untuk run baru

    run_id = run_id or str(uuid.uuid4())[:8]
    n_gen = n_generations or settings.N_GENERATIONS
    p_size = pop_size or settings.POP_SIZE

    log.info('Evolution starting', run_id=run_id, n_generations=n_gen, pop_size=p_size)

    # ── Inisialisasi komponen ─────────────────────────────────────────
    checkpoint_mgr = CheckpointManager(run_id)
    
    generator = DEAPGenerator()
    constraint_eval = HardConstraintEvaluator()
    fitness_scorer = FitnessScorer()
    metrics_calc = MetricsCalculator()
    backtest_runner = RayBacktestRunner()

    # ── Hall of Fame (top 10 sepanjang masa) ─────────────────────────
    hall_of_fame: list = []
    total_evaluated = 0
    total_passed = 0

    # ── Populasi awal atau resume ─────────────────────────────────────
    start_gen = 0
    population = []
    if resume:
        chk = checkpoint_mgr.load_latest()
        if chk:
            population = [generator.dna_from_dict(d) for d in chk.population]
            hall_of_fame = chk.hall_of_fame
            start_gen = chk.generation + 1
            total_evaluated = chk.total_evaluated
            total_passed = chk.total_passed
            log.info('Resumed from checkpoint', from_generation=start_gen)
        else:
            log.warning('Resume requested but no checkpoint found — starting fresh')
            population = generator.initialize_population(p_size)
    else:
        population = generator.initialize_population(p_size)

    # ── Evolution loop ────────────────────────────────────────────────
    for gen in range(start_gen, n_gen):
        if STOP_FLAG:
            log.warning('STOP_FLAG set — keluar dari evolution loop', generation=gen)
            break

        log.info('Generation start', gen=gen, pop_size=len(population))

        # Step 1: Parallel backtest via Ray
        # evaluate_population() return list[(StrategyDNA, pl.DataFrame)]
        eval_results = []
        eval_results = await backtest_runner.evaluate_population(population, gen)
        total_evaluated += len(eval_results)

        # Step 2 & 3: Constraint check + Fitness scoring
        survivors = []
        for dna, monthly_metrics in eval_results:
            constraint_result = constraint_eval.evaluate(monthly_metrics)
            if constraint_result['passed']:
                score = fitness_scorer.compute(monthly_metrics)
                dna.fitness_score = score
                dna.status = 'passed'
                survivors.append(dna)
                total_passed += 1
            else:
                dna.status = 'eliminated'
                dna.eliminated_reason = ' | '.join(constraint_result['flags'])

        log.info('Generation evaluated',
                 gen=gen,
                 total=len(eval_results),
                 survivors=len(survivors),
                 eliminated=len(eval_results) - len(survivors))

        if not survivors and eval_results:
            log.warning('Zero survivors — all eliminated. Reinitializing subset.')
            population = generator.initialize_population(p_size)
            continue

        # Update Hall of Fame
        all_candidates = hall_of_fame + [vars(d) for d in survivors]
        hall_of_fame = sorted(
            all_candidates,
            key=lambda x: x.get('fitness_score', 0),
            reverse=True
        )[:10]

        # Step 4: Next generation via DEAP
        if survivors:
            population = generator.next_generation(survivors, p_size)

        # Step 5: Checkpoint setiap 5 generasi atau saat stop
        if gen % 5 == 0 or STOP_FLAG:
            checkpoint = EvolutionCheckpoint(
                run_id=run_id,
                generation=gen,
                population=[vars(d) for d in population] if population else [],
                hall_of_fame=hall_of_fame,
                optuna_storage=settings.OPTUNA_STORAGE,
                config_snapshot=settings.model_dump(),
                total_evaluated=total_evaluated,
                total_passed=total_passed,
            )
            checkpoint_mgr.save(checkpoint)

    log.info('Evolution complete',
             run_id=run_id,
             generations_run=gen - start_gen + 1,
             total_evaluated=total_evaluated,
             total_passed=total_passed,
             hof_best_score=hall_of_fame[0].get('fitness_score') if hall_of_fame else 0)


# ── Entrypoint ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Geoxiao Evolution Runner')
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--generations', type=int, default=None)
    parser.add_argument('--pop-size', type=int, default=None)
    args = parser.parse_args()

    ray.init(address=settings.RAY_ADDRESS or 'auto', ignore_reinit_error=True)
    log.info('Ray initialized', address=settings.RAY_ADDRESS or 'auto')

    asyncio.run(run_evolution(
        run_id=args.run_id,
        resume=args.resume,
        n_generations=args.generations,
        pop_size=args.pop_size,
    ))
