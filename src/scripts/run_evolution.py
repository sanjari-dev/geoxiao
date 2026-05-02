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
from src.evolution.fitness import HardConstraintEvaluator, FitnessScorer
from src.evolution.persistence import CheckpointManager, EvolutionCheckpoint
from src.analytics.metrics_calculator import MetricsCalculator
from src.data.repositories.strategy_repo import StrategyRepository
from src.data.repositories.trial_repo import TrialRepository
from src.data.schema_bootstrap import ensure_postgres_schema

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


@ray.remote
def _evaluate_dna_clickhouse_remote(
    dna_dict: dict,
    trial_id: str,
) -> tuple[dict, object] | None:
    """
    Ray worker for one ClickHouse vectorized backtest.

    Deferred imports are intentional: Ray serializes this function to workers,
    and importing the ClickHouse/PostgreSQL stack inside the worker keeps the
    driver process free from worker-only runtime objects.
    """

    import asyncio
    import traceback

    import structlog

    from src.analytics.metrics_calculator import MetricsCalculator
    from src.backtest.clickhouse_evaluator import ClickHouseEvaluator
    from src.data.repositories.trade_repo import TradeRepository
    from src.strategy.base_strategy import StrategyDNA

    worker_log = structlog.get_logger("clickhouse_eval_worker")
    dna = StrategyDNA(
        **{
            k: v
            for k, v in dna_dict.items()
            if k in StrategyDNA.__dataclass_fields__
        }
    )
    dna.params = dna_dict.get("params") or dna_dict.get("params_json") or {}

    loop = asyncio.new_event_loop()
    trade_repo = TradeRepository()

    try:
        asyncio.set_event_loop(loop)

        def persist_trade_batch(batch: list[dict]) -> None:
            loop.run_until_complete(trade_repo.batch_insert(batch))

        evaluator = ClickHouseEvaluator()
        evaluator.evaluate_stream(
            dna,
            trial_id=trial_id,
            persist_trade_sink=persist_trade_batch,
        )

        monthly_metrics = MetricsCalculator().compute_monthly_metrics(trial_id)
        return vars(dna).copy(), monthly_metrics

    except Exception as e:
        worker_log.error(
            "ClickHouse worker evaluation failed",
            dna_id=str(dna.id)[:8],
            trial_id=trial_id,
            error=str(e),
            tb=traceback.format_exc()[-1000:],
        )
        return None
    finally:
        try:
            loop.run_until_complete(trade_repo.close())
        finally:
            loop.close()
            asyncio.set_event_loop(None)


def _fetch_optuna_sample_data(symbol: str):
    """Fetch a small ClickHouse sample for Optuna proxy evaluation."""

    import clickhouse_connect
    import polars as pl

    client = clickhouse_connect.get_client(
        host=settings.CH_HOST,
        port=settings.CH_PORT,
        database=settings.CH_DATABASE,
        username=settings.CH_USER,
        password=settings.CH_PASSWORD,
        connect_timeout=10,
        send_receive_timeout=300,
    )
    query = f"""
        SELECT
            timestamp,
            bid,
            ask,
            bid_volume AS bid_size,
            ask_volume AS ask_size
        FROM `{settings.CH_DATABASE}`.`ticks`
        WHERE instrument = '{symbol.replace("'", "\\'")}'
        ORDER BY timestamp ASC
        LIMIT 50000
    """
    try:
        return pl.from_pandas(client.query_df(query))
    except Exception as e:
        log.warning("Optuna sample data fetch failed", symbol=symbol, error=str(e))
        return pl.DataFrame()


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
    ensure_postgres_schema()

    # ── Inisialisasi komponen ─────────────────────────────────────────
    checkpoint_mgr = CheckpointManager(run_id)
    
    generator = DEAPGenerator()
    constraint_eval = HardConstraintEvaluator()
    fitness_scorer = FitnessScorer()
    metrics_calc = MetricsCalculator()
    strategy_repo = StrategyRepository()
    trial_repo = TrialRepository()
    sample_data_cache = {}

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

        # Step 1: Optuna tuning, then parallel ClickHouse backtest via Ray.
        tuned_population = []
        for dna in population:
            try:
                tuner = OptunaTuner(
                    study_name=f'gen{gen}',
                    n_trials=settings.OPTUNA_N_TRIALS,
                )
                generator_for_class = DEAPGenerator(symbol=dna.symbol, timeframe=dna.timeframe)
                if dna.symbol not in sample_data_cache:
                    sample_data_cache[dna.symbol] = _fetch_optuna_sample_data(
                        dna.symbol,
                    )
                sample_data = sample_data_cache[dna.symbol]
                if not sample_data.is_empty():
                    dna = tuner.tune(
                        dna=dna,
                        tick_data=sample_data,
                        strategy_class_factory=generator_for_class.to_strategy_class,
                    )
                tuned_population.append(dna)
            except Exception as e:
                log.warning('Optuna tuning failed — using default params',
                            dna_id=dna.id[:8], error=str(e))
                tuned_population.append(dna)

        for dna in tuned_population:
            await strategy_repo.save(dna)

        ray_futures = []
        for dna in tuned_population:
            trial_id = str(uuid.uuid4())
            await trial_repo.save({
                'id': trial_id,
                'strategy_id': dna.id,
                'optuna_trial_id': -1,
                'study_name': f'gen{gen}_{dna.id[:8]}',
                'params_json': dna.params,
            })
            ray_futures.append(
                _evaluate_dna_clickhouse_remote.remote(
                    vars(dna).copy(),
                    trial_id,
                )
            )

        log.info('ClickHouse Ray tasks dispatched', generation=gen, count=len(ray_futures))

        raw_results = ray.get(ray_futures, timeout=3600) if ray_futures else []
        eval_results = []
        for result in raw_results:
            if result is None:
                continue
            dna_dict, monthly_metrics = result
            eval_results.append((generator.dna_from_dict(dna_dict), monthly_metrics))

        total_evaluated += len(eval_results)

        # Step 2 & 3: Constraint check + Fitness scoring
        survivors = []
        for dna, monthly_metrics in eval_results:
            constraint_result = constraint_eval.evaluate(monthly_metrics)
            trial_id = (
                str(monthly_metrics['trial_id'][0])
                if 'trial_id' in monthly_metrics.columns and not monthly_metrics.is_empty()
                else None
            )
            summary = metrics_calc.get_summary_stats(monthly_metrics)
            total_pips = (
                float(monthly_metrics['net_pips'].sum())
                if 'net_pips' in monthly_metrics.columns
                else None
            )
            trade_count = (
                int(monthly_metrics['trade_count'].sum())
                if 'trade_count' in monthly_metrics.columns
                else None
            )

            if constraint_result['passed']:
                score = fitness_scorer.compute(monthly_metrics)
                dna.fitness_score = score
                dna.status = 'passed'
                dna.eliminated_reason = None
                survivors.append(dna)
                total_passed += 1
            else:
                dna.status = 'eliminated'
                dna.eliminated_reason = ' | '.join(constraint_result['flags'])

            if trial_id:
                metrics_calc.update_constraint_result(
                    trial_id,
                    passed=constraint_result['passed'],
                    flags=constraint_result.get('flags', []),
                )
                await trial_repo.update_results(
                    trial_id,
                    profit_factor=summary.get('avg_profit_factor'),
                    total_pips=total_pips,
                    max_drawdown_pips=summary.get('max_drawdown'),
                    trade_count=trade_count,
                    fitness_score=dna.fitness_score if constraint_result['passed'] else None,
                    eliminated_reason=dna.eliminated_reason,
                )
            await strategy_repo.update_status(dna.id, dna.status)

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

    ensure_postgres_schema()
    ray.init(address=settings.RAY_ADDRESS or 'auto', ignore_reinit_error=True)
    log.info('Ray initialized', address=settings.RAY_ADDRESS or 'auto')

    asyncio.run(run_evolution(
        run_id=args.run_id,
        resume=args.resume,
        n_generations=args.generations,
        pop_size=args.pop_size,
    ))
