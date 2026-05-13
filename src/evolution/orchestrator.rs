use rayon::prelude::*;
use uuid::Uuid;

use crate::{
    dashboard::telemetry::SharedTelemetry,
    error::EsseError,
    gp::{
        ast::{AstNode, Primitive, Terminal},
        population::{Individual, StrategyParams},
    },
};

const DUMMY_POPULATION_SIZE: usize = 32;
const GENERATION_LIMIT: usize = 1;

pub async fn run_evolution_loop(telemetry: SharedTelemetry) -> Result<(), EsseError> {
    let mut population = build_dummy_population();

    for generation in 0..GENERATION_LIMIT {
        population = tokio::task::spawn_blocking(move || {
            population.par_iter_mut().for_each(|individual| {
                let structural_bonus = individual.ast.node_count() as f64 * 0.001;
                individual.fitness = Some(1.0 + structural_bonus);
            });

            population
        })
        .await?;

        let population_size = population.len();
        let best_fitness = population
            .iter()
            .filter_map(|individual| individual.fitness)
            .fold(f64::NEG_INFINITY, f64::max);
        let fitness_sum = population
            .iter()
            .filter_map(|individual| individual.fitness)
            .sum::<f64>();
        let avg_fitness = if population_size == 0 {
            0.0
        } else {
            fitness_sum / population_size as f64
        };

        {
            let mut state = telemetry.write();
            state.current_generation = generation;
            state.population_size = population_size;
            state.best_fitness = if best_fitness.is_finite() {
                best_fitness
            } else {
                0.0
            };
            state.avg_fitness = avg_fitness;
            state.elimination_funnel.generated = population_size;
        }
    }

    Ok(())
}

fn build_dummy_population() -> Vec<Individual> {
    (0..DUMMY_POPULATION_SIZE)
        .map(|index| Individual {
            id: Uuid::from_u128(index as u128 + 1),
            ast: dummy_ast(),
            params: StrategyParams {
                sl_pips: 20.0,
                tp_pips: 40.0,
                signal_threshold: 0.5,
                feature_window: 30,
            },
            fitness: None,
            oos_metrics: None,
            regime_profile: None,
            generation_born: 0,
        })
        .collect()
}

fn dummy_ast() -> AstNode {
    AstNode::Binary {
        op: Primitive::Add,
        left: Box::new(AstNode::Leaf(Terminal::TickVelocity)),
        right: Box::new(AstNode::Leaf(Terminal::Constant(1.0))),
    }
}
