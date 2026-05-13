use std::fs;

use serde::Serialize;
use uuid::Uuid;

use crate::{
    error::EsseError,
    gp::population::{Individual, OosMetrics, RegimeProfile, StrategyParams},
};

#[derive(Debug, Clone, Serialize)]
pub struct ExportedStrategy {
    pub id: Uuid,
    pub ast_json: String,
    pub params: StrategyParams,
    pub oos_metrics: OosMetrics,
    pub regime_profile: RegimeProfile,
}

pub fn top_n_strategy_selection_engine(population: &[Individual], top_n: usize) -> Vec<Individual> {
    let mut ranked = population
        .iter()
        .filter(|individual| individual.fitness.is_some())
        .cloned()
        .collect::<Vec<_>>();

    ranked.sort_by(|a, b| {
        b.fitness
            .unwrap_or(f64::NEG_INFINITY)
            .total_cmp(&a.fitness.unwrap_or(f64::NEG_INFINITY))
    });
    ranked.truncate(top_n);
    ranked
}

pub fn export_to_json(strategies: &[Individual], file_path: &str) -> Result<(), EsseError> {
    let exported = strategies
        .iter()
        .map(|strategy| {
            let ast_json = serde_json::to_string(&strategy.ast)
                .map_err(|error| EsseError::RuntimeError(error.to_string()))?;
            let oos_metrics = strategy.oos_metrics.clone().ok_or_else(|| {
                EsseError::RuntimeError(format!(
                    "strategy {} is missing oos_metrics for export",
                    strategy.id
                ))
            })?;
            let regime_profile = strategy.regime_profile.clone().ok_or_else(|| {
                EsseError::RuntimeError(format!(
                    "strategy {} is missing regime_profile for export",
                    strategy.id
                ))
            })?;

            Ok(ExportedStrategy {
                id: strategy.id,
                ast_json,
                params: strategy.params.clone(),
                oos_metrics,
                regime_profile,
            })
        })
        .collect::<Result<Vec<_>, EsseError>>()?;

    let json = serde_json::to_string_pretty(&exported)
        .map_err(|error| EsseError::RuntimeError(error.to_string()))?;
    fs::write(file_path, json).map_err(|error| EsseError::RuntimeError(error.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use uuid::Uuid;

    use super::*;
    use crate::gp::{
        ast::{AstNode, Terminal},
        population::{OosMetrics, RegimeLabel, RegimeProfile},
    };

    #[test]
    fn top_n_selection_drops_none_and_sorts_descending() {
        let population = vec![
            individual_with_fitness(Uuid::from_u128(1), Some(1.2)),
            individual_with_fitness(Uuid::from_u128(2), None),
            individual_with_fitness(Uuid::from_u128(3), Some(3.4)),
            individual_with_fitness(Uuid::from_u128(4), Some(2.1)),
        ];

        let selected = top_n_strategy_selection_engine(&population, 2);

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].id, Uuid::from_u128(3));
        assert_eq!(selected[1].id, Uuid::from_u128(4));
        assert!(selected
            .iter()
            .all(|individual| individual.fitness.is_some()));
    }

    fn individual_with_fitness(id: Uuid, fitness: Option<f64>) -> Individual {
        Individual {
            id,
            ast: AstNode::Leaf(Terminal::Obi),
            params: StrategyParams {
                sl_pips: 20.0,
                tp_pips: 40.0,
                signal_threshold: 0.5,
                feature_window: 30,
            },
            fitness,
            oos_metrics: Some(OosMetrics {
                avg_profit_factor: 1.8,
                regime_stability_score: 0.7,
                avg_drawdown_pips: 120.0,
                windows_passed: 3,
                windows_total: 4,
            }),
            regime_profile: Some(RegimeProfile {
                target_regime: RegimeLabel::HighVolTrending,
                survival_map: HashMap::from([(RegimeLabel::HighVolTrending, 0.9)]),
            }),
            generation_born: 0,
        }
    }
}
