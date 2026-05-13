#![cfg(feature = "gpu-cuda")]

use std::sync::Arc;

use cudarc::driver::CudaContext;

use crate::{error::EsseError, features::FeatureRow, gp::population::Individual};

pub struct GpuEvaluator {
    context: Arc<CudaContext>,
}

impl GpuEvaluator {
    pub fn new() -> Result<Self, EsseError> {
        let context = CudaContext::new(0)?;
        Ok(Self { context })
    }

    pub fn evaluate_population_batch(
        &self,
        population: &[Individual],
        features: &[FeatureRow],
    ) -> Result<Vec<f64>, EsseError> {
        // RULE-GPU-03: the feature matrix should be uploaded to VRAM only once per generation.
        let _ = features;
        let stream = self.context.default_stream();
        let results_dev = stream.alloc_zeros::<f64>(population.len())?;

        // Placeholder: this is where AST compilation to PTX and the launch! kernel dispatch would go.

        let results = stream.clone_dtoh(&results_dev)?;
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::gp::{
        ast::{AstNode, Terminal},
        population::{Individual, OosMetrics, RegimeLabel, RegimeProfile, StrategyParams},
    };
    use uuid::Uuid;

    #[test]
    #[ignore = "requires CUDA-capable hardware and drivers"]
    fn initializes_gpu_evaluator_and_runs_dummy_batch() -> Result<(), EsseError> {
        let evaluator = GpuEvaluator::new()?;
        let population = vec![dummy_individual(Uuid::from_u128(1))];
        let features = vec![FeatureRow::default(); 4];

        let scores = evaluator.evaluate_population_batch(&population, &features)?;

        assert_eq!(scores.len(), population.len());
        Ok(())
    }

    fn dummy_individual(id: Uuid) -> Individual {
        Individual {
            id,
            ast: AstNode::Leaf(Terminal::Obi),
            params: StrategyParams {
                sl_pips: 20.0,
                tp_pips: 40.0,
                signal_threshold: 0.5,
                feature_window: 30,
            },
            fitness: Some(1.0),
            oos_metrics: Some(OosMetrics {
                avg_profit_factor: 1.5,
                regime_stability_score: 0.7,
                avg_drawdown_pips: 100.0,
                windows_passed: 3,
                windows_total: 4,
            }),
            regime_profile: Some(RegimeProfile {
                target_regime: RegimeLabel::RandomNoise,
                survival_map: HashMap::from([(RegimeLabel::RandomNoise, 1.0)]),
            }),
            generation_born: 0,
        }
    }
}
