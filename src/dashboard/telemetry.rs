use std::sync::Arc;

use parking_lot::RwLock;
use serde::Serialize;

pub type SharedTelemetry = Arc<RwLock<TelemetryState>>;

#[derive(Debug, Default, Clone, Serialize)]
pub struct TelemetryState {
    pub current_generation: usize,
    pub population_size: usize,
    pub best_fitness: f64,
    pub avg_fitness: f64,
    pub diversity_score: f64,
    pub ram_usage_mb: f64,
    pub elimination_funnel: EliminationFunnel,
    pub gpu_vram_usage_mb: f64,
    pub rows_per_second: f64,
    pub top_features: Vec<(String, f64)>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct EliminationFunnel {
    pub generated: usize,
    pub failed_hard_filter: usize,
    pub failed_oos: usize,
    pub failed_degradation: usize,
    pub golden_set_count: usize,
}
