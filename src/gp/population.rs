use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::gp::ast::AstNode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    pub id: Uuid,
    pub ast: AstNode,
    pub params: StrategyParams,
    pub fitness: Option<f64>,
    pub oos_metrics: Option<OosMetrics>,
    pub regime_profile: Option<RegimeProfile>,
    pub generation_born: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyParams {
    pub sl_pips: f64,
    pub tp_pips: f64,
    pub signal_threshold: f64,
    pub feature_window: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OosMetrics {
    pub avg_profit_factor: f64,
    pub regime_stability_score: f64,
    pub avg_drawdown_pips: f64,
    pub windows_passed: usize,
    pub windows_total: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegimeLabel {
    HighVolTrending,
    LowVolTrending,
    HighVolRanging,
    LowVolRanging,
    RandomNoise,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeProfile {
    pub target_regime: RegimeLabel,
    pub survival_map: HashMap<RegimeLabel, f64>,
}
