pub mod density;
pub mod higher_moments;
pub mod hurst;
pub mod momentum;
pub mod obi;
pub mod pipeline;
pub mod spread;
pub mod tick_velocity;
pub mod volume_skew;

#[derive(Debug, Clone, Default)]
pub struct FeatureRow {
    pub timestamp_ns: i64,
    pub obi: f64,
    pub tick_velocity: f64,
    pub spread_dynamics: f64,
    pub tick_density: f64,
    pub volume_clock_skew: f64,
    pub mid_momentum: f64,
    pub rolling_skewness: f64,
    pub rolling_kurtosis: f64,
    pub volume_weighted_spread: f64,
    pub hurst_exponent: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct WindowConfig {
    pub short_window: usize,
    pub long_window: usize,
    pub hurst_window: usize,
}
