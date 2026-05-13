use crate::{
    data::types::Tick,
    features::{
        density::compute_tick_density,
        higher_moments::{compute_rolling_kurtosis, compute_rolling_skewness},
        hurst::compute_hurst,
        momentum::compute_mid_momentum,
        obi::compute_obi,
        spread::{compute_spread_dynamics, compute_volume_weighted_spread},
        tick_velocity::compute_tick_velocity,
        volume_skew::compute_volume_clock_skew,
        FeatureRow, WindowConfig,
    },
};

pub fn compute_features(ticks: &[Tick], config: &WindowConfig) -> FeatureRow {
    let Some(last_tick) = ticks.last() else {
        return FeatureRow::default();
    };

    let hurst_ticks = if config.hurst_window > 0 && ticks.len() > config.hurst_window {
        &ticks[ticks.len() - config.hurst_window..]
    } else {
        ticks
    };
    let prices = hurst_ticks
        .iter()
        .map(|tick| tick.mid())
        .collect::<Vec<_>>();

    FeatureRow {
        timestamp_ns: match last_tick.timestamp.timestamp_nanos_opt() {
            Some(timestamp_ns) => timestamp_ns,
            None => 0,
        },
        obi: compute_obi(ticks),
        tick_velocity: compute_tick_velocity(ticks),
        spread_dynamics: compute_spread_dynamics(ticks),
        tick_density: compute_tick_density(ticks),
        volume_clock_skew: compute_volume_clock_skew(ticks),
        mid_momentum: compute_mid_momentum(ticks, config.short_window, config.long_window),
        rolling_skewness: compute_rolling_skewness(ticks),
        rolling_kurtosis: compute_rolling_kurtosis(ticks),
        volume_weighted_spread: compute_volume_weighted_spread(ticks),
        hurst_exponent: compute_hurst(&prices),
    }
}
