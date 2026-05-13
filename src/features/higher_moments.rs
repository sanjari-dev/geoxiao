use crate::data::types::Tick;

const MIN_STD_DEV: f64 = 1e-10;

pub fn compute_rolling_skewness(ticks: &[Tick]) -> f64 {
    let Some((mean, std_dev)) = mean_and_std_dev(ticks) else {
        return 0.0;
    };

    if std_dev < MIN_STD_DEV {
        return 0.0;
    }

    let third_moment = ticks
        .iter()
        .map(|tick| (tick.mid() - mean).powi(3))
        .sum::<f64>()
        / ticks.len() as f64;

    third_moment / std_dev.powi(3)
}

pub fn compute_rolling_kurtosis(ticks: &[Tick]) -> f64 {
    let Some((mean, std_dev)) = mean_and_std_dev(ticks) else {
        return 0.0;
    };

    if std_dev < MIN_STD_DEV {
        return 0.0;
    }

    let fourth_moment = ticks
        .iter()
        .map(|tick| (tick.mid() - mean).powi(4))
        .sum::<f64>()
        / ticks.len() as f64;

    fourth_moment / std_dev.powi(4) - 3.0
}

fn mean_and_std_dev(ticks: &[Tick]) -> Option<(f64, f64)> {
    if ticks.is_empty() {
        return None;
    }

    let mean = ticks.iter().map(Tick::mid).sum::<f64>() / ticks.len() as f64;
    let variance = ticks
        .iter()
        .map(|tick| {
            let diff = tick.mid() - mean;
            diff * diff
        })
        .sum::<f64>()
        / ticks.len() as f64;

    Some((mean, variance.sqrt()))
}

#[cfg(test)]
mod tests {
    use chrono::{LocalResult, TimeZone, Utc};

    use super::*;
    use crate::data::types::Instrument;

    #[test]
    fn computes_known_higher_moments() -> Result<(), String> {
        let ticks = vec![test_tick(0, 1.0)?, test_tick(1, 2.0)?, test_tick(2, 3.0)?];

        assert!(compute_rolling_skewness(&ticks).abs() < f64::EPSILON);
        assert!((compute_rolling_kurtosis(&ticks) + 1.5).abs() < 1e-12);
        Ok(())
    }

    fn test_tick(seconds: i64, mid: f64) -> Result<Tick, String> {
        let timestamp = match Utc.timestamp_opt(seconds, 0) {
            LocalResult::Single(timestamp) => timestamp,
            _ => return Err("failed to build test timestamp".to_string()),
        };

        Ok(Tick {
            timestamp,
            instrument: Instrument::EURUSD,
            bid: mid,
            ask: mid,
            bid_volume: 1,
            ask_volume: 1,
        })
    }
}
