use crate::data::types::Tick;

pub fn compute_spread_dynamics(ticks: &[Tick]) -> f64 {
    let len = ticks.len();
    if len < 2 {
        return 0.0;
    }

    let mean = ticks.iter().map(Tick::spread).sum::<f64>() / len as f64;
    let variance = ticks
        .iter()
        .map(|tick| {
            let diff = tick.spread() - mean;
            diff * diff
        })
        .sum::<f64>()
        / (len - 1) as f64;

    variance.sqrt()
}

pub fn compute_volume_weighted_spread(ticks: &[Tick]) -> f64 {
    let (weighted_spread, total_volume) =
        ticks.iter().fold((0.0, 0.0), |(weighted, total), tick| {
            let volume = (tick.bid_volume as u64 + tick.ask_volume as u64) as f64;
            (weighted + tick.spread() * volume, total + volume)
        });

    if total_volume == 0.0 {
        return 0.0;
    }

    weighted_spread / total_volume
}

#[cfg(test)]
mod tests {
    use chrono::{LocalResult, TimeZone, Utc};

    use super::*;
    use crate::data::types::Instrument;

    #[test]
    fn computes_known_spread_statistics() -> Result<(), String> {
        let ticks = vec![
            test_tick(0, 1.0, 2.0, 1, 1)?,
            test_tick(1, 1.0, 3.0, 2, 2)?,
            test_tick(2, 1.0, 4.0, 3, 3)?,
        ];

        assert!((compute_spread_dynamics(&ticks) - 1.0).abs() < f64::EPSILON);
        assert!((compute_volume_weighted_spread(&ticks) - (14.0 / 6.0)).abs() < 1e-12);
        Ok(())
    }

    fn test_tick(
        seconds: i64,
        bid: f64,
        ask: f64,
        bid_volume: u32,
        ask_volume: u32,
    ) -> Result<Tick, String> {
        let timestamp = match Utc.timestamp_opt(seconds, 0) {
            LocalResult::Single(timestamp) => timestamp,
            _ => return Err("failed to build test timestamp".to_string()),
        };

        Ok(Tick {
            timestamp,
            instrument: Instrument::EURUSD,
            bid,
            ask,
            bid_volume,
            ask_volume,
        })
    }
}
