use crate::data::types::Tick;

pub fn compute_tick_density(ticks: &[Tick]) -> f64 {
    let Some(first_tick) = ticks.first() else {
        return 0.0;
    };
    let Some(last_tick) = ticks.last() else {
        return 0.0;
    };

    let time_diff_seconds = elapsed_seconds(first_tick, last_tick);
    if time_diff_seconds == 0.0 {
        return ticks.len() as f64;
    }

    ticks.len() as f64 / time_diff_seconds
}

fn elapsed_seconds(first_tick: &Tick, last_tick: &Tick) -> f64 {
    let Some(first_ns) = first_tick.timestamp.timestamp_nanos_opt() else {
        return 0.0;
    };
    let Some(last_ns) = last_tick.timestamp.timestamp_nanos_opt() else {
        return 0.0;
    };

    (last_ns - first_ns) as f64 / 1_000_000_000.0
}

#[cfg(test)]
mod tests {
    use chrono::{LocalResult, TimeZone, Utc};

    use super::*;
    use crate::data::types::Instrument;

    #[test]
    fn computes_known_tick_density() -> Result<(), String> {
        let ticks = vec![test_tick(0)?, test_tick(1)?, test_tick(2)?, test_tick(3)?];

        assert!((compute_tick_density(&ticks) - (4.0 / 3.0)).abs() < f64::EPSILON);
        assert_eq!(compute_tick_density(&[test_tick(0)?]), 1.0);
        Ok(())
    }

    fn test_tick(seconds: i64) -> Result<Tick, String> {
        let timestamp = match Utc.timestamp_opt(seconds, 0) {
            LocalResult::Single(timestamp) => timestamp,
            _ => return Err("failed to build test timestamp".to_string()),
        };

        Ok(Tick {
            timestamp,
            instrument: Instrument::EURUSD,
            bid: 1.0,
            ask: 1.1,
            bid_volume: 1,
            ask_volume: 1,
        })
    }
}
