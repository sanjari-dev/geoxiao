use crate::data::types::Tick;

pub fn compute_tick_velocity(ticks: &[Tick]) -> f64 {
    let Some(first_tick) = ticks.first() else {
        return 0.0;
    };
    let Some(last_tick) = ticks.last() else {
        return 0.0;
    };

    let time_diff_seconds = elapsed_seconds(first_tick, last_tick);
    if time_diff_seconds == 0.0 {
        return 0.0;
    }

    (last_tick.mid() - first_tick.mid()) / time_diff_seconds
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
    fn computes_known_tick_velocity() -> Result<(), String> {
        let ticks = vec![test_tick(0, 1.0, 1.2)?, test_tick(2, 2.0, 2.2)?];

        assert!((compute_tick_velocity(&ticks) - 0.5).abs() < f64::EPSILON);
        Ok(())
    }

    fn test_tick(seconds: i64, bid: f64, ask: f64) -> Result<Tick, String> {
        let timestamp = match Utc.timestamp_opt(seconds, 0) {
            LocalResult::Single(timestamp) => timestamp,
            _ => return Err("failed to build test timestamp".to_string()),
        };

        Ok(Tick {
            timestamp,
            instrument: Instrument::EURUSD,
            bid,
            ask,
            bid_volume: 1,
            ask_volume: 1,
        })
    }
}
