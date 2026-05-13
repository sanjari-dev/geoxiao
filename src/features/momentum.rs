use crate::data::types::Tick;

pub fn compute_mid_momentum(ticks: &[Tick], short_window: usize, long_window: usize) -> f64 {
    average_mid(recent_slice(ticks, short_window)) - average_mid(recent_slice(ticks, long_window))
}

fn recent_slice(ticks: &[Tick], window: usize) -> &[Tick] {
    if window == 0 || ticks.len() <= window {
        ticks
    } else {
        &ticks[ticks.len() - window..]
    }
}

fn average_mid(ticks: &[Tick]) -> f64 {
    if ticks.is_empty() {
        return 0.0;
    }

    ticks.iter().map(Tick::mid).sum::<f64>() / ticks.len() as f64
}

#[cfg(test)]
mod tests {
    use chrono::{LocalResult, TimeZone, Utc};

    use super::*;
    use crate::data::types::Instrument;

    #[test]
    fn computes_known_mid_momentum() -> Result<(), String> {
        let ticks = vec![
            test_tick(0, 1.0)?,
            test_tick(1, 2.0)?,
            test_tick(2, 3.0)?,
            test_tick(3, 4.0)?,
        ];

        assert!((compute_mid_momentum(&ticks, 2, 4) - 1.0).abs() < f64::EPSILON);
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
