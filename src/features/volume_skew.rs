use crate::data::types::Tick;

pub fn compute_volume_clock_skew(ticks: &[Tick]) -> f64 {
    let (total_bid, total_ask) = ticks.iter().fold((0_u64, 0_u64), |(bid, ask), tick| {
        (bid + tick.bid_volume as u64, ask + tick.ask_volume as u64)
    });

    total_bid as f64 / ((total_bid + total_ask) as f64 + 0.5)
}

#[cfg(test)]
mod tests {
    use chrono::{LocalResult, TimeZone, Utc};

    use super::*;
    use crate::data::types::Instrument;

    #[test]
    fn computes_known_volume_clock_skew() -> Result<(), String> {
        let ticks = vec![test_tick(60, 40)?];

        assert!((compute_volume_clock_skew(&ticks) - (60.0 / 100.5)).abs() < f64::EPSILON);
        Ok(())
    }

    fn test_tick(bid_volume: u32, ask_volume: u32) -> Result<Tick, String> {
        let timestamp = match Utc.timestamp_opt(0, 0) {
            LocalResult::Single(timestamp) => timestamp,
            _ => return Err("failed to build test timestamp".to_string()),
        };

        Ok(Tick {
            timestamp,
            instrument: Instrument::EURUSD,
            bid: 1.0,
            ask: 1.1,
            bid_volume,
            ask_volume,
        })
    }
}
