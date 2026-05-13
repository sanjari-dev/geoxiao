use crate::data::types::Tick;

pub fn compute_obi(ticks: &[Tick]) -> f64 {
    let total_bid: u64 = ticks.iter().map(|tick| tick.bid_volume as u64).sum();
    let total_ask: u64 = ticks.iter().map(|tick| tick.ask_volume as u64).sum();
    let total = total_bid + total_ask;

    if total == 0 {
        return 0.0;
    }

    (total_bid as f64 - total_ask as f64) / total as f64
}

#[cfg(test)]
mod tests {
    use chrono::{LocalResult, TimeZone, Utc};

    use super::*;
    use crate::data::types::Instrument;

    #[test]
    fn computes_known_order_book_imbalance() -> Result<(), String> {
        let ticks = vec![test_tick(100, 50)?, test_tick(50, 50)?];

        let obi = compute_obi(&ticks);

        assert!((obi - 0.2).abs() < f64::EPSILON);
        Ok(())
    }

    #[test]
    fn returns_zero_when_volume_total_is_zero() -> Result<(), String> {
        let ticks = vec![test_tick(0, 0)?];

        assert_eq!(compute_obi(&ticks), 0.0);
        Ok(())
    }

    fn test_tick(bid_volume: u32, ask_volume: u32) -> Result<Tick, String> {
        let timestamp = match Utc.timestamp_opt(1_800_000_000, 0) {
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
