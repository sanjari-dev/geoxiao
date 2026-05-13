use rand::Rng;

use crate::data::types::Tick;

const PIP_SIZE: f64 = 0.0001;

pub fn inject_noise(ticks: &mut [Tick], rng: &mut impl Rng) {
    for tick in ticks {
        if rng.gen_bool(0.3) {
            let widening = rng.gen_range(1..=5) as f64 * PIP_SIZE;
            tick.bid -= widening;
            tick.ask += widening;
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::{LocalResult, TimeZone, Utc};
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::data::types::Instrument;

    #[test]
    fn inject_noise_only_widens_or_preserves_spread() -> Result<(), String> {
        let mut ticks = (0..100).map(test_tick).collect::<Result<Vec<_>, _>>()?;
        let original = ticks.clone();
        let mut rng = StdRng::seed_from_u64(42);

        inject_noise(&mut ticks, &mut rng);

        let mut widened_count = 0;
        for (before, after) in original.iter().zip(ticks.iter()) {
            let before_spread = before.spread();
            let after_spread = after.spread();
            assert!(after_spread >= before_spread);
            if after_spread > before_spread {
                widened_count += 1;
            }
        }

        assert!(widened_count > 0);
        Ok(())
    }

    fn test_tick(index: usize) -> Result<Tick, String> {
        let timestamp = match Utc.timestamp_opt(index as i64, 0) {
            LocalResult::Single(timestamp) => timestamp,
            _ => return Err("failed to build test timestamp".to_string()),
        };

        Ok(Tick {
            timestamp,
            instrument: Instrument::EURUSD,
            bid: 1.1000,
            ask: 1.1002,
            bid_volume: 10,
            ask_volume: 12,
        })
    }
}
