use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Instrument {
    XAUUSD = 1,
    AUDJPY = 2,
    CHFJPY = 3,
    CADJPY = 4,
    GBPJPY = 5,
    EURJPY = 6,
    NZDJPY = 7,
    USDJPY = 8,
    GBPUSD = 9,
    EURUSD = 10,
    USDCHF = 11,
    USDCAD = 12,
    AUDUSD = 13,
    NZDUSD = 14,
    EURGBP = 15,
    EURNZD = 16,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Timeframe {
    M1 = 1,
    M2 = 2,
    M3 = 3,
    M4 = 4,
    M5 = 5,
    M6 = 6,
    M10 = 7,
    M12 = 8,
    M15 = 9,
    M20 = 10,
    M30 = 11,
    H1 = 12,
    H2 = 13,
    H3 = 14,
    H4 = 15,
    H6 = 16,
    H8 = 17,
    H12 = 18,
    D1 = 19,
    W1 = 20,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    pub timestamp: DateTime<Utc>,
    pub instrument: Instrument,
    pub bid: f64,
    pub ask: f64,
    pub bid_volume: u32,
    pub ask_volume: u32,
}

impl Tick {
    #[inline(always)]
    pub fn mid(&self) -> f64 {
        (self.bid + self.ask) * 0.5
    }

    #[inline(always)]
    pub fn spread(&self) -> f64 {
        self.ask - self.bid
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub instrument: Instrument,
    pub timeframe: Timeframe,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub tick_count: u32,
    pub total_bid_volume: u64,
    pub total_ask_volume: u64,
    pub min_spread: f64,
    pub max_spread: f64,
    pub avg_spread: f64,
    pub vwap: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_tick_to_json() -> Result<(), serde_json::Error> {
        let tick = Tick {
            timestamp: DateTime::parse_from_rfc3339("2026-05-13T09:30:00Z")
                .map_err(serde::de::Error::custom)?
                .with_timezone(&Utc),
            instrument: Instrument::EURUSD,
            bid: 1.08342,
            ask: 1.08355,
            bid_volume: 125,
            ask_volume: 140,
        };

        let json = serde_json::to_string(&tick)?;
        let value: serde_json::Value = serde_json::from_str(&json)?;

        assert_eq!(value["timestamp"], "2026-05-13T09:30:00Z");
        assert_eq!(value["instrument"], "EURUSD");
        assert_eq!(value["bid"], 1.08342);
        assert_eq!(value["ask"], 1.08355);
        assert_eq!(value["bid_volume"], 125);
        assert_eq!(value["ask_volume"], 140);
        assert!((tick.mid() - 1.083485).abs() < f64::EPSILON);
        assert!((tick.spread() - 0.00013).abs() < f64::EPSILON);

        Ok(())
    }
}
