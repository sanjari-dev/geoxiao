use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use clickhouse::{query::RowCursor, Client, Compression, Row};
use futures::{stream, Stream};
use serde::Deserialize;

use crate::{
    data::types::{Instrument, Tick},
    error::EsseError,
};

const DEFAULT_WARMUP_ROWS: usize = 5_000;

pub struct ClickHouseDataSource {
    client: Client,
    warmup_rows: usize,
}

impl ClickHouseDataSource {
    pub fn new(url: &str, database: &str) -> Self {
        let client = Client::default()
            .with_url(url)
            .with_database(database)
            .with_compression(Compression::Lz4);

        Self {
            client,
            warmup_rows: DEFAULT_WARMUP_ROWS,
        }
    }

    pub async fn stream_ticks_for_month(
        &self,
        instrument: Instrument,
        year: i32,
        month: u32,
    ) -> impl Stream<Item = Result<Tick, EsseError>> {
        let (month_start, month_end) = match month_bounds(year, month) {
            Ok(bounds) => bounds,
            Err(error) => return tick_stream(TickStreamState::Error(Some(error))),
        };

        let month_start = format_clickhouse_datetime64(month_start);
        let month_end = format_clickhouse_datetime64(month_end);
        let instrument_id = instrument as u8;

        let cursor = self
            .client
            .query(
                r#"
                SELECT
                    toDateTime64(timestamp, 6, 'UTC') AS timestamp,
                    toUInt8(instrument) AS instrument,
                    bid,
                    ask,
                    bid_volume,
                    ask_volume
                FROM ticks
                WHERE instrument = ?
                  AND timestamp >= ifNull(
                    (
                        SELECT minOrNull(timestamp)
                        FROM (
                            SELECT timestamp
                            FROM ticks
                            WHERE instrument = ?
                              AND timestamp < parseDateTime64BestEffort(?, 6, 'UTC')
                            ORDER BY timestamp DESC
                            LIMIT ?
                        )
                    ),
                    parseDateTime64BestEffort(?, 6, 'UTC')
                  )
                  AND timestamp < parseDateTime64BestEffort(?, 6, 'UTC')
                ORDER BY timestamp
                "#,
            )
            .bind(instrument_id)
            .bind(instrument_id)
            .bind(&month_start)
            .bind(self.warmup_rows)
            .bind(&month_start)
            .bind(&month_end)
            .fetch::<TickRow>();

        match cursor {
            Ok(cursor) => tick_stream(TickStreamState::Cursor(cursor)),
            Err(error) => tick_stream(TickStreamState::Error(Some(error.into()))),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FeatureState;

#[derive(Debug, Clone, Default)]
pub struct StrategyStates;

#[derive(Debug, Clone, Default)]
pub struct ChunkContext {
    pub last_ticks: Vec<Tick>,
    pub feature_state: FeatureState,
    pub strategy_states: StrategyStates,
}

#[derive(Debug, Row, Deserialize)]
struct TickRow {
    #[serde(with = "clickhouse::serde::chrono::datetime64::micros")]
    timestamp: DateTime<Utc>,
    instrument: InstrumentId,
    bid: f64,
    ask: f64,
    bid_volume: u32,
    ask_volume: u32,
}

impl From<TickRow> for Tick {
    fn from(row: TickRow) -> Self {
        Self {
            timestamp: row.timestamp,
            instrument: row.instrument.0,
            bid: row.bid,
            ask: row.ask,
            bid_volume: row.bid_volume,
            ask_volume: row.ask_volume,
        }
    }
}

#[derive(Debug)]
struct InstrumentId(Instrument);

impl<'de> Deserialize<'de> for InstrumentId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        instrument_from_id(value)
            .map(Self)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown instrument id {value}")))
    }
}

enum TickStreamState {
    Cursor(RowCursor<TickRow>),
    Error(Option<EsseError>),
}

fn tick_stream(initial_state: TickStreamState) -> impl Stream<Item = Result<Tick, EsseError>> {
    stream::unfold(initial_state, |state| async move {
        match state {
            TickStreamState::Cursor(mut cursor) => match cursor.next().await {
                Ok(Some(row)) => Some((Ok(row.into()), TickStreamState::Cursor(cursor))),
                Ok(None) => None,
                Err(error) => Some((Err(error.into()), TickStreamState::Error(None))),
            },
            TickStreamState::Error(Some(error)) => Some((Err(error), TickStreamState::Error(None))),
            TickStreamState::Error(None) => None,
        }
    })
}

fn month_bounds(year: i32, month: u32) -> Result<(DateTime<Utc>, DateTime<Utc>), EsseError> {
    if !(1..=12).contains(&month) {
        return Err(EsseError::DataParseError(format!(
            "invalid month {month}; expected 1 through 12"
        )));
    }

    let start_date = NaiveDate::from_ymd_opt(year, month, 1).ok_or_else(|| {
        EsseError::DataParseError(format!("invalid year/month combination: {year}-{month:02}"))
    })?;

    let (end_year, end_month) = if month == 12 {
        let next_year = year.checked_add(1).ok_or_else(|| {
            EsseError::DataParseError(format!("year overflow while computing month end: {year}"))
        })?;
        (next_year, 1)
    } else {
        (year, month + 1)
    };

    let end_date = NaiveDate::from_ymd_opt(end_year, end_month, 1).ok_or_else(|| {
        EsseError::DataParseError(format!(
            "invalid next month combination: {end_year}-{end_month:02}"
        ))
    })?;

    let start = start_date.and_hms_opt(0, 0, 0).ok_or_else(|| {
        EsseError::DataParseError(format!("invalid month start time: {year}-{month:02}-01"))
    })?;
    let end = end_date.and_hms_opt(0, 0, 0).ok_or_else(|| {
        EsseError::DataParseError(format!(
            "invalid month end time: {end_year}-{end_month:02}-01"
        ))
    })?;

    Ok((Utc.from_utc_datetime(&start), Utc.from_utc_datetime(&end)))
}

fn format_clickhouse_datetime64(timestamp: DateTime<Utc>) -> String {
    timestamp.format("%Y-%m-%d %H:%M:%S%.6f").to_string()
}

fn instrument_from_id(value: u8) -> Option<Instrument> {
    match value {
        1 => Some(Instrument::XAUUSD),
        2 => Some(Instrument::AUDJPY),
        3 => Some(Instrument::CHFJPY),
        4 => Some(Instrument::CADJPY),
        5 => Some(Instrument::GBPJPY),
        6 => Some(Instrument::EURJPY),
        7 => Some(Instrument::NZDJPY),
        8 => Some(Instrument::USDJPY),
        9 => Some(Instrument::GBPUSD),
        10 => Some(Instrument::EURUSD),
        11 => Some(Instrument::USDCHF),
        12 => Some(Instrument::USDCAD),
        13 => Some(Instrument::AUDUSD),
        14 => Some(Instrument::NZDUSD),
        15 => Some(Instrument::EURGBP),
        16 => Some(Instrument::EURNZD),
        _ => None,
    }
}

#[allow(dead_code)]
fn assert_chunk_context_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ChunkContext>();
}
