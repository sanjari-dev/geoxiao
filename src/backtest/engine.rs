use chrono::{DateTime, Utc};

use crate::{
    data::types::Tick,
    features::FeatureRow,
    gp::{ast::AstNode, population::StrategyParams},
};

const PIP_SIZE: f64 = 0.0001;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    Hold,
    Buy,
}

impl Default for Signal {
    fn default() -> Self {
        Self::Hold
    }
}

#[derive(Debug, Clone)]
pub struct Position {
    pub entry_time: DateTime<Utc>,
    pub entry_price: f64,
    pub sl_price: f64,
    pub tp_price: f64,
    pub max_hold_ticks: usize,
    pub hold_ticks: usize,
}

#[derive(Debug, Clone)]
pub struct StrategyState {
    pub open_position: Option<Position>,
    pub equity: f64,
    pub last_signal: Signal,
}

impl Default for StrategyState {
    fn default() -> Self {
        Self {
            open_position: None,
            equity: 0.0,
            last_signal: Signal::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TradeResult {
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub gross_pips: f64,
    pub net_pips: f64,
    pub exit_reason: String,
}

pub fn process_tick(
    state: &mut StrategyState,
    feature: &FeatureRow,
    tick: &Tick,
    ast: &AstNode,
    params: &StrategyParams,
    broker_spread_pips: f64,
    broker_commission: f64,
) -> Option<TradeResult> {
    let raw_signal = ast.evaluate(feature);
    let current_signal = if raw_signal > params.signal_threshold {
        Signal::Buy
    } else {
        Signal::Hold
    };
    let should_enter = state.last_signal == Signal::Hold && current_signal == Signal::Buy;
    state.last_signal = current_signal;

    if let Some(position) = state.open_position.as_mut() {
        position.hold_ticks += 1;

        let exit_reason = if tick.bid <= position.sl_price {
            Some("SL_HIT")
        } else if tick.ask >= position.tp_price {
            Some("TP_HIT")
        } else if position.hold_ticks >= position.max_hold_ticks {
            Some("TIMEOUT")
        } else {
            None
        };

        if let Some(exit_reason) = exit_reason {
            let entry_time = position.entry_time;
            let gross_pips = (tick.bid - position.entry_price) / PIP_SIZE;
            let net_pips = gross_pips - broker_spread_pips - broker_commission;

            state.equity += net_pips;
            state.open_position = None;

            return Some(TradeResult {
                entry_time,
                exit_time: tick.timestamp,
                gross_pips,
                net_pips,
                exit_reason: exit_reason.to_string(),
            });
        }
    }

    if should_enter && state.open_position.is_none() {
        let entry_price = tick.ask;
        state.open_position = Some(Position {
            entry_time: tick.timestamp,
            entry_price,
            sl_price: entry_price - params.sl_pips * PIP_SIZE,
            tp_price: entry_price + params.tp_pips * PIP_SIZE,
            max_hold_ticks: params.feature_window * 10,
            hold_ticks: 0,
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use chrono::{LocalResult, TimeZone};

    use super::*;
    use crate::{data::types::Instrument, gp::ast::Terminal};

    #[test]
    fn enters_on_signal_transition_and_exits_on_take_profit() -> Result<(), String> {
        let ast = AstNode::Leaf(Terminal::Constant(1.0));
        let params = StrategyParams {
            sl_pips: 10.0,
            tp_pips: 5.0,
            signal_threshold: 0.5,
            feature_window: 2,
        };
        let feature = FeatureRow::default();
        let mut state = StrategyState::default();

        let entry_tick = test_tick(0, 1.1000, 1.1002)?;
        let entry_result = process_tick(&mut state, &feature, &entry_tick, &ast, &params, 1.0, 0.5);

        assert!(entry_result.is_none());
        assert!(state.open_position.is_some());
        assert_eq!(state.last_signal, Signal::Buy);

        let exit_tick = test_tick(1, 1.1010, 1.1012)?;
        let trade = process_tick(&mut state, &feature, &exit_tick, &ast, &params, 1.0, 0.5)
            .ok_or_else(|| "expected TP trade result".to_string())?;

        assert_eq!(trade.entry_time, entry_tick.timestamp);
        assert_eq!(trade.exit_time, exit_tick.timestamp);
        assert_eq!(trade.exit_reason, "TP_HIT");
        assert!((trade.gross_pips - 8.0).abs() < 1e-10);
        assert!((trade.net_pips - 6.5).abs() < 1e-10);
        assert!((state.equity - 6.5).abs() < 1e-10);
        assert!(state.open_position.is_none());

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
