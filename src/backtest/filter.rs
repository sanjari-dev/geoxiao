#[derive(Debug, Clone)]
pub struct WindowMetrics {
    pub trade_count: usize,
    pub avg_sl_pips: f64,
    pub max_drawdown_pips: f64,
    pub loss_count: usize,
}

pub fn passes_hard_filters(metrics: &WindowMetrics) -> bool {
    if metrics.trade_count < 10 || metrics.trade_count > 20 {
        return false;
    }

    if metrics.avg_sl_pips < 10.0 || metrics.avg_sl_pips > 50.0 {
        return false;
    }

    if metrics.max_drawdown_pips > 500.0 {
        return false;
    }

    if metrics.loss_count == 0 {
        return false;
    }

    true
}

pub fn passes_degradation_filter(oos_results: &[f64]) -> bool {
    if oos_results.is_empty() {
        return false;
    }

    let fail_count = oos_results
        .iter()
        .filter(|profit_factor| **profit_factor < 1.5)
        .count();
    let fail_ratio = fail_count as f64 / oos_results.len() as f64;

    fail_ratio < 0.80
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_filters_accept_valid_metrics() {
        let metrics = WindowMetrics {
            trade_count: 15,
            avg_sl_pips: 25.0,
            max_drawdown_pips: 250.0,
            loss_count: 3,
        };

        assert!(passes_hard_filters(&metrics));
    }

    #[test]
    fn hard_filters_reject_invalid_metrics() {
        let valid = WindowMetrics {
            trade_count: 15,
            avg_sl_pips: 25.0,
            max_drawdown_pips: 250.0,
            loss_count: 3,
        };

        let mut too_few_trades = valid.clone();
        too_few_trades.trade_count = 9;
        assert!(!passes_hard_filters(&too_few_trades));

        let mut too_many_trades = valid.clone();
        too_many_trades.trade_count = 21;
        assert!(!passes_hard_filters(&too_many_trades));

        let mut stop_loss_too_small = valid.clone();
        stop_loss_too_small.avg_sl_pips = 9.99;
        assert!(!passes_hard_filters(&stop_loss_too_small));

        let mut stop_loss_too_large = valid.clone();
        stop_loss_too_large.avg_sl_pips = 50.01;
        assert!(!passes_hard_filters(&stop_loss_too_large));

        let mut drawdown_too_large = valid.clone();
        drawdown_too_large.max_drawdown_pips = 500.01;
        assert!(!passes_hard_filters(&drawdown_too_large));

        let mut no_losses = valid;
        no_losses.loss_count = 0;
        assert!(!passes_hard_filters(&no_losses));
    }

    #[test]
    fn degradation_filter_checks_oos_profit_factor_fail_ratio() {
        assert!(!passes_degradation_filter(&[]));
        assert!(passes_degradation_filter(&[1.4, 1.6, 1.7, 1.8]));
        assert!(!passes_degradation_filter(&[1.4, 1.3, 1.2, 1.1, 1.8]));
    }
}
