#[derive(Debug, Clone)]
pub struct FitnessComponents {
    pub avg_oos_profit_factor: f64,
    pub oos_regime_stability: f64,
    pub risk_drawdown_penalty: f64,
    pub is_average_performance: f64,
}

impl FitnessComponents {
    pub fn compute_score(&self) -> f64 {
        (0.40 * self.avg_oos_profit_factor) + (0.25 * self.oos_regime_stability)
            - (0.20 * self.risk_drawdown_penalty)
            + (0.15 * self.is_average_performance)
    }
}

pub fn ulcer_index(equity_curve: &[f64]) -> f64 {
    let Some((&first_equity, rest)) = equity_curve.split_first() else {
        return 0.0;
    };

    let mut peak = first_equity;
    let mut sum_sq = drawdown_squared(first_equity, peak);

    for &equity in rest {
        if equity > peak {
            peak = equity;
        }

        sum_sq += drawdown_squared(equity, peak);
    }

    (sum_sq / equity_curve.len() as f64).sqrt()
}

fn drawdown_squared(equity: f64, peak: f64) -> f64 {
    let drawdown = (peak - equity) / peak.abs().max(1e-10) * 100.0;
    drawdown * drawdown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_weighted_fitness_score() {
        let components = FitnessComponents {
            avg_oos_profit_factor: 2.0,
            oos_regime_stability: 0.8,
            risk_drawdown_penalty: 0.5,
            is_average_performance: 1.0,
        };

        assert!((components.compute_score() - 1.05).abs() < f64::EPSILON);
    }

    #[test]
    fn computes_ulcer_index_for_known_equity_curve() {
        let equity_curve = [100.0, 90.0, 80.0, 95.0, 110.0];
        let index = ulcer_index(&equity_curve);

        assert!((index - 10.246950765959598).abs() < 1e-12);
        assert_eq!(ulcer_index(&[]), 0.0);
    }
}
