pub fn compute_hurst(prices: &[f64]) -> f64 {
    let n = prices.len();
    if n < 20 {
        return 0.5;
    }

    let returns: Vec<f64> = prices
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect();
    let sizes: Vec<usize> = (2..=n / 2).filter(|size| size.is_power_of_two()).collect();

    if sizes.len() < 2 {
        return 0.5;
    }

    let mut log_sizes = Vec::new();
    let mut log_rs = Vec::new();

    for size in &sizes {
        if let Some(rs) = rescaled_range(&returns[..returns.len().min(*size)]) {
            log_sizes.push((*size as f64).ln());
            log_rs.push(rs.ln());
        }
    }

    if log_sizes.len() < 2 {
        return 0.5;
    }

    ols_slope(&log_sizes, &log_rs).clamp(0.0, 1.0)
}

fn rescaled_range(data: &[f64]) -> Option<f64> {
    let n = data.len();
    if n == 0 {
        return None;
    }

    let mean = data.iter().sum::<f64>() / n as f64;
    let deviations: Vec<f64> = data.iter().map(|&value| value - mean).collect();
    let cumulative: Vec<f64> = deviations
        .iter()
        .scan(0.0, |accumulator, &value| {
            *accumulator += value;
            Some(*accumulator)
        })
        .collect();

    let range = cumulative.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - cumulative.iter().cloned().fold(f64::INFINITY, f64::min);
    let std_dev = (data
        .iter()
        .map(|&value| (value - mean).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();

    if std_dev < 1e-10 {
        return None;
    }

    Some(range / std_dev)
}

fn ols_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sxx: f64 = x.iter().map(|a| a * a).sum();

    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_random_noise_for_insufficient_data() {
        let prices = vec![1.0; 19];

        assert_eq!(compute_hurst(&prices), 0.5);
    }

    #[test]
    fn computes_known_rescaled_range() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let rs = rescaled_range(&data);

        assert!(matches!(rs, Some(value) if (value - 1.7888543819998317).abs() < 1e-12));
    }

    #[test]
    fn computes_known_ols_slope() {
        let x = [1.0, 2.0, 3.0];
        let y = [2.0, 4.0, 6.0];

        assert!((ols_slope(&x, &y) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn constant_returns_default_to_random_noise() {
        let prices: Vec<f64> = (0..32).map(|index| index as f64).collect();

        assert_eq!(compute_hurst(&prices), 0.5);
    }
}
