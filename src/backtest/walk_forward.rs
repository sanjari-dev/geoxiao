#[derive(Debug, Clone)]
pub struct WalkForwardWindow {
    pub window_id: usize,
    pub is_months: Vec<(i32, u32)>,
    pub oos_months: Vec<(i32, u32)>,
}

pub fn generate_windows(
    start_year: i32,
    start_month: u32,
    total_months: usize,
    is_size: usize,
    oos_size: usize,
    step: usize,
) -> Vec<WalkForwardWindow> {
    if start_month == 0 || start_month > 12 || is_size == 0 || oos_size == 0 || step == 0 {
        return Vec::new();
    }

    let mut windows = Vec::new();
    let mut offset = 0;

    while offset + is_size + oos_size <= total_months {
        let is_months = (offset..offset + is_size)
            .map(|month_offset| add_months(start_year, start_month, month_offset))
            .collect();
        let oos_months = (offset + is_size..offset + is_size + oos_size)
            .map(|month_offset| add_months(start_year, start_month, month_offset))
            .collect();

        windows.push(WalkForwardWindow {
            window_id: windows.len(),
            is_months,
            oos_months,
        });

        offset += step;
    }

    windows
}

fn add_months(start_year: i32, start_month: u32, offset: usize) -> (i32, u32) {
    let zero_based_start_month = start_month - 1;
    let absolute_month = zero_based_start_month as usize + offset;
    let year = start_year + (absolute_month / 12) as i32;
    let month = (absolute_month % 12) as u32 + 1;

    (year, month)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_expected_walk_forward_windows() {
        let windows = generate_windows(2026, 1, 10, 6, 2, 2);

        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].window_id, 0);
        assert_eq!(
            windows[0].is_months,
            vec![
                (2026, 1),
                (2026, 2),
                (2026, 3),
                (2026, 4),
                (2026, 5),
                (2026, 6)
            ]
        );
        assert_eq!(windows[0].oos_months, vec![(2026, 7), (2026, 8)]);

        assert_eq!(windows[1].window_id, 1);
        assert_eq!(
            windows[1].is_months,
            vec![
                (2026, 3),
                (2026, 4),
                (2026, 5),
                (2026, 6),
                (2026, 7),
                (2026, 8)
            ]
        );
        assert_eq!(windows[1].oos_months, vec![(2026, 9), (2026, 10)]);
    }

    #[test]
    fn wraps_months_across_year_boundaries() {
        let windows = generate_windows(2026, 11, 4, 2, 1, 1);

        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].is_months, vec![(2026, 11), (2026, 12)]);
        assert_eq!(windows[0].oos_months, vec![(2027, 1)]);
        assert_eq!(windows[1].is_months, vec![(2026, 12), (2027, 1)]);
        assert_eq!(windows[1].oos_months, vec![(2027, 2)]);
    }
}
