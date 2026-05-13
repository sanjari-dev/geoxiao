use crate::gp::ast::{AstNode, Primitive, Terminal};

pub fn genotypic_distance(a: &AstNode, b: &AstNode) -> f64 {
    let labels_a = flatten_ast(a);
    let labels_b = flatten_ast(b);
    let max_len = labels_a.len().max(labels_b.len());

    if max_len == 0 {
        return 0.0;
    }

    let distance = levenshtein_distance(&labels_a, &labels_b);
    (distance as f64 / max_len as f64).clamp(0.0, 1.0)
}

pub fn phenotypic_similarity(trades_a: &[(i64, i64, bool)], trades_b: &[(i64, i64, bool)]) -> f64 {
    if trades_a.is_empty() || trades_b.is_empty() {
        return 0.0;
    }

    let matching_count = trades_a
        .iter()
        .filter(|trade_a| {
            trades_b.iter().any(|trade_b| {
                trade_a.2 == trade_b.2
                    && overlap_ratio((trade_a.0, trade_a.1), (trade_b.0, trade_b.1)) > 0.5
            })
        })
        .count();

    matching_count as f64 / trades_a.len().max(trades_b.len()) as f64
}

fn flatten_ast(tree: &AstNode) -> Vec<String> {
    let mut labels = Vec::new();
    flatten_ast_inner(tree, &mut labels);
    labels
}

fn flatten_ast_inner(tree: &AstNode, labels: &mut Vec<String>) {
    match tree {
        AstNode::Leaf(terminal) => labels.push(terminal_label(terminal).to_string()),
        AstNode::Unary { op, child } => {
            labels.push(primitive_label(op).to_string());
            flatten_ast_inner(child, labels);
        }
        AstNode::Binary { op, left, right } => {
            labels.push(primitive_label(op).to_string());
            flatten_ast_inner(left, labels);
            flatten_ast_inner(right, labels);
        }
    }
}

fn levenshtein_distance(a: &[String], b: &[String]) -> usize {
    let rows = a.len() + 1;
    let cols = b.len() + 1;
    let mut matrix = vec![vec![0; cols]; rows];

    for (row, values) in matrix.iter_mut().enumerate() {
        values[0] = row;
    }

    for col in 0..cols {
        matrix[0][col] = col;
    }

    for row in 1..rows {
        for col in 1..cols {
            let substitution_cost = if a[row - 1] == b[col - 1] { 0 } else { 1 };
            matrix[row][col] = (matrix[row - 1][col] + 1)
                .min(matrix[row][col - 1] + 1)
                .min(matrix[row - 1][col - 1] + substitution_cost);
        }
    }

    matrix[a.len()][b.len()]
}

fn overlap_ratio(a: (i64, i64), b: (i64, i64)) -> f64 {
    let overlap_start = a.0.max(b.0);
    let overlap_end = a.1.min(b.1);

    if overlap_end < overlap_start {
        return 0.0;
    }

    let overlap = overlap_end - overlap_start;
    let union = a.1.max(b.1) - a.0.min(b.0);

    if union <= 0 {
        return 0.0;
    }

    overlap as f64 / union as f64
}

fn primitive_label(primitive: &Primitive) -> &'static str {
    match primitive {
        Primitive::Add => "Add",
        Primitive::Sub => "Sub",
        Primitive::Mul => "Mul",
        Primitive::Div => "Div",
        Primitive::Max2 => "Max2",
        Primitive::Min2 => "Min2",
        Primitive::Neg => "Neg",
        Primitive::Square => "Square",
        Primitive::Cube => "Cube",
        Primitive::Log => "Log",
        Primitive::Sqrt => "Sqrt",
        Primitive::Sigmoid => "Sigmoid",
        Primitive::Sign => "Sign",
    }
}

fn terminal_label(terminal: &Terminal) -> &'static str {
    match terminal {
        Terminal::Obi => "Obi",
        Terminal::TickVelocity => "TickVelocity",
        Terminal::SpreadDynamics => "SpreadDynamics",
        Terminal::TickDensity => "TickDensity",
        Terminal::VolumeClockSkew => "VolumeClockSkew",
        Terminal::MidMomentum => "MidMomentum",
        Terminal::RollingSkewness => "RollingSkewness",
        Terminal::RollingKurtosis => "RollingKurtosis",
        Terminal::VolumeWeightedSpread => "VolumeWeightedSpread",
        Terminal::HurstExponent => "HurstExponent",
        Terminal::Constant(_) => "Constant",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genotypic_distance_handles_identical_slight_and_different_trees() {
        let tree = add_tree(Terminal::TickVelocity, Terminal::Constant(5.0));
        let identical = tree.clone();
        let slightly_different = add_tree(Terminal::TickVelocity, Terminal::Obi);
        let completely_different = AstNode::Unary {
            op: Primitive::Log,
            child: Box::new(AstNode::Leaf(Terminal::HurstExponent)),
        };

        assert_eq!(genotypic_distance(&tree, &identical), 0.0);
        assert!((genotypic_distance(&tree, &slightly_different) - (1.0 / 3.0)).abs() < 1e-12);
        assert!(genotypic_distance(&tree, &completely_different) >= 0.9);
    }

    #[test]
    fn phenotypic_similarity_matches_exact_overlapping_trades() {
        let trades_a = [(0, 10, true), (20, 30, false)];
        let trades_b = [(0, 10, true), (20, 30, false)];

        assert_eq!(phenotypic_similarity(&trades_a, &trades_b), 1.0);
    }

    #[test]
    fn phenotypic_similarity_rejects_non_overlapping_trades() {
        let trades_a = [(0, 10, true)];
        let trades_b = [(11, 20, true), (0, 10, false)];

        assert_eq!(phenotypic_similarity(&trades_a, &trades_b), 0.0);
    }

    #[test]
    fn phenotypic_similarity_counts_partial_overlaps_above_threshold() {
        let trades_a = [(0, 10, true), (20, 30, true)];
        let trades_b = [(2, 9, true), (24, 40, true)];

        assert_eq!(phenotypic_similarity(&trades_a, &trades_b), 0.5);
    }

    fn add_tree(left: Terminal, right: Terminal) -> AstNode {
        AstNode::Binary {
            op: Primitive::Add,
            left: Box::new(AstNode::Leaf(left)),
            right: Box::new(AstNode::Leaf(right)),
        }
    }
}
