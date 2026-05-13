use rand::Rng;

use crate::gp::ast::{AstNode, Primitive, Terminal};

const CONSTANT_MIN: f64 = -100.0;
const CONSTANT_MAX: f64 = 100.0;

pub fn grow_tree(max_depth: usize, rng: &mut impl Rng) -> AstNode {
    let depth_limit = max_depth.max(2);

    loop {
        let tree = grow_subtree(depth_limit, true, rng);
        if tree.is_valid() {
            return tree;
        }
    }
}

pub fn crossover(parent_a: &AstNode, parent_b: &AstNode, rng: &mut impl Rng) -> (AstNode, AstNode) {
    let original_a = parent_a.clone();
    let original_b = parent_b.clone();

    let paths_a = collect_paths(parent_a);
    let paths_b = collect_paths(parent_b);

    let path_a = &paths_a[rng.gen_range(0..paths_a.len())];
    let path_b = &paths_b[rng.gen_range(0..paths_b.len())];

    let subtree_a = subtree_at(parent_a, path_a).clone();
    let subtree_b = subtree_at(parent_b, path_b).clone();

    let child_a = replace_subtree(parent_a, path_a, &subtree_b);
    let child_b = replace_subtree(parent_b, path_b, &subtree_a);

    let final_a = if child_a.is_valid() {
        child_a
    } else {
        original_a
    };
    let final_b = if child_b.is_valid() {
        child_b
    } else {
        original_b
    };

    (final_a, final_b)
}

pub fn mutate(tree: &mut AstNode, rng: &mut impl Rng, mutation_rate: f64) {
    let original = tree.clone();
    let bounded_rate = mutation_rate.clamp(0.0, 1.0);

    mutate_subtree(tree, rng, bounded_rate);

    if !tree.is_valid() {
        *tree = original;
    }
}

fn grow_subtree(remaining_depth: usize, force_internal: bool, rng: &mut impl Rng) -> AstNode {
    if remaining_depth <= 1 {
        return AstNode::Leaf(random_terminal(rng));
    }

    if !force_internal && rng.gen_bool(0.3) {
        return AstNode::Leaf(random_terminal(rng));
    }

    if rng.gen_bool(0.5) {
        AstNode::Unary {
            op: random_unary_primitive(rng),
            child: Box::new(grow_subtree(remaining_depth - 1, false, rng)),
        }
    } else {
        AstNode::Binary {
            op: random_binary_primitive(rng),
            left: Box::new(grow_subtree(remaining_depth - 1, false, rng)),
            right: Box::new(grow_subtree(remaining_depth - 1, false, rng)),
        }
    }
}

fn mutate_subtree(tree: &mut AstNode, rng: &mut impl Rng, mutation_rate: f64) {
    match tree {
        AstNode::Leaf(terminal) => {
            if rng.gen_bool(mutation_rate) {
                *terminal = random_terminal(rng);
            }
        }
        AstNode::Unary { op, child } => {
            if rng.gen_bool(mutation_rate) {
                *op = random_unary_primitive(rng);
            }
            if rng.gen_bool(mutation_rate) {
                **child = grow_subtree(child.depth().max(1), false, rng);
            } else {
                mutate_subtree(child, rng, mutation_rate);
            }
        }
        AstNode::Binary { op, left, right } => {
            if rng.gen_bool(mutation_rate) {
                *op = random_binary_primitive(rng);
            }
            if rng.gen_bool(mutation_rate) {
                **left = grow_subtree(left.depth().max(1), false, rng);
            } else {
                mutate_subtree(left, rng, mutation_rate);
            }
            if rng.gen_bool(mutation_rate) {
                **right = grow_subtree(right.depth().max(1), false, rng);
            } else {
                mutate_subtree(right, rng, mutation_rate);
            }
        }
    }
}

fn random_terminal(rng: &mut impl Rng) -> Terminal {
    match rng.gen_range(0..11) {
        0 => Terminal::Obi,
        1 => Terminal::TickVelocity,
        2 => Terminal::SpreadDynamics,
        3 => Terminal::TickDensity,
        4 => Terminal::VolumeClockSkew,
        5 => Terminal::MidMomentum,
        6 => Terminal::RollingSkewness,
        7 => Terminal::RollingKurtosis,
        8 => Terminal::VolumeWeightedSpread,
        9 => Terminal::HurstExponent,
        _ => Terminal::Constant(clamp_constant(rng.gen_range(CONSTANT_MIN..=CONSTANT_MAX))),
    }
}

fn random_unary_primitive(rng: &mut impl Rng) -> Primitive {
    match rng.gen_range(0..7) {
        0 => Primitive::Neg,
        1 => Primitive::Square,
        2 => Primitive::Cube,
        3 => Primitive::Log,
        4 => Primitive::Sqrt,
        5 => Primitive::Sigmoid,
        _ => Primitive::Sign,
    }
}

fn random_binary_primitive(rng: &mut impl Rng) -> Primitive {
    match rng.gen_range(0..6) {
        0 => Primitive::Add,
        1 => Primitive::Sub,
        2 => Primitive::Mul,
        3 => Primitive::Div,
        4 => Primitive::Max2,
        _ => Primitive::Min2,
    }
}

fn clamp_constant(value: f64) -> f64 {
    value.clamp(CONSTANT_MIN, CONSTANT_MAX)
}

fn collect_paths(tree: &AstNode) -> Vec<Vec<usize>> {
    let mut paths = Vec::new();
    let mut current = Vec::new();
    collect_paths_inner(tree, &mut current, &mut paths);
    paths
}

fn collect_paths_inner(tree: &AstNode, current: &mut Vec<usize>, paths: &mut Vec<Vec<usize>>) {
    paths.push(current.clone());

    match tree {
        AstNode::Leaf(_) => {}
        AstNode::Unary { child, .. } => {
            current.push(0);
            collect_paths_inner(child, current, paths);
            current.pop();
        }
        AstNode::Binary { left, right, .. } => {
            current.push(0);
            collect_paths_inner(left, current, paths);
            current.pop();

            current.push(1);
            collect_paths_inner(right, current, paths);
            current.pop();
        }
    }
}

fn subtree_at<'a>(tree: &'a AstNode, path: &[usize]) -> &'a AstNode {
    if path.is_empty() {
        return tree;
    }

    match tree {
        AstNode::Leaf(_) => tree,
        AstNode::Unary { child, .. } => subtree_at(child, &path[1..]),
        AstNode::Binary { left, right, .. } => match path[0] {
            0 => subtree_at(left, &path[1..]),
            _ => subtree_at(right, &path[1..]),
        },
    }
}

fn replace_subtree(tree: &AstNode, path: &[usize], replacement: &AstNode) -> AstNode {
    if path.is_empty() {
        return replacement.clone();
    }

    match tree {
        AstNode::Leaf(_) => tree.clone(),
        AstNode::Unary { op, child } => AstNode::Unary {
            op: op.clone(),
            child: Box::new(replace_subtree(child, &path[1..], replacement)),
        },
        AstNode::Binary { op, left, right } => match path[0] {
            0 => AstNode::Binary {
                op: op.clone(),
                left: Box::new(replace_subtree(left, &path[1..], replacement)),
                right: right.clone(),
            },
            _ => AstNode::Binary {
                op: op.clone(),
                left: left.clone(),
                right: Box::new(replace_subtree(right, &path[1..], replacement)),
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    #[test]
    fn grow_tree_always_returns_valid_tree() {
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..1000 {
            let tree = grow_tree(6, &mut rng);
            assert!(tree.is_valid());
        }
    }

    #[test]
    fn mutate_preserves_validity() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut tree = grow_tree(6, &mut rng);

        mutate(&mut tree, &mut rng, 0.5);

        assert!(tree.is_valid());
    }

    #[test]
    fn crossover_preserves_validity() {
        let mut rng = StdRng::seed_from_u64(99);
        let parent_a = grow_tree(6, &mut rng);
        let parent_b = grow_tree(6, &mut rng);

        let (child_a, child_b) = crossover(&parent_a, &parent_b, &mut rng);

        assert!(child_a.is_valid());
        assert!(child_b.is_valid());
    }
}
