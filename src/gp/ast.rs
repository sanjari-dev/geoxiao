use serde::{Deserialize, Serialize};

use crate::features::FeatureRow;

pub const MAX_DEPTH: usize = 6;
pub const MAX_NODES: usize = 60;
pub const MIN_NODES: usize = 2;

const PROTECTED_EPSILON: f64 = 1e-10;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Terminal {
    Obi,
    TickVelocity,
    SpreadDynamics,
    TickDensity,
    VolumeClockSkew,
    MidMomentum,
    RollingSkewness,
    RollingKurtosis,
    VolumeWeightedSpread,
    HurstExponent,
    Constant(f64),
}

impl Terminal {
    #[inline]
    pub fn evaluate(&self, f: &FeatureRow) -> f64 {
        match self {
            Self::Obi => f.obi,
            Self::TickVelocity => f.tick_velocity,
            Self::SpreadDynamics => f.spread_dynamics,
            Self::TickDensity => f.tick_density,
            Self::VolumeClockSkew => f.volume_clock_skew,
            Self::MidMomentum => f.mid_momentum,
            Self::RollingSkewness => f.rolling_skewness,
            Self::RollingKurtosis => f.rolling_kurtosis,
            Self::VolumeWeightedSpread => f.volume_weighted_spread,
            Self::HurstExponent => f.hurst_exponent,
            Self::Constant(value) => *value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Primitive {
    Add,
    Sub,
    Mul,
    Div,
    Max2,
    Min2,
    Neg,
    Square,
    Cube,
    Log,
    Sqrt,
    Sigmoid,
    Sign,
}

impl Primitive {
    #[inline]
    pub fn apply_unary(&self, v: f64) -> f64 {
        match self {
            Self::Neg => -v,
            Self::Square => v * v,
            Self::Cube => v * v * v,
            Self::Log => {
                if v <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    v.ln()
                }
            }
            Self::Sqrt => {
                if v < 0.0 {
                    0.0
                } else {
                    v.sqrt()
                }
            }
            Self::Sigmoid => 1.0 / (1.0 + (-v).exp()),
            Self::Sign => v.signum(),
            _ => v,
        }
    }

    #[inline]
    pub fn apply_binary(&self, l: f64, r: f64) -> f64 {
        match self {
            Self::Add => l + r,
            Self::Sub => l - r,
            Self::Mul => l * r,
            Self::Div => {
                if r.abs() < PROTECTED_EPSILON {
                    0.0
                } else {
                    l / r
                }
            }
            Self::Max2 => l.max(r),
            Self::Min2 => l.min(r),
            _ => 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AstNode {
    Leaf(Terminal),
    Unary {
        op: Primitive,
        child: Box<AstNode>,
    },
    Binary {
        op: Primitive,
        left: Box<AstNode>,
        right: Box<AstNode>,
    },
}

impl AstNode {
    #[inline]
    pub fn evaluate(&self, features: &FeatureRow) -> f64 {
        match self {
            Self::Leaf(terminal) => terminal.evaluate(features),
            Self::Unary { op, child } => op.apply_unary(child.evaluate(features)),
            Self::Binary { op, left, right } => {
                op.apply_binary(left.evaluate(features), right.evaluate(features))
            }
        }
    }

    pub fn node_count(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Unary { child, .. } => 1 + child.node_count(),
            Self::Binary { left, right, .. } => 1 + left.node_count() + right.node_count(),
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Unary { child, .. } => 1 + child.depth(),
            Self::Binary { left, right, .. } => 1 + left.depth().max(right.depth()),
        }
    }

    pub fn is_valid(&self) -> bool {
        let node_count = self.node_count();
        node_count >= MIN_NODES && node_count <= MAX_NODES && self.depth() <= MAX_DEPTH
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_manual_add_tree() {
        let ast = AstNode::Binary {
            op: Primitive::Add,
            left: Box::new(AstNode::Leaf(Terminal::TickVelocity)),
            right: Box::new(AstNode::Leaf(Terminal::Constant(5.0))),
        };
        let features = FeatureRow {
            tick_velocity: 2.5,
            ..FeatureRow::default()
        };

        assert!((ast.evaluate(&features) - 7.5).abs() < f64::EPSILON);
        assert_eq!(ast.node_count(), 3);
        assert_eq!(ast.depth(), 2);
        assert!(ast.is_valid());
    }

    #[test]
    fn uses_protected_math() {
        assert_eq!(Primitive::Div.apply_binary(10.0, 0.0), 0.0);
        assert_eq!(Primitive::Log.apply_unary(0.0), f64::NEG_INFINITY);
        assert_eq!(Primitive::Sqrt.apply_unary(-1.0), 0.0);
    }
}
