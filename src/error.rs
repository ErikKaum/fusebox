// defines a small Error enum for shape/dtype mismatches and “unsupported for this exercise” cases.

use core::fmt;

use crate::{dtype::DType, shape::Shape};

/// Errors from building/typing our tiny StableHLO graph.
#[derive(Debug, Clone)]
pub enum Error {
    /// Expected a tensor of a specific rank.
    RankMismatch {
        op: &'static str,
        expected: usize,
        got: usize,
    },
    /// Dimension mismatch, e.g. matmul inner dims don't match.
    DimMismatch {
        op: &'static str,
        axis_a: usize,
        dim_a: i64,
        axis_b: usize,
        dim_b: i64,
    },
    /// Dtype mismatch (for now enforce equal dtypes).
    DTypeMismatch {
        op: &'static str,
        a: DType,
        b: DType,
    },
    /// Shapes must match exactly (keep add strict for now).
    ShapeMismatch {
        op: &'static str,
        a: Shape,
        b: Shape,
    },
    /// Mismatch between safetensor file and module struct
    MissingWeight {
        key: String,
    },
    UnsupportedDType {
        key: String,
        dtype: String,
    },
    /// A feature intentionally not implement in yet.
    Unsupported {
        op: &'static str,
        msg: &'static str,
    },
}

impl Error {
    pub(crate) fn rank(op: &'static str, expected: usize, got: usize) -> Self {
        Error::RankMismatch { op, expected, got }
    }

    pub(crate) fn dtype(op: &'static str, a: DType, b: DType) -> Self {
        Error::DTypeMismatch { op, a, b }
    }

    pub(crate) fn shape(op: &'static str, a: &Shape, b: &Shape) -> Self {
        Error::ShapeMismatch {
            op,
            a: a.clone(),
            b: b.clone(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::RankMismatch { op, expected, got } => {
                write!(f, "{op}: expected rank {expected}, got {got}")
            }
            Error::DimMismatch {
                op,
                axis_a,
                dim_a,
                axis_b,
                dim_b,
            } => {
                write!(
                    f,
                    "{op}: dimension mismatch (a[axis {axis_a}]={dim_a} vs b[axis {axis_b}]={dim_b})"
                )
            }
            Error::DTypeMismatch { op, a, b } => {
                write!(f, "{op}: dtype mismatch ({a} vs {b})")
            }
            Error::ShapeMismatch { op, a, b } => {
                write!(f, "{op}: shape mismatch ({a} vs {b})")
            }
            Error::MissingWeight { key } => {
                write!(f, "missing weight {:?}", key)
            }
            Error::UnsupportedDType { key, dtype } => {
                write!(f, "unsupported dtype {} for {:?}", dtype, key)
            }
            Error::Unsupported { op, msg } => {
                write!(f, "{op}: unsupported ({msg})")
            }
        }
    }
}

impl std::error::Error for Error {}
