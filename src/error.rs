use core::fmt;

use crate::{dtype::DType, shape::Shape};

#[derive(Debug, Clone)]
pub enum Error {
    RankMismatch {
        op: &'static str,
        expected: usize,
        got: usize,
    },
    DimMismatch {
        op: &'static str,
        axis_a: usize,
        dim_a: i64,
        axis_b: usize,
        dim_b: i64,
    },
    DTypeMismatch {
        op: &'static str,
        a: DType,
        b: DType,
    },
    ShapeMismatch {
        op: &'static str,
        a: Shape,
        b: Shape,
    },
    BroadcastError {
        a: Shape,
        b: Shape,
    },
    MissingWeight {
        key: String,
    },
    UnsupportedDType {
        key: String,
        dtype: String,
    },
    Unsupported {
        op: &'static str,
        msg: &'static str,
    },
    GraphMismatch,
    InvalidParam {
        msg: String,
    },
    RuntimeError(String),
    CompilationError(String),
    ValidationError(String),
    Scoped {
        scope: String,
        inner: Box<Error>,
    },
}

impl Error {
    pub(crate) fn rank(op: &'static str, expected: usize, got: usize) -> Self {
        Error::RankMismatch { op, expected, got }
    }

    pub(crate) fn dtype(op: &'static str, a: DType, b: DType) -> Self {
        Error::DTypeMismatch { op, a, b }
    }

    #[allow(dead_code)]
    pub(crate) fn shape(op: &'static str, a: &Shape, b: &Shape) -> Self {
        Error::ShapeMismatch {
            op,
            a: a.clone(),
            b: b.clone(),
        }
    }

    pub fn with_scope(self, scope: impl Into<String>) -> Self {
        let scope = scope.into();
        if scope.is_empty() {
            return self;
        }
        Error::Scoped {
            scope,
            inner: Box::new(self),
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
            Error::BroadcastError { a, b } => {
                write!(f, "cannot broadcast shapes {a} and {b}")
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
            Error::GraphMismatch => {
                write!(
                    f,
                    "cannot combine tensors from different computation graphs"
                )
            }
            Error::InvalidParam { msg } => {
                write!(f, "invalid parameter: {msg}")
            }
            Error::RuntimeError(msg) => write!(f, "runtime error: {msg}"),
            Error::CompilationError(msg) => write!(f, "compilation error: {msg}"),
            Error::ValidationError(msg) => write!(f, "validation error: {msg}"),
            Error::Scoped { scope, inner } => {
                write!(f, "in {scope}: {inner}")
            }
        }
    }
}

impl std::error::Error for Error {}
