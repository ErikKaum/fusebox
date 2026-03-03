use core::fmt;

use serde::{Deserialize, Serialize};

use crate::dtype::DType;

/// Tensor shape: dimension sizes + element type.
///
/// Uses `i64` dims to match MLIR's 64-bit dimension representation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    /// Dimension sizes. Empty = rank-0 tensor.
    ///
    /// We use i64 because MLIR tensor dimensions are represented as 64-bit integers.
    pub dims: Vec<i64>,
    pub dtype: DType,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<i64>>, dtype: DType) -> Self {
        Self {
            dims: dims.into(),
            dtype,
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dim(&self, axis: usize) -> i64 {
        self.dims[axis]
    }

    /// Returns the MLIR tensor type spelling, e.g.:
    /// - rank 0: tensor<f32>
    /// - rank 2: tensor<32x768xf32>
    pub fn mlir_tensor_type(&self) -> String {
        if self.dims.is_empty() {
            return format!("tensor<{}>", self.dtype.mlir_str());
        }

        let mut s = String::from("tensor<");
        for (i, d) in self.dims.iter().copied().enumerate() {
            if i > 0 {
                s.push('x');
            }
            s.push_str(&d.to_string());
        }
        s.push('x');
        s.push_str(self.dtype.mlir_str());
        s.push('>');
        s
    }
}

impl fmt::Display for Shape {
    /// For now, Display just prints the MLIR tensor type form.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.mlir_tensor_type())
    }
}
