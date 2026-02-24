use core::fmt;

use crate::{dtype::DType, shape::Shape, value::ValueId};

/// A symbolic tensor: shape metadata + an SSA value id in the graph.
///
/// This is NOT a buffer and does not contain any data.
/// Think: "a reference to the result of some operation".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub shape: Shape,
    pub value: ValueId,
}

impl Tensor {
    pub fn new(shape: Shape, value: ValueId) -> Self {
        Self { shape, value }
    }

    pub fn dtype(&self) -> DType {
        self.shape.dtype
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }
}

impl fmt::Display for Tensor {
    /// Helpful for debugging: prints like "%3 : tensor<32x768xf32>"
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} : {}", self.value, self.shape)
    }
}
