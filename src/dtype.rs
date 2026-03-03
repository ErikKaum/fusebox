use core::fmt;

use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F16,
    BF16,
    F32,
    I32,
    Bool,
}

impl DType {
    pub fn mlir_str(self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::I32 => "i32",
            DType::Bool => "i1",
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32)
    }

    pub fn is_integer(self) -> bool {
        matches!(self, DType::I32)
    }

    pub fn byte_size(self) -> usize {
        match self {
            DType::F16 | DType::BF16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::Bool => 1,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.mlir_str())
    }
}
