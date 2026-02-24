// defines the minimal set of dtypes we support (start with F32 only, but leave the enum open).

use core::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    BF16,
    F32,
}

impl DType {
    pub fn mlir_str(self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.mlir_str())
    }
}
