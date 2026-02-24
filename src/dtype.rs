// defines the minimal set of dtypes we support (start with F32 only, but leave the enum open).

use core::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
}

impl DType {
    /// How this dtype is spelled in MLIR/StableHLO textual form.
    pub fn mlir_str(self) -> &'static str {
        match self {
            DType::F32 => "f32",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.mlir_str())
    }
}
