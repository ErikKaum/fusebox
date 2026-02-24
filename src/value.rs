use core::fmt;

/// An SSA (static single assignment) id for a value produced in the graph.
///
/// We store an integer, and only turn it into MLIR text ("%0", "%1", ...)
/// when we print the module. This keeps the rest of the code free of string
/// formatting concerns.
/// Instead of storing for example the string %42 everywhere (which is brittle and annoying), we store ValueId(42) and only render it when printing.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

impl ValueId {
    /// Formats this value as an MLIR SSA name, e.g. "%0".
    pub fn mlir_name(self) -> String {
        format!("%{}", self.0)
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // We avoid allocating in Display; printer code can use mlir_name()
        // if it wants a String.
        write!(f, "%{}", self.0)
    }
}
