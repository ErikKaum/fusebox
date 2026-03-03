// src/module_api.rs
use crate::error::Error;
use crate::shape::Shape;
use crate::trace::TraceCx;

/// Something that can answer: “what is the shape of weight `full_name`?”
///
/// This is intentionally shape-only, no data.
/// In practice it will be backed by a safetensors header map.
pub trait ShapeProvider {
    fn shape_of(&self, full_name: &str) -> Result<Option<Shape>, Error>;

    /// Returns true if any known weight key starts with `prefix`.
    /// Used by the `Module` derive to discover `Vec<T>` layer counts.
    fn has_prefix(&self, prefix: &str) -> bool {
        let _ = prefix;
        false
    }
}

/// Implemented by structs that can allocate their weights as graph parameters.
///
/// The derive macro will generate this for you.
/// You keep `forward` as an inherent method (exactly like your ideal API).
pub trait Module: Sized {
    fn trace(cx: &mut TraceCx, name: &str, shapes: &dyn ShapeProvider) -> Result<Self, Error>;
}
